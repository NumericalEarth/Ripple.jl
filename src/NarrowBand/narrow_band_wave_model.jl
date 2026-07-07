import Oceananigans
import Oceananigans: AbstractModel, fields, prognostic_fields
import Oceananigans.Architectures: architecture
import Oceananigans.Advection: WENO, materialize_advection
import Oceananigans.Grids: topology, Flat, halo_size
import Oceananigans.Fields: CenterField, interior
import Oceananigans.TimeSteppers: Clock

#####
##### NarrowBandWaveModel — the Onuki & Fujiwara (2026) amplitude model
#####
##### Prognoses the reconstituted amplitude G ≡ [1 + α(∇ₕ² + κ²)]A as a real
##### pair (Gʳ, Gⁱ) and diagnoses A = M⁻¹G as a real pair (Aʳ, Aⁱ) once per RK3
##### stage. Storing G rather than A makes the dispersive term local; the mass
##### matrix never appears in the stepper (see docs/design/narrow_band_model_plan.md,
##### §3.2). Phase 1 supports the wave-only regime (`velocities = nothing`).

mutable struct NarrowBandWaveModel{Arch, G, D, S, Adv, Vel, F, C} <: AbstractModel{Nothing, Arch}
    grid :: G                    # (Periodic, Periodic, Flat) wave grid (= solver grid)
    dispersion :: D              # NarrowBandDispersion carrier coefficients
    helmholtz_solver :: S        # diagnostic A = M⁻¹G solver
    advection :: Adv             # Doppler-transport scheme (used from Phase 2)
    velocities :: Vel            # nothing (wave-only) in Phase 1
    Gr :: F                      # prognostic reconstituted amplitude, real part
    Gi :: F                      # prognostic reconstituted amplitude, imag part
    Ar :: F                      # diagnostic amplitude, real part
    Ai :: F                      # diagnostic amplitude, imag part
    Gr_tendency :: F
    Gi_tendency :: F
    G0r :: F                     # RK3 stage-0 scratch
    G0i :: F
    timestepper :: Symbol
    clock :: C
end

# Water depth for the carrier dispersion. An explicit `depth` wins; otherwise it
# is read from a vertically resolved grid (flat bottom, rigid lid at z = 0), and
# a Flat-vertical grid requires the user to name it.
function narrow_band_depth(depth, grid)
    depth === nothing || return depth
    topology(grid, 3) === Flat &&
        throw(ArgumentError("the grid has a Flat vertical topology; pass `depth` " *
                            "(a positive number or InfiniteDepth()) to set the carrier dispersion"))
    return grid.Lz
end

"""
    NarrowBandWaveModel(grid; κ, depth=nothing, gravity=9.81,
                        velocities=nothing, advection=nothing, timestepper=:RK3,
                        clock=Clock(time=0.0))

Construct a narrow-band amplitude wave model on the horizontal footprint of
`grid`. The carrier wavenumber `κ` is required; the frequency `ω`, group
velocity, reconstitution parameter `α`, and vertical structure follow from `κ`
and `depth`. When `depth` is omitted it is taken from a vertically resolved
`grid` (its vertical extent); pass `InfiniteDepth()` for deep water.

The prognostic fields are the reconstituted-amplitude pair `(Gr, Gi)`; the
amplitude pair `(Ar, Ai)` is diagnosed each stage by a screened-Poisson solve.
Only `velocities = nothing` (the wave-only regime) is supported so far.
"""
function NarrowBandWaveModel(grid;
                             κ=nothing,
                             depth=nothing,
                             gravity=9.81,
                             velocities=nothing,
                             advection=WENO(),
                             timestepper=:RK3,
                             clock=Clock(time=0.0))
    κ === nothing && throw(ArgumentError("NarrowBandWaveModel requires a carrier wavenumber `κ`"))
    timestepper === :RK3 ||
        throw(ArgumentError("NarrowBandWaveModel currently supports only `timestepper = :RK3`; got $timestepper"))

    depth = narrow_band_depth(depth, grid)
    dispersion = NarrowBandDispersion(κ, depth; gravity)
    solver = NarrowBandHelmholtzSolver(grid, dispersion)
    wave_grid = solver.grid

    # Resolve deferred advection settings (e.g. WENO's per-backend weight
    # computation) for the wave grid, exactly as an Oceananigans model would.
    advection = materialize_advection(advection, wave_grid)

    coupling = nothing
    if velocities !== nothing
        topology(grid, 3) === Flat &&
            throw(ArgumentError("prescribed `velocities` require a vertically resolved grid " *
                                "for the depth-weighted projection; the given grid is Flat in z"))
        Hx, Hy, _ = halo_size(wave_grid)
        (Hx ≥ 3 && Hy ≥ 3) ||
            throw(ArgumentError("prescribed `velocities` need a horizontal halo of at least 3 " *
                                "(WENO transport + the refraction stencil); got halo $(halo_size(wave_grid)[1:2])"))
        coupling = build_narrow_band_velocities(velocities, grid, wave_grid, dispersion)
    end

    Gr = CenterField(wave_grid); Gi = CenterField(wave_grid)
    Ar = CenterField(wave_grid); Ai = CenterField(wave_grid)
    Gr_tendency = CenterField(wave_grid); Gi_tendency = CenterField(wave_grid)
    G0r = CenterField(wave_grid); G0i = CenterField(wave_grid)

    Arch = typeof(architecture(wave_grid))
    return NarrowBandWaveModel{Arch, typeof(wave_grid), typeof(dispersion), typeof(solver),
                               typeof(advection), typeof(coupling), typeof(Gr), typeof(clock)}(
        wave_grid, dispersion, solver, advection, coupling,
        Gr, Gi, Ar, Ai, Gr_tendency, Gi_tendency, G0r, G0i, timestepper, clock)
end

fields(model::NarrowBandWaveModel) = (Gr=model.Gr, Gi=model.Gi, Ar=model.Ar, Ai=model.Ai)
prognostic_fields(model::NarrowBandWaveModel) = (Gr=model.Gr, Gi=model.Gi)
Base.eltype(model::NarrowBandWaveModel) = eltype(model.grid)
architecture(model::NarrowBandWaveModel) = architecture(model.grid)

"""
    amplitude(model::NarrowBandWaveModel)

Return the complex amplitude `A = Aʳ + i Aⁱ` over the model interior, assembled
from the two real diagnostic fields. Call after `update_state!` (or a step) so
the diagnostic `A` is current.
"""
amplitude(model::NarrowBandWaveModel) = interior(model.Ar) .+ im .* interior(model.Ai)

"""
    reconstituted_amplitude(model::NarrowBandWaveModel)

Return the complex reconstituted amplitude `G = Gʳ + i Gⁱ` over the model
interior — the prognostic state (all that is needed to restart the model).
"""
reconstituted_amplitude(model::NarrowBandWaveModel) = interior(model.Gr) .+ im .* interior(model.Gi)

function Base.show(io::IO, model::NarrowBandWaveModel)
    Nx, Ny, _ = size(model.grid)
    print(io, "NarrowBandWaveModel on ", string(typeof(architecture(model))), ":", '\n',
              "├── grid: ", Nx, "×", Ny, " (", summary(model.grid), ")", '\n',
              "├── ", summary(model.dispersion), '\n',
              "├── velocities: ", model.velocities === nothing ? "nothing (wave-only)" : summary(model.velocities), '\n',
              "└── timestepper: ", model.timestepper)
end
