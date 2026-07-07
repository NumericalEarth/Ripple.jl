import Oceananigans.Operators: ∇²ᶜᶜᶜ
import Oceananigans.Utils: launch!
import Oceananigans.Architectures: architecture
import Oceananigans.BoundaryConditions: fill_halo_regions!
import KernelAbstractions: @kernel, @index

#####
##### Wave-side operators for the narrow-band amplitude model
#####
##### The prognostic field is the reconstituted amplitude G ≡ [1 + α(∇ₕ² + κ²)]A;
##### the amplitude A is diagnosed each stage by inverting that operator (see
##### helmholtz_solve.jl). `reconstitute!` is the forward operator (A → G), used
##### to initialize G from a prescribed amplitude. The tendency ∂ₜG is assembled
##### term by term; in the wave-only regime (velocities = nothing) only the local
##### dispersion term survives.

# Forward reconstitution G = [1 + α(∇ₕ² + κ²)]A. Using αm = 1 + ακ² (m the
# screened-Poisson symbol), this is G = α(m A + ∇²A) — the exact inverse of the
# diagnostic solve, discretized with the same centered Laplacian the solver's
# eigenvalues represent, so `reconstitute!` and `solve_amplitude!` round-trip.
@kernel function _reconstitute!(G, A, grid, α, m)
    i, j, k = @index(Global, NTuple)
    @inbounds G[i, j, k] = α * (m * A[i, j, k] + ∇²ᶜᶜᶜ(i, j, k, grid, A))
end

"""
    reconstitute!(G, solver::NarrowBandHelmholtzSolver, A)

Apply the reconstitution operator `G = [1 + α(∇ₕ² + κ²)]A` to a real component
`A`, writing the result into `G`. The halo regions of `A` are filled first so
the centered Laplacian is well defined at the boundary.
"""
function reconstitute!(G, solver::NarrowBandHelmholtzSolver, A)
    grid = solver.grid
    fill_halo_regions!(A)
    launch!(architecture(grid), grid, :xyz, _reconstitute!, G, A, grid, solver.α, solver.m)
    return G
end

# Local dispersion tendency (wave-only regime, U = 0):
#     ∂ₜG = (i ω_κ / 2κα)(G - A) = iβ(G - A),   β = ω_κ / (2κα) < 0.
# In real components: ∂ₜGʳ = -β(Gⁱ - Aⁱ),  ∂ₜGⁱ = +β(Gʳ - Aʳ).
@kernel function _dispersion_tendency!(Gr_t, Gi_t, Gr, Gi, Ar, Ai, β)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Gr_t[i, j, k] = -β * (Gi[i, j, k] - Ai[i, j, k])
        Gi_t[i, j, k] =  β * (Gr[i, j, k] - Ar[i, j, k])
    end
end

@inline dispersion_frequency_coefficient(d::NarrowBandDispersion) = d.ω_κ / (2 * d.κ * d.α)

"""
    compute_tendencies!(model::NarrowBandWaveModel)

Diagnose the amplitude `A` from the prognostic reconstituted amplitude `G`
(one screened-Poisson solve per real component), then assemble the tendency
`∂ₜG` from the current `(G, A)` and the model's velocity coupling.
"""
function compute_tendencies!(model::NarrowBandWaveModel)
    solver = model.helmholtz_solver
    solve_amplitude!(model.Ar, solver, model.Gr)
    solve_amplitude!(model.Ai, solver, model.Gi)
    compute_wave_tendencies!(model, model.velocities)
    return nothing
end

# Wave-only regime: dispersion is the whole right-hand side.
function compute_wave_tendencies!(model::NarrowBandWaveModel, ::Nothing)
    grid = model.grid
    β = convert(eltype(grid), dispersion_frequency_coefficient(model.dispersion))
    launch!(architecture(grid), grid, :xyz, _dispersion_tendency!,
            model.Gr_tendency, model.Gi_tendency,
            model.Gr, model.Gi, model.Ar, model.Ai, β)
    return nothing
end
