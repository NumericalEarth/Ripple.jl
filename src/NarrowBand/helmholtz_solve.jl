import Oceananigans
import Oceananigans.Architectures: architecture
import Oceananigans.Grids: RectilinearGrid, Flat, Periodic, topology, halo_size
import Oceananigans.Fields: CenterField, interior
import Oceananigans.Solvers: FFTBasedPoissonSolver, solve!

#####
##### Diagnostic inversion of the reconstitution operator [1 + α(∇ₕ² + κ²)]
#####
##### The narrow-band model prognoses the reconstituted amplitude G and
##### diagnoses the physical amplitude A once per stage from
#####
#####     [1 + α(∇ₕ² + κ²)] A = G .
#####
##### Rearranged, this is the generalized ("screened") Poisson equation
#####
#####     (∇ₕ² + m) A = G / α ,     m = (1 + ακ²) / α < 0 ,
#####
##### solved exactly by `Oceananigans.Solvers.FFTBasedPoissonSolver` on a
##### companion grid whose vertical direction is `Flat` — the `Flat`
##### eigenvalues are zero, so the 3-D solver degenerates to the 2-D horizontal
##### solve. Because α < 0 and 1 + ακ² > 0, we have m < 0 strictly, so every
##### discrete mode satisfies λ - m > 0 and the operator is nonsingular.

"""
    amplitude_solver_grid(grid)

Build the `(Periodic, Periodic, Flat)` companion grid on which the narrow-band
amplitude solve runs, sharing the horizontal extent and resolution of the
mean-flow `grid`. Errors unless the horizontal topology is periodic and the
horizontal spacing is uniform (both required by the FFT-based solve).
"""
function amplitude_solver_grid(grid)
    arch = architecture(grid)
    FT = eltype(grid)
    Nx, Ny, _ = size(grid)
    tx, ty, _ = topology(grid)

    (tx === Periodic && ty === Periodic) ||
        throw(ArgumentError("the narrow-band FFT amplitude solve requires a horizontally " *
                            "periodic grid; got horizontal topology ($tx, $ty)"))

    _assert_uniform_horizontal_spacing(grid)

    x = Oceananigans.Grids.x_domain(grid)
    y = Oceananigans.Grids.y_domain(grid)
    Hx, Hy, _ = halo_size(grid)

    return RectilinearGrid(arch, FT; size=(Nx, Ny), x, y, halo=(Hx, Hy),
                           topology=(Periodic, Periodic, Flat))
end

function _assert_uniform_horizontal_spacing(grid)
    Δx = xspacings(grid)
    Δy = yspacings(grid)
    all(≈(first(Δx)), Δx) ||
        throw(ArgumentError("the narrow-band FFT amplitude solve requires uniform x spacing"))
    all(≈(first(Δy)), Δy) ||
        throw(ArgumentError("the narrow-band FFT amplitude solve requires uniform y spacing"))
    return nothing
end

"""
    NarrowBandHelmholtzSolver{S, G, FT}

Wraps the stock `FFTBasedPoissonSolver` that diagnoses the narrow-band amplitude
`A` from the reconstituted amplitude `G` by inverting `[1 + α(∇ₕ² + κ²)]`. Built
from a mean-flow `grid` and a [`NarrowBandDispersion`](@ref); see
[`solve_amplitude!`](@ref).
"""
struct NarrowBandHelmholtzSolver{S, G, FT}
    poisson_solver :: S   # FFTBasedPoissonSolver on the Flat-z companion grid
    grid :: G             # (Periodic, Periodic, Flat) companion grid
    α :: FT               # reconstitution parameter
    m :: FT               # screened-Poisson symbol (1 + ακ²) / α
end

"""
    NarrowBandHelmholtzSolver(grid, dispersion::NarrowBandDispersion)

Construct the diagnostic amplitude solver for the mean-flow `grid` and carrier
`dispersion`. The companion FFT grid is built by [`amplitude_solver_grid`](@ref)
and the screened-Poisson symbol `m` by [`screened_poisson_symbol`](@ref).
"""
function NarrowBandHelmholtzSolver(grid, dispersion::NarrowBandDispersion)
    companion = amplitude_solver_grid(grid)
    poisson_solver = FFTBasedPoissonSolver(companion)
    m = screened_poisson_symbol(dispersion)
    α = convert(eltype(companion), dispersion.α)
    return NarrowBandHelmholtzSolver(poisson_solver, companion,
                                     α, convert(eltype(companion), m))
end

"""
    amplitude_field(solver::NarrowBandHelmholtzSolver)

Allocate a real `Field` on the solver's companion grid, sized to hold one real
component (real or imaginary part) of the amplitude `A`.
"""
amplitude_field(solver::NarrowBandHelmholtzSolver) = CenterField(solver.grid)

"""
    solve_amplitude!(A, solver::NarrowBandHelmholtzSolver, G)

Diagnose one real component of the amplitude `A` from the corresponding real
component of the reconstituted amplitude `G` by solving
`[1 + α(∇ₕ² + κ²)] A = G`. `A` and `G` are real `Field`s on the solver's
companion grid; the solve is performed in place through the wrapped
`FFTBasedPoissonSolver`, whose complex storage is loaded with `G / α` and
inverted with symbol `m`. The real and imaginary parts of a complex amplitude
are solved by two separate calls.
"""
function solve_amplitude!(A, solver::NarrowBandHelmholtzSolver, G)
    poisson_solver = solver.poisson_solver
    storage = poisson_solver.storage      # complex, size (Nx, Ny, 1)
    Gi = interior(G)
    @. storage = complex(Gi / solver.α)
    solve!(A, poisson_solver, storage, solver.m)
    return A
end
