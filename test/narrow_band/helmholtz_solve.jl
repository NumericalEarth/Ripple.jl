import Oceananigans
import Oceananigans.Operators: ∇²ᶜᶜᶜ
import Oceananigans.Utils: launch!
import Oceananigans.Architectures: architecture
import KernelAbstractions: @kernel, @index
using Random

# Residual of the reconstitution operator: r = (1 + ακ²) A + α ∇²ᶜᶜᶜ A - G,
# which must vanish to solver tolerance when A = [1 + α(∇² + κ²)]⁻¹ G.
@kernel function _helmholtz_residual!(r, A, G, grid, one_plus_ακ², α)
    i, j, k = @index(Global, NTuple)
    @inbounds r[i, j, k] = one_plus_ακ² * A[i, j, k] + α * ∇²ᶜᶜᶜ(i, j, k, grid, A) - G[i, j, k]
end

function helmholtz_residual(A, G, solver, dispersion)
    grid = solver.grid
    Ripple.fill_halo_regions!(A)
    r = amplitude_field(solver)
    launch!(architecture(grid), grid, :xyz, _helmholtz_residual!,
            r, A, G, grid, 1 + dispersion.α * dispersion.κ^2, dispersion.α)
    return maximum(abs, interior(r))
end

@testset "Narrow-band Helmholtz (screened-Poisson) solve" begin
    @testset "Companion grid degenerates to 2-D" begin
        grid = RectilinearGrid(CPU(); size=(16, 16, 4), x=(0, 2π), y=(0, 2π),
                               z=(-1, 0), topology=(Periodic, Periodic, Bounded))
        companion = amplitude_solver_grid(grid)
        @test Oceananigans.Grids.topology(companion) === (Periodic, Periodic, Flat)
        @test size(companion) == (16, 16, 1)
        @test Oceananigans.Grids.x_domain(companion) == Oceananigans.Grids.x_domain(grid)
        @test Oceananigans.Grids.y_domain(companion) == Oceananigans.Grids.y_domain(grid)
    end

    @testset "Operator/eigenvalue consistency on random fields" begin
        for depth in (InfiniteDepth(), 1.0), κ in (0.5, 1.0)
            d = NarrowBandDispersion(κ, depth)
            grid = RectilinearGrid(CPU(); size=(32, 24, 4), x=(0, 4π), y=(0, 3π),
                                   z=(-2, 0), topology=(Periodic, Periodic, Bounded))
            solver = NarrowBandHelmholtzSolver(grid, d)

            Random.seed!(20260707)
            for component in 1:2   # real and imaginary parts
                A = amplitude_field(solver)
                G = amplitude_field(solver)
                interior(G) .= randn(size(interior(G))...)
                Ripple.fill_halo_regions!(G)

                solve_amplitude!(A, solver, G)

                res = helmholtz_residual(A, G, solver, d)
                @test res < 1e-9 * maximum(abs, interior(G))
            end
        end
    end

    @testset "Plane-wave eigenvalue" begin
        Nx = Ny = 16
        Lx = Ly = 2π
        grid = RectilinearGrid(CPU(); size=(Nx, Ny, 2), x=(0, Lx), y=(0, Ly),
                               z=(-1, 0), topology=(Periodic, Periodic, Bounded))

        for depth in (InfiniteDepth(), 1.0)
            d = NarrowBandDispersion(1.0, depth)
            solver = NarrowBandHelmholtzSolver(grid, d)
            m = screened_poisson_symbol(d)

            kx, ky = 3, 2
            Δx, Δy = Lx / Nx, Ly / Ny
            λ = (2sin(kx * π / Nx) / Δx)^2 + (2sin(ky * π / Ny) / Δy)^2

            A = amplitude_field(solver)
            G = amplitude_field(solver)
            set!(G, (x, y) -> cos(kx * x + ky * y))
            Ripple.fill_halo_regions!(G)
            solve_amplitude!(A, solver, G)

            # Single Fourier mode: A = G / (α (m - λ)) pointwise.
            expected = interior(G) ./ (d.α * (m - λ))
            @test interior(A) ≈ expected rtol = 1e-9
        end
    end

    @testset "Constructor validation" begin
        d = NarrowBandDispersion(1.0, InfiniteDepth())

        nonperiodic = RectilinearGrid(CPU(); size=(8, 8, 2), x=(0, 1), y=(0, 1),
                                      z=(-1, 0), topology=(Bounded, Periodic, Bounded))
        @test_throws ArgumentError NarrowBandHelmholtzSolver(nonperiodic, d)
        @test_throws ArgumentError amplitude_solver_grid(nonperiodic)
    end
end
