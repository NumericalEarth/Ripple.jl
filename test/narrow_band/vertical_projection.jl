import Oceananigans.Grids: znodes, Face

@testset "Narrow-band vertical projections Ū, Ũ" begin
    # Exact finite-volume weights integrate polynomial and exponential columns
    # to their analytic depth integrals ∫Φ²U dz and (1/κ²)∫Φ_z²U dz.
    @testset "Cell integrals vs analytic (finite depth)" begin
        κ = 1.5
        h = 2.0
        # Full-depth ∫Φ² dz = 1 by construction.
        Nz = 200
        zf = range(-h, 0; length=Nz + 1)
        total_Φ² = sum(Ripple.phi_squared_cell_integral(κ, h, zf[k], zf[k+1]) for k in 1:Nz)
        @test total_Φ² ≈ 1 rtol = 1e-12

        # ∫Φ² e^{2κz} dz assembled from cell weights (U = e^{2κz} sampled at centers)
        # converges to the exact integral as Nz grows.
        exact = 0.0
        # exact ∫_{-h}^0 Φ² e^{2κz} dz via fine quadrature
        zc_fine = range(-h, 0; length=200001)
        Φ²_fine = Ripple.vertical_structure.(zc_fine, κ, h) .^ 2 .* exp.(2κ .* zc_fine)
        exact = (sum(Φ²_fine) - (Φ²_fine[1] + Φ²_fine[end]) / 2) * (h / (length(zc_fine) - 1))

        for Nz in (50, 100, 200)
            zf = collect(range(-h, 0; length=Nz + 1))
            zc = (zf[1:end-1] .+ zf[2:end]) ./ 2
            w = [Ripple.phi_squared_cell_integral(κ, h, zf[k], zf[k+1]) for k in 1:Nz]
            approx = sum(w .* exp.(2κ .* zc))
            @test approx ≈ exact rtol = 5e-3
        end
    end

    @testset "Deep-water Ũ = Ū and exponential weights" begin
        κ = 2.0
        zf = collect(range(-5.0, 0; length=41))
        for k in 1:(length(zf) - 1)
            wŪ = Ripple.phi_squared_cell_integral(κ, InfiniteDepth(), zf[k], zf[k+1])
            wŨ = Ripple.phi_z_squared_cell_integral(κ, InfiniteDepth(), zf[k], zf[k+1])
            @test wŪ ≈ wŨ                                       # Φ_z² = κ²Φ² in deep water
            @test wŪ ≈ exp(2κ * zf[k+1]) - exp(2κ * zf[k])
        end
    end

    @testset "Finite depth → deep water as κh grows" begin
        κ = 1.0
        for h in (5.0, 15.0, 40.0)
            zf = (-0.5, -0.25)   # a near-surface cell
            w_finite = Ripple.phi_squared_cell_integral(κ, h, zf[1], zf[2])
            w_deep = Ripple.phi_squared_cell_integral(κ, InfiniteDepth(), zf[1], zf[2])
            rtol = h ≥ 40 ? 1e-6 : 1e-1
            @test w_finite ≈ w_deep rtol = rtol
        end
    end

    @testset "VerticalProjectionWeights on a grid" begin
        grid = RectilinearGrid(CPU(); size=(4, 4, 16), x=(0, 1), y=(0, 1), z=(-2, 0),
                               topology=(Periodic, Periodic, Bounded))
        d = NarrowBandDispersion(1.5, 2.0)
        w = Ripple.VerticalProjectionWeights(d, grid)
        @test length(w.Ū_weights) == 16
        @test sum(w.Ū_weights) ≈ 1 rtol = 1e-12       # ∫Φ² dz = 1
        @test all(isfinite, w.Ū_weights)
        @test all(isfinite, w.Ũ_weights)
    end
end
