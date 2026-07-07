@testset "Narrow-band Stokes drift diagnostics" begin
    grid = RectilinearGrid(CPU(); size=(64, 64, 4), x=(0, 2π), y=(0, 2π), z=(-1000, 0),
                           topology=(Periodic, Periodic, Bounded), halo=(3, 3, 3))
    κ = 1.0
    a = 1.0
    model = NarrowBandWaveModel(grid; κ, depth=InfiniteDepth())
    set!(model; A=(x, y) -> a * cis(κ * x))       # plane wave, |k| = κ
    d = model.dispersion

    S = stokes_functionals(model)

    @testset "Plane-wave functionals" begin
        # S₁ = 2a²κ³/ω, T₁ = 2a²κ/ω, and S₂ = T₂ = 0 (no y-dependence).
        @test maximum(abs, interior(S.T1) .- 2a^2 * κ / d.ω) < 1e-2 * (2a^2 * κ / d.ω)
        @test maximum(abs, interior(S.S1) .- 2a^2 * κ^3 / d.ω) < 1e-2 * (2a^2 * κ^3 / d.ω)
        @test maximum(abs, interior(S.T2)) < 1e-8
        @test maximum(abs, interior(S.S2)) < 1e-8
    end

    @testset "Surface-elevation dictionary" begin
        a₀ = surface_elevation_amplitude(model)     # 2ωC|A|/g, uniform for |A| = a
        expected = 2 * d.ω * d.C * a / d.gravity
        @test all(x -> isapprox(x, expected; rtol=1e-8), a₀)
    end

    @testset "Reconstruction matches classical deep-water Stokes at the surface" begin
        # Uˢ₁(0) = Φ²(0) S₁ + Φ_z²(0) T₁ should equal a₀² ω κ (deep water).
        Φ²0  = vertical_structure(0.0, κ, InfiniteDepth())^2
        Φz²0 = vertical_structure_derivative(0.0, κ, InfiniteDepth())^2
        us0  = Φ²0 * (2a^2 * κ^3 / d.ω) + Φz²0 * (2a^2 * κ / d.ω)
        a₀   = 2 * d.ω * d.C * a / d.gravity
        @test us0 ≈ a₀^2 * d.ω * κ rtol = 1e-10
    end
end
