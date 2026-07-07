@testset "Narrow-band dispersion" begin
    gravity = 9.81

    # A spread of carrier wavenumbers and depths covering κh ∈ [10⁻², 10²].
    κs = (0.5, 1.0, 2.0)
    κhs = (1e-2, 1e-1, 1.0, 5.0, 1e1, 1e2)

    @testset "κ-derivatives vs finite differences" begin
        for κ in κs, κh in κhs
            h = κh / κ
            ω    = carrier_frequency(κ, h; gravity)
            ω_κ  = carrier_group_velocity(κ, h; gravity)
            ω_κκ = carrier_frequency_curvature(κ, h; gravity)

            # Central finite differences of ω(κ) at fixed depth h.
            dκ = 1e-5 * κ
            ωp = carrier_frequency(κ + dκ, h; gravity)
            ωm = carrier_frequency(κ - dκ, h; gravity)
            ω0 = carrier_frequency(κ, h; gravity)

            ω_κ_fd  = (ωp - ωm) / 2dκ
            @test ω_κ ≈ ω_κ_fd rtol = 1e-6
            @test ω ≈ ω0

            # ω_κκ: larger step to avoid roundoff cancellation in the 2nd difference.
            δκ = 1e-3 * κ
            ω_κκ_fd = (carrier_frequency(κ + δκ, h; gravity)
                       - 2 * carrier_frequency(κ, h; gravity)
                       + carrier_frequency(κ - δκ, h; gravity)) / δκ^2
            @test ω_κκ ≈ ω_κκ_fd rtol = 1e-4
        end
    end

    @testset "Deep-water limits" begin
        for κ in κs
            ω_inf    = carrier_frequency(κ, InfiniteDepth(); gravity)
            ω_κ_inf  = carrier_group_velocity(κ, InfiniteDepth(); gravity)
            ω_κκ_inf = carrier_frequency_curvature(κ, InfiniteDepth(); gravity)

            @test ω_inf ≈ sqrt(gravity * κ)
            @test ω_κ_inf ≈ ω_inf / 2κ            # group speed = c/2
            @test ω_κκ_inf ≈ -ω_inf / 4κ^2
            @test reconstitution_parameter(κ, InfiniteDepth(); gravity) ≈ -3 / (8κ^2)
            @test vertical_structure_constant(κ, InfiniteDepth(); gravity) ≈ sqrt(2κ)

            # Finite depth converges to the deep-water branch as κh → ∞.
            h = 100 / κ
            @test carrier_frequency(κ, h; gravity) ≈ ω_inf rtol = 1e-6
            @test carrier_group_velocity(κ, h; gravity) ≈ ω_κ_inf rtol = 1e-6
            @test reconstitution_parameter(κ, h; gravity) ≈ -3 / (8κ^2) rtol = 1e-6
        end
    end

    @testset "Shallow-water limit α → -1/(4κ²)" begin
        for κ in κs
            h = 1e-3 / κ   # κh = 1e-3
            @test reconstitution_parameter(κ, h; gravity) ≈ -1 / (4κ^2) rtol = 1e-4
        end
    end

    @testset "α independent of gravity" begin
        for κ in κs, κh in κhs
            h = κh / κ
            @test reconstitution_parameter(κ, h; gravity=9.81) ≈
                  reconstitution_parameter(κ, h; gravity=1.0)
        end
    end

    @testset "Screened-Poisson well-posedness" begin
        for κ in κs, κh in κhs
            h = κh / κ
            d = NarrowBandDispersion(κ, h; gravity)
            one_plus_ακ² = 1 + d.α * d.κ^2
            # Strictly positive (nonsingular solve) and < 1 (since α < 0). The
            # deep- and shallow-water endpoints are 5/8 and 3/4, but 1 + ακ² can
            # dip below 5/8 at intermediate depths — only positivity matters.
            @test 0 < one_plus_ακ² < 1
            m = screened_poisson_symbol(d)
            @test m < 0                       # screened, not singular
            @test d.α * m ≈ one_plus_ακ²      # m = (1 + ακ²)/α
        end

        d_inf = NarrowBandDispersion(1.0, InfiniteDepth(); gravity)
        @test 1 + d_inf.α * d_inf.κ^2 ≈ 5//8
    end

    @testset "Vertical structure normalization ∫Φ²dz = 1" begin
        for κ in κs, κh in (0.1, 1.0, 10.0)
            h = κh / κ
            zc = range(-h, 0; length=20001)
            Φ² = vertical_structure.(zc, κ, h; gravity) .^ 2
            # Trapezoidal integral over [-h, 0].
            integral = (sum(Φ²) - (Φ²[1] + Φ²[end]) / 2) * (h / (length(zc) - 1))
            @test integral ≈ 1 rtol = 1e-4

            # Analytic identity for C: C² = cosh²(κh) / (h/2 + sinh(2κh)/(4κ)).
            C_analytic = sqrt(cosh(κ * h)^2 / (h / 2 + sinh(2κ * h) / 4κ))
            @test vertical_structure_constant(κ, h; gravity) ≈ C_analytic
        end

        # Deep water: ∫_{-∞}^{0} 2κ e^{2κz} dz = 1.
        κ = 1.0
        H = 25 / κ
        zc = range(-H, 0; length=20001)
        Φ² = vertical_structure.(zc, κ, Ref(InfiniteDepth()); gravity) .^ 2
        integral = (sum(Φ²) - (Φ²[1] + Φ²[end]) / 2) * (H / (length(zc) - 1))
        @test integral ≈ 1 rtol = 1e-4
    end

    @testset "Φ_z: deep-water Φ_z = κΦ" begin
        κ = 1.5
        for z in (-0.1, -0.5, -1.0)
            @test vertical_structure_derivative(z, κ, InfiniteDepth(); gravity) ≈
                  κ * vertical_structure(z, κ, InfiniteDepth(); gravity)
        end
    end

    @testset "Type preservation" begin
        d32 = NarrowBandDispersion(0.1f0, 5.0f0)
        @test d32 isa NarrowBandDispersion{Float32}
        @test d32.ω isa Float32
        @test d32.α isa Float32

        d_inf32 = NarrowBandDispersion(0.1f0, InfiniteDepth(); gravity=9.81f0)
        @test d_inf32 isa NarrowBandDispersion{Float32}
    end

    @testset "Constructor validation" begin
        @test_throws ArgumentError NarrowBandDispersion(-1.0, 10.0)
        @test_throws ArgumentError NarrowBandDispersion(0.0, 10.0)
        @test_throws ArgumentError NarrowBandDispersion(1.0, -5.0)
        @test_throws ArgumentError NarrowBandDispersion(1.0, 0.0)
    end
end
