@testset "Drag parameterizations" begin
    @testset "BulkWindDrag :linear (Wu 1982)" begin
        d = BulkWindDrag(:linear)
        # Cd(U10=10) = (0.8 + 0.65) × 1e-3 = 1.45e-3 → u* = U10·√Cd ≈ 0.381
        @test isapprox(drag_coefficient(d, 10.0), 1.45e-3; rtol=1e-12)
        @test isapprox(friction_velocity(d, 10.0), 10.0 * sqrt(1.45e-3); rtol=1e-12)

        # u* monotone in U10 over the regime where Cd > 0
        @test friction_velocity(d, 20.0) > friction_velocity(d, 10.0)
        @test friction_velocity(d, 5.0)  > 0
    end

    @testset "BulkWindDrag :capped (Hwang 2001)" begin
        d = BulkWindDrag(:capped)
        # u* should keep growing through hurricane winds (cap protects from Cd<0)
        @test friction_velocity(d, 30.0) > friction_velocity(d, 20.0)
        @test friction_velocity(d, 50.0) > friction_velocity(d, 30.0)

        # Cd floor protects the curve from going negative at U10 ~ 60 m/s
        @test drag_coefficient(d, 60.0) >= 4e-4
    end

    @testset "WaveSupportedDrag stub" begin
        d = WaveSupportedDrag(; alpha0=0.0095, z_u=10.0)
        @test d isa Ripple.AbstractDrag
        @test d.alpha0 == 0.0095
        @test d.z_u == 10.0
        # Unimplemented — should throw until ST3 wind input lands.
        @test_throws ErrorException drag_coefficient(d, 10.0)
        @test_throws ErrorException friction_velocity(d, 10.0, 0.5)
    end
end

@testset "DiagnosticTail" begin
    tail = DiagnosticTail(; power=5.0, f_cut=0.625, f_max=10.0)
    @test tail.power == 5.0
    @test tail.f_cut == 0.625
    @test tail.f_max == 10.0

    # Tail factor at k=k_max is 1 (continuity), and decays as k→∞.
    @test isapprox(action_tail_factor(tail, 1.0), 1.0; rtol=1e-12)
    @test action_tail_factor(tail, 2.0) < 1.0
    @test action_tail_factor(tail, 4.0) < action_tail_factor(tail, 2.0)

    # m=5 power-law: N(k) ∝ k^(-7/2) along the tail.
    @test isapprox(action_tail_factor(tail, 2.0), 2.0^(-7/2); rtol=1e-12)
end

@testset "DynamicSubstepLimiter" begin
    lim = DynamicSubstepLimiter()
    @test lim.X_p == 0.15
    @test lim.X_r == 0.10
    @test lim.X_f == 0.05

    # ΔN_p scales as 1/(σ·k³): doubling σ halves the bound; doubling k cuts by 8.
    σ, k = 1.0, 0.5
    bound = parametric_action_bound(lim, σ, k)
    @test bound > 0
    @test isapprox(parametric_action_bound(lim, 2σ, k), bound / 2; rtol=1e-12)
    @test isapprox(parametric_action_bound(lim, σ, 2k), bound / 8; rtol=1e-12)
end

@testset "Bundle prepare_sources default" begin
    # prepare_sources returns `nothing` for non-bundle terms; bundles override
    # to return a NamedTuple of per-grid-point state.
    @test Ripple.prepare_sources(NoSource(), nothing) === nothing
    @test Ripple.prepare_sources(nothing, nothing) === nothing
end
