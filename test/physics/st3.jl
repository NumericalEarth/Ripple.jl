@testset "PressureCorrelationInput construction" begin
    drag = BulkWindDrag(:linear)
    inp = PressureCorrelationInput(; drag=drag, wind=10.0)

    @test inp isa AbstractWindInput
    @test inp.β_max == 1.2
    @test inp.z_α == 0.011
    @test inp.p_in == 2.0
    @test inp.α₀ == 0.0095
    @test inp.von_karman == 0.4
    @test inp.gravity == 9.81

    inp_st4 = PressureCorrelationInput(; drag=drag, wind=10.0, β_max=1.5, p_in=2.5)
    @test inp_st4.β_max == 1.5
    @test inp_st4.p_in == 2.5
end

@testset "PressureCorrelationInput on a model" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(; frequency=range(0.1, 0.3; length=4), φ=collect(range(0, 2π * 7/8; length=8)))

    inp = PressureCorrelationInput(; drag=BulkWindDrag(:linear), wind=15.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing,
                                physics=inp, timestepper=:ForwardEuler)
    set!(model, N=1e-3)

    # Sin must be non-negative when wind aligns with wave direction and inverse wave
    # age u*/C > 1. With wind = 15 m/s and the low-frequency cell (≈0.1 Hz, C ≈ 15 m/s),
    # we should see growth on at least the wind-aligned bin.
    compute_tendencies!(model)
    G = interior(model.tendencies)
    @test maximum(G) > 0          # some bin grows
    @test minimum(G) >= 0         # no negative contributions (pure growth)

    # Zero wind → zero tendency.
    inp_calm = PressureCorrelationInput(; drag=BulkWindDrag(:linear), wind=0.0)
    model_calm = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing,
                                     physics=inp_calm, timestepper=:ForwardEuler)
    set!(model_calm, N=1e-3)
    compute_tendencies!(model_calm)
    @test all(interior(model_calm.tendencies) .== 0)
end

@testset "MeanSpectrumWhitecapping construction" begin
    d = MeanSpectrumWhitecapping()
    @test d isa AbstractDissipation
    @test d.C_ds == -2.1            # BJA default
    @test d.δ₁ == 0.4
    @test d.δ₂ == 0.6
    @test d.p == 0.5
end

@testset "MeanSpectrumWhitecapping on a model" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(; frequency=range(0.1, 0.3; length=4), φ=collect(range(0, 2π * 7/8; length=8)))

    diss = MeanSpectrumWhitecapping()
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing,
                                physics=diss, timestepper=:SemiImplicitEuler)
    set!(model, N=1e-2)

    # Dissipation must produce only damping (positive part is zero).
    compute_tendencies!(model)
    G = interior(model.tendencies)
    @test all(G .<= 0)              # all cells dissipated or unchanged

    # Action density must decrease under semi-implicit stepping.
    initial_action = maximum(interior(model.action))
    time_step!(model, 1.0)
    @test maximum(interior(model.action)) < initial_action
end

@testset "MeanSpectrumPhysics bundle" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(; frequency=range(0.1, 0.3; length=4), φ=collect(range(0, 2π * 7/8; length=8)))

    inp = PressureCorrelationInput(; drag=BulkWindDrag(:linear), wind=15.0)
    diss = MeanSpectrumWhitecapping()
    bundle = MeanSpectrumPhysics(; wind_input=inp, dissipation=diss)

    @test bundle isa AbstractPhysicsBundle
    @test bundle isa AbstractPhysicsTerm
    @test bundle.wind_input === inp
    @test bundle.dissipation === diss

    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing,
                                physics=bundle, timestepper=:SemiImplicitEuler)
    set!(model, N=1e-3)
    compute_tendencies!(model)
    # Bundle is the sum: input growth - dissipation damping.
    # We can't predict the sign without knowing the balance, but the tendency
    # must be finite and the model must step without throwing.
    @test all(isfinite, interior(model.tendencies))

    # One semi-implicit step should leave N non-negative.
    time_step!(model, 1.0)
    @test all(interior(model.action) .>= 0)
end
