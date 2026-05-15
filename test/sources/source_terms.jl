@testset "Source terms" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = CartesianWaveVectorGrid(Float64; kx=range(0.4, 0.8; length=3), ky=range(-0.1, 0.1; length=3))

    relaxation = RelaxationToSpectrum((x, y, kx, ky) -> 2.0; timescale=1.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=relaxation, timestepper=:ForwardEuler)
    set!(model, N=0.0)
    time_step!(model, 0.5)
    @test all(interior(model.action) .≈ 1.0)

    balanced = SourceTermSet(LinearWindInput(rate=0.2), BottomFriction(rate=0.2))
    @test length(balanced) == 2
    @test !isempty(balanced)
    @test balanced[1] isa LinearWindInput
    @test balanced[2] isa BottomFriction
    @test collect(balanced) == collect(balanced.terms)
    @test isempty(SourceTermSet())

    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=balanced, timestepper=:ForwardEuler)
    set!(model, N=1.5)
    compute_tendencies!(model)
    @test maximum(abs, interior(model.tendencies)) < 1e-14

    damping = BottomFriction(rate=100.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=damping, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    time_step!(model, 1.0)
    @test all(interior(model.action) .== 0)
end

@testset "Directional wind and saturation sources" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0, pi/2, pi, 3pi/2])

    wind = ExponentialWindInput(rate=0.5, direction=0.0, spreading_power=2)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=wind, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    compute_tendencies!(model)

    wind_weights = [Ripple.wind_directional_weight(cgrid, 1, n, 0.0, 2)
                    for n in eachindex(cgrid.φ)]
    @test model.tendencies[1, 1, 1, 1] ≈ 0.5 * wind_weights[1]
    @test model.tendencies[1, 1, 1, 2] ≈ 0.5 * wind_weights[2]
    @test abs(model.tendencies[1, 1, 1, 3]) < 1e-14
    @test model.tendencies[1, 1, 1, 4] ≈ 0.5 * wind_weights[4]
    @test wind_weights[1] < 1
    @test wind_weights[2] > 0
    @test wind_weights[1] ≈ 1 / 2 + 1 / pi
    @test wind_weights[2] ≈ 1 / 4 - 1 / (2pi)

    invalid_spreading = ExponentialWindInput(rate=0.5, direction=0.0, spreading_power=1.5)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=invalid_spreading)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    damping = SaturationDissipation(rate=2.0, threshold=0.1, power=1.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=damping, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    compute_tendencies!(model)
    @test all(interior(model.tendencies) .< 0)

    damping = SaturationDissipation(rate=2.0, threshold=100.0, power=1.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=damping, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    compute_tendencies!(model)
    @test maximum(abs, interior(model.tendencies)) < 1e-14
end

@testset "Power-law wind input" begin
    grid = RectilinearGrid(CPU(); size=(2, 1, 1), x=(0, 2), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0, pi/2, pi, 3pi/2])

    speed = reshape([2.0, 4.0], 2, 1)
    direction = reshape([0.0, pi/2], 2, 1)
    wind = PowerLawWindInput(rate=0.25,
                             speed=speed,
                             direction=direction,
                             reference_speed=2.0,
                             speed_power=2.0,
                             spreading_power=1.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=wind, timestepper=:ForwardEuler)
    set!(model, N=3.0)
    compute_tendencies!(model)

    x_wind_weights = [Ripple.wind_directional_weight(cgrid, 1, n, 0.0, 1)
                      for n in eachindex(cgrid.φ)]
    y_wind_weights = [Ripple.wind_directional_weight(cgrid, 1, n, pi/2, 1)
                      for n in eachindex(cgrid.φ)]
    @test model.tendencies[1, 1, 1, 1] ≈ 0.75 * x_wind_weights[1]
    @test model.tendencies[1, 1, 1, 2] ≈ 0.75 * x_wind_weights[2]
    @test model.tendencies[2, 1, 1, 1] ≈ 3.0 * y_wind_weights[1]
    @test model.tendencies[2, 1, 1, 2] ≈ 3.0 * y_wind_weights[2]

    positive, damping = source_split(wind, model, 2, 1, 1, 2)
    @test positive ≈ 3.0 * y_wind_weights[2]
    @test damping == 0

    storm_grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 2), y=(0, 2), z=(0, 1))
    storm = IdealizedHurricaneWind(center=(0.0, 1.0), vmax=2.0, rmax=1.0, radius=4.0)
    storm_wind = PowerLawWindInput(rate=0.25,
                                   wind=storm,
                                   reference_speed=2.0,
                                   speed_power=1.0,
                                   spreading_power=1.0)
    storm_model = SpectralWaveModel(storm_grid, cgrid;
                      horizontal_advection=nothing,
                      sources=storm_wind,
                      timestepper=:ForwardEuler)
    set!(storm_model, N=4.0)
    compute_tendencies!(storm_model)
    storm_weights = [Ripple.wind_directional_weight(cgrid, 1, n, pi/2, 1)
                     for n in eachindex(cgrid.φ)]
    @test storm_model.tendencies[1, 1, 1, 1] ≈ storm_weights[1]
    @test storm_model.tendencies[1, 1, 1, 2] ≈ storm_weights[2]

    negative = PowerLawWindInput(rate=-0.5,
                                 speed=2.0,
                                 direction=0.0,
                                 reference_speed=2.0,
                                 speed_power=1.0,
                                 spreading_power=1.0)
    damping_model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=nothing,
                      sources=negative,
                      timestepper=:SemiImplicitEuler)
    set!(damping_model, N=3.0)
    positive, damping = source_split(negative, damping_model, 1, 1, 1, 1)
    @test positive == 0
    @test damping ≈ 0.5 * x_wind_weights[1]
    time_step!(damping_model, 1.0)
    @test damping_model.action[1, 1, 1, 1] ≈ 3.0 / (1 + 0.5 * x_wind_weights[1])
    @test all(interior(damping_model.action) .>= 0)

    invalid = PowerLawWindInput(rate=0.1, speed=-1.0)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=invalid)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    invalid_reference = PowerLawWindInput(rate=0.1, reference_speed=0.0)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=invalid_reference)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Wave-age wind input" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[0.25, 1.0], φ=[0.0, pi])

    wind = WaveAgeWindInput(rate=0.5,
                            speed=2.0,
                            direction=0.0,
                            inverse_wave_age_threshold=1.0,
                            power=1.0,
                            spreading_power=0.0,
                            gravity=1.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=wind, timestepper=:ForwardEuler)
    set!(model, N=3.0)
    compute_tendencies!(model)

    @test model.tendencies[1, 1, 1, 1] == 0
    @test model.tendencies[1, 1, 1, 2] == 0
    @test model.tendencies[1, 1, 2, 1] ≈ 1.5
    @test model.tendencies[1, 1, 2, 2] == 0

    positive, damping = source_split(wind, model, 1, 1, 2, 1)
    @test positive ≈ 1.5
    @test damping == 0

    opposing = WaveAgeWindInput(rate=0.5,
                                speed=2.0,
                                direction=pi,
                                inverse_wave_age_threshold=1.0,
                                spreading_power=0.0,
                                gravity=1.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=opposing)
    set!(model, N=3.0)
    compute_tendencies!(model)
    @test model.tendencies[1, 1, 2, 1] == 0
    @test model.tendencies[1, 1, 2, 2] ≈ 1.5

    negative = WaveAgeWindInput(rate=-0.5,
                                speed=2.0,
                                direction=0.0,
                                inverse_wave_age_threshold=1.0,
                                spreading_power=0.0,
                                gravity=1.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=negative, timestepper=:SemiImplicitEuler)
    set!(model, N=3.0)
    positive, damping = source_split(negative, model, 1, 1, 2, 1)
    @test positive == 0
    @test damping ≈ 0.5
    time_step!(model, 1.0)
    @test model.action[1, 1, 2, 1] ≈ 2.0

    invalid = WaveAgeWindInput(rate=0.1, speed=2.0, inverse_wave_age_threshold=-1.0)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=invalid)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Time-dependent wind source parameters" begin
    grid = RectilinearGrid(CPU(); size=(2, 1, 1), x=(0, 2), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0, pi/2])

    moving_rate(x, y, t) = x + 10t
    rotating_direction(t) = t == 0 ? 0.0 : pi / 2
    wind = ExponentialWindInput(rate=moving_rate, direction=rotating_direction, spreading_power=1)

    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=wind, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    compute_tendencies!(model)
    initial_weights = [Ripple.wind_directional_weight(cgrid, 1, n, 0.0, 1)
                       for n in eachindex(cgrid.φ)]
    @test model.tendencies[1, 1, 1, 1] ≈ 0.5 * initial_weights[1]
    @test model.tendencies[2, 1, 1, 1] ≈ 1.5 * initial_weights[1]
    @test model.tendencies[1, 1, 1, 2] ≈ 0.5 * initial_weights[2]

    time_step!(model, 0.1)
    compute_tendencies!(model)
    rotated_weights = [Ripple.wind_directional_weight(cgrid, 1, n, pi/2, 1)
                       for n in eachindex(cgrid.φ)]
    @test model.tendencies[1, 1, 1, 1] ≈ source_tendency(wind, model, 1, 1, 1, 1)
    @test model.tendencies[1, 1, 1, 1] > 0
    @test model.tendencies[1, 1, 1, 2] ≈ source_tendency(wind, model, 1, 1, 1, 2)
    @test sum(rotated_weights) ≈ 2 / pi
    @test model.tendencies[1, 1, 1, 2] > 0

    damping = SaturationDissipation(rate=(x, y, t) -> 1 + t, threshold=0.1, power=1)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=damping, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    model.clock.time = 0.2
    compute_tendencies!(model)
    @test all(interior(model.tendencies) .< 0)
end

@testset "Whitecapping dissipation" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[0.5, 1.0, 2.0], φ=[0.0])

    whitecapping = WhitecappingDissipation(rate=0.2,
                                           saturation_threshold=0.1,
                                           saturation_power=1.0,
                                           wavenumber_power=2.0,
                                           reference_wavenumber=1.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=whitecapping, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    compute_tendencies!(model)

    spectral_factor = [spectral_radial_power_average(cgrid, m, 1, 2)
                       for m in eachindex(cgrid.κ)]
    @test model.tendencies[1, 1, 1, 1] < 0
    @test model.tendencies[1, 1, 2, 1] < model.tendencies[1, 1, 1, 1]
    @test model.tendencies[1, 1, 3, 1] < model.tendencies[1, 1, 2, 1]
    @test model.tendencies[1, 1, 3, 1] / model.tendencies[1, 1, 2, 1] ≈ spectral_factor[3] / spectral_factor[2]

    weak = WhitecappingDissipation(rate=0.2, saturation_threshold=100.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=weak, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    compute_tendencies!(model)
    @test maximum(abs, interior(model.tendencies)) < 1e-14
end

@testset "Frequency dissipation" begin
    grid = RectilinearGrid(CPU(); size=(2, 1, 1), x=(0, 2), y=(0, 1), z=(0, 1))
    φ = range(0, 2pi; length=5)[1:4]
    cgrid = FrequencyDirectionGrid(; frequency=[0.1, 0.2, 0.4], φ=φ)
    rate = reshape([0.5, -0.25], 2, 1)
    source = FrequencyDissipation(rate=rate,
                                  reference_frequency=0.2,
                                  power=2)
    no_advection = nothing
    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=no_advection,
                      sources=source,
                      timestepper=:ForwardEuler)
    set!(model, N=2.0)
    compute_tendencies!(model)

    expected_rates = [0.5 * spectral_frequency_power_average(cgrid, m, 1, 2) / 0.2^2
                      for m in eachindex(cgrid.frequency)]
    @test model.tendencies[1, 1, 1, 1] ≈ -expected_rates[1] * 2.0
    @test model.tendencies[1, 1, 2, 1] ≈ -expected_rates[2] * 2.0
    @test model.tendencies[1, 1, 3, 1] ≈ -expected_rates[3] * 2.0
    @test model.tendencies[2, 1, 2, 1] ≈ 0.25 * spectral_frequency_power_average(cgrid, 2, 1, 2) / 0.2^2 * 2.0

    positive, damping = source_split(source, model, 1, 1, 3, 1)
    @test positive == 0
    @test damping ≈ expected_rates[3]
    positive, damping = source_split(source, model, 2, 1, 2, 1)
    @test positive ≈ 0.25 * spectral_frequency_power_average(cgrid, 2, 1, 2) / 0.2^2 * 2.0
    @test damping == 0

    damping_model = SpectralWaveModel(RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1)), cgrid;
                      horizontal_advection=no_advection,
                      sources=FrequencyDissipation(rate=0.5,
                      reference_frequency=0.2,
                      power=1),
                      timestepper=:SemiImplicitEuler)
    set!(damping_model, N=2.0)
    initial = copy(interior(damping_model.action))
    time_step!(damping_model, 0.25)
    for n in eachindex(cgrid.φ), m in eachindex(cgrid.frequency)
        damping = 0.5 * spectral_frequency_power_average(cgrid, m, n, 1) / 0.2
        @test damping_model.action[1, 1, m, n] ≈ initial[1, 1, m, n] / (1 + 0.25damping)
    end
    @test all(interior(damping_model.action) .>= 0)

    pgrid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0])
    bad_model = SpectralWaveModel(RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1)), pgrid;
                      horizontal_advection=nothing,
                      sources=source)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    invalid_reference = FrequencyDissipation(rate=0.1, reference_frequency=0.0)
    bad_model = SpectralWaveModel(RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1)), cgrid;
                      horizontal_advection=no_advection,
                      sources=invalid_reference)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    invalid_power = FrequencyDissipation(rate=0.1, power=-1)
    bad_model = SpectralWaveModel(RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1)), cgrid;
                      horizontal_advection=no_advection,
                      sources=invalid_power)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Wavenumber dissipation" begin
    grid = RectilinearGrid(CPU(); size=(2, 1, 1), x=(0, 2), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[0.5, 1.0, 2.0], φ=[0.0])
    rate = reshape([0.4, -0.2], 2, 1)
    source = WavenumberDissipation(rate=rate,
                                   reference_wavenumber=1.0,
                                   power=2)
    no_advection = nothing
    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=no_advection,
                      sources=source,
                      timestepper=:ForwardEuler)
    set!(model, N=2.0)
    compute_tendencies!(model)

    expected_rates = [0.4 * spectral_radial_power_average(cgrid, m, 1, 2)
                      for m in eachindex(cgrid.κ)]
    @test model.tendencies[1, 1, 1, 1] ≈ -expected_rates[1] * 2.0
    @test model.tendencies[1, 1, 2, 1] ≈ -expected_rates[2] * 2.0
    @test model.tendencies[1, 1, 3, 1] ≈ -expected_rates[3] * 2.0
    @test model.tendencies[2, 1, 2, 1] ≈ 0.2 * spectral_radial_power_average(cgrid, 2, 1, 2) * 2.0

    positive, damping = source_split(source, model, 1, 1, 3, 1)
    @test positive == 0
    @test damping ≈ expected_rates[3]
    positive, damping = source_split(source, model, 2, 1, 2, 1)
    @test positive ≈ 0.2 * spectral_radial_power_average(cgrid, 2, 1, 2) * 2.0
    @test damping == 0

    damping_model = SpectralWaveModel(RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1)), cgrid;
                      horizontal_advection=no_advection,
                      sources=WavenumberDissipation(rate=0.5,
                      reference_wavenumber=1.0,
                      power=1),
                      timestepper=:SemiImplicitEuler)
    set!(damping_model, N=2.0)
    initial = copy(interior(damping_model.action))
    time_step!(damping_model, 0.25)
    for n in eachindex(cgrid.φ), m in eachindex(cgrid.κ)
        damping = 0.5 * spectral_radial_power_average(cgrid, m, n, 1)
        @test damping_model.action[1, 1, m, n] ≈ initial[1, 1, m, n] / (1 + 0.25damping)
    end
    @test all(interior(damping_model.action) .>= 0)

    square_grid = CartesianWaveVectorGrid(Float64; kx=[0.0], ky=[0.0],
                                          kx_faces=[-1.0, 1.0], ky_faces=[-1.0, 1.0])
    cartesian_model = SpectralWaveModel(RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1)), square_grid;
                      horizontal_advection=no_advection,
                      sources=WavenumberDissipation(rate=0.3,
                      reference_wavenumber=1.0,
                      power=1),
                      timestepper=:ForwardEuler)
    set!(cartesian_model, N=2.0)
    compute_tendencies!(cartesian_model)
    @test cartesian_model.tendencies[1, 1, 1, 1] ≈ -0.3 * spectral_radial_power_average(square_grid, 1, 1, 1) * 2.0

    unsupported = WavenumberDissipation(rate=0.1, power=3)
    bad_model = SpectralWaveModel(RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1)), square_grid;
                      horizontal_advection=nothing,
                      sources=unsupported)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    invalid_reference = WavenumberDissipation(rate=0.1, reference_wavenumber=0.0)
    bad_model = SpectralWaveModel(RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1)), cgrid;
                      horizontal_advection=nothing,
                      sources=invalid_reference)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    invalid_power = WavenumberDissipation(rate=0.1, power=-1)
    bad_model = SpectralWaveModel(RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1)), cgrid;
                      horizontal_advection=nothing,
                      sources=invalid_power)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Mean-frequency dissipation" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    φ = range(0, 2pi; length=5)[1:4]
    cgrid = FrequencyDirectionGrid(; frequency=[0.1, 0.2, 0.4], φ=φ)
    no_advection = nothing
    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=no_advection,
                      sources=MeanFrequencyDissipation(rate=0.5, reference_frequency=0.2, power=2),
                      timestepper=:ForwardEuler)
    set!(model, N=(x, y, kx, ky) -> abs(hypot(kx, ky) - cgrid.κ[3]) < 1e-12 ? 2.0 : 0.5)

    fmean = mean_frequency(model.action)[1, 1]
    _, damping = source_split(model.sources, model, 1, 1, 2, 1)
    @test damping ≈ 0.5 * (fmean / 0.2)^2

    compute_tendencies!(model)
    @test model.tendencies[1, 1, 2, 1] ≈ -damping * model.action[1, 1, 2, 1]

    initial = model.action[1, 1, 2, 1]
    time_step!(model, 1.0)
    @test model.action[1, 1, 2, 1] == 0

    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=no_advection,
                      sources=MeanFrequencyDissipation(rate=0.5, reference_frequency=0.2),
                      timestepper=:SemiImplicitEuler)
    set!(model, N=initial)
    compute_tendencies!(model)
    tendency = model.tendencies[1, 1, 2, 1]
    _, damping = source_split(model.sources, model, 1, 1, 2, 1)
    expected = (initial + tendency + damping * initial) / (1 + damping)
    time_step!(model, 1.0)
    @test model.action[1, 1, 2, 1] ≈ expected

    pgrid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0])
    model = SpectralWaveModel(grid, pgrid; horizontal_advection=nothing, sources=MeanFrequencyDissipation(rate=0.1))
    set!(model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(model)
end

@testset "Peak-frequency dissipation" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    φ = range(0, 2pi; length=5)[1:4]
    cgrid = FrequencyDirectionGrid(; frequency=[0.1, 0.2, 0.4], φ=φ)
    no_advection = nothing
    source = PeakFrequencyDissipation(rate=0.3,
                                      reference_frequency=0.2,
                                      power=2)
    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=no_advection,
                      sources=source,
                      timestepper=:ForwardEuler)
    set!(model, N=(x, y, kx, ky) -> abs(hypot(kx, ky) - cgrid.κ[3]) < 1e-12 ? 4.0 : 1.0)

    fpeak = peak_frequency(model.action)[1, 1]
    _, damping = source_split(source, model, 1, 1, 2, 1)
    @test fpeak == 0.4
    @test damping ≈ 0.3 * (fpeak / 0.2)^2
    compute_tendencies!(model)
    @test all(interior(model.tendencies) .≈ -damping .* interior(model.action))

    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=no_advection,
                      sources=source,
                      timestepper=:SemiImplicitEuler)
    set!(model, N=(x, y, kx, ky) -> abs(hypot(kx, ky) - cgrid.κ[3]) < 1e-12 ? 4.0 : 1.0)
    initial = copy(interior(model.action))
    time_step!(model, 0.25)
    @test all(interior(model.action) .≈ initial ./ (1 + 0.25damping))
    @test all(interior(model.action) .>= 0)

    pgrid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0])
    bad_model = SpectralWaveModel(grid, pgrid; horizontal_advection=nothing, sources=source)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    invalid_reference = PeakFrequencyDissipation(rate=0.1, reference_frequency=0.0)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=no_advection, sources=invalid_reference)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    invalid_power = PeakFrequencyDissipation(rate=0.1, power=-1)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=no_advection, sources=invalid_power)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Mean-square-wavenumber dissipation" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(;
                                κ=[0.4, 0.9, 1.6],
                                κ_faces=[0.2, 0.65, 1.2, 2.0],
                                φ=range(0, 2pi; length=9)[1:8])
    source = MeanSquareWavenumberDissipation(rate=0.25,
                                             reference_wavenumber=1.0,
                                             power=2)
    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=nothing,
                      sources=source,
                      timestepper=:SemiImplicitEuler)
    set!(model, N=(x, y, kx, ky) -> hypot(kx, ky) > 1.0 ? 2.0 : 0.5)

    rms_k = root_mean_square_wavenumber(model.action)[1, 1]
    _, damping = source_split(source, model, 1, 1, 2, 1)
    @test damping ≈ 0.25 * rms_k^2
    compute_tendencies!(model)
    @test model.tendencies[1, 1, 2, 1] ≈ -damping * model.action[1, 1, 2, 1]

    initial = copy(interior(model.action))
    time_step!(model, 0.5)
    @test all(interior(model.action) .≈ initial ./ (1 + 0.5damping))
    @test all(interior(model.action) .>= 0)

    invalid_reference = MeanSquareWavenumberDissipation(rate=0.1, reference_wavenumber=0.0)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=invalid_reference)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    invalid_power = MeanSquareWavenumberDissipation(rate=0.1, power=-1)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=invalid_power)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Peak-wavenumber dissipation" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[0.5, 1.0, 2.0], φ=[0.0, pi])
    source = PeakWavenumberDissipation(rate=0.3,
                                       reference_wavenumber=1.0,
                                       power=2)
    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=nothing,
                      sources=source,
                      timestepper=:ForwardEuler)
    set!(model, N=(x, y, kx, ky) -> hypot(kx, ky) == 2.0 ? 4.0 : 1.0)

    _, damping = source_split(source, model, 1, 1, 2, 1)
    @test damping ≈ 0.3 * 2.0^2
    compute_tendencies!(model)
    @test all(interior(model.tendencies) .≈ -damping .* interior(model.action))

    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=nothing,
                      sources=source,
                      timestepper=:SemiImplicitEuler)
    set!(model, N=(x, y, kx, ky) -> hypot(kx, ky) == 2.0 ? 4.0 : 1.0)
    initial = copy(interior(model.action))
    time_step!(model, 0.25)
    @test all(interior(model.action) .≈ initial ./ (1 + 0.25damping))
    @test all(interior(model.action) .>= 0)

    invalid_reference = PeakWavenumberDissipation(rate=0.1, reference_wavenumber=0.0)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=invalid_reference)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    invalid_power = PeakWavenumberDissipation(rate=0.1, power=-1)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=invalid_power)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Depth-limited breaking" begin
    grid = RectilinearGrid(CPU(); size=(2, 1, 1), x=(0, 2), y=(0, 1), z=(0, 1))
    cgrid = CartesianWaveVectorGrid(Float64; kx=[0.0], ky=[0.0])

    depth = reshape([2.0, 10.0], 2, 1)
    breaking = DepthLimitedBreaking(rate=0.4, depth=depth, gamma=0.8, power=2)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=breaking, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    compute_tendencies!(model)

    Hs = significant_wave_height(model.action)
    expected_damping = 0.4 * max(Hs[1, 1] / (0.8 * depth[1, 1]) - 1, 0)^2
    _, damping = source_split(breaking, model, 1, 1, 1, 1)
    @test damping ≈ expected_damping
    @test model.tendencies[1, 1, 1, 1] ≈ -expected_damping
    @test model.tendencies[2, 1, 1, 1] == 0

    stiff = DepthLimitedBreaking(rate=10.0, depth=1.0, gamma=1.0, power=1)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=stiff, timestepper=:SemiImplicitEuler)
    set!(model, N=1.0)
    _, damping = source_split(stiff, model, 1, 1, 1, 1)
    time_step!(model, 1.0)
    @test model.action[1, 1, 1, 1] ≈ 1 / (1 + damping)

    invalid = DepthLimitedBreaking(rate=0.1, depth=1.0, gamma=0.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=invalid)
    set!(model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(model)
end

@testset "Depth-dependent bottom friction" begin
    grid = RectilinearGrid(CPU(); size=(2, 1, 1), x=(0, 2), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[0.5, 1.0], φ=[0.0])

    depth = reshape([0.5, 2.0], 2, 1)
    friction = BottomFriction(rate=0.2,
                              depth=depth,
                              reference_depth=1.0,
                              wavenumber_power=1.0,
                              reference_wavenumber=0.5)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=friction, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    compute_tendencies!(model)

    factor_1 = spectral_radial_power_average(cgrid, 1, 1, 1) / 0.5
    factor_2 = spectral_radial_power_average(cgrid, 2, 1, 1) / 0.5
    @test model.tendencies[1, 1, 1, 1] ≈ -0.2 * (1 / depth[1, 1]) * factor_1
    @test model.tendencies[2, 1, 1, 1] ≈ -0.2 * (1 / depth[2, 1]) * factor_1
    @test model.tendencies[1, 1, 2, 1] / model.tendencies[1, 1, 1, 1] ≈ factor_2 / factor_1

    disabled = BottomFriction(rate=0.0, depth=(x, y, t) -> 0.1 + t)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=disabled, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    compute_tendencies!(model)
    @test maximum(abs, interior(model.tendencies)) < 1e-14

    stiff = BottomFriction(rate=10.0, depth=0.5, reference_depth=1.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=stiff, timestepper=:SemiImplicitEuler)
    set!(model, N=1.0)
    time_step!(model, 1.0)
    @test all(interior(model.action) .≈ 1 / 21)
end

@testset "Ice and swell damping" begin
    grid = RectilinearGrid(CPU(); size=(2, 1, 1), x=(0, 2), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[0.5, 1.0, 2.0], φ=[0.0])

    concentration = reshape([0.5, 2.0], 2, 1)
    ice = IceDamping(rate=0.2,
                     concentration=concentration,
                     wavenumber_power=2.0,
                     reference_wavenumber=0.5)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=ice, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    compute_tendencies!(model)

    factors = [spectral_radial_power_average(cgrid, m, 1, 2) / 0.5^2
               for m in eachindex(cgrid.κ)]
    @test model.tendencies[1, 1, 1, 1] ≈ -0.2 * 0.5 * factors[1]
    @test model.tendencies[2, 1, 1, 1] ≈ -0.2 * 1.0 * factors[1]
    @test model.tendencies[1, 1, 2, 1] ≈ -0.2 * 0.5 * factors[2]
    @test model.tendencies[1, 1, 3, 1] ≈ -0.2 * 0.5 * factors[3]

    cgrid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0, pi/2, pi])
    swell = SwellDissipation(rate=0.3, direction=0.0, spreading_power=2)
    model = SpectralWaveModel(RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1)), cgrid;
                      horizontal_advection=nothing,
                      sources=swell,
                      timestepper=:ForwardEuler)
    set!(model, N=1.0)
    compute_tendencies!(model)

    swell_weights = [1 - Ripple.wind_directional_weight(cgrid, 1, n, 0.0, 2)
                     for n in eachindex(cgrid.φ)]
    @test model.tendencies[1, 1, 1, 1] ≈ -0.3 * swell_weights[1]
    @test model.tendencies[1, 1, 1, 2] ≈ -0.3 * swell_weights[2]
    @test model.tendencies[1, 1, 1, 3] ≈ -0.3 * swell_weights[3]
    @test swell_weights[1] > 0
    @test 0.95 < swell_weights[3] < 1
end

@testset "Mean-direction dissipation" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0, pi/2, pi, 3pi/2])
    source = MeanDirectionDissipation(rate=0.25, power=1)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=source, timestepper=:ForwardEuler)
    set!(model, N=1.0)
    model.action[1, 1, 1, 1] = 10.0
    compute_tendencies!(model)

    @test abs(model.tendencies[1, 1, 1, 1]) < 1e-14
    @test model.tendencies[1, 1, 1, 2] ≈ -0.25
    @test model.tendencies[1, 1, 1, 3] ≈ -0.5
    @test model.tendencies[1, 1, 1, 4] ≈ -0.25

    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=source, timestepper=:SemiImplicitEuler)
    set!(model, N=1.0)
    model.action[1, 1, 1, 1] = 10.0
    time_step!(model, 1.0)
    @test model.action[1, 1, 1, 1] ≈ 10.0
    @test model.action[1, 1, 1, 2] ≈ 1 / 1.25
    @test model.action[1, 1, 1, 3] ≈ 1 / 1.5
    @test model.action[1, 1, 1, 4] ≈ 1 / 1.25
    @test all(interior(model.action) .>= 0)

    isotropic = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=source)
    set!(isotropic, N=1.0)
    compute_tendencies!(isotropic)
    @test maximum(abs, interior(isotropic.tendencies)) < 1e-14

    invalid = MeanDirectionDissipation(rate=0.1, power=-1)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=invalid)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Directional diffusion source" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    φ = range(0, 2pi; length=9)[1:8]
    cgrid = PolarWaveVectorGrid(; κ=[1.0], φ=φ)
    diffusion = DirectionalDiffusion(rate=0.1)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=diffusion, timestepper=:ForwardEuler)
    set!(model, N=(x, y, kx, ky) -> ky > 0 && abs(kx) < 1e-12 ? 10.0 : 1.0)

    initial_action = total_action(model.action)
    compute_tendencies!(model)
    weighted_tendency = sum(model.tendencies[1, 1, 1, n] * spectral_weight(cgrid, 1, n)
                            for n in eachindex(cgrid.φ))
    @test abs(weighted_tendency) < 1e-12
    @test model.tendencies[1, 1, 1, 3] < 0
    @test model.tendencies[1, 1, 1, 2] > 0
    @test model.tendencies[1, 1, 1, 4] > 0

    positive, damping = source_split(diffusion, model, 1, 1, 1, 3)
    @test source_tendency(diffusion, model, 1, 1, 1, 3) ≈ positive - damping * model.action[1, 1, 1, 3]
    @test damping > 0

    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=diffusion, timestepper=:SemiImplicitEuler)
    set!(model, N=(x, y, kx, ky) -> ky > 0 && abs(kx) < 1e-12 ? 10.0 : 1.0)
    time_step!(model, 0.1)
    @test all(interior(model.action) .>= 0)
    @test total_action(model.action) ≈ initial_action

    cartesian = CartesianWaveVectorGrid(Float64; kx=[0.0], ky=[0.0])
    bad_model = SpectralWaveModel(grid, cartesian; horizontal_advection=nothing, sources=diffusion)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    negative = DirectionalDiffusion(rate=-0.1)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=negative)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Directional advection source" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    φ = range(0, 2pi; length=9)[1:8]
    cgrid = PolarWaveVectorGrid(; κ=[1.0], φ=φ)
    advection = DirectionalAdvection(velocity=0.04)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=advection, timestepper=:ForwardEuler)
    set!(model, N=(x, y, kx, ky) -> kx > 0 && abs(ky) < 1e-12 ? 10.0 : 1.0)

    initial_action = total_action(model.action)
    compute_tendencies!(model)
    weighted_tendency = sum(model.tendencies[1, 1, 1, n] * spectral_weight(cgrid, 1, n)
                            for n in eachindex(cgrid.φ))
    @test abs(weighted_tendency) < 1e-12
    @test model.tendencies[1, 1, 1, 1] < 0
    @test model.tendencies[1, 1, 1, 2] > 0

    positive, damping = source_split(advection, model, 1, 1, 1, 2)
    @test source_tendency(advection, model, 1, 1, 1, 2) ≈ positive - damping * model.action[1, 1, 1, 2]
    @test damping > 0

    time_step!(model, 0.01)
    @test all(interior(model.action) .>= 0)
    @test total_action(model.action) ≈ initial_action

    clockwise = DirectionalAdvection(angular_velocity=-0.04)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=clockwise, timestepper=:ForwardEuler)
    set!(model, N=(x, y, kx, ky) -> ky < 0 && abs(kx) < 1e-12 ? 10.0 : 1.0)
    compute_tendencies!(model)
    weighted_tendency = sum(model.tendencies[1, 1, 1, n] * spectral_weight(cgrid, 1, n)
                            for n in eachindex(cgrid.φ))
    @test abs(weighted_tendency) < 1e-12
    @test model.tendencies[1, 1, 1, 7] < 0
    @test model.tendencies[1, 1, 1, 6] > 0

    fgrid = FrequencyDirectionGrid(; frequency=[0.1, 0.2], φ=φ)
    model = SpectralWaveModel(grid, fgrid; horizontal_advection=nothing, sources=DirectionalAdvection(velocity=0.02))
    set!(model, N=(x, y, kx, ky) -> 1 + max(kx, 0))
    compute_tendencies!(model)
    weighted_tendency = sum(model.tendencies[1, 1, m, n] * spectral_weight(fgrid, m, n)
                            for n in eachindex(fgrid.φ), m in eachindex(fgrid.frequency))
    @test abs(weighted_tendency) < 1e-12

    one_direction = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0])
    model = SpectralWaveModel(grid, one_direction; horizontal_advection=nothing, sources=advection)
    set!(model, N=1.0)
    compute_tendencies!(model)
    @test maximum(abs, interior(model.tendencies)) == 0

    cartesian = CartesianWaveVectorGrid(Float64; kx=[0.0], ky=[0.0])
    bad_model = SpectralWaveModel(grid, cartesian; horizontal_advection=nothing, sources=advection)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    bounded_direction = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0, pi],
                                            topology=(NoFlux(), Bounded()))
    bad_model = SpectralWaveModel(grid, bounded_direction; horizontal_advection=nothing, sources=advection)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Radial diffusion source" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[0.5, 1.0, 2.0], φ=[0.0])
    diffusion = RadialDiffusion(rate=0.05)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=diffusion, timestepper=:ForwardEuler)
    set!(model, N=(x, y, kx, ky) -> abs(hypot(kx, ky) - 1.0) < 1e-12 ? 10.0 : 1.0)

    initial_action = total_action(model.action)
    compute_tendencies!(model)
    weighted_tendency = sum(model.tendencies[1, 1, m, 1] * spectral_weight(cgrid, m, 1)
                            for m in eachindex(cgrid.κ))
    @test abs(weighted_tendency) < 1e-12
    @test model.tendencies[1, 1, 2, 1] < 0
    @test model.tendencies[1, 1, 1, 1] > 0
    @test model.tendencies[1, 1, 3, 1] > 0

    positive, damping = source_split(diffusion, model, 1, 1, 2, 1)
    @test source_tendency(diffusion, model, 1, 1, 2, 1) ≈ positive - damping * model.action[1, 1, 2, 1]
    @test damping > 0

    time_step!(model, 0.01)
    @test all(interior(model.action) .>= 0)
    @test total_action(model.action) ≈ initial_action

    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=RadialDiffusion(rate=0.05), timestepper=:SemiImplicitEuler)
    set!(model, N=(x, y, kx, ky) -> abs(hypot(kx, ky) - 1.0) < 1e-12 ? 10.0 : 1.0)
    time_step!(model, 0.1)
    @test all(interior(model.action) .>= 0)

    cartesian = CartesianWaveVectorGrid(Float64; kx=[0.0], ky=[0.0])
    bad_model = SpectralWaveModel(grid, cartesian; horizontal_advection=nothing, sources=diffusion)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    periodic_radial = PolarWaveVectorGrid(; κ=[0.5, 1.0], φ=[0.0],
                                          topology=(Periodic(), Periodic()))
    bad_model = SpectralWaveModel(grid, periodic_radial; horizontal_advection=nothing, sources=diffusion)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    negative = RadialDiffusion(rate=-0.1)
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=negative)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Radial advection source" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[0.5, 1.0, 2.0], φ=[0.0])
    advection = RadialAdvection(velocity=0.03)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=advection, timestepper=:ForwardEuler)
    set!(model, N=(x, y, kx, ky) -> abs(hypot(kx, ky) - 0.5) < 1e-12 ? 10.0 : 1.0)

    initial_action = total_action(model.action)
    compute_tendencies!(model)
    weighted_tendency = sum(model.tendencies[1, 1, m, 1] * spectral_weight(cgrid, m, 1)
                            for m in eachindex(cgrid.κ))
    @test abs(weighted_tendency) < 1e-12
    @test model.tendencies[1, 1, 1, 1] < 0
    @test model.tendencies[1, 1, 2, 1] > 0

    positive, damping = source_split(advection, model, 1, 1, 2, 1)
    @test source_tendency(advection, model, 1, 1, 2, 1) ≈ positive - damping * model.action[1, 1, 2, 1]
    @test damping > 0

    time_step!(model, 0.01)
    @test all(interior(model.action) .>= 0)
    @test total_action(model.action) ≈ initial_action

    downshift = RadialAdvection(velocity=-0.03)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=downshift, timestepper=:ForwardEuler)
    set!(model, N=(x, y, kx, ky) -> abs(hypot(kx, ky) - 2.0) < 1e-12 ? 10.0 : 1.0)
    compute_tendencies!(model)
    weighted_tendency = sum(model.tendencies[1, 1, m, 1] * spectral_weight(cgrid, m, 1)
                            for m in eachindex(cgrid.κ))
    @test abs(weighted_tendency) < 1e-12
    @test model.tendencies[1, 1, 3, 1] < 0
    @test model.tendencies[1, 1, 2, 1] > 0

    fgrid = FrequencyDirectionGrid(; frequency=[0.1, 0.2, 0.4], φ=[0.0])
    model = SpectralWaveModel(grid, fgrid; horizontal_advection=nothing, sources=RadialAdvection(velocity=0.01))
    set!(model, N=(x, y, kx, ky) -> 1 + hypot(kx, ky))
    compute_tendencies!(model)
    weighted_tendency = sum(model.tendencies[1, 1, m, 1] * spectral_weight(fgrid, m, 1)
                            for m in eachindex(fgrid.frequency))
    @test abs(weighted_tendency) < 1e-12

    cartesian = CartesianWaveVectorGrid(Float64; kx=[0.0], ky=[0.0])
    bad_model = SpectralWaveModel(grid, cartesian; horizontal_advection=nothing, sources=advection)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    periodic_radial = PolarWaveVectorGrid(; κ=[0.5, 1.0], φ=[0.0],
                                          topology=(Periodic(), Periodic()))
    bad_model = SpectralWaveModel(grid, periodic_radial; horizontal_advection=nothing, sources=advection)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Nonlinear spectral transfer source" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[0.5, 1.0, 2.0], φ=[0.0])
    transfer = NonlinearSpectralTransfer((
        SpectralTransferInteraction((1, 1), (3, 1); rate=0.25),
    ); power=2)
    no_advection = nothing
    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=no_advection,
                      sources=transfer,
                      timestepper=:ForwardEuler)
    set!(model, N=0.0)
    model.action[1, 1, 1, 1] = 2.0
    model.action[1, 1, 2, 1] = 1.0
    model.action[1, 1, 3, 1] = 0.5

    initial_action = total_action(model.action)
    compute_tendencies!(model)

    donor_weight = spectral_weight(cgrid, 1, 1)
    receiver_weight = spectral_weight(cgrid, 3, 1)
    expected_transfer = 0.25 * 2.0^2
    @test model.tendencies[1, 1, 1, 1] ≈ -expected_transfer
    @test model.tendencies[1, 1, 2, 1] == 0
    @test model.tendencies[1, 1, 3, 1] ≈ expected_transfer * donor_weight / receiver_weight

    weighted_tendency = sum(model.tendencies[1, 1, m, 1] * spectral_weight(cgrid, m, 1)
                            for m in eachindex(cgrid.κ))
    @test abs(weighted_tendency) < 1e-14

    positive, damping = source_split(transfer, model, 1, 1, 1, 1)
    @test positive == 0
    @test damping ≈ 0.25 * 2.0
    positive, damping = source_split(transfer, model, 1, 1, 3, 1)
    @test positive ≈ expected_transfer * donor_weight / receiver_weight
    @test damping == 0

    time_step!(model, 0.1)
    @test all(interior(model.action) .>= 0)
    @test total_action(model.action) ≈ initial_action
    @test model.action[1, 1, 1, 1] < 2.0
    @test model.action[1, 1, 3, 1] > 0.5

    invalid = NonlinearSpectralTransfer((
        SpectralTransferInteraction((1, 1), (4, 1); rate=0.1),
    ))
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=invalid)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    negative_rate = NonlinearSpectralTransfer((
        SpectralTransferInteraction((1, 1), (2, 1); rate=-0.1),
    ))
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=negative_rate)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Nonlinear invariant transfer source" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0, pi/2, pi, 3pi/2])
    stencil = SpectralTransferStencil(((1, 1), (1, 3), (1, 2), (1, 4)),
                                      (-1, -1, 1, 1);
                                      rate=0.05)
    transfer = NonlinearInvariantTransfer((stencil,); power=1)
    no_advection = nothing
    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=no_advection,
                      sources=transfer,
                      timestepper=:ForwardEuler)
    set!(model, N=(x, y, kx, ky) -> abs(ky) < 1e-12 ? 4.0 : 1.0)

    initial_action = total_action(model.action)
    initial_mx, initial_my = first_moment(model.action)
    compute_tendencies!(model)

    action_tendency = sum(model.tendencies[1, 1, 1, n] * spectral_weight(cgrid, 1, n)
                          for n in eachindex(cgrid.φ))
    mx_tendency = sum(model.tendencies[1, 1, 1, n] * spectral_weight(cgrid, 1, n) *
                      k_components(cgrid, 1, n)[1] for n in eachindex(cgrid.φ))
    my_tendency = sum(model.tendencies[1, 1, 1, n] * spectral_weight(cgrid, 1, n) *
                      k_components(cgrid, 1, n)[2] for n in eachindex(cgrid.φ))
    @test abs(action_tendency) < 1e-14
    @test abs(mx_tendency) < 1e-14
    @test abs(my_tendency) < 1e-14
    @test model.tendencies[1, 1, 1, 1] < 0
    @test model.tendencies[1, 1, 1, 3] < 0
    @test model.tendencies[1, 1, 1, 2] > 0
    @test model.tendencies[1, 1, 1, 4] > 0

    time_step!(model, 0.05)
    final_mx, final_my = first_moment(model.action)
    @test total_action(model.action) ≈ initial_action
    @test final_mx[1, 1] ≈ initial_mx[1, 1] atol=1e-14
    @test final_my[1, 1] ≈ initial_my[1, 1] atol=1e-14
    @test all(interior(model.action) .>= 0)

    @test_throws ArgumentError SpectralTransferStencil(((1, 1), (1, 2)), (-1, 0.5); rate=0.1)

    invalid = NonlinearInvariantTransfer((SpectralTransferStencil(((1, 1), (1, 5)), (-1, 1); rate=0.1),))
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=invalid)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Triad spectral transfer source" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(; frequency=[0.1, 0.2, 0.3], φ=[0.0])
    triad = TriadSpectralTransfer((
        TriadTransferInteraction((1, 1), (2, 1), (3, 1); rate=0.03),
    ); power=1)
    no_advection = nothing
    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=no_advection,
                      sources=triad,
                      timestepper=:ForwardEuler)
    set!(model, N=0.5)
    model.action[1, 1, 1, 1] = 4.0
    model.action[1, 1, 2, 1] = 3.0
    model.action[1, 1, 3, 1] = 0.25

    compute_tendencies!(model)

    weights = spectral_weights(cgrid)
    expected_flux = 0.03 * 4.0 * 3.0
    @test model.tendencies[1, 1, 1, 1] ≈ -expected_flux / weights[1, 1]
    @test model.tendencies[1, 1, 2, 1] ≈ -expected_flux / weights[2, 1]
    @test model.tendencies[1, 1, 3, 1] ≈ expected_flux / weights[3, 1]

    energy_tendency = sum(model.tendencies[1, 1, m, 1] * weights[m, 1] * cgrid.frequency[m]
                          for m in eachindex(cgrid.frequency))
    action_tendency = sum(model.tendencies[1, 1, m, 1] * weights[m, 1]
                          for m in eachindex(cgrid.frequency))
    @test abs(energy_tendency) < 1e-14
    @test action_tendency ≈ -expected_flux

    positive, damping = source_split(triad, model, 1, 1, 1, 1)
    @test positive == 0
    @test damping ≈ expected_flux / (weights[1, 1] * model.action[1, 1, 1, 1])
    positive, damping = source_split(triad, model, 1, 1, 3, 1)
    @test positive ≈ expected_flux / weights[3, 1]
    @test damping == 0

    initial_energy = sum(model.action[1, 1, m, 1] * weights[m, 1] * cgrid.frequency[m]
                         for m in eachindex(cgrid.frequency))
    initial_child = model.action[1, 1, 3, 1]
    time_step!(model, 0.01)
    final_energy = sum(model.action[1, 1, m, 1] * weights[m, 1] * cgrid.frequency[m]
                       for m in eachindex(cgrid.frequency))
    @test final_energy ≈ initial_energy
    @test model.action[1, 1, 1, 1] < 4.0
    @test model.action[1, 1, 2, 1] < 3.0
    @test model.action[1, 1, 3, 1] > initial_child
    @test all(interior(model.action) .>= 0)

    subharmonic = TriadSpectralTransfer((
        TriadTransferInteraction((1, 1), (1, 1), (2, 1); rate=0.02),
    ))
    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=no_advection,
                      sources=subharmonic,
                      timestepper=:ForwardEuler)
    set!(model, N=2.0)
    compute_tendencies!(model)
    subharmonic_flux = 0.02 * 2.0^2
    @test model.tendencies[1, 1, 1, 1] ≈ -2subharmonic_flux / weights[1, 1]
    @test model.tendencies[1, 1, 2, 1] ≈ subharmonic_flux / weights[2, 1]

    nonresonant = TriadSpectralTransfer((
        TriadTransferInteraction((1, 1), (1, 1), (3, 1); rate=0.02),
    ))
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=no_advection, sources=nonresonant)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    invalid = TriadSpectralTransfer((
        TriadTransferInteraction((1, 1), (2, 1), (4, 1); rate=0.02),
    ))
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=no_advection, sources=invalid)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    negative_rate = TriadSpectralTransfer((
        TriadTransferInteraction((1, 1), (2, 1), (3, 1); rate=-0.02),
    ))
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=no_advection, sources=negative_rate)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    polar = PolarWaveVectorGrid(; κ=[1.0, 2.0], φ=[0.0])
    bad_model = SpectralWaveModel(grid, polar; horizontal_advection=nothing, sources=triad)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Discrete interaction approximation source" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(;
                                   frequency=[0.2],
                                   φ=[0.0, pi/2, pi, 3pi/2])
    dia = DiscreteInteractionApproximation((
        QuadrupletTransferInteraction((1, 1), (1, 3), (1, 2), (1, 4); rate=0.05),
    ); power=1)
    no_advection = nothing
    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=no_advection,
                      sources=dia,
                      timestepper=:ForwardEuler)
    set!(model, N=0.5)
    model.action[1, 1, 1, 1] = 4.0
    model.action[1, 1, 1, 3] = 3.0
    model.action[1, 1, 1, 2] = 0.25
    model.action[1, 1, 1, 4] = 0.75

    initial_action = total_action(model.action)
    initial_energy = sum(model.action[1, 1, 1, n] * spectral_weight(cgrid, 1, n) *
                         cgrid.frequency[1] for n in eachindex(cgrid.φ))
    initial_mx, initial_my = first_moment(model.action)
    initial_receivers = model.action[1, 1, 1, 2] + model.action[1, 1, 1, 4]
    compute_tendencies!(model)

    weights = spectral_weights(cgrid)
    expected_flux = 0.05 * 4.0 * 3.0
    @test model.tendencies[1, 1, 1, 1] ≈ -expected_flux / weights[1, 1]
    @test model.tendencies[1, 1, 1, 3] ≈ -expected_flux / weights[1, 3]
    @test model.tendencies[1, 1, 1, 2] ≈ expected_flux / weights[1, 2]
    @test model.tendencies[1, 1, 1, 4] ≈ expected_flux / weights[1, 4]

    action_tendency = sum(model.tendencies[1, 1, 1, n] * spectral_weight(cgrid, 1, n)
                          for n in eachindex(cgrid.φ))
    energy_tendency = sum(model.tendencies[1, 1, 1, n] * spectral_weight(cgrid, 1, n) *
                          cgrid.frequency[1] for n in eachindex(cgrid.φ))
    mx_tendency = sum(model.tendencies[1, 1, 1, n] * spectral_weight(cgrid, 1, n) *
                      k_components(cgrid, 1, n)[1] for n in eachindex(cgrid.φ))
    my_tendency = sum(model.tendencies[1, 1, 1, n] * spectral_weight(cgrid, 1, n) *
                      k_components(cgrid, 1, n)[2] for n in eachindex(cgrid.φ))
    @test abs(action_tendency) < 1e-14
    @test abs(energy_tendency) < 1e-14
    @test abs(mx_tendency) < 1e-14
    @test abs(my_tendency) < 1e-14

    positive, damping = source_split(dia, model, 1, 1, 1, 1)
    @test positive == 0
    @test damping ≈ expected_flux / (weights[1, 1] * model.action[1, 1, 1, 1])
    positive, damping = source_split(dia, model, 1, 1, 1, 2)
    @test positive ≈ expected_flux / weights[1, 2]
    @test damping == 0

    time_step!(model, 0.01)
    final_mx, final_my = first_moment(model.action)
    final_energy = sum(model.action[1, 1, 1, n] * spectral_weight(cgrid, 1, n) *
                       cgrid.frequency[1] for n in eachindex(cgrid.φ))
    final_receivers = model.action[1, 1, 1, 2] + model.action[1, 1, 1, 4]
    @test total_action(model.action) ≈ initial_action
    @test final_energy ≈ initial_energy
    @test final_mx[1, 1] ≈ initial_mx[1, 1] atol=1e-14
    @test final_my[1, 1] ≈ initial_my[1, 1] atol=1e-14
    @test final_receivers > initial_receivers
    @test all(interior(model.action) .>= 0)

    nonresonant = DiscreteInteractionApproximation((
        QuadrupletTransferInteraction((1, 1), (1, 2), (1, 3), (1, 4); rate=0.05),
    ))
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=no_advection, sources=nonresonant)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    invalid = DiscreteInteractionApproximation((
        QuadrupletTransferInteraction((1, 1), (1, 3), (1, 2), (1, 5); rate=0.05),
    ))
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=no_advection, sources=invalid)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    negative_rate = DiscreteInteractionApproximation((
        QuadrupletTransferInteraction((1, 1), (1, 3), (1, 2), (1, 4); rate=-0.05),
    ))
    bad_model = SpectralWaveModel(grid, cgrid; horizontal_advection=no_advection, sources=negative_rate)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)

    polar = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0, pi/2, pi, 3pi/2])
    bad_model = SpectralWaveModel(grid, polar; horizontal_advection=nothing, sources=dia)
    set!(bad_model, N=1.0)
    @test_throws ArgumentError compute_tendencies!(bad_model)
end

@testset "Source positive/damping split" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0])

    sources = (
        nothing,
        RelaxationToSpectrum((x, y, kx, ky) -> 2.0; timescale=4.0),
        LinearWindInput(rate=0.3),
        LinearWindInput(rate=-0.2),
        ExponentialWindInput(rate=0.5, direction=0.0, spreading_power=2),
        PowerLawWindInput(rate=0.2, speed=2.0, reference_speed=1.0),
        WaveAgeWindInput(rate=0.2, speed=2.0, inverse_wave_age_threshold=0.0),
        SaturationDissipation(rate=0.1, threshold=0.1),
        WhitecappingDissipation(rate=0.1, saturation_threshold=0.1),
        FrequencyDissipation(rate=0.1, reference_frequency=0.1),
        WavenumberDissipation(rate=0.1),
        MeanFrequencyDissipation(rate=0.1, reference_frequency=0.1),
        PeakFrequencyDissipation(rate=0.1, reference_frequency=0.1),
        MeanSquareWavenumberDissipation(rate=0.1),
        PeakWavenumberDissipation(rate=0.1),
        MeanDirectionDissipation(rate=0.1),
        DepthLimitedBreaking(rate=0.1, depth=0.5),
        BottomFriction(rate=0.2, depth=0.5, reference_depth=1.0),
        IceDamping(rate=0.2, concentration=0.5),
        SwellDissipation(rate=0.2, direction=pi),
        DirectionalDiffusion(rate=0.2),
        DirectionalAdvection(velocity=0.2),
        RadialDiffusion(rate=0.2),
        RadialAdvection(velocity=0.2),
        NonlinearSpectralTransfer((SpectralTransferInteraction((1, 1), (1, 1); rate=0.2),)),
        NonlinearInvariantTransfer((SpectralTransferStencil(((1, 1), (1, 1)), (-1, 1); rate=0.2),)),
        TriadSpectralTransfer((TriadTransferInteraction((1, 1), (1, 1), (2, 1); rate=0.2),)),
        DiscreteInteractionApproximation((QuadrupletTransferInteraction((1, 1), (1, 3), (1, 2), (1, 4); rate=0.2),)),
        SourceTermSet((LinearWindInput(rate=0.3), BottomFriction(rate=0.1))),
    )

    for source in sources
        spectral_grid = source isa Union{FrequencyDissipation, MeanFrequencyDissipation, PeakFrequencyDissipation, TriadSpectralTransfer} ?
            FrequencyDirectionGrid(; frequency=[0.1], φ=[0.0]) : cgrid
        spectral_grid = source isa TriadSpectralTransfer ?
            FrequencyDirectionGrid(; frequency=[0.1, 0.2], φ=[0.0]) : spectral_grid
        spectral_grid = source isa DiscreteInteractionApproximation ?
            FrequencyDirectionGrid(; frequency=[0.1], φ=[0.0, pi/2, pi, 3pi/2]) : spectral_grid
        model = SpectralWaveModel(grid, spectral_grid; horizontal_advection=nothing, sources=source, timestepper=:ForwardEuler)
        set!(model, N=1.25)
        positive, damping = source_split(source, model, 1, 1, 1, 1)
        @test source_tendency(source, model, 1, 1, 1, 1) ≈ positive - damping * model.action[1, 1, 1, 1]
        @test implicit_source_rate(source, model, 1, 1, 1, 1) == damping
        @test damping >= 0
    end
end

@testset "Semi-implicit source update" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = CartesianWaveVectorGrid(Float64; kx=[0.5], ky=[0.0])

    damping = BottomFriction(rate=100.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=damping, timestepper=:SemiImplicitEuler)
    set!(model, N=1.0)
    time_step!(model, 1.0)
    @test model.action[1, 1, 1, 1] ≈ 1 / 101

    relaxation = RelaxationToSpectrum((x, y, kx, ky) -> 2.0; timescale=1.0)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing, sources=relaxation, timestepper=:SemiImplicitEuler)
    set!(model, N=0.0)
    time_step!(model, 0.5)
    @test model.action[1, 1, 1, 1] ≈ 2 / 3

    pgrid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0])
    whitecapping = WhitecappingDissipation(rate=0.5,
                                           saturation_threshold=1.0,
                                           saturation_power=1.0,
                                           wavenumber_power=0.0)
    model = SpectralWaveModel(grid, pgrid; horizontal_advection=nothing, sources=whitecapping, timestepper=:SemiImplicitEuler)
    set!(model, N=1.0)
    λ = Ripple.implicit_source_rate(whitecapping, model, 1, 1, 1, 1)
    time_step!(model, 0.25)
    @test model.action[1, 1, 1, 1] ≈ 1 / (1 + 0.25λ)
    @test model.action[1, 1, 1, 1] > 0
end
