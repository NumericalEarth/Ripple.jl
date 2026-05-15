@testset "Idealized wind forcing" begin
    ccw = StationaryVortexWind(; center=(0.0, 0.0),
                                diameter=4.0,
                                speed=0.2,
                                rotation=Counterclockwise())
    cw = StationaryVortexWind(; center=(0.0, 0.0),
                               diameter=4.0,
                               speed=0.2,
                               rotation=Clockwise())

    @test wind_speed(ccw, 1.0, 0.0) > 0
    @test wind_angle(ccw, 1.0, 0.0) ≈ pi / 2
    @test wind_angle(cw, 1.0, 0.0) ≈ -pi / 2
    @test wind_velocity(ccw, 0.0, 0.0) == (0.0, 0.0)

    track = LinearStormTrack([0.0, 1.0, 3.0],
                             [(0.0, 0.0), (2.0, 1.0), (4.0, -1.0)])
    @test track(-1.0) == (0.0, 0.0)
    @test track(0.5) == (1.0, 0.5)
    @test track(2.0) == (3.0, 0.0)
    @test track(4.0) == (4.0, -1.0)

    storm = IdealizedHurricaneWind(; center=t -> (t, 0.0),
                                    vmax=0.3,
                                    rmax=1.0,
                                    radius=4.0,
                                    inflow_angle=pi / 12,
                                    background=(0.05, 0.0))
    vx, vy = wind_velocity(storm, 2.0, 0.0, 1.0)
    @test isfinite(vx)
    @test isfinite(vy)
    @test wind_speed(storm, 2.0, 0.0, 1.0) > 0

    holland = HollandHurricaneWind(; center=t -> (t, 0.0),
                                    vmax=0.3,
                                    rmax=1.0,
                                    radius=4.0,
                                    shape_parameter=1.5,
                                    inflow_angle=0.0,
                                    background=(0.0, 0.0),
                                    rotation=Counterclockwise())
    @test wind_speed(holland, 1.0, 0.0, 0.0) ≈ 0.3
    @test wind_speed(holland, 2.0, 0.0, 1.0) ≈ 0.3
    @test wind_speed(holland, 0.0, 0.0, 0.0) == 0
    @test wind_speed(holland, 3.0, 0.0, 0.0) < 0.3
    @test wind_angle(holland, 1.0, 0.0, 0.0) ≈ pi / 2

    holland_cw = HollandHurricaneWind(; vmax=0.3,
                                       rmax=1.0,
                                       radius=4.0,
                                       rotation=Clockwise())
    @test wind_angle(holland_cw, 1.0, 0.0) ≈ -pi / 2

    holland_inflow = HollandHurricaneWind(; vmax=0.3,
                                           rmax=1.0,
                                           radius=4.0,
                                           inflow_angle=pi / 12,
                                           background=(0.05, 0.0))
    hvx, hvy = wind_velocity(holland_inflow, 2.0, 0.0, 0.0)
    @test isfinite(hvx)
    @test isfinite(hvy)
    @test wind_speed(holland_inflow, 2.0, 0.0, 0.0) > 0

    tracked_holland = HollandHurricaneWind(; center=track,
                                            vmax=0.3,
                                            rmax=1.0,
                                            radius=4.0)
    @test wind_speed(tracked_holland, 3.0, 1.0, 1.0) ≈ 0.3

    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 2), y=(-0.5, 0.5), z=(0, 1))
    spectral_grid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0, pi / 2])
    model = SpectralWaveModel(; advection=nothing, grid,
                                spectral_grid,
                                sources=ExponentialWindInput(rate=ccw,
                                                              direction=ccw,
                                                              spreading_power=2))
    set!(model, N=1.0)
    compute_tendencies!(model)

    ccw_direction = wind_angle(ccw, 1.0, 0.0, model.clock.time)
    ccw_speed = wind_speed(ccw, 1.0, 0.0, model.clock.time)
    ccw_weights = [Ripple.wind_directional_weight(spectral_grid, 1, n, ccw_direction, 2)
                   for n in 1:2]

    @test model.tendencies[1, 1, 1, 1] ≈ ccw_speed * ccw_weights[1]
    @test model.tendencies[1, 1, 1, 2] ≈ ccw_speed * ccw_weights[2]
    @test ccw_weights ≈ fill(0.25, 2)

    holland_source = HollandHurricaneWind(; center=(0.0, 0.0),
                                           vmax=0.3,
                                           rmax=1.0,
                                           radius=4.0)
    model = SpectralWaveModel(; advection=nothing, grid,
                                spectral_grid,
                                sources=ExponentialWindInput(rate=holland_source,
                                                              direction=holland_source,
                                                              spreading_power=2))
    set!(model, N=1.0)
    compute_tendencies!(model)

    holland_direction = wind_angle(holland_source, 1.0, 0.0, model.clock.time)
    holland_speed = wind_speed(holland_source, 1.0, 0.0, model.clock.time)
    holland_weights = [Ripple.wind_directional_weight(spectral_grid, 1, n, holland_direction, 2)
                       for n in 1:2]

    @test model.tendencies[1, 1, 1, 1] ≈ holland_speed * holland_weights[1]
    @test model.tendencies[1, 1, 1, 2] ≈ holland_speed * holland_weights[2]
    @test holland_weights ≈ fill(0.25, 2)

    @test_throws ArgumentError StationaryVortexWind(; diameter=0.0)
    @test_throws ArgumentError IdealizedHurricaneWind(; rmax=2.0, radius=1.0)
    @test_throws ArgumentError HollandHurricaneWind(; vmax=-1.0)
    @test_throws ArgumentError HollandHurricaneWind(; rmax=2.0, radius=1.0)
    @test_throws ArgumentError HollandHurricaneWind(; shape_parameter=0.0)
    @test_throws ArgumentError LinearStormTrack([0.0, 0.0], [(0.0, 0.0), (1.0, 1.0)])
    @test_throws ArgumentError LinearStormTrack([0.0], [(0.0, 0.0), (1.0, 1.0)])
    @test_throws ArgumentError LinearStormTrack([0.0], [(0.0, 0.0, 0.0)])
end
