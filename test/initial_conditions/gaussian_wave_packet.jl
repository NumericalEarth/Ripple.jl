@testset "Gaussian wave-packet initial condition" begin
    grid = RectilinearGrid(CPU(); size=(3, 2, 1), x=(0, 3), y=(0, 2), z=(0, 1))
    cgrid = CartesianWaveVectorGrid(Float64; kx=[0.4, 0.8], ky=[0.0])
    N = WaveActionField(grid, cgrid)
    packet = GaussianWavePacket(; x0=1.5,
                                y0=0.5,
                                kx0=0.8,
                                ky0=0.0,
                                spatial_width=0.5,
                                spectral_width=0.2,
                                amplitude=2.0)

    set!(N, packet)

    @test N[2, 1, 2, 1] ≈ 2.0
    @test N[1, 1, 2, 1] < N[2, 1, 2, 1]
    @test N[2, 1, 1, 1] < N[2, 1, 2, 1]
    @test all(interior(N) .>= 0)
    @test isfinite(total_action(N))

    pgrid = PolarWaveVectorGrid(;
                                κ=[0.5, 1.0],
                                φ=[0.0, pi/2, pi, 3pi/2])
    P = WaveActionField(grid, pgrid)
    polar_packet = GaussianWavePacket(; x0=1.5,
                                      y0=0.5,
                                      kx0=0.0,
                                      ky0=1.0,
                                      spatial_width=0.5,
                                      spectral_width=0.2,
                                      amplitude=1.0)
    set!(P, polar_packet)
    @test peak_wavenumber(P)[2, 1] == 1.0
    @test peak_direction(P)[2, 1] ≈ pi / 2

    calm = WaveActionField(grid, cgrid)
    set!(calm, GaussianWavePacket(; amplitude=0.0))
    @test total_action(calm) == 0

    @test_throws ArgumentError GaussianWavePacket(; spatial_width=0.0)
    @test_throws ArgumentError GaussianWavePacket(; spectral_width=0.0)
    @test_throws ArgumentError GaussianWavePacket(; amplitude=-1.0)
end
