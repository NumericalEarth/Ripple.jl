@testset "JONSWAP initial condition" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(Float64;
                                   frequency=range(0.08, 0.20; length=49),
                                   theta=range(0, 2pi; length=73)[1:72])
    N = WaveActionField(grid, cgrid)
    spectrum = JONSWAPSpectrum(; Hs=2.0, Tp=8.0, direction=pi / 2, spread=0.25)

    set!(N, spectrum)

    @test all(interior(N) .>= 0)
    @test isfinite(total_action(N))
    @test significant_wave_height(N)[1, 1] > 0
    @test abs(peak_frequency(N)[1, 1] - inv(8.0)) <= cgrid.frequency[2] - cgrid.frequency[1]
    @test peak_direction(N)[1, 1] ≈ pi / 2

    calm = WaveActionField(grid, cgrid)
    set!(calm, JONSWAPSpectrum(; Hs=0.0))
    @test total_action(calm) == 0
    @test significant_wave_height(calm)[1, 1] == 0

    @test_throws ArgumentError JONSWAPSpectrum(; Hs=-1.0)
    @test_throws ArgumentError JONSWAPSpectrum(; Tp=0.0)
    @test_throws ArgumentError JONSWAPSpectrum(; spread=0.0)
    @test_throws ArgumentError JONSWAPSpectrum(; gamma=0.0)
end
