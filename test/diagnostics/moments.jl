@testset "Spectral diagnostics" begin
    grid = RectilinearGrid(CPU(); size=(2, 2, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    φ = range(0, 2pi; length=9)[1:8]
    cgrid = PolarWaveVectorGrid(; κ=[1.0], φ=φ)
    N = WaveActionField(grid, cgrid)
    set!(N, 1.0)

    mx, my = first_moment(N)
    mxx, mxy, myy = second_moment(N)
    @test maximum(abs.(mx)) < 1e-12
    @test maximum(abs.(my)) < 1e-12
    @test maximum(abs.(mxy)) < 1e-12
    @test maximum(abs.(mxx .- myy)) < 1e-12
    @test all(mxx .+ myy .≈ pi / 2 * (cgrid.κ_faces[end]^4 - cgrid.κ_faces[1]^4))
    expected_mean_square = (pi / 2 * (cgrid.κ_faces[end]^4 - cgrid.κ_faces[1]^4)) /
                           (pi * (cgrid.κ_faces[end]^2 - cgrid.κ_faces[1]^2))
    @test all(mean_square_wavenumber(N) .≈ expected_mean_square)
    @test all(root_mean_square_wavenumber(N) .≈ sqrt(expected_mean_square))
    @test all(m0(N) .> 0)
    @test all(significant_wave_height(N) .> 0)

    set!(N, (x, y, κ, φ) -> abs(φ - pi / 2) < 1e-12 ? 10.0 : 1.0)
    @test all(peak_direction(N) .≈ pi / 2)
    @test all(peak_wavenumber(N) .≈ 1.0)
    @test all(deep_water_peak_phase_speed(N; gravity=9.0) .≈ 3.0)
    @test all(wave_age(N, 1.5; gravity=9.0) .≈ 2.0)
    expected_energy = zeros(2, 2)
    expected_group_speed = zeros(2, 2)
    for j in 1:2, i in 1:2
        expected_energy[i, j] = sum(N[i, j, m, n] * sqrt(9.0) *
                                    spectral_radial_power_measure(cgrid, m, n, 1 / 2)
                                    for n in eachindex(cgrid.φ), m in eachindex(cgrid.κ))
        expected_group_speed[i, j] = sum(N[i, j, m, n] * sqrt(9.0) / 2 *
                                         spectral_radial_power_measure(cgrid, m, n, -1 / 2)
                                         for n in eachindex(cgrid.φ), m in eachindex(cgrid.κ)) /
                                     m0(N)[i, j]
    end
    @test all(deep_water_energy_density(N; gravity=9.0) .≈ expected_energy)
    @test all(mean_deep_water_group_speed(N; gravity=9.0) .≈ expected_group_speed)
    @test total_deep_water_energy(N; gravity=9.0) ≈
          sum(deep_water_energy_density(N; gravity=9.0)) * xspacings(grid)[1] * yspacings(grid)[1]

    wind = StationaryVortexWind(; center=(0.0, 0.0), diameter=4.0, speed=2.0)
    @test all(isfinite, wave_age(N, wind; gravity=9.0))
end

@testset "Frequency-direction diagnostics" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    φ = range(0, 2pi; length=9)[1:8]
    cgrid = FrequencyDirectionGrid(; frequency=range(0.08, 0.2; length=4), φ=φ)
    N = WaveActionField(grid, cgrid)
    set!(N, 1.0)

    mx, my = first_moment(N)
    mxx, mxy, myy = second_moment(N)
    @test abs(mx[1, 1]) < 1e-12
    @test abs(my[1, 1]) < 1e-12
    @test abs(mxy[1, 1]) < 1e-12
    trace = pi / 2 * (cgrid.κ_faces[end]^4 - cgrid.κ_faces[1]^4)
    @test mxx[1, 1] + myy[1, 1] ≈ trace
    @test m0(N)[1, 1] ≈ sum(spectral_weights(cgrid))
    @test mean_square_wavenumber(N)[1, 1] ≈ trace / m0(N)[1, 1]
    @test root_mean_square_wavenumber(N)[1, 1] ≈ sqrt(trace / m0(N)[1, 1])

    expected_mean = sum(spectral_frequency_power_measure(cgrid, m, n, 1)
                        for n in eachindex(cgrid.φ), m in eachindex(cgrid.frequency)) /
                    sum(spectral_weights(cgrid))
    @test mean_frequency(N)[1, 1] ≈ expected_mean
    @test mean_period(N)[1, 1] ≈ inv(expected_mean)

    peak_kappa = radial_wavenumber(cgrid, 3, 1)
    peak_frequency_value = cgrid.frequency[3]
    set!(N, (x, y, f, φ) -> abs(f - peak_frequency_value) < 1e-12 ? 10.0 : 1.0)
    @test peak_frequency(N)[1, 1] == cgrid.frequency[3]
    @test peak_period(N)[1, 1] == inv(cgrid.frequency[3])
    @test peak_wavenumber(N)[1, 1] == cgrid.κ[3]
    @test deep_water_peak_phase_speed(N; gravity=9.81)[1, 1] ≈ sqrt(9.81 / cgrid.κ[3])
    @test wave_age(N, fill(2.0, 1, 1); gravity=9.81)[1, 1] ≈ sqrt(9.81 / cgrid.κ[3]) / 2
    energy = deep_water_energy_density(N; gravity=9.81)
    expected_energy = sum(N[1, 1, m, n] * sqrt(9.81) *
                          spectral_radial_power_measure(cgrid, m, n, 1 / 2)
                          for n in eachindex(cgrid.φ), m in eachindex(cgrid.frequency))
    expected_group_speed = sum(N[1, 1, m, n] * sqrt(9.81) / 2 *
                               spectral_radial_power_measure(cgrid, m, n, -1 / 2)
                               for n in eachindex(cgrid.φ), m in eachindex(cgrid.frequency)) /
                           m0(N)[1, 1]
    @test energy[1, 1] ≈ expected_energy
    @test mean_deep_water_group_speed(N; gravity=9.81)[1, 1] ≈ expected_group_speed
    @test energy[1, 1] > 0
    @test total_deep_water_energy(N; gravity=9.81) ≈ energy[1, 1]

    set!(N, (x, y, f, φ) -> abs(φ - pi / 2) < 1e-12 ? 10.0 : 1.0)
    @test peak_direction(N)[1, 1] == pi / 2

    pgrid = PolarWaveVectorGrid(; κ=[1.0], φ=φ)
    P = WaveActionField(grid, pgrid)
    set!(P, 1.0)
    @test_throws ArgumentError mean_frequency(P)
    @test_throws ArgumentError peak_frequency(P)
end
