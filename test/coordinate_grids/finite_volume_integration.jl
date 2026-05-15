import Oceananigans

@testset "Spectral finite-volume cell measures" begin
    physical_grid = RectilinearGrid(CPU();
                                    size=(2, 3, 1),
                                    x=(0, 1),
                                    y=(0, 2),
                                    z=(0, 1),
                                    topology=(Bounded, Periodic, Bounded))
    @test Oceananigans.Grids.topology(physical_grid) == (Bounded, Periodic, Bounded)

    type_cartesian = CartesianWaveVectorGrid(Float64;
                                             kx=[-0.2, 0.2],
                                             ky=[0.1],
                                             topology=(Periodic, Bounded))
    @test type_cartesian.topology[1] isa Periodic
    @test type_cartesian.topology[2] isa Bounded

    type_polar = PolarWaveVectorGrid(;
                                     κ=[0.4, 0.8],
                                     φ=[0.0, pi],
                                     topology=(NoFlux, Periodic))
    @test type_polar.topology[1] isa NoFlux
    @test type_polar.topology[2] isa Periodic

    type_frequency = FrequencyDirectionGrid(;
                                            frequency=[0.1, 0.2],
                                            φ=[0.0, pi],
                                            topology=(Bounded, Periodic))
    @test type_frequency.topology[1] isa Bounded
    @test type_frequency.topology[2] isa Periodic
    @test_throws Exception RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1), topology=(Periodic,))
    @test_throws ArgumentError CartesianWaveVectorGrid(Float64; kx=[0.0], ky=[0.0], topology=(:periodic, Bounded))

    polar = PolarWaveVectorGrid(Float32;
                                κ=[0.4, 0.8],
                                φ=[0.0f0, Float32(pi)],
                                κ_faces=[0.2, 0.6, 1.0],
                                φ_faces=Float32[-pi/2, pi/2, 3pi/2],
                                topology=(NoFlux, Periodic))
    @test eltype(polar.κ) === Float32
    @test eltype(polar.φ) === Float32
    @test architecture(polar) isa CPU
    @test_throws ArgumentError PolarWaveVectorGrid(; κ=[0.4])
    @test_throws ArgumentError PolarWaveVectorGrid(; φ=[0.0])

    fgrid_check = FrequencyDirectionGrid(;
                                         frequency=[0.1, 0.2],
                                         φ=[0.0, pi],
                                         frequency_faces=[0.05, 0.15, 0.25],
                                         φ_faces=[-pi/2, pi/2, 3pi/2],
                                         topology=(Bounded, Periodic))
    @test architecture(fgrid_check) isa CPU
    @test_throws ArgumentError FrequencyDirectionGrid(; φ=[0.0])
    @test_throws ArgumentError FrequencyDirectionGrid(; frequency=[0.1])

    cgrid = CartesianWaveVectorGrid(Float64; kx=range(-1, 1; length=5), ky=range(-2, 2; length=9))
    @test coordinate_size(cgrid) == (5, 9)
    @test cgrid.topology[1] isa Bounded
    @test cgrid.topology[2] isa Bounded
    @test all(diff(coordinate_centers(cgrid, 1)) .> 0)
    @test all(diff(coordinate_centers(cgrid, 2)) .> 0)
    @test all(diff(coordinate_faces(cgrid, 1)) .> 0)
    @test all(diff(coordinate_faces(cgrid, 2)) .> 0)
    @test spectral_cell_measures(cgrid) == spectral_weights(cgrid)
    @test spectral_cell_measure(cgrid, 2, 3) == spectral_weight(cgrid, 2, 3)
    @test sum(spectral_cell_measures(cgrid)) ≈ (maximum(cgrid.kx_faces) - minimum(cgrid.kx_faces)) *
                                               (maximum(cgrid.ky_faces) - minimum(cgrid.ky_faces))

    xfaces = [-1.2, -0.4, 0.25, 1.1]
    yfaces = [-2.0, -0.3, 0.8, 1.7]
    xcenters = [(xfaces[i] + xfaces[i+1]) / 2 for i in 1:(length(xfaces)-1)]
    ycenters = [(yfaces[j] + yfaces[j+1]) / 2 for j in 1:(length(yfaces)-1)]
    exact_grid = CartesianWaveVectorGrid(Float64; kx=xcenters, ky=ycenters,
                                         kx_faces=xfaces, ky_faces=yfaces)
    cell_average = [1 + 2 * (xfaces[m] + xfaces[m+1]) / 2 -
                        3 * (yfaces[n] + yfaces[n+1]) / 2 +
                        0.4 * ((xfaces[m] + xfaces[m+1]) / 2) *
                              ((yfaces[n] + yfaces[n+1]) / 2)
                    for m in eachindex(xcenters), n in eachindex(ycenters)]
    exact_integral = (x, y) -> x * y + x^2 * y - 1.5 * x * y^2 + 0.1 * x^2 * y^2
    expected = exact_integral(xfaces[end], yfaces[end]) -
               exact_integral(xfaces[1], yfaces[end]) -
               exact_integral(xfaces[end], yfaces[1]) +
               exact_integral(xfaces[1], yfaces[1])
    @test integrate_spectrum(cell_average, exact_grid) ≈ expected atol=1e-14

    first_x = sum(spectral_first_moment_measures(exact_grid, m, n)[1]
                  for m in eachindex(xcenters), n in eachindex(ycenters))
    first_y = sum(spectral_first_moment_measures(exact_grid, m, n)[2]
                  for m in eachindex(xcenters), n in eachindex(ycenters))
    second_xx = sum(spectral_second_moment_measures(exact_grid, m, n)[1]
                    for m in eachindex(xcenters), n in eachindex(ycenters))
    second_xy = sum(spectral_second_moment_measures(exact_grid, m, n)[2]
                    for m in eachindex(xcenters), n in eachindex(ycenters))
    second_yy = sum(spectral_second_moment_measures(exact_grid, m, n)[3]
                    for m in eachindex(xcenters), n in eachindex(ycenters))
    @test first_x ≈ (xfaces[end]^2 - xfaces[1]^2) * (yfaces[end] - yfaces[1]) / 2 atol=1e-14
    @test first_y ≈ (xfaces[end] - xfaces[1]) * (yfaces[end]^2 - yfaces[1]^2) / 2 atol=1e-14
    @test second_xx ≈ (xfaces[end]^3 - xfaces[1]^3) * (yfaces[end] - yfaces[1]) / 3 atol=1e-14
    @test second_xy ≈ (xfaces[end]^2 - xfaces[1]^2) * (yfaces[end]^2 - yfaces[1]^2) / 4 atol=1e-14
    @test second_yy ≈ (xfaces[end] - xfaces[1]) * (yfaces[end]^3 - yfaces[1]^3) / 3 atol=1e-14
    @test_throws ArgumentError integrate_spectrum(ones(2, 2), exact_grid)
    @test_throws ArgumentError CartesianWaveVectorGrid(Float64; kx=[0.0, 0.0], ky=[0.0])
    @test_throws ArgumentError CartesianWaveVectorGrid(Float64; kx=[0.0], ky=[0.0], kx_faces=[-1.0, 0.0, 1.0])
    @test_throws ArgumentError CartesianWaveVectorGrid(Float64; kx=[0.0], ky=[0.0], kx_faces=[1.0, 0.0])
    @test_throws ArgumentError CartesianWaveVectorGrid(Float64; kx=[0.0], ky=[0.0], kx_faces=[0.1, 1.0])

    pgrid = PolarWaveVectorGrid(; κ=range(0.25, 1.0; length=4), φ=range(0, 2pi; length=9)[1:8])
    @test pgrid.topology[1] isa NoFlux
    @test pgrid.topology[2] isa Periodic
    @test all(diff(coordinate_centers(pgrid, 1)) .> 0)
    @test all(diff(coordinate_centers(pgrid, 2)) .> 0)
    @test all(diff(coordinate_faces(pgrid, 1)) .> 0)
    @test all(diff(coordinate_faces(pgrid, 2)) .> 0)
    area = pi * (pgrid.κ_faces[end]^2 - pgrid.κ_faces[1]^2)
    @test sum(spectral_cell_measures(pgrid)) ≈ area
    @test k_components(pgrid, 2, 1)[2] ≈ 0

    radial_cell_average = [spectral_radial_power_average(pgrid, m, n, 2)
                           for m in eachindex(pgrid.κ), n in eachindex(pgrid.φ)]
    exact_radial_integral = pi / 2 * (pgrid.κ_faces[end]^4 - pgrid.κ_faces[1]^4)
    @test integrate_spectrum(radial_cell_average, pgrid) ≈ exact_radial_integral atol=1e-14
    inverse_root_average = [spectral_radial_power_average(pgrid, m, n, -1 / 2)
                            for m in eachindex(pgrid.κ), n in eachindex(pgrid.φ)]
    exact_inverse_root_integral = 2pi * (pgrid.κ_faces[end]^(3 / 2) -
                                         pgrid.κ_faces[1]^(3 / 2)) / (3 / 2)
    @test integrate_spectrum(inverse_root_average, pgrid) ≈ exact_inverse_root_integral atol=1e-14
    polar_first_x = sum(spectral_first_moment_measures(pgrid, m, n)[1]
                        for m in eachindex(pgrid.κ), n in eachindex(pgrid.φ))
    polar_first_y = sum(spectral_first_moment_measures(pgrid, m, n)[2]
                        for m in eachindex(pgrid.κ), n in eachindex(pgrid.φ))
    polar_trace = sum(spectral_second_moment_measures(pgrid, m, n)[1] +
                      spectral_second_moment_measures(pgrid, m, n)[3]
                      for m in eachindex(pgrid.κ), n in eachindex(pgrid.φ))
    @test abs(polar_first_x) < 1e-14
    @test abs(polar_first_y) < 1e-14
    @test polar_trace ≈ exact_radial_integral atol=1e-14
    shallow_radial_grid = PolarWaveVectorGrid(; κ=[1e-4, 0.8], φ=[0.0])
    @test shallow_radial_grid.κ_faces[1] == 0
    @test all(diff(shallow_radial_grid.κ_faces) .> 0)
    @test all(spectral_cell_measures(shallow_radial_grid) .> 0)
    @test_throws ArgumentError PolarWaveVectorGrid(; κ=[0.0], φ=[0.0])
    @test_throws ArgumentError PolarWaveVectorGrid(; κ=[1.0], φ=[0.0], κ_faces=[-0.1, 2.0])
    @test_throws ArgumentError PolarWaveVectorGrid(; κ=[1.0, 0.5], φ=[0.0])
    @test_throws ArgumentError PolarWaveVectorGrid(; κ=[1.0], φ=[0.0, pi], φ_faces=[-0.5, 0.5])
    @test_throws ArgumentError PolarWaveVectorGrid(; κ=[1.0], φ=[0.0], φ_faces=[0.1, 1.0])
    @test_throws ArgumentError PolarWaveVectorGrid(; κ=[1.0], φ=Float64[])

    square_grid = CartesianWaveVectorGrid(Float64; kx=[0.0], ky=[0.0],
                                          kx_faces=[-1.0, 1.0], ky_faces=[-1.0, 1.0])
    exact_square_average = (sqrt(2) + asinh(1)) / 3
    @test spectral_radial_power_average(square_grid, 1, 1, 1) ≈ exact_square_average atol=1e-14
    @test spectral_radial_power_measure(square_grid, 1, 1, 2) ≈ 8 / 3 atol=1e-14
    @test_throws ArgumentError spectral_radial_power_measure(square_grid, 1, 1, 3)
    @test_throws ArgumentError spectral_radial_power_average(square_grid, 1, 1, -1 / 2)
    @test_throws ArgumentError spectral_radial_power_average(pgrid, 1, 1, -2)

    fgrid = FrequencyDirectionGrid(; frequency=range(0.08, 0.2; length=4), φ=range(0, 2pi; length=9)[1:8])
    @test coordinate_size(fgrid) == (4, 8)
    @test fgrid.topology[1] isa Bounded
    @test fgrid.topology[2] isa Periodic
    @test all(diff(coordinate_centers(fgrid, 1)) .> 0)
    @test all(diff(coordinate_centers(fgrid, 2)) .> 0)
    @test all(diff(coordinate_faces(fgrid, 1)) .> 0)
    @test all(diff(coordinate_faces(fgrid, 2)) .> 0)
    @test fgrid.κ[2] ≈ (2pi * fgrid.frequency[2])^2 / 9.81
    @test fgrid.κ_faces[2] ≈ (2pi * fgrid.frequency_faces[2])^2 / 9.81
    farea = pi * (((2pi * fgrid.frequency_faces[end])^2 / 9.81)^2 -
                  ((2pi * fgrid.frequency_faces[1])^2 / 9.81)^2)
    @test sum(spectral_cell_measures(fgrid)) ≈ farea
    frequency_radial_average = [spectral_radial_power_average(fgrid, m, n, 1)
                                for m in eachindex(fgrid.frequency), n in eachindex(fgrid.φ)]
    exact_frequency_radial_integral = 2pi / 3 * (fgrid.κ_faces[end]^3 - fgrid.κ_faces[1]^3)
    @test integrate_spectrum(frequency_radial_average, fgrid) ≈ exact_frequency_radial_integral atol=1e-14
    frequency_power_average = [spectral_frequency_power_average(fgrid, m, n, 2)
                               for m in eachindex(fgrid.frequency), n in eachindex(fgrid.φ)]
    α = fgrid.κ_faces[end] / fgrid.frequency_faces[end]^2
    exact_frequency_power_integral = 2pi * 2 * α^2 *
                                     (fgrid.frequency_faces[end]^6 - fgrid.frequency_faces[1]^6) / 6
    @test integrate_spectrum(frequency_power_average, fgrid) ≈ exact_frequency_power_integral atol=1e-14
    @test spectral_frequency_power_average(fgrid, 1, 1, 0) ≈ 1
    @test_throws ArgumentError spectral_frequency_power_average(pgrid, 1, 1, 1)
    single_frequency_grid = FrequencyDirectionGrid(; frequency=[0.1], φ=[0.0])
    @test single_frequency_grid.frequency_faces[1] == 0
    @test all(diff(single_frequency_grid.frequency_faces) .> 0)
    @test all(diff(single_frequency_grid.κ_faces) .> 0)
    @test all(spectral_cell_measures(single_frequency_grid) .> 0)
    @test_throws ArgumentError FrequencyDirectionGrid(; frequency=[0.0], φ=[0.0])
    @test_throws ArgumentError FrequencyDirectionGrid(; frequency=[0.1], φ=[0.0], frequency_faces=[-0.1, 0.2])
    @test_throws ArgumentError FrequencyDirectionGrid(; frequency=[0.2, 0.1], φ=[0.0])
    @test_throws ArgumentError FrequencyDirectionGrid(; frequency=[0.1], φ=[0.0], frequency_faces=[0.2, 0.3])
    @test_throws ArgumentError FrequencyDirectionGrid(; frequency=[0.1], φ=[0.0], φ_faces=[0.1, 1.0])
    @test_throws ArgumentError FrequencyDirectionGrid(; frequency=[0.1], φ=[0.0], gravity=0.0)
    @test_throws ArgumentError FrequencyDirectionGrid(; frequency=[0.1], φ=Float64[])
end
