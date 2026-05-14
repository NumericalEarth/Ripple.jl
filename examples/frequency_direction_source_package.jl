# # Frequency-Direction Source Package
#
# Operational source packages are easiest to inspect in a source-only column.
# `advection=nothing` leaves the spectrum local while semi-implicit source
# splitting grows wind-aligned waves and damps opposing waves.

using Oceananigans, Ripple, CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "..", "scripts", "example_visuals.jl"))

output_dir = example_output_directory("frequency_direction_source_package")
plot_paths = String[]
animation_paths = String[]

grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
frequencies = range(0.08, 0.30; length=16)
directions_radians = range(0, 2pi; length=25)[1:24]
spectral_grid = FrequencyDirectionGrid(Float64;
                                       frequency=frequencies,
                                       φ=directions_radians)

sources = SourceTermSet(
    PowerLawWindInput(rate=0.2,
                      speed=12.0,
                      direction=0.0,
                      reference_speed=10.0,
                      speed_power=1.0,
                      spreading_power=2.0),
    PeakFrequencyDissipation(rate=0.03,
                             reference_frequency=0.18,
                             power=1.0),
)

model = SpectralWaveModel(; grid,
                            spectral_grid,
                            advection=nothing,
                            sources,
                            timestepper=:SemiImplicitEuler)

peak_m = argmin(abs.(collect(frequencies) .- 0.18))
east_n = argmin(abs.(collect(directions_radians) .- 0.0))
north_n = argmin(abs.(collect(directions_radians) .- pi / 2))
west_n = argmin(abs.(collect(directions_radians) .- pi))
south_n = argmin(abs.(collect(directions_radians) .- 3pi / 2))

frequency_from_kappa(kappa) = sqrt(9.81 * kappa) / (2pi)
angular_distance(a, b) = abs(atan(sin(a - b), cos(a - b)))

set!(model, N=(x, y, kx, ky) -> begin
    frequency = frequency_from_kappa(hypot(kx, ky))
    direction = mod(atan(ky, kx), 2pi)
    at_peak_frequency = abs(frequency - frequencies[peak_m]) < 1e-12
    at_peak_frequency && angular_distance(direction, directions_radians[east_n]) < 1e-12 && return 2.0
    at_peak_frequency && angular_distance(direction, directions_radians[north_n]) < 1e-12 && return 0.4
    at_peak_frequency && angular_distance(direction, directions_radians[south_n]) < 1e-12 && return 0.4
    return 0.05
end)

initial_total_action = total_action(model.action)
initial_peak_frequency = peak_frequency(model.action)[1, 1]
initial_mean_frequency = mean_frequency(model.action)[1, 1]
initial_east_peak = model.action[1, 1, peak_m, east_n]
initial_west_peak = model.action[1, 1, peak_m, west_n]
initial_directional_peak = [model.action[1, 1, peak_m, n] for n in 1:length(spectral_grid.φ)]

east_positive, east_damping = source_split(sources, model, 1, 1, peak_m, east_n)
west_positive, west_damping = source_split(sources, model, 1, 1, peak_m, west_n)

@assert initial_peak_frequency > 0
@assert east_positive > 0
@assert east_damping > 0
@assert west_positive == 0
@assert west_damping == east_damping

dt = 0.5
frames = [column_spectrum_matrix(model.action)]
time_step!(model, dt)

final_total_action = total_action(model.action)
final_peak_frequency = peak_frequency(model.action)[1, 1]
final_mean_frequency = mean_frequency(model.action)[1, 1]
final_east_peak = model.action[1, 1, peak_m, east_n]
final_west_peak = model.action[1, 1, peak_m, west_n]

expected_east_peak = (initial_east_peak + dt * east_positive) / (1 + dt * east_damping)
expected_west_peak = initial_west_peak / (1 + dt * west_damping)

step_count = example_mode() == :small ? 4 : 24
for n in 1:step_count
    time_step!(model, dt)
    if n == step_count || n % max(1, step_count ÷ 6) == 0
        push!(frames, column_spectrum_matrix(model.action))
    end
end

final_directional_peak = [model.action[1, 1, peak_m, n] for n in 1:length(spectral_grid.φ)]
directions = collect(directions_radians) .* 180 ./ pi

push!(plot_paths,
      write_line(joinpath(output_dir, "directional_peak_bin.png"),
                 directions, (initial_directional_peak, final_directional_peak);
                 title="Wind-aligned growth at peak frequency",
                 xlabel="direction degrees", ylabel="action",
                 names=("initial", "final")))

push!(animation_paths,
      write_heatmap_animation(joinpath(output_dir, "source_package_spectrum.mp4"),
                              frames;
                              title="Frequency-direction source evolution",
                              xlabel="frequency bin", ylabel="direction bin"))

@assert final_peak_frequency > 0
@assert final_east_peak ≈ expected_east_peak
@assert final_west_peak ≈ expected_west_peak
@assert final_east_peak > initial_east_peak
@assert final_west_peak < initial_west_peak
@assert final_total_action > initial_total_action
@assert isfinite(final_mean_frequency)
@assert final_mean_frequency > 0
@assert all(interior(model.action) .>= 0)
@assert all(isfile, plot_paths)
@assert all(isfile, animation_paths)

@show initial_total_action
@show final_total_action
@show initial_peak_frequency
@show initial_mean_frequency
@show final_mean_frequency
@show plot_paths
@show animation_paths
