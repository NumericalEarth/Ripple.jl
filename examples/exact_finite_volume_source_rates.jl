# # Exact Finite-Volume Source Rates
#
# Ripple stores spectral cell averages. Power-law source rates are therefore
# integrated over the finite cell volume, not sampled at bin centers. This
# example compares the active finite-volume factors with midpoint factors while
# still rendering a resolved frequency-direction column.

using Oceananigans, Ripple, CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "..", "scripts", "example_visuals.jl"))

output_dir = example_output_directory("exact_finite_volume_source_rates")
plot_paths = String[]
animation_paths = String[]

grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
spectral_grid = FrequencyDirectionGrid(Float64;
                                       frequency=range(0.08, 0.32; length=12),
                                       φ=range(0, 2pi; length=25)[1:24])

frequency_source = FrequencyDissipation(rate=0.4,
                                        reference_frequency=0.16,
                                        power=2)
wavenumber_source = WavenumberDissipation(rate=0.1,
                                          reference_wavenumber=spectral_grid.κ[2],
                                          power=2)
sources = SourceTermSet(frequency_source, wavenumber_source)

model = SpectralWaveModel(; grid,
                            spectral_grid,
                            advection=nothing,
                            sources,
                            timestepper=:SemiImplicitEuler)

set!(model, N=1.0)

m, n = 6, 1
initial_bin_action = model.action[1, 1, m, n]
frames = [column_spectrum_matrix(model.action)]

frequency_factor = spectral_frequency_power_average(spectral_grid, m, n, frequency_source.power) /
                   frequency_source.reference_frequency^frequency_source.power
wavenumber_factor = spectral_radial_power_average(spectral_grid, m, n, wavenumber_source.power) /
                    wavenumber_source.reference_wavenumber^wavenumber_source.power

center_frequency_factor = (spectral_grid.frequency[m] / frequency_source.reference_frequency)^frequency_source.power
center_wavenumber_factor = (spectral_grid.κ[m] / wavenumber_source.reference_wavenumber)^wavenumber_source.power

expected_damping = frequency_source.rate * frequency_factor +
                   wavenumber_source.rate * wavenumber_factor

positive, damping = source_split(sources, model, 1, 1, m, n)

@assert positive == 0
@assert damping ≈ expected_damping
@assert abs(frequency_factor - center_frequency_factor) > 1e-3
@assert abs(wavenumber_factor - center_wavenumber_factor) > 1e-3

dt = 0.4
time_step!(model, dt)
push!(frames, column_spectrum_matrix(model.action))

expected_bin_action = initial_bin_action / (1 + dt * expected_damping)
low_fraction = model.action[1, 1, 1, 1]
high_fraction = model.action[1, 1, size(model.action, 3), 1]

factor_bins = [1.0, 2.0]
finite_volume_factors = [frequency_factor, wavenumber_factor]
midpoint_factors = [center_frequency_factor, center_wavenumber_factor]

push!(plot_paths,
      write_line(joinpath(output_dir, "finite_volume_vs_midpoint_rates.png"),
                 factor_bins, (finite_volume_factors, midpoint_factors);
                 title="Finite-volume source factors",
                 xlabel="factor family", ylabel="normalized rate",
                 names=("finite volume", "midpoint")))

push!(animation_paths,
      write_heatmap_animation(joinpath(output_dir, "damping_spectrum_before_after.mp4"),
                              frames;
                              title="Exact cell-average damping",
                              xlabel="frequency bin", ylabel="direction bin"))

@assert model.action[1, 1, m, n] ≈ expected_bin_action
@assert all(interior(model.action) .>= 0)
@assert high_fraction < low_fraction
@assert all(isfile, plot_paths)
@assert all(isfile, animation_paths)

@show frequency_factor
@show center_frequency_factor
@show wavenumber_factor
@show center_wavenumber_factor
@show plot_paths
@show animation_paths
