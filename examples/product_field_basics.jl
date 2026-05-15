# # Product Field Basics
#
# Ripple fields live on a physical grid crossed with a spectral grid. This first
# example builds a wave-action field directly, initializes it from a function,
# and turns the zeroth moment into a static plot and a short animation.

using Oceananigans, Ripple, CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "..", "scripts", "example_visuals.jl"))

output_dir = example_output_directory("product_field_basics")
plot_paths = String[]
animation_paths = String[]

grid = RectilinearGrid(CPU(); size=(40, 24, 1), x=(0, 1), y=(0, 1), z=(0, 1))
spectral_grid = PolarWaveVectorGrid(Float64;
                                    κ=range(0.1, 1.0; length=14),
                                    φ=range(0, 2pi; length=33)[1:32])

N = WaveActionField(grid, spectral_grid)

set!(N, (x, y, κ, φ) -> exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.035) *
                              exp(-((κ - 0.5)^2) / 0.08) *
                              (1 + 0.25cos(φ)))

height = significant_wave_height(N)
moment = field_m0_matrix(N)

push!(plot_paths,
      write_heatmap(joinpath(output_dir, "zeroth_moment.png"),
                    N;
                    title="Zeroth moment",
                    xlabel="x cell", ylabel="y cell"))

frames = [scale .* moment for scale in range(0.15, 1.0; length=6)]
push!(animation_paths,
      write_heatmap_animation(joinpath(output_dir, "action_spinup.mp4"),
                              frames;
                              title="Wave action amplitude ramp",
                              xlabel="x cell", ylabel="y cell"))

@assert all(isfile, plot_paths)
@assert all(isfile, animation_paths)

@show size(N)
@show location(N)
@show coordinate_location(N)
@show maximum(height)
@show plot_paths
@show animation_paths
