# # Source-Only Fetch-Limited Growth
#
# Source-only models use `horizontal_advection=nothing`, matching the
# Oceananigans and Breeze convention that absent physics is represented by
# `nothing`. This column combines wind input and whitecapping dissipation and
# approaches an analytic equilibrium.

using Oceananigans, Ripple, CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "..", "scripts", "example_visuals.jl"))

output_dir = example_output_directory("source_only_fetch_limited_growth")
plot_paths = String[]
animation_paths = String[]
output_paths = String[]

case = only(filter(c -> c.name == :fetch_limited_source_balance,
                   default_validation_cases()))
result = run_validation(case)

grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
spectral_grid = PolarWaveVectorGrid(Float64; κ=[1.0], φ=[0.0])

target_growth_rate = 1.2
directional_growth_weight = Ripple.wind_directional_weight(spectral_grid, 1, 1, 0.0, 2)
growth_rate = target_growth_rate / directional_growth_weight
whitecapping_rate = 4.8
saturation_threshold = 0.5

sources = SourceTermSet((
    ExponentialWindInput(rate=growth_rate, direction=0.0, spreading_power=2),
    WhitecappingDissipation(rate=whitecapping_rate,
                             saturation_threshold=saturation_threshold,
                             saturation_power=1.0,
                             wavenumber_power=0.0),
))

model = SpectralWaveModel(grid, spectral_grid;
                          horizontal_advection=nothing,
                          sources,
                          timestepper=:SemiImplicitEuler)

weight = spectral_weight(spectral_grid, 1, 1)
equilibrium_m0 = saturation_threshold * (1 + target_growth_rate / whitecapping_rate)
set!(model, N=saturation_threshold / (20 * weight))

step_count = example_mode() == :small ? 40 : 240
dt = 0.02
times = [model.clock.time]
moments = [m0(model.action)[1, 1, 1]]

for n in 1:step_count
    time_step!(model, dt)
    push!(times, model.clock.time)
    push!(moments, m0(model.action)[1, 1, 1])
end

frame_indices = unique(round.(Int, range(1, length(moments); length=min(length(moments), 12))))
line_frames = [vcat(moments[1:i], fill(moments[i], length(moments) - i))
               for i in frame_indices]

push!(plot_paths,
      write_line(joinpath(output_dir, "fetch_limited_growth.png"),
                 times, (moments, fill(equilibrium_m0, length(times)));
                 title="Fetch-limited source balance",
                 xlabel="time", ylabel="m0",
                 names=("column m0", "analytic equilibrium")))

push!(animation_paths,
      write_line_animation(joinpath(output_dir, "fetch_limited_growth_animation.mp4"),
                           times, line_frames;
                           title="Approach to source equilibrium",
                           xlabel="time", ylabel="m0"))

writer_model = SpectralWaveModel(grid, spectral_grid;
                                 horizontal_advection=nothing,
                                 sources,
                                 timestepper=:SemiImplicitEuler)
set!(writer_model, N=saturation_threshold / (20 * weight))
writer_simulation = Simulation(writer_model; Δt=dt, stop_iteration=1, verbose=false)
writer_path = joinpath(output_dir, "fetch_limited_growth.jld2")
writer_simulation.output_writers[:action] =
    JLD2Writer(writer_model, (; N=writer_model.action);
               filename=writer_path,
               schedule=IterationInterval(1),
               overwrite_existing=true)
run!(writer_simulation)
push!(output_paths, writer_path)

@assert validation_passed(result)
@assert last(moments) <= equilibrium_m0 + 1e-2
@assert all(diff(moments) .>= -1e-12)
@assert all(isfile, plot_paths)
@assert all(isfile, animation_paths)
@assert all(isfile, output_paths)

@show validation_passed(result)
@show result.metrics[:equilibrium_tendency_error]
@show result.metrics[:final_m0_error]
@show result.metrics[:monotonicity_error]
@show last(moments)
@show plot_paths
@show animation_paths
