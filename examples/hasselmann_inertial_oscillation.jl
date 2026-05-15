# # Hasselmann Column Growth
#
# This column repeats the validation problem at tutorial scale: a relaxation
# source spins up a directional spectrum, then the validation suite checks the
# induced Q-integrated pseudomomentum and inertial-current response.

using Oceananigans, Ripple, CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "..", "scripts", "example_visuals.jl"))

output_dir = example_output_directory("hasselmann_inertial_oscillation")
plot_paths = String[]
animation_paths = String[]

case = only(filter(c -> c.name == :hasselmann_column, default_validation_cases()))
result = run_validation(case)

grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
spectral_grid = PolarWaveVectorGrid(Float64;
                                    κ=range(0.35, 1.15; length=16),
                                    φ=range(0, 2pi; length=33)[1:32])

target(x, y, κ, φ) = begin
    direction = max(cos(φ), 0.0)^4
    exp(-((κ - 0.75) / 0.22)^2) * direction
end

alpha = 1.3
model = SpectralWaveModel(; grid,
                            spectral_grid,
                            advection=nothing,
                            sources=RelaxationToSpectrum(target; timescale=inv(alpha)),
                            timestepper=:ForwardEuler)
set!(model, N=0.0)

target_action = WaveActionField(grid, spectral_grid)
set!(target_action, target)
target_total_action = total_action(target_action)

final_time = example_mode() == :small ? 0.04 : 0.4
dt = example_mode() == :small ? 1e-3 : 5e-4
step_count = round(Int, final_time / dt)
times = [model.clock.time]
actions = [total_action(model.action)]
analytic_actions = [target_total_action * (1 - exp(-alpha * model.clock.time))]
frames = [column_spectrum_matrix(model.action)]

for n in 1:step_count
    time_step!(model, dt)
    if n == step_count || n % max(1, step_count ÷ 24) == 0
        push!(times, model.clock.time)
        push!(actions, total_action(model.action))
        push!(analytic_actions, target_total_action * (1 - exp(-alpha * model.clock.time)))
    end
    if n == step_count || n % max(1, step_count ÷ 6) == 0
        push!(frames, column_spectrum_matrix(model.action))
    end
end

push!(plot_paths,
      write_line(joinpath(output_dir, "hasselmann_action_growth.png"),
                 times, (actions, analytic_actions);
                 title="Relaxation toward Hasselmann column spectrum",
                 xlabel="time", ylabel="total action",
                 names=("Ripple", "analytic")))

push!(animation_paths,
      write_heatmap_animation(joinpath(output_dir, "hasselmann_spectrum_growth.mp4"),
                              frames;
                              title="Directional spectrum spinup",
                              xlabel="radial bin", ylabel="direction bin"))

@assert validation_passed(result)
@assert all(isfinite, interior(model.action))
@assert all(isfile, plot_paths)
@assert all(isfile, animation_paths)

@show validation_passed(result)
@show result.metrics[:action_error]
@show result.metrics[:pseudomomentum_error]
@show result.metrics[:current_error]
@show result.metrics[:kinetic_error]
@show plot_paths
@show animation_paths
