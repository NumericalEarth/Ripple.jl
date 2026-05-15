# # Bounded Wave Packet Dispersion
#
# A one-dimensional bounded physical domain is the simplest place to see
# transport. We initialize a compact packet near the left boundary with a
# finite-width wavenumber spectrum. All waves travel in the positive x direction,
# but long waves have larger deep-water group velocity than short waves, so the
# packet spreads as it moves across the domain.

using Oceananigans, Ripple, CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "..", "scripts", "example_visuals.jl"))

output_dir = example_output_directory("bounded_wave_packet_dispersion")
plot_paths = String[]
animation_paths = String[]

small = example_mode() == :small
Nx = small ? 96 : 192
Ny = 6
Nk = small ? 10 : 18
Lx = 384.0
grid = RectilinearGrid(CPU();
                       size=(Nx, Ny, 1),
                       halo=(3, 3, 1),
                       x=(0, Lx),
                       y=(0, 1),
                       z=(-1, 0),
                       topology=(Bounded, Periodic, Bounded))

kappas = range(0.35, 1.25; length=Nk)
theta_width = pi / 18
spectral_grid = PolarWaveVectorGrid(Float64;
                                    κ=kappas,
                                    φ=[0.0],
                                    φ_faces=[-theta_width / 2, theta_width / 2])

model = SpectralWaveModel(; grid,
                            spectral_grid,
                            advection=WENO(order=5),
                            timestepper=:RK3)

packet_left = 24.0
packet_right = 64.0
set!(model, N=(x, y, kx, ky) -> packet_left <= x < packet_right ? 1.0 : 0.0)

x = collect(xnodes(grid))
m0_profile() = vec(interior(m0(model.action))[:, 1, 1])
x_kappa_profile() = x_kappa_phase_space_matrix(model.action; j=1, n=1)

function packet_center(profile)
    mass = sum(profile)
    return sum(x .* profile) / mass
end

function packet_width(profile)
    mass = sum(profile)
    center = packet_center(profile)
    return sqrt(sum((x .- center).^2 .* profile) / mass)
end

dt = 0.25
final_time = small ? 30.0 : 90.0
step_count = round(Int, final_time / dt)
sample_count = small ? 12 : 24
sample_interval = max(1, step_count ÷ (sample_count - 1))

times = [model.clock.time]
profiles = [m0_profile()]
phase_frames = [x_kappa_profile()]

initial_center = packet_center(first(profiles))
initial_width = packet_width(first(profiles))
initial_action = total_action(model.action)

for step in 1:step_count
    time_step!(model, dt)
    if step == step_count || step % sample_interval == 0
        push!(times, model.clock.time)
        push!(profiles, m0_profile())
        push!(phase_frames, x_kappa_profile())
    end
end

hovmoller = reduce(vcat, transpose.(profiles))
final_center = packet_center(last(profiles))
final_width = packet_width(last(profiles))
final_action = total_action(model.action)

push!(plot_paths,
      write_heatmap(joinpath(output_dir, "packet_hovmoller.png"),
                    hovmoller;
                    title="Bounded-domain wave-packet transport",
                    xlabel="x cell", ylabel="time sample"))

push!(plot_paths,
      write_heatmap(joinpath(output_dir, "final_x_kappa_action.png"),
                    last(phase_frames);
                    title="Final x-kappa action density",
                    xlabel="x cell", ylabel="kappa bin"))

push!(animation_paths,
      write_heatmap_animation(joinpath(output_dir, "x_kappa_packet_dispersion.mp4"),
                              phase_frames;
                              title="Frequency-dependent wave-packet transport",
                              xlabel="x cell", ylabel="kappa bin",
                              fps=5))

@assert final_center > initial_center + 5
@assert final_width > initial_width
@assert isapprox(final_action, initial_action; rtol=2e-2)
@assert all(isfinite, interior(model.action))
@assert all(interior(model.action) .>= -1e-12)
@assert all(isfile, plot_paths)
@assert all(isfile, animation_paths)

@show initial_center
@show final_center
@show initial_width
@show final_width
@show initial_action
@show final_action
@show plot_paths
@show animation_paths
