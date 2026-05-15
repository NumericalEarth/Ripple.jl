# # Wave Refraction Through A Barotropic Vortex
#
# This example refracts an initially uniform, narrow-banded wave-action field
# through a barotropic Gaussian vortex. The vortex velocity is the
# Lagrangian-mean current `uᴸ`. Two physical effects act on the action:
# Doppler-shifted physical transport at `cg + uᴸ`, and kinematic refraction
# `∇_k·(c_k N)` driven by gradients of `uᴸ`. Both are applied in a single
# fused KA kernel that uses 5th-order WENO in all four directions, and the
# model is advanced with SSP-RK3. The resulting movie shows the spectrum
# evolving spatially through `m₀`, `κᵣₘₛ`, and the mean direction.

using Oceananigans, Ripple, CairoMakie
using Printf

Base.include(@__MODULE__, joinpath(@__DIR__, "..", "scripts", "example_visuals.jl"))

output_dir = example_output_directory("vortex_refraction")
plot_paths = String[]
animation_paths = String[]

# ## Grid setup
#
# A 64²×16 (or smaller, when `RIPPLE_EXAMPLE_MODE=small`) horizontal periodic
# grid with halo wide enough for WENO5. Spectral grid: four κ rings and
# sixteen direction bins.

small = example_mode() == :small
Nx = small ? 24 : 64
Ny = small ? 24 : 64
Nz = small ? 4 : 16
Lx = Ly = 80.0
Lz = 1.0

grid = RectilinearGrid(CPU();
                       size = (Nx, Ny, Nz),
                       halo = (3, 3, 3),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

spectral_grid = PolarWaveVectorGrid(; κ = range(0.30, 0.50; length = 4),
                                      φ = range(0, 2pi; length = 17)[1:16])

# ## Vortex velocity field
#
# A barotropic (depth-uniform) Gaussian-cored vortex. The azimuthal velocity
# peaks at `r = a` with magnitude `U₀`.

xc, yc = Lx / 2, Ly / 2
a = Lx / 8
U0 = 0.4

xs = xnodes(grid)
ys = ynodes(grid)
u_field = zeros(Nx, Ny, Nz)
v_field = zeros(Nx, Ny, Nz)
for k in 1:Nz, j in 1:Ny, i in 1:Nx
    dx = xs[i] - xc
    dy = ys[j] - yc
    r = hypot(dx, dy)
    if r > 0
        uθ = U0 * (r / a) * exp(0.5 - 0.5 * (r / a)^2)
        u_field[i, j, k] = -uθ * dy / r
        v_field[i, j, k] = +uθ * dx / r
    end
end

# ## Wave model
#
# The new `velocities` kwarg builds the wave-current coupling internally;
# here we pass the prescribed vortex as a NamedTuple. Physical advection is
# disabled (`advection=nothing`) because the fused refraction kernel handles
# physical and spectral transport together.

model = SpectralWaveModel(; grid,
                            spectral_grid,
                            velocities = (; u = u_field, v = v_field),
                            advection = nothing,
                            sources = nothing,
                            timestepper = :ForwardEuler)
coupling = model.coupling

# Narrow-banded Gaussian initial condition, uniform in (x, y), peaking at
# `κ ≈ 0.4` and direction `φ ≈ 0` (waves travelling in `+x`).

κ0 = 0.4; σκ = 0.05; σφ = 0.30
set!(model, N = (x, y, kx, ky) -> begin
    κ = hypot(kx, ky)
    φ = atan(ky, kx)
    exp(-((κ - κ0) / σκ)^2 - (sin(φ / 2)^2) / σφ^2)
end)

G  = similar(model.action)
N1 = similar(model.action)
N2 = similar(model.action)

# ## Time loop
#
# 400 SSP-RK3 steps of `dt = 0.05 s` for a total of 20 s. Snapshots are
# stored every few steps for the animation. In smoke-test mode we run a
# much shorter trajectory.

dt = 0.05
total_time = small ? 1.0 : 20.0
steps = Int(round(total_time / dt))
frame_stride = max(1, steps ÷ (small ? 4 : 60))

_diag_matrix(f) = Array(interior(f))[:, :, 1]

m0_frames = Matrix{Float64}[_diag_matrix(m0(model.action))]
krms_frames = Matrix{Float64}[_diag_matrix(root_mean_square_wavenumber(model.action))]
dir_frames = Matrix{Float64}[_diag_matrix(mean_direction(model.action))]
times = Float64[0.0]

for step in 1:steps
    rk3_step!(model, coupling, G, N1, N2, dt)
    if step % frame_stride == 0 || step == steps
        push!(m0_frames, _diag_matrix(m0(model.action)))
        push!(krms_frames, _diag_matrix(root_mean_square_wavenumber(model.action)))
        push!(dir_frames, _diag_matrix(mean_direction(model.action)))
        push!(times, model.clock.time)
    end
end

# ## Visualization
#
# A static plot of the vortex velocity magnitude and a three-panel movie
# of `m₀`, `κᵣₘₛ`, and the mean direction.

let speed = hypot.(view(u_field, :, :, 1), view(v_field, :, :, 1))
    path = joinpath(output_dir, "vortex_speed.png")
    fig = Figure(size = (640, 540))
    ax = Axis(fig[1, 1]; title = "Barotropic vortex |U| (m/s)",
              xlabel = "x (m)", ylabel = "y (m)", aspect = 1)
    hm = heatmap!(ax, xs, ys, speed; colormap = :viridis)
    Colorbar(fig[1, 2], hm)
    save(path, fig)
    push!(plot_paths, path)
end

m0_limits = (minimum(minimum.(m0_frames)), maximum(maximum.(m0_frames)))
krms_limits = (minimum(minimum.(krms_frames)), maximum(maximum.(krms_frames)))
dir_lim = max(maximum(maximum.(abs, dir_frames)), 0.05)

m0_obs = Observable(m0_frames[1])
krms_obs = Observable(krms_frames[1])
dir_obs = Observable(dir_frames[1])
title_obs = Observable("t = 0.00 s")

fig = Figure(size = (1320, 480))
ax1 = Axis(fig[1, 1]; title = "m₀ (total action)", xlabel = "x", ylabel = "y", aspect = 1)
ax2 = Axis(fig[1, 3]; title = "κᵣₘₛ (rad/m)", xlabel = "x", ylabel = "y", aspect = 1)
ax3 = Axis(fig[1, 5]; title = "mean direction (rad)", xlabel = "x", ylabel = "y", aspect = 1)
hm1 = heatmap!(ax1, xs, ys, m0_obs; colormap = :viridis, colorrange = m0_limits)
hm2 = heatmap!(ax2, xs, ys, krms_obs; colormap = :magma, colorrange = krms_limits)
hm3 = heatmap!(ax3, xs, ys, dir_obs; colormap = :balance, colorrange = (-dir_lim, dir_lim))
Colorbar(fig[1, 2], hm1); Colorbar(fig[1, 4], hm2); Colorbar(fig[1, 6], hm3)
Label(fig[0, :], title_obs; fontsize = 18, halign = :center)

movie_path = joinpath(output_dir, "vortex_refraction.mp4")
record(fig, movie_path, eachindex(m0_frames); framerate = small ? 4 : 12) do idx
    m0_obs[] = m0_frames[idx]
    krms_obs[] = krms_frames[idx]
    dir_obs[] = dir_frames[idx]
    title_obs[] = @sprintf("Wave action through barotropic vortex — t = %5.2f s", times[idx])
end
push!(animation_paths, movie_path)

@assert all(isfinite, interior(model.action))
@assert all(isfile, plot_paths)
@assert all(isfile, animation_paths)

@show length(m0_frames)
@show plot_paths
@show animation_paths
