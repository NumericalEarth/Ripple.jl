# # Wave Refraction Through A Barotropic Vortex
#
# This example refracts an initially uniform, narrow-banded wave-action
# field through a barotropic Gaussian vortex. The vortex velocity is the
# Lagrangian-mean current ``u^L``. Two physical effects act on the action:
# Doppler-shifted physical transport at ``c_g + u^L``, and kinematic
# refraction ``\nabla_k \cdot (c_k N)`` driven by gradients of ``u^L``.
# Both are applied in a single fused KA kernel that uses 5th-order WENO in
# all four directions, and the model is advanced with RK3 via an
# `Oceananigans.Simulation`. The resulting movie shows the spectrum
# evolving spatially through ``m_0``, ``\kappa_\mathrm{rms}``, and the
# mean direction.

using Oceananigans, Ripple
using CairoMakie
using Printf

CairoMakie.activate!(type = "png")

# ## Grid setup
#
# A 64²×16 horizontal periodic grid with halo wide enough for WENO5.
# Spectral grid: four ``\kappa`` rings and sixteen direction bins.

Nx = Ny = 64
Nz = 16
Lx = Ly = 80.0
Lz = 1.0

grid = RectilinearGrid(CPU();
                       size     = (Nx, Ny, Nz),
                       halo     = (3, 3, 3),
                       x        = (0, Lx),
                       y        = (0, Ly),
                       z        = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

spectral_grid = PolarWaveVectorGrid(; κ = range(0.30, 0.50; length = 4),
                                      φ = range(0, 2pi; length = 17)[1:16])

# ## Vortex velocity field
#
# A barotropic (depth-uniform) Gaussian-cored vortex. The azimuthal velocity
# peaks at ``r = a`` with magnitude ``U_0``.

xc, yc = Lx / 2, Ly / 2
a  = Lx / 8
U0 = 0.4

xs = xnodes(grid)
ys = ynodes(grid)
u_field = zeros(Nx, Ny, Nz)
v_field = zeros(Nx, Ny, Nz)
for k in 1:Nz, j in 1:Ny, i in 1:Nx
    dx = xs[i] - xc
    dy = ys[j] - yc
    r  = hypot(dx, dy)
    if r > 0
        uθ = U0 * (r / a) * exp(0.5 - 0.5 * (r / a)^2)
        u_field[i, j, k] = -uθ * dy / r
        v_field[i, j, k] = +uθ * dx / r
    end
end

# Static plot of the vortex speed field.

let speed = hypot.(view(u_field, :, :, 1), view(v_field, :, :, 1))
    fig = Figure(size = (640, 540))
    ax  = Axis(fig[1, 1]; title  = "Barotropic vortex |U| (m/s)",
                           xlabel = "x (m)", ylabel = "y (m)", aspect = 1)
    hm  = heatmap!(ax, xs, ys, speed; colormap = :viridis)
    Colorbar(fig[1, 2], hm)
    fig
end

# ## Wave model
#
# The `velocities` kwarg builds the wave-current coupling internally; here
# we pass the prescribed vortex as a NamedTuple. The fused refraction
# kernel handles physical and spectral transport together when both CWCM
# coupling and `spectral_advection` are set (both defaults).

model = SpectralWaveModel(grid, spectral_grid;
                          velocities  = (; u = u_field, v = v_field),
                          sources     = nothing,
                          timestepper = :RK3);

# Narrow-banded Gaussian initial condition, uniform in ``(x, y)``, peaking
# at ``\kappa \approx 0.4`` and direction ``\varphi \approx 0`` (waves
# travelling in ``+x``).

κ0 = 0.4
σκ = 0.05
σφ = 0.30
set!(model, N = (x, y, kx, ky) -> begin
    κ = hypot(kx, ky)
    φ = atan(ky, kx)
    exp(-((κ - κ0) / σκ)^2 - (sin(φ / 2)^2) / σφ^2)
end);

# ## Time loop
#
# 400 RK3 steps of ``\Delta t = 0.05\,\mathrm{s}`` for a total of 20 s, run
# via an `Oceananigans.Simulation`. Snapshots are captured between
# segments for the animation.

dt           = 0.05
total_time   = 20.0
steps        = Int(round(total_time / dt))
frame_stride = max(1, steps ÷ 60)

slab(f) = Array(interior(f))[:, :, 1]

m0_frames   = Matrix{Float64}[slab(m0(model.action))]
krms_frames = Matrix{Float64}[slab(root_mean_square_wavenumber(model.action))]
dir_frames  = Matrix{Float64}[slab(mean_direction(model.action))]
times       = Float64[0.0]

remaining = steps
while remaining > 0
    chunk             = min(frame_stride, remaining)
    target_iteration  = model.clock.iteration + chunk
    simulation        = Simulation(model; Δt = dt, stop_iteration = target_iteration, verbose = false)
    run!(simulation)
    push!(m0_frames,   slab(m0(model.action)))
    push!(krms_frames, slab(root_mean_square_wavenumber(model.action)))
    push!(dir_frames,  slab(mean_direction(model.action)))
    push!(times,       model.clock.time)
    global remaining -= chunk
end

# ## Three-panel animation
#
# ``m_0`` (total action), ``\kappa_\mathrm{rms}``, and the mean direction
# evolve together as the packet bends through the vortex.

m0_limits   = (minimum(minimum.(m0_frames)),   maximum(maximum.(m0_frames)))
krms_limits = (minimum(minimum.(krms_frames)), maximum(maximum.(krms_frames)))
dir_lim     = max(maximum(maximum.(abs, dir_frames)), 0.05)

m0_obs    = Observable(m0_frames[1])
krms_obs  = Observable(krms_frames[1])
dir_obs   = Observable(dir_frames[1])
title_obs = Observable("t = 0.00 s")

fig = Figure(size = (1320, 480))
ax1 = Axis(fig[1, 1]; title = "m₀ (total action)",       xlabel = "x", ylabel = "y", aspect = 1)
ax2 = Axis(fig[1, 3]; title = "κᵣₘₛ (rad/m)",            xlabel = "x", ylabel = "y", aspect = 1)
ax3 = Axis(fig[1, 5]; title = "mean direction (rad)",    xlabel = "x", ylabel = "y", aspect = 1)
hm1 = heatmap!(ax1, xs, ys, m0_obs;   colormap = :viridis, colorrange = m0_limits)
hm2 = heatmap!(ax2, xs, ys, krms_obs; colormap = :magma,   colorrange = krms_limits)
hm3 = heatmap!(ax3, xs, ys, dir_obs;  colormap = :balance, colorrange = (-dir_lim, dir_lim))
Colorbar(fig[1, 2], hm1); Colorbar(fig[1, 4], hm2); Colorbar(fig[1, 6], hm3)
Label(fig[0, :], title_obs; fontsize = 18, halign = :center)

record(fig, "vortex_refraction.mp4", eachindex(m0_frames); framerate = 12) do idx
    m0_obs[]    = m0_frames[idx]
    krms_obs[]  = krms_frames[idx]
    dir_obs[]   = dir_frames[idx]
    title_obs[] = @sprintf("Wave action through barotropic vortex — t = %5.2f s", times[idx])
end
nothing #hide

# ![](vortex_refraction.mp4)
