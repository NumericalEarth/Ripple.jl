# # 3D Langmuir Circulation
#
# This example exercises Oceananigans' Craik–Leibovich `UniformStokesDrift`
# inside a fully 3D `NonhydrostaticModel` with surface stress. The
# Craik–Leibovich vortex force
#
# ```math
# \boldsymbol{F}_{\text{CL}} = (\nabla \times \boldsymbol{u}^s) \times \boldsymbol{u}^E
# ```
#
# is unstable against horizontally varying perturbations: vertically sheared
# `uˢ(z)` couples to the surface jet driven by the wind stress, and a
# spanwise instability rolls the flow into pairs of counter-rotating
# **Langmuir cells** aligned with the wind direction (Craik & Leibovich
# 1976; Skyllingstad & Denbo 1995; McWilliams, Sullivan & Moeng 1997).
#
# In contrast to the horizontally uniform Ekman setup, here the vortex force
# is *not* zero (`wᴱ ≠ 0` once cells form), so the Stokes drift drives the
# dynamics. The signature is a band of strong spanwise `v` and downwelling
# `w` aligned with the wind.

using Oceananigans, Ripple
using CairoMakie, Printf, Random, Statistics
CairoMakie.activate!(type = "png")

# ## Setup
#
# Tank-scale 3D box: a few wave-decay scales `h_s` deep and an `O(10 h_s)`
# footprint, enough for several spanwise cell widths.

Nx, Ny, Nz = 32, 32, 32
Lx, Ly, Lz = 8.0, 8.0, 4.0
ν          = 1.0e-4
τ_x        = -1.0e-4              # negative-flux Oceananigans convention → drives +u
Uˢ         = 0.06
h_s        = 0.5

grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz), halo=(3, 3, 3),
                       x=(0, Lx), y=(0, Ly), z=(-Lz, 0),
                       topology=(Periodic, Periodic, Bounded))

stokes_drift = UniformStokesDrift(grid;
    ∂z_uˢ = (z, t) -> (Uˢ / h_s) * exp(z / h_s),
    ∂z_vˢ = (z, t) -> 0.0)

u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τ_x))

model = NonhydrostaticModel(grid;
                            advection = Centered(),
                            closure   = ScalarDiffusivity(ν=ν),
                            stokes_drift,
                            boundary_conditions = (; u=u_bcs))

## Surface-trapped noise to seed the spanwise instability. The vertical
## decay scale matches the Stokes drift so the noise lives where the
## instability mechanism (CL vortex force) is active.
Random.seed!(2026)
noise(x, y, z) = 1.0e-3 * randn() * exp(z / h_s)
set!(model; u=noise, v=noise, w=noise)

# ## Integrate
#
# The Langmuir-cell growth timescale is roughly `T_L ≈ √(h_s · Lz / (u_τ · Uˢ))`
# for surface friction velocity `u_τ = √|τ_x|`. With these parameters
# `T_L ≈ 1 s`, so we run for `~100 T_L`.

Δt        = 1.0
T_run     = 600.0
nsteps    = Int(round(T_run / Δt))
frame_stride = max(1, nsteps ÷ 80)

frames = (times = Float64[],
          v_xy  = Matrix{Float64}[],   # spanwise v at near-surface depth
          w_yz  = Matrix{Float64}[])   # downwelling cells, x-averaged y–z slice

function snapshot!()
    v_int = interior(model.velocities.v)
    w_int = interior(model.velocities.w)
    push!(frames.times, model.clock.time)
    ## v near surface (4 cells below top), averaged in x makes spanwise
    ## modulation clear but we plot the raw 2D field instead for richness:
    k_near = Nz - 3
    push!(frames.v_xy, Array(v_int[:, :, k_near]))
    ## w averaged in x to show the y-z roll pattern.
    w_yz = dropdims(mean(w_int; dims=1); dims=1)
    push!(frames.w_yz, Array(w_yz))
    return nothing
end
snapshot!()

for step in 1:nsteps
    time_step!(model, Δt)
    step % frame_stride == 0 && snapshot!()
end

@info "Langmuir simulation complete" final_time=model.clock.time max_u=maximum(abs, interior(model.velocities.u)) max_v=maximum(abs, interior(model.velocities.v)) max_w=maximum(abs, interior(model.velocities.w))

# ## Static snapshot at end of run

xs = collect(xnodes(grid, Center()))  # m
ys = collect(ynodes(grid, Center()))
zs = collect(znodes(grid, Center()))

let
    fig = Figure(size=(1100, 460))
    vmax = maximum(abs, frames.v_xy[end])
    wmax = maximum(abs, frames.w_yz[end])
    ax_v = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x (m)", ylabel="y (m)",
                title=@sprintf("v near surface (z ≈ %.2f m)", zs[end-3]))
    hm_v = heatmap!(ax_v, xs, ys, frames.v_xy[end];
                    colormap=:balance, colorrange=(-vmax, vmax))
    Colorbar(fig[1, 2], hm_v)

    ax_w = Axis(fig[1, 3]; aspect=DataAspect(), xlabel="y (m)", ylabel="z (m)",
                title="w  (x-averaged y–z slice)")
    hm_w = heatmap!(ax_w, ys, zs, frames.w_yz[end];
                    colormap=:balance, colorrange=(-wmax, wmax))
    Colorbar(fig[1, 4], hm_w)
    save("three_dimensional_langmuir.png", fig)
end

# ![](three_dimensional_langmuir.png)

# ## Animation
#
# Spanwise `v` at near-surface (left) and x-averaged `w` in the y–z plane
# (right). The growing roll structure is the Langmuir signature of the CL
# vortex force.

let
    n_frames = length(frames.times)
    vmax = maximum(maximum(abs, f) for f in frames.v_xy)
    wmax = maximum(maximum(abs, f) for f in frames.w_yz)

    fig2 = Figure(size=(1100, 460))
    ax_v = Axis(fig2[1, 1]; aspect=DataAspect(), xlabel="x (m)", ylabel="y (m)")
    obs_v = Observable(zeros(Nx, Ny))
    hm_v  = heatmap!(ax_v, xs, ys, obs_v;
                     colormap=:balance, colorrange=(-vmax, vmax))
    Colorbar(fig2[1, 2], hm_v)

    ax_w = Axis(fig2[1, 3]; aspect=DataAspect(), xlabel="y (m)", ylabel="z (m)")
    obs_w = Observable(zeros(Ny, Nz))
    hm_w  = heatmap!(ax_w, ys, zs, obs_w;
                     colormap=:balance, colorrange=(-wmax, wmax))
    Colorbar(fig2[1, 4], hm_w)

    title_obs = Observable("")
    Label(fig2[0, :], title_obs; fontsize=14, halign=:center)

    record(fig2, "three_dimensional_langmuir.mp4", 1:n_frames; framerate=14) do n
        obs_v[]    = frames.v_xy[n]
        obs_w[]    = frames.w_yz[n]
        title_obs[] = @sprintf("v (z ≈ %.2f m) and  ⟨w⟩_x ,  t = %5.1f s",
                               zs[end-3], frames.times[n])
    end
end

# ![](three_dimensional_langmuir.mp4)
