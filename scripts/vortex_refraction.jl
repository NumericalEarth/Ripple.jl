using Oceananigans, Ripple, CairoMakie
using Printf

Nx = Ny = 64
Nz = 16
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

# Barotropic Gaussian-cored vortex, depth-uniform.
xc, yc = Lx / 2, Ly / 2
a = Lx / 8       # core radius
U0 = 0.4         # peak azimuthal speed

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

model = SpectralWaveModel(; grid,
                            spectral_grid,
                            velocities = (; u=u_field, v=v_field),
                            advection = nothing,
                            sources = nothing,
                            timestepper = :ForwardEuler)
coupling = model.coupling

κ0 = 0.4; σκ = 0.05; σφ = 0.30
set!(model, N = (x, y, κ, φ) ->
     exp(-((κ - κ0) / σκ)^2 - (sin(φ / 2)^2) / σφ^2))

G  = similar(model.action)
N1 = similar(model.action)
N2 = similar(model.action)

dt = 0.05
total_time = 20.0
steps = Int(round(total_time / dt))
frame_stride = max(1, steps ÷ 60)

m0_frames = Matrix{Float64}[]
krms_frames = Matrix{Float64}[]
dir_frames = Matrix{Float64}[]
times = Float64[]

function snapshot!()
    push!(m0_frames, m0(model.action))
    push!(krms_frames, root_mean_square_wavenumber(model.action))
    push!(dir_frames, mean_direction(model.action))
    push!(times, model.clock.time)
end

snapshot!()

t0 = time()
for step in 1:steps
    rk3_step!(model, coupling, G, N1, N2, dt)
    if step % frame_stride == 0 || step == steps
        snapshot!()
        @info @sprintf("step %4d/%d  t=%.2fs  m0∈[%.4f, %.4f]  κᵣₘₛ∈[%.4f, %.4f]",
                       step, steps, model.clock.time,
                       minimum(m0_frames[end]), maximum(m0_frames[end]),
                       minimum(krms_frames[end]), maximum(krms_frames[end]))
    end
end
@info "simulation done" wall_clock_s = round(time() - t0; digits=1) frames = length(m0_frames)

output_dir = mkpath(joinpath(@__DIR__, "..", "vortex_refraction_output"))

let speed = hypot.(view(u_field, :, :, 1), view(v_field, :, :, 1))
    fig = Figure(size = (640, 540))
    ax = Axis(fig[1, 1]; title = "Barotropic vortex |U| (m/s)",
              xlabel = "x (m)", ylabel = "y (m)", aspect = 1)
    hm = heatmap!(ax, xs, ys, speed; colormap = :viridis)
    Colorbar(fig[1, 2], hm)
    save(joinpath(output_dir, "vortex_speed.png"), fig)
end

m0_limits = (minimum(minimum.(m0_frames)), maximum(maximum.(m0_frames)))
krms_limits = (minimum(minimum.(krms_frames)), maximum(maximum.(krms_frames)))

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
dir_lim = max(maximum(maximum.(abs, dir_frames)), 0.05)
hm3 = heatmap!(ax3, xs, ys, dir_obs; colormap = :balance, colorrange = (-dir_lim, dir_lim))
Colorbar(fig[1, 2], hm1)
Colorbar(fig[1, 4], hm2)
Colorbar(fig[1, 6], hm3)
Label(fig[0, :], title_obs; fontsize = 18, halign = :center)

movie_path = joinpath(output_dir, "vortex_refraction.mp4")
record(fig, movie_path, eachindex(m0_frames); framerate = 12) do idx
    m0_obs[] = m0_frames[idx]
    krms_obs[] = krms_frames[idx]
    dir_obs[] = dir_frames[idx]
    title_obs[] = @sprintf("Wave action through barotropic vortex — t = %5.2f s", times[idx])
end

@info "wrote movie" movie_path
@info "final m0 stats"    minimum(m0_frames[end])    maximum(m0_frames[end])
@info "final κᵣₘₛ stats"  minimum(krms_frames[end])  maximum(krms_frames[end])
@info "final dir stats"   minimum(dir_frames[end])   maximum(dir_frames[end])
