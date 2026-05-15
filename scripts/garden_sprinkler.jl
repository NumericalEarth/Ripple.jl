# Tolman (2002) Fig 1(a): Garden Sprinkler Effect test case.
#
# A localized initial swell in (x, y, frequency, direction) space is
# propagated for 5 days at the conventional operational resolution
# (Δθ = 15°, γ = 1.10). With a coarse directional grid, each discrete
# direction propagates as its own beam at slightly different speeds /
# angles, so the continuous swell field disintegrates into discrete
# beams — the "garden sprinkler" effect.
#
# Reference: H.L. Tolman, "Alleviating the Garden Sprinkler Effect in
# wind wave models", Ocean Modelling 4 (2002) 269–289, Fig 1(a).

using Oceananigans, Ripple, CairoMakie
using Printf

const km = 1.0e3
const day = 86400.0
const g = 9.81

# ## Physical grid: 4500 km × 3500 km at Δx = Δy = 100 km, bounded.
Lx = 4500 * km
Ly = 3500 * km
Δ  = 100 * km
Nx = Int(Lx / Δ)   # 45
Ny = Int(Ly / Δ)   # 35

grid = RectilinearGrid(CPU();
                       size = (Nx, Ny, 1),
                       halo = (3, 3, 3),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-1, 0),
                       topology = (Bounded, Bounded, Bounded))

# ## Spectral grid: 24 directions, 20 logarithmically spaced frequencies
# with γ = 1.10. The Gaussian frequency width is 0.01 Hz around 0.1 Hz,
# so a few bins on either side capture the energy; the full range
# 0.05 – 0.32 Hz is operational-style.
γ = 1.10
Nf = 20
f0 = 0.05
frequencies = [f0 * γ^(i - 1) for i in 1:Nf]
Nφ = 24
φ_centers = collect(range(0, 2π; length = Nφ + 1))[1:Nφ]

spectral_grid = FrequencyDirectionGrid(; frequency = frequencies, φ = φ_centers)

# ## Initial action: Gaussian in (x, y), Gaussian in f, cos²(θ - θ_m).
# Centred at (500, 500) km with spatial spread 150 km, peak frequency
# 0.1 Hz, mean direction 30° (from +x axis).
x0 = 500 * km
y0 = 500 * km
σ_xy = 150 * km
f_m = 0.10
σ_f  = 0.01
θ_m  = deg2rad(30)

function initial_action(x, y, kx, ky)
    κ = hypot(kx, ky)
    ω = sqrt(g * κ)
    f = ω / (2π)
    φ = atan(ky, kx)
    spatial = exp(-((x - x0)^2 + (y - y0)^2) / (2 * σ_xy^2))
    freq    = exp(-((f - f_m)^2) / (2 * σ_f^2))
    direct  = max(cos(φ - θ_m), 0)^2
    return spatial * freq * direct
end

model = SpectralWaveModel(; grid,
                            spectral_grid,
                            velocities = ZeroVelocities(),
                            sources = nothing,
                            timestepper = :ForwardEuler)
set!(model, N = initial_action)

# Quick check: H_s magnitude at the initial peak
Hs0 = significant_wave_height(model.action)
@info "initial Hs" peak = maximum(Hs0) location = argmax(Hs0)

# ## Time loop: 5 days with a CFL-safe time step for c_g ≲ 15 m/s.
total_time = 5 * day
dt = 1500.0            # ~25 min, CFL ≈ 0.22 at the highest cg
steps = Int(ceil(total_time / dt))
frame_stride = max(1, steps ÷ 60)

Hs_frames = Matrix{Float64}[]
times = Float64[]
push!(Hs_frames, copy(significant_wave_height(model.action)))
push!(times, 0.0)

t0 = time()
for step in 1:steps
    time_step!(model, dt)
    if step % frame_stride == 0 || step == steps
        push!(Hs_frames, copy(significant_wave_height(model.action)))
        push!(times, model.clock.time)
        if step == steps || step % (frame_stride * 4) == 0
            @info @sprintf("step %4d/%d  t=%.2f d  Hs∈[%.4f, %.4f]",
                           step, steps, model.clock.time / day,
                           minimum(Hs_frames[end]), maximum(Hs_frames[end]))
        end
    end
end
@info "wall time" seconds = round(time() - t0; digits=1)

# ## Visualization: contour plot of H_s at t = 5 days (matches Tolman Fig 1).
output_dir = mkpath(joinpath(@__DIR__, "..", "garden_sprinkler_output"))

xs_km = collect(xnodes(grid)) ./ km
ys_km = collect(ynodes(grid)) ./ km

let
    Hs_final = Hs_frames[end]
    Hs_peak  = maximum(Hs_final)
    fig = Figure(size = (820, 660))
    ax = Axis(fig[1, 1];
              title = @sprintf("Garden Sprinkler test: Hs at t = 5 d  (peak %.2f m)", Hs_peak),
              xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
    hm = heatmap!(ax, xs_km, ys_km, Hs_final;
                  colormap = :viridis, colorrange = (0, Hs_peak))
    levels = collect(range(0.05 * Hs_peak, 0.95 * Hs_peak; length = 10))
    contour!(ax, xs_km, ys_km, Hs_final;
             levels = levels, color = :black, linewidth = 0.6)
    Colorbar(fig[1, 2], hm; label = "Hs (m)")
    save(joinpath(output_dir, "garden_sprinkler_Hs_t5d.png"), fig)
    @info "Hs map written" path = joinpath(output_dir, "garden_sprinkler_Hs_t5d.png")
end

# Animation: H_s as the swell propagates (and disintegrates) over 5 days.
let
    Hs_max = maximum(maximum.(Hs_frames))
    Hs_obs = Observable(Hs_frames[1])
    title_obs = Observable("t = 0.00 d")
    fig = Figure(size = (820, 660))
    ax = Axis(fig[1, 1];
              xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
    hm = heatmap!(ax, xs_km, ys_km, Hs_obs;
                  colormap = :viridis, colorrange = (0, Hs_max))
    Colorbar(fig[1, 2], hm; label = "Hs (m)")
    Label(fig[0, :], title_obs; fontsize = 18, halign = :center)

    movie_path = joinpath(output_dir, "garden_sprinkler.mp4")
    record(fig, movie_path, eachindex(Hs_frames); framerate = 10) do idx
        Hs_obs[] = Hs_frames[idx]
        title_obs[] = @sprintf("Garden Sprinkler test — t = %.2f d", times[idx] / day)
    end
    @info "movie written" path = movie_path
end

@info "final Hs stats" min = minimum(Hs_frames[end]) max = maximum(Hs_frames[end])
