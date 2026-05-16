# Generate a publication-quality MP4 of the translating-hurricane example.
# Heavier than the smoke-test version: 60×60 grid, 24 freq × 16 dir, 36 h.
#
# Run from the repo root:
#   julia --project=. scripts/translating_hurricane_animation.jl

using Oceananigans, Ripple
using CairoMakie
using Printf
CairoMakie.activate!(type = "png")

const OUTPUT_PATH = joinpath(@__DIR__, "..", "translating_hurricane.mp4")

# ## Configuration
Nx = Ny   = 60
Lx = Ly   = 1.8e6                            # 1800 km basin
T_FINAL   = 36 * 3600.0                      # 36 hours
DT        = 90.0
FRAME_DT  = 30 * 60.0                        # 30-min frames → 72 frames

# Spectral grid: 24 frequencies × 16 directions.
NFREQ = 24
NDIR  = 16
f0    = 0.04118
xfr   = 1.10
frequency_centers = [f0 * xfr^(k - 1) for k in 1:NFREQ]
direction_centers = collect(range(0, 2π * (NDIR - 1) / NDIR; length = NDIR))

grid = RectilinearGrid(CPU();
                       size     = (Nx, Ny, 1),
                       halo     = (3, 3, 3),
                       x        = (0, Lx),
                       y        = (0, Ly),
                       z        = (-1.0, 0.0),
                       topology = (Periodic, Periodic, Bounded))

spectral_grid = FrequencyDirectionGrid(; frequency = frequency_centers,
                                         φ         = direction_centers)

# ## Translating Holland hurricane
U_translation = 7.0
y_track       = 0.40 * Ly
track_start   = (0.15 * Lx,                          y_track)
track_end     = (0.15 * Lx + U_translation * T_FINAL, y_track)
storm_track   = LinearStormTrack([0.0, T_FINAL], [track_start, track_end])

hurricane = HollandHurricaneWind(; center          = storm_track,
                                   vmax            = 50.0,
                                   rmax            = 40.0e3,
                                   radius          = 500.0e3,
                                   shape_parameter = 1.5,
                                   inflow_angle    = deg2rad(20),
                                   rotation        = Counterclockwise())

# ## Physics + model
wind_input  = PressureCorrelationInput(; drag      = BulkWindDrag(:linear),
                                         wind      = hurricane,
                                         direction = 0.0)
dissipation = LocalSaturationDissipation(; B_r     = 1.05e-2,
                                           σ_power = 1.0)
nonlinear   = HasselmannDIA(; C = 1.5e7)
physics     = MeanSpectrumPhysics(; wind_input, dissipation, nonlinear)

model = SpectralWaveModel(grid, spectral_grid;
                          advection   = WENO(),
                          physics,
                          timestepper = :SemiImplicitEuler)

total_weight = sum(spectral_weight(spectral_grid, m, n) for m in 1:NFREQ, n in 1:NDIR)
set!(model, N = 1.0e-3 / total_weight)

# ## Pre-compute wind speed snapshots
xs = collect(xnodes(grid)) ./ 1e3
ys = collect(ynodes(grid)) ./ 1e3

function wind_speed_field(t)
    field = Matrix{Float64}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        x = (i - 0.5) * Lx / Nx
        y = (j - 0.5) * Ly / Ny
        field[i, j] = wind_speed(hurricane, x, y, t)
    end
    return field
end

# ## Time integration with frame capture
times      = Float64[model.clock.time]
hs_frames  = Matrix{Float64}[Array(interior(significant_wave_height(model.action)))[:, :, 1]]
wnd_frames = Matrix{Float64}[wind_speed_field(model.clock.time)]
track_xs   = Float64[hurricane.center(model.clock.time)[1] / 1e3]
track_ys   = Float64[hurricane.center(model.clock.time)[2] / 1e3]

let next_output = FRAME_DT
    while model.clock.time < T_FINAL
        time_step!(model, DT)
        if model.clock.time >= next_output - DT / 2
            push!(times,      model.clock.time)
            push!(hs_frames,  Array(interior(significant_wave_height(model.action)))[:, :, 1])
            push!(wnd_frames, wind_speed_field(model.clock.time))
            sx, sy = hurricane.center(model.clock.time)
            push!(track_xs, sx / 1e3)
            push!(track_ys, sy / 1e3)
            next_output += FRAME_DT
            @info "frame" t_hours = model.clock.time / 3600 Hs_max = maximum(hs_frames[end]) U10_max = maximum(wnd_frames[end])
        end
    end
end

# ## Animation
hs_max  = maximum(maximum, hs_frames)
wnd_max = maximum(maximum, wnd_frames)

hs_obs    = Observable(hs_frames[1])
wnd_obs   = Observable(wnd_frames[1])
track_obs = Observable((track_xs[1:1], track_ys[1:1]))
storm_obs = Observable(([track_xs[1]], [track_ys[1]]))
title_obs = Observable("t = 0.0 h")

fig = Figure(size = (1280, 600), backgroundcolor = :white)
Label(fig[0, :], title_obs; fontsize = 22, halign = :center, font = :bold)

ax1 = Axis(fig[1, 1];
           title  = "Wind speed |U₁₀| (m/s)",
           xlabel = "x (km)", ylabel = "y (km)",
           aspect = DataAspect())
hm1 = heatmap!(ax1, xs, ys, wnd_obs;
               colormap = :magma, colorrange = (0, wnd_max))
lines!(ax1, lift(t -> t[1], track_obs), lift(t -> t[2], track_obs);
       color = :white, linewidth = 2)
scatter!(ax1, lift(t -> t[1], storm_obs), lift(t -> t[2], storm_obs);
         color = :white, marker = :star5, markersize = 22, strokewidth = 1.5,
         strokecolor = :black)
Colorbar(fig[1, 2], hm1; label = "U₁₀ (m/s)")

ax2 = Axis(fig[1, 3];
           title  = "Significant wave height Hs (m)",
           xlabel = "x (km)", ylabel = "y (km)",
           aspect = DataAspect())
hm2 = heatmap!(ax2, xs, ys, hs_obs;
               colormap = :viridis, colorrange = (0, hs_max))
lines!(ax2, lift(t -> t[1], track_obs), lift(t -> t[2], track_obs);
       color = :white, linewidth = 2)
scatter!(ax2, lift(t -> t[1], storm_obs), lift(t -> t[2], storm_obs);
         color = :red, marker = :star5, markersize = 22, strokewidth = 1.5,
         strokecolor = :black)
Colorbar(fig[1, 4], hm2; label = "Hs (m)")

record(fig, OUTPUT_PATH, eachindex(hs_frames); framerate = 12) do idx
    hs_obs[]    = hs_frames[idx]
    wnd_obs[]   = wnd_frames[idx]
    track_obs[] = (track_xs[1:idx], track_ys[1:idx])
    storm_obs[] = ([track_xs[idx]], [track_ys[idx]])
    title_obs[] = @sprintf("Translating Holland hurricane (U_t = %.0f m/s) — t = %5.1f h",
                            U_translation, times[idx] / 3600)
end

@info "Animation written" path = OUTPUT_PATH frames = length(hs_frames) hs_max wnd_max
