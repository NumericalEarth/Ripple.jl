# Generate a publication-quality MP4 of the translating-hurricane example.
# Heavier than the smoke-test version: bigger grid, fuller spectrum, longer
# integration. Output is streamed to JLD2 via an Oceananigans `Simulation`
# and animated by loading back through `FieldTimeSeries`.
#
# Run from the repo root:
#   julia --project=. scripts/translating_hurricane_animation.jl

using Oceananigans, Ripple
using Oceananigans.Units
using CairoMakie
using Printf
CairoMakie.activate!(type = "png")

const OUTPUT_PATH = joinpath(@__DIR__, "..", "translating_hurricane.jld2")
const MOVIE_PATH  = joinpath(@__DIR__, "..", "translating_hurricane.mp4")

# ## Configuration
#
# DIA is omitted in the bundle below; the remaining cost (wind input +
# saturation dissipation + WENO transport) is cheap enough to push the
# resolution and integration time considerably further than the smoke
# example.
Nx = Ny = 48
Lx = Ly = 2400kilometers
T_FINAL = 3days
DT      = 5minutes
FRAME_DT = 1hour

NFREQ = 20
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

# ## Translating Holland hurricane (diagonal track)
U_translation = 7.0
track_start   = (0.15Lx, 0.20Ly)
track_end     = (0.15Lx + U_translation * T_FINAL / sqrt(2),
                 0.20Ly + U_translation * T_FINAL / sqrt(2))
storm_track   = LinearStormTrack([0.0, T_FINAL], [track_start, track_end])

hurricane = HollandHurricaneWind(; center          = storm_track,
                                   vmax            = 50.0,
                                   rmax            = 40kilometers,
                                   radius          = 500kilometers,
                                   shape_parameter = 1.5,
                                   inflow_angle    = deg2rad(20),
                                   rotation        = Counterclockwise())

# ## Physics + model
#
# Forcing + dissipation only; no quadruplet nonlinear transfer.
wind_input  = PressureCorrelationInput(; drag      = BulkWindDrag(:linear),
                                         wind      = hurricane,
                                         direction = 0.0)
dissipation = LocalSaturationDissipation(; B_r     = 1.05e-2,
                                           σ_power = 1.0)
sources     = MeanSpectrumPhysics(; wind_input, dissipation)

model = SpectralWaveModel(grid, spectral_grid;
                          advection   = WENO(),
                          sources,
                          timestepper = :SemiImplicitEuler)

total_weight = sum(spectral_weight(spectral_grid, m, n) for m in 1:NFREQ, n in 1:NDIR)
set!(model, N = 1.0e-3 / total_weight)

simulation = Simulation(model; Δt = DT, stop_time = T_FINAL, verbose = true)

# Hourly diagnostic snapshots of `Hs` and `mean_direction`.
Hs       = significant_wave_height(model.action)
mean_dir = mean_direction(model.action)

simulation.output_writers[:diagnostics] =
    JLD2Writer(model, (; Hs, mean_dir);
               filename          = OUTPUT_PATH,
               schedule          = TimeInterval(FRAME_DT),
               overwrite_existing = true)

run!(simulation)

# ## Load snapshots back
Hs_ts    = FieldTimeSeries(OUTPUT_PATH, "Hs")
times    = Hs_ts.times
nframes  = length(times)

# Wind speed evaluated from the storm struct at each saved time (cheap; no
# need to write it to disk).
xs = collect(xnodes(grid)) ./ 1kilometer
ys = collect(ynodes(grid)) ./ 1kilometer

function wind_speed_field(t)
    field = Matrix{Float64}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        x = (i - 0.5) * Lx / Nx
        y = (j - 0.5) * Ly / Ny
        field[i, j] = wind_speed(hurricane, x, y, t)
    end
    return field
end

storm_xy   = [hurricane.center(t) for t in times]
track_xs   = [pt[1] / 1kilometer for pt in storm_xy]
track_ys   = [pt[2] / 1kilometer for pt in storm_xy]

hs_frames  = [Array(interior(Hs_ts[i]))[:, :, 1] for i in 1:nframes]
wnd_frames = [wind_speed_field(t)                for t in times]
hs_max     = maximum(maximum, hs_frames)
wnd_max    = maximum(maximum, wnd_frames)

# ## Animation
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

record(fig, MOVIE_PATH, eachindex(hs_frames); framerate = 12) do idx
    hs_obs[]    = hs_frames[idx]
    wnd_obs[]   = wnd_frames[idx]
    track_obs[] = (track_xs[1:idx], track_ys[1:idx])
    storm_obs[] = ([track_xs[idx]], [track_ys[idx]])
    title_obs[] = @sprintf("Translating Holland hurricane (U_t = %.0f m/s) — t = %5.1f h",
                            U_translation, times[idx] / 1hour)
end

@info "Animation written" path = MOVIE_PATH frames = nframes hs_max wnd_max
