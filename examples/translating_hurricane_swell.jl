# # Swell Generation by a Translating Idealized Hurricane
#
# A translating tropical cyclone (TC) generates a wide-angle swell field
# whose transverse structure is the headline feature here. The salient
# effects captured by this configuration:
#
# 1. **Storm-following sea state** under the eyewall.
# 2. **Extended (or "trapped") fetch** on the right of the track: when the
#    storm translation speed ``U_t`` is close to the deep-water group speed
#    of the dominant waves, those waves stay inside the forcing region for
#    many radii and grow well above the stationary-storm limit
#    [Bowyer2005](@cite).
# 3. **Transverse swell fans**: low-frequency, narrow-banded waves radiate
#    away from the track at large oblique angles. With a long zonal domain
#    they can be followed for thousands of kilometres on either side
#    [Young2003](@cite), [Young2006](@cite). The 2:1 zonal:meridional
#    domain aspect ratio and the six-day integration make this directional
#    spreading the dominant visual feature of the final map.
#
# Physics here is a wind-input + dissipation pair. The quadruplet nonlinear
# transfer is deliberately omitted — both because it's the dominant CPU
# cost when included and because the geometry of interest here (where the
# swell goes) is set by forcing + whitecapping plus group-velocity
# transport, not by spectral peak downshifting.
#
# - **Wind input** [Janssen1991](@cite): `PressureCorrelationInput`.
# - **Dissipation** [Ardhuin2010](@cite): `LocalSaturationDissipation`.
# - **Hurricane wind** [Holland1980](@cite): `HollandHurricaneWind` translated
#   along a straight zonal track.

using Oceananigans, Ripple
using Oceananigans.Units
using CairoMakie
using Printf
CairoMakie.activate!(type = "png")

# ## Domain
#
# 6000 × 3000 km periodic basin, single vertical level (deep water). The
# zonal direction is twice the meridional so the transverse swell wings
# have ample y-extent to develop on either side of the track. ``y`` is
# centred on zero so the storm sits at ``y = 0``. The 64 × 32 grid gives
# ~94 km cells — coarse, but enough to resolve the transverse fan
# geometry; double the resolution if rendering a paper figure on GPU.

Nx = 64
Ny = 32
Lx = 6000kilometers
Ly = 3000kilometers

grid = RectilinearGrid(CPU();
                       size     = (Nx, Ny, 1),
                       halo     = (3, 3, 3),
                       x        = (0, Lx),
                       y        = (-Ly/2, Ly/2),
                       z        = (-1.0, 0.0),
                       topology = (Periodic, Periodic, Bounded));

# Spectral grid: 12 logarithmically spaced frequencies × 12 directions.
# 12 directions (30° bins) keeps the transverse-spreading fan visible
# without making the per-step source-term loop too expensive.

NFREQ = 12
NDIR  = 12
f0    = 0.04118
xfr   = 1.15
spectral_grid = FrequencyDirectionGrid(;
    frequency = [f0 * xfr^(k - 1) for k in 1:NFREQ],
    φ         = collect(range(0, 2π * (NDIR - 1) / NDIR; length = NDIR)));

# ## Translating Holland hurricane
#
# Track is purely zonal at ``U_t = 7\,\mathrm{m/s}`` — close to the
# deep-water group speed for ``f \approx 0.1\,\mathrm{Hz}``, placing us
# near the Bowyer–MacAfee extended-fetch resonance. The storm enters
# from the left edge and traverses ~3600 km in six days, ending well
# short of the periodic seam.

T_FINAL       = 6days
U_translation = 7.0
track_start   = (0.10Lx, 0.0)
track_end     = (0.10Lx + U_translation * T_FINAL, 0.0)
storm_track   = LinearStormTrack([0.0, T_FINAL], [track_start, track_end])

hurricane = HollandHurricaneWind(; center          = storm_track,
                                   vmax            = 50.0,
                                   rmax            = 40kilometers,
                                   radius          = 500kilometers,
                                   shape_parameter = 1.5,
                                   inflow_angle    = deg2rad(20),
                                   rotation        = Counterclockwise());

# ## Physics bundle
#
# `PrecomputedSources` co-optimizes the terms via `prepare_sources`, which
# runs two KernelAbstractions kernels once per tendency pass to precompute
# the wave-supported-stress cap (for wind input) and the bulk spectral
# moments (used by mean-spectrum-based whitecapping diagnostics).

wind_input  = PressureCorrelationInput(; drag      = BulkWindDrag(:linear),
                                         wind      = hurricane,
                                         direction = 0.0)
dissipation = LocalSaturationDissipation(; B_r     = 1.05e-2,
                                           σ_power = 1.0)
sources     = PrecomputedSources(; wind_input, dissipation)

# ## Model + simulation
#
# `advection = WENO()` sets both horizontal and spectral advection;
# horizontal transport is essential here — the transverse swell wings
# propagate over thousands of kilometres and numerical diffusion would
# smear them out.

model = SpectralWaveModel(grid, spectral_grid;
                          advection   = WENO(),
                          sources,
                          timestepper = :SemiImplicitEuler);

total_weight = sum(spectral_weight(spectral_grid, m, n) for m in 1:NFREQ, n in 1:NDIR)
set!(model, N = 1.0e-3 / total_weight);

# Δt = 30 min is well under the CFL bound (~130 min for these cell sizes
# and peak group velocities) and avoids spending most of the wall clock
# inside the per-step kernels.
simulation = Simulation(model; Δt = 30minutes, stop_time = T_FINAL, verbose = false);

# ## Output writer
#
# 2-D diagnostic fields snapshotted every twelve hours (13 frames over six
# days). Saving the full 4-D action would be ~28 MB per snapshot at this
# resolution; the bulk moments are ~75 KB. Keep the snapshot interval
# coarse so the resulting documentation page stays under the rendered-HTML
# size threshold.

output_path = "translating_hurricane_swell.jld2"

Hs       = significant_wave_height(model.action)
fpeak    = peak_frequency(model.action)
mean_dir = mean_direction(model.action)

simulation.output_writers[:diagnostics] =
    JLD2Writer(model, (; Hs, fpeak, mean_dir);
               filename          = output_path,
               schedule          = TimeInterval(12hours),
               overwrite_existing = true);

run!(simulation);

# ## Load snapshots back as `FieldTimeSeries`
#
# Each `FieldTimeSeries` is a 4-D array indexed `(i, j, k, t_index)` with
# `.times` carrying the snapshot times.

Hs_ts       = FieldTimeSeries(output_path, "Hs")
fpeak_ts    = FieldTimeSeries(output_path, "fpeak")
mean_dir_ts = FieldTimeSeries(output_path, "mean_dir")
times       = Hs_ts.times
nframes     = length(times)

xs = collect(xnodes(grid)) ./ 1kilometer
ys = collect(ynodes(grid)) ./ 1kilometer
storm_xy = [hurricane.center(t) for t in times]

# ## Animation
#
# Wave-speed left, ``H_s`` right. Hourly frames over the six-day run;
# embedded directly in the rendered documentation page. The transverse
# fan of swell on either side of the track is the headline visual
# feature — wave energy radiated from the storm at large oblique angles
# propagates meridionally for thousands of kilometres on both sides.

hs_max  = maximum(maximum.(interior.(Hs_ts[t] for t in 1:nframes)))
wnd_max = 0.0
wnd_frames = Matrix{Float64}[]
for t in times
    field = Matrix{Float64}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        x = (i - 0.5) * Lx / Nx
        y = -Ly/2 + (j - 0.5) * Ly / Ny
        field[i, j] = wind_speed(hurricane, x, y, t)
    end
    push!(wnd_frames, field)
    global wnd_max = max(wnd_max, maximum(field))
end

track_xs = [pt[1] / 1kilometer for pt in storm_xy]
track_ys = [pt[2] / 1kilometer for pt in storm_xy]

hs_obs    = Observable(Array(interior(Hs_ts[1]))[:, :, 1])
wnd_obs   = Observable(wnd_frames[1])
storm_obs = Observable(([track_xs[1]], [track_ys[1]]))
trail_obs = Observable((track_xs[1:1], track_ys[1:1]))
title_obs = Observable("t = 0.0 d")

fig = Figure(size = (1280, 500), backgroundcolor = :white)
Label(fig[0, :], title_obs; fontsize = 20, halign = :center, font = :bold)

ax1 = Axis(fig[1, 1];
           title  = "Wind speed |U₁₀| (m/s)",
           xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
hm1 = heatmap!(ax1, xs, ys, wnd_obs; colormap = :magma, colorrange = (0, wnd_max))
lines!(ax1, lift(t -> t[1], trail_obs), lift(t -> t[2], trail_obs);
       color = :white, linewidth = 2)
scatter!(ax1, lift(t -> t[1], storm_obs), lift(t -> t[2], storm_obs);
         color = :white, marker = :star5, markersize = 18,
         strokewidth = 1.0, strokecolor = :black)
Colorbar(fig[1, 2], hm1; label = "U₁₀ (m/s)")

ax2 = Axis(fig[1, 3];
           title  = "Significant wave height Hs (m)",
           xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
hm2 = heatmap!(ax2, xs, ys, hs_obs; colormap = :viridis, colorrange = (0, hs_max))
lines!(ax2, lift(t -> t[1], trail_obs), lift(t -> t[2], trail_obs);
       color = :white, linewidth = 2)
scatter!(ax2, lift(t -> t[1], storm_obs), lift(t -> t[2], storm_obs);
         color = :red, marker = :star5, markersize = 18,
         strokewidth = 1.0, strokecolor = :black)
Colorbar(fig[1, 4], hm2; label = "Hs (m)")

record(fig, "translating_hurricane_swell.mp4", 1:nframes; framerate = 6) do idx
    hs_obs[]    = Array(interior(Hs_ts[idx]))[:, :, 1]
    wnd_obs[]   = wnd_frames[idx]
    trail_obs[] = (track_xs[1:idx], track_ys[1:idx])
    storm_obs[] = ([track_xs[idx]], [track_ys[idx]])
    title_obs[] = @sprintf("Translating Holland TC (U_t = %.0f m/s) — t = %4.1f d",
                            U_translation, times[idx] / 1day)
end
nothing #hide

# ![](translating_hurricane_swell.mp4)
