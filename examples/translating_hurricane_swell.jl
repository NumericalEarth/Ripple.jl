# # Swell Generation by a Translating Idealized Hurricane
#
# A translating tropical cyclone (TC) is a textbook generator of long-period
# swell. The salient features captured here are:
#
# 1. **Storm-following sea state** under the eyewall.
# 2. **Extended (or "trapped") fetch** on the right of the track: when the
#    storm translation speed ``U_t`` is close to the deep-water group speed
#    of the dominant waves, those waves stay inside the forcing region for
#    many radii and grow well above the stationary-storm limit
#    [Bowyer2005](@cite).
# 3. **Far-field swell wake**: low-frequency, narrow-banded waves that
#    radiate forward and to the right of the storm and persist long after
#    the storm has passed [Young2003](@cite), [Young2006](@cite).
#
# Features 1–2 develop within hours; the swell wake takes a day or more
# to populate the far field. We integrate for two days to capture both.
#
# Physics here is a wind-input + dissipation pair. We deliberately omit the
# quadruplet nonlinear transfer because (a) it's the dominant CPU cost when
# included and (b) the storm-wake geometry of interest here is set by
# forcing and whitecapping, not by spectral peak downshifting.
#
# - **Wind input** [Janssen1991](@cite): `PressureCorrelationInput`.
# - **Dissipation** [Ardhuin2010](@cite): `LocalSaturationDissipation`.
# - **Hurricane wind** [Holland1980](@cite): `HollandHurricaneWind` translated
#   along a straight track.

using Oceananigans, Ripple
using Oceananigans.Units
using CairoMakie
using Printf
CairoMakie.activate!(type = "png")

# ## Domain
#
# 1500 × 1500 km periodic basin, single vertical level (deep water).

Nx = Ny = 24
Lx = Ly = 1500kilometers

grid = RectilinearGrid(CPU();
                       size     = (Nx, Ny, 1),
                       halo     = (3, 3, 3),
                       x        = (0, Lx),
                       y        = (0, Ly),
                       z        = (-1.0, 0.0),
                       topology = (Periodic, Periodic, Bounded))

# Spectral grid: 12 logarithmically spaced frequencies × 8 directions.

NFREQ = 12
NDIR  = 8
f0    = 0.04118
xfr   = 1.15
spectral_grid = FrequencyDirectionGrid(;
    frequency = [f0 * xfr^(k - 1) for k in 1:NFREQ],
    φ         = collect(range(0, 2π * (NDIR - 1) / NDIR; length = NDIR)))

# ## Translating Holland hurricane
#
# Track is purely zonal at ``U_t = 7\,\mathrm{m/s}`` — close to the
# deep-water group speed for ``f \approx 0.1\,\mathrm{Hz}``, placing us
# near the Bowyer–MacAfee extended-fetch resonance.

T_FINAL       = 2days
U_translation = 7.0
y_track       = 0.40 * Ly
track_start   = (0.15 * Lx,                          y_track)
track_end     = (0.15 * Lx + U_translation * T_FINAL, y_track)
storm_track   = LinearStormTrack([0.0, T_FINAL], [track_start, track_end])

hurricane = HollandHurricaneWind(; center          = storm_track,
                                   vmax            = 50.0,
                                   rmax            = 40kilometers,
                                   radius          = 500kilometers,
                                   shape_parameter = 1.5,
                                   inflow_angle    = deg2rad(20),
                                   rotation        = Counterclockwise())

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
# horizontal transport is essential here — swell wakes propagate hundreds
# of kilometres beyond the storm, and numerical diffusion would smear them.

model = SpectralWaveModel(grid, spectral_grid;
                          advection   = WENO(),
                          sources,
                          timestepper = :SemiImplicitEuler)

total_weight = sum(spectral_weight(spectral_grid, m, n) for m in 1:NFREQ, n in 1:NDIR)
set!(model, N = 1.0e-3 / total_weight)

simulation = Simulation(model; Δt = 5minutes, stop_time = T_FINAL, verbose = false)

# ## Output writer
#
# 2-D diagnostic fields snapshotted every two hours. Saving the full 4-D
# action would be ~5 MB per snapshot at this resolution; saving the three
# 2-D bulk moments is ~20 KB and is what visualisations actually want.

output_path = "translating_hurricane_swell.jld2"

Hs       = significant_wave_height(model.action)
fpeak    = peak_frequency(model.action)
mean_dir = mean_direction(model.action)

simulation.output_writers[:diagnostics] =
    JLD2Writer(model, (; Hs, fpeak, mean_dir);
               filename          = output_path,
               schedule          = TimeInterval(2hours),
               overwrite_existing = true)

run!(simulation)

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

# ## Mosaic of ``H_s`` at six times across the integration

hs_max  = maximum(maximum.(interior.(Hs_ts[t] for t in 1:nframes)))
mosaic  = Figure(size = (1200, 800))
ntiles  = min(nframes, 6)
idxs    = round.(Int, range(1, nframes; length = ntiles))
for (ti, idx) in enumerate(idxs)
    r = (ti - 1) ÷ 3 + 1
    c = (ti - 1) % 3 + 1
    ax = Axis(mosaic[r, c];
              title  = @sprintf("t = %.1f h", times[idx] / 1hour),
              xlabel = "x (km)",
              ylabel = "y (km)",
              aspect = DataAspect())
    heatmap!(ax, xs, ys, Array(interior(Hs_ts[idx]))[:, :, 1];
             colormap = :viridis, colorrange = (0, hs_max))
    sx, sy = storm_xy[idx]
    scatter!(ax, [sx / 1kilometer], [sy / 1kilometer];
             color = :red, marker = :star5, markersize = 18)
end
Colorbar(mosaic[:, 4];
         colormap   = :viridis,
         colorrange = (0, hs_max),
         label      = "Hs (m)")
mosaic

# ## Final-time ``H_s`` with storm-track overlay

fig = Figure(size = (900, 700))
ax  = Axis(fig[1, 1];
           xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect(),
           title  = @sprintf("Hs at t = %.1f d, Holland TC, U_t = %.1f m/s",
                              times[end] / 1day, U_translation))
hm  = heatmap!(ax, xs, ys, Array(interior(Hs_ts[nframes]))[:, :, 1];
               colormap = :viridis, colorrange = (0, hs_max))
Colorbar(fig[1, 2], hm; label = "Hs (m)")
track_xs = [pt[1] / 1kilometer for pt in storm_xy]
track_ys = [pt[2] / 1kilometer for pt in storm_xy]
lines!(ax, track_xs, track_ys; color = :white, linewidth = 2)
scatter!(ax, [track_xs[end]], [track_ys[end]];
         color = :red, marker = :star5, markersize = 25)
fig
