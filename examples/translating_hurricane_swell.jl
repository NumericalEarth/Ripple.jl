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
# These features show up in operational hindcasts of Bonnie
# [Wright2001](@cite), Ivan [Moon2003](@cite), and other major hurricanes;
# the fetch-growth scaling is reviewed in [Hwang2016](@cite).
#
# The state-of-the-art physics used here mirrors WAVEWATCH III's ST3/ST4
# practice:
#
# - **Wind input** [Janssen1991](@cite): `PressureCorrelationInput` — Janssen
#   quasi-linear pressure correlation with a per-grid-point wave-supported
#   stress cap.
# - **Dissipation** [Ardhuin2010](@cite): `LocalSaturationDissipation` —
#   saturation-based whitecapping, ``\propto (B/B_r - 1)^p``.
# - **Nonlinear interactions** [Hasselmann1985](@cite): `HasselmannDIA` —
#   discrete interaction approximation with bilinear receiver spread.
# - **Hurricane wind** [Holland1980](@cite): `HollandHurricaneWind`, advected
#   along a straight track at constant ``U_t``.

using Oceananigans, Ripple
using CairoMakie
using Printf
CairoMakie.activate!(type = "png")

# ## Domain
#
# 1200 × 1200 km periodic basin, single vertical level (deep water). The
# resolution here is intentionally light so this example runs in a few
# seconds as part of the smoke suite; production-scale runs would push the
# horizontal grid to 60²–120² with 25 frequencies and 24 directions.

Nx = Ny = 24
Lx = Ly = 1.2e6
T_FINAL  = 6 * 3600.0
DT       = 120.0
SNAPSHOT = 1 * 3600.0

grid = RectilinearGrid(CPU();
                       size     = (Nx, Ny, 1),
                       halo     = (3, 3, 3),
                       x        = (0, Lx),
                       y        = (0, Ly),
                       z        = (-1.0, 0.0),
                       topology = (Periodic, Periodic, Bounded))

NFREQ = 12
NDIR  = 8
f0    = 0.04118
xfr   = 1.15
spectral_grid = FrequencyDirectionGrid(;
    frequency = [f0 * xfr^(k - 1) for k in 1:NFREQ],
    φ         = collect(range(0, 2π * (NDIR - 1) / NDIR; length = NDIR)))

# ## Translating Holland hurricane
#
# `LinearStormTrack` interpolates the storm center linearly between
# waypoints. The track is purely zonal at ``U_t = 7\,\mathrm{m/s}`` — close
# to the deep-water group speed for ``f \approx 0.1\,\mathrm{Hz}``, putting
# us near the Bowyer–MacAfee extended-fetch resonance.

U_translation = 7.0
y_track       = 0.40 * Ly
track_start   = (0.15 * Lx,                                  y_track)
track_end     = (0.15 * Lx + U_translation * T_FINAL,        y_track)
storm_track   = LinearStormTrack([0.0, T_FINAL], [track_start, track_end])

hurricane = HollandHurricaneWind(; center          = storm_track,
                                   vmax            = 50.0,
                                   rmax            = 40.0e3,
                                   radius          = 500.0e3,
                                   shape_parameter = 1.5,
                                   inflow_angle    = deg2rad(20),
                                   rotation        = Counterclockwise())

# ## Physics bundle
#
# `MeanSpectrumPhysics` co-optimizes the three terms via `prepare_physics`,
# which runs three KernelAbstractions kernels once per time step to
# precompute (i) the wave-supported-stress cap for the wind input, (ii)
# bulk spectral moments needed by mean-spectrum-based terms, and (iii) the
# DIA nonlinear transfer field.

wind_input  = PressureCorrelationInput(; drag      = BulkWindDrag(:linear),
                                         wind      = hurricane,
                                         direction = 0.0)
dissipation = LocalSaturationDissipation(; B_r     = 1.05e-2,
                                           σ_power = 1.0)
nonlinear   = HasselmannDIA(; C = 1.5e7)
physics     = MeanSpectrumPhysics(; wind_input, dissipation, nonlinear)

# ## Model
#
# `advection = WENO()` is a shortcut that sets both horizontal and spectral
# advection; here we only need horizontal transport, but WENO is essential
# — swell wakes propagate hundreds of kilometres beyond the storm, and
# numerical diffusion would smear them.

model = SpectralWaveModel(grid, spectral_grid;
                          advection   = WENO(),
                          physics,
                          timestepper = :SemiImplicitEuler)

# Spin-up seed: a small uniform action density.
total_weight = sum(spectral_weight(spectral_grid, m, n) for m in 1:NFREQ, n in 1:NDIR)
set!(model, N = 1.0e-3 / total_weight)

# ## Time integration
#
# 24-hour run, sampling Hs and the storm location every 3 h.

times    = Float64[model.clock.time]
hs_snaps = Matrix{Float64}[Array(interior(significant_wave_height(model.action)))[:, :, 1]]
storm_xy = Tuple{Float64, Float64}[hurricane.center(model.clock.time)]

let next_output = SNAPSHOT
    while model.clock.time < T_FINAL
        time_step!(model, DT)
        if model.clock.time >= next_output - DT / 2
            push!(times,    model.clock.time)
            push!(hs_snaps, Array(interior(significant_wave_height(model.action)))[:, :, 1])
            push!(storm_xy, hurricane.center(model.clock.time))
            next_output += SNAPSHOT
        end
    end
end

# ## Snapshot mosaic
#
# Hs at six times. The right-front quadrant develops noticeably higher Hs
# than the left side (extended-fetch resonance), and a long swell wake
# trails the storm.

xs = collect(xnodes(grid)) ./ 1e3
ys = collect(ynodes(grid)) ./ 1e3
hs_max = maximum(maximum, hs_snaps)

mosaic = Figure(size = (1200, 800))
ntiles = min(length(hs_snaps), 6)
idxs   = round.(Int, range(1, length(hs_snaps); length = ntiles))
for (ti, idx) in enumerate(idxs)
    r = (ti - 1) ÷ 3 + 1
    c = (ti - 1) % 3 + 1
    ax = Axis(mosaic[r, c];
              title  = @sprintf("t = %.1f h", times[idx] / 3600),
              xlabel = "x (km)",
              ylabel = "y (km)",
              aspect = DataAspect())
    heatmap!(ax, xs, ys, hs_snaps[idx];
             colormap = :viridis, colorrange = (0, hs_max))
    sx, sy = storm_xy[idx]
    scatter!(ax, [sx / 1e3], [sy / 1e3];
             color = :red, marker = :star5, markersize = 18)
end
Colorbar(mosaic[:, 4];
         colormap = :viridis,
         colorrange = (0, hs_max),
         label = "Hs (m)")
mosaic

# ## Final-time Hs with storm track overlay
#
# The white line traces the storm history; the red star marks its current
# location at ``t = 24\,\mathrm{h}``.

fig = Figure(size = (900, 700))
ax  = Axis(fig[1, 1];
           xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect(),
           title  = @sprintf("Hs at t = %.1f h, Holland TC, U_t = %.1f m/s",
                              times[end] / 3600, U_translation))
hm  = heatmap!(ax, xs, ys, hs_snaps[end];
               colormap = :viridis, colorrange = (0, hs_max))
Colorbar(fig[1, 2], hm; label = "Hs (m)")
track_xs = [pt[1] / 1e3 for pt in storm_xy]
track_ys = [pt[2] / 1e3 for pt in storm_xy]
lines!(ax, track_xs, track_ys; color = :white, linewidth = 2)
scatter!(ax, [track_xs[end]], [track_ys[end]];
         color = :red, marker = :star5, markersize = 25)
fig
