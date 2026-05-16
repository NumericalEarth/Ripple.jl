# # Source-Only Fetch-Limited Growth
#
# Source-only models use `horizontal_advection=nothing`, matching the
# Oceananigans and Breeze convention that absent physics is `nothing`. This
# 1×1 column combines `ExponentialWindInput` (wind-driven growth) with
# `WhitecappingDissipation` (saturation-limited dissipation) and approaches
# the analytic equilibrium
#
# ```math
# m_0^\star = N_\mathrm{sat}\,(1 + r_w / r_d)
# ```
#
# where ``r_w`` is the wind growth rate and ``r_d`` the whitecapping rate.

using Oceananigans, Ripple
using CairoMakie
CairoMakie.activate!(type = "png")

# ## Grid + spectral grid
#
# A single physical cell and a single spectral cell — this is purely a
# source-term ODE in disguise.

grid          = RectilinearGrid(CPU(); size = (1, 1, 1), x = (0, 1), y = (0, 1), z = (0, 1))
spectral_grid = PolarWaveVectorGrid(Float64; κ = [1.0], φ = [0.0])

# ## Sources and equilibrium

target_growth_rate           = 1.2
directional_growth_weight    = Ripple.wind_directional_weight(spectral_grid, 1, 1, 0.0, 2)
growth_rate                  = target_growth_rate / directional_growth_weight
whitecapping_rate            = 4.8
saturation_threshold         = 0.5

sources = SourceTermSet((
    ExponentialWindInput(rate = growth_rate, direction = 0.0, spreading_power = 2),
    WhitecappingDissipation(rate                 = whitecapping_rate,
                            saturation_threshold = saturation_threshold,
                            saturation_power     = 1.0,
                            wavenumber_power     = 0.0),
))

model = SpectralWaveModel(grid, spectral_grid;
                          horizontal_advection = nothing,
                          sources,
                          timestepper = :SemiImplicitEuler);

weight         = spectral_weight(spectral_grid, 1, 1)
equilibrium_m0 = saturation_threshold * (1 + target_growth_rate / whitecapping_rate)
set!(model, N = saturation_threshold / (20 * weight));

# ## Time integration
#
# 240 semi-implicit Euler steps of ``\Delta t = 0.02\,\mathrm{s}``.

dt         = 0.02
step_count = 240
times      = [model.clock.time]
moments    = [m0(model.action)[1, 1, 1]]

for _ in 1:step_count
    time_step!(model, dt)
    push!(times,   model.clock.time)
    push!(moments, m0(model.action)[1, 1, 1])
end

# ## Plot

fig = Figure(size = (720, 360))
ax  = Axis(fig[1, 1]; title  = "Fetch-limited source balance",
                       xlabel = "time",
                       ylabel = "m₀")
lines!(ax, times, moments;                         label = "column m₀")
lines!(ax, times, fill(equilibrium_m0, length(times)); label = "analytic equilibrium", linestyle = :dash)
axislegend(ax; position = :rb)
fig
