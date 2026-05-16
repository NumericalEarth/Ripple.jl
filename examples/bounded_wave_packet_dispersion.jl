# # Bounded Wave Packet Dispersion
#
# A one-dimensional bounded physical domain is the simplest place to see
# transport. We initialize a compact packet near the left boundary with a
# finite-width wavenumber spectrum. All waves travel in the positive ``x``
# direction, but for deep-water gravity waves the group velocity is
# ``c_g(\kappa) = \tfrac{1}{2}\sqrt{g/\kappa}`` — long waves are fast,
# short waves are slow. The packet therefore *fans out*: at any observer
# downstream the long-wave component arrives first, and as time passes
# the local spectrum drifts toward shorter wavelengths.

using Oceananigans, Ripple
using CairoMakie
CairoMakie.activate!(type = "png")

# ## Grid setup
#
# Bounded in ``x`` and ``z``, periodic in ``y``. The default `WENO(order=5)`
# advection needs `halo=(3, 3, 1)`. We keep ``N_y`` small because the
# physics is one-dimensional.

Nx = 96
Ny = 6
Nk = 10
Lx = 384.0

grid = RectilinearGrid(CPU();
                       size     = (Nx, Ny, 1),
                       halo     = (3, 3, 1),
                       x        = (0, Lx),
                       y        = (0, 1),
                       z        = (-1, 0),
                       topology = (Bounded, Periodic, Bounded))

# Spectral grid: a thin wedge of ``N_\kappa`` bins, all pointing in ``+x``
# (single-bin ``\varphi``).

kappas        = range(0.35, 1.25; length = Nk)
theta_width   = pi / 18
spectral_grid = PolarWaveVectorGrid(Float64;
                                    κ       = kappas,
                                    φ       = [0.0],
                                    φ_faces = [-theta_width / 2, theta_width / 2])

# ## Model and initial packet
#
# Compact top-hat support in ``x \in [24, 64]``, uniform in ``\kappa``.

model = SpectralWaveModel(grid, spectral_grid;
                          horizontal_advection = WENO(order = 5),
                          timestepper          = :RK3);

packet_left  = 24.0
packet_right = 64.0
set!(model, N = (x, y, kx, ky) -> packet_left <= x < packet_right ? 1.0 : 0.0);

# ## Time stepping
#
# Run long enough for even the slowest waves to clear an observer near
# the downstream end. For ``\kappa_\mathrm{max} = 1.25``,
# ``c_g \approx 1.4\,\mathrm{m/s}``, so a 280 m journey takes ~200 s.

x_nodes = collect(xnodes(grid))
x_obs   = 280.0
i_obs   = argmin(abs.(x_nodes .- x_obs))

m0_profile() = vec(interior(m0(model.action))[:, 1, 1])

function observer_mean_kappa()
    f = root_mean_square_wavenumber(model.action)
    return @inbounds Array(interior(f))[i_obs, 1, 1]
end

dt              = 0.5
step_count      = 400
sample_interval = 4

times    = [model.clock.time]
profiles = [m0_profile()]
κ_obs    = [observer_mean_kappa()]

for step in 1:step_count
    time_step!(model, dt)
    if step == step_count || step % sample_interval == 0
        push!(times,    model.clock.time)
        push!(profiles, m0_profile())
        push!(κ_obs,    observer_mean_kappa())
    end
end

# ## Hovmöller of ``m_0``
#
# Action mass on the (``x``, ``t``) plane. The fan of leading and trailing
# edges spread linearly: each ``\kappa`` component contributes a wedge at
# slope ``c_g(\kappa)``. The dashed vertical line marks the observer
# column ``x = `` $(round(x_nodes[i_obs]; digits=1)) m used below.

hovmoller = reduce(vcat, transpose.(profiles))

fig1 = Figure(size = (720, 360))
ax1  = Axis(fig1[1, 1]; title  = "m₀(x, t)",
                        xlabel = "x (m)",
                        ylabel = "t (s)")
hm1  = heatmap!(ax1, x_nodes, times, transpose(hovmoller); colormap = :viridis)
vlines!(ax1, [x_nodes[i_obs]]; color = :white, linestyle = :dash)
Colorbar(fig1[1, 2], hm1)
fig1

# ## Wavenumber at a fixed observer
#
# At the observer column ``x = `` $(round(x_nodes[i_obs]; digits=1)) m,
# the local mean wavenumber is initially zero (no action present), jumps
# up to ``\kappa \approx 0.35`` (longest waves arrive first), and then
# climbs monotonically toward ``\kappa \approx 1.25`` as the slower,
# shorter-wavelength components catch up. The dashed reference is the
# stationary-phase prediction
#
# ```math
# \kappa(t) = \frac{g\,t^2}{4 (x_\mathrm{obs} - x_\mathrm{src})^2}
# ```
#
# obtained by inverting ``c_g(\kappa) = (x_\mathrm{obs} - x_\mathrm{src})/t``.

x_src = 0.5 * (packet_left + packet_right)
D     = x_nodes[i_obs] - x_src
g     = 9.81
κ_theory = [t > 0 ? min(g * t^2 / (4 * D^2), kappas[end]) : 0.0 for t in times]

fig2 = Figure(size = (720, 360))
ax2  = Axis(fig2[1, 1];
            title  = "Mean wavenumber at x = $(round(x_nodes[i_obs]; digits=1)) m",
            xlabel = "t (s)",
            ylabel = "κ (rad/m)")
lines!(ax2, times, κ_obs;   label = "model")
lines!(ax2, times, κ_theory; label = "stationary phase: g t² / (4 D²)", linestyle = :dash)
hlines!(ax2, [kappas[1], kappas[end]]; color = :gray, linestyle = :dot)
axislegend(ax2; position = :rb)
fig2
