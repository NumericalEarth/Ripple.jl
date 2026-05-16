# # Bounded Wave Packet Dispersion
#
# A one-dimensional bounded physical domain is the simplest place to see
# transport. We initialize a compact packet near the left boundary with a
# finite-width wavenumber spectrum. All waves travel in the positive ``x``
# direction, but long waves have larger deep-water group velocity than
# short waves, so the packet spreads as it moves across the domain.

using Oceananigans, Ripple
using CairoMakie
CairoMakie.activate!(type = "png")

# ## Grid setup
#
# Bounded in ``x`` and ``z``, periodic in ``y``. The default `WENO(order=5)`
# advection needs `halo=(3, 3, 1)`.

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

# Spectral grid: a thin wedge of 18 ``\kappa`` bins, all pointing in ``+x``
# (single-bin ``\varphi``).

kappas        = range(0.35, 1.25; length = Nk)
theta_width   = pi / 18
spectral_grid = PolarWaveVectorGrid(Float64;
                                    κ       = kappas,
                                    φ       = [0.0],
                                    φ_faces = [-theta_width / 2, theta_width / 2])

# ## Model and initial packet

model = SpectralWaveModel(grid, spectral_grid;
                          horizontal_advection = WENO(order = 5),
                          timestepper          = :RK3);

packet_left  = 24.0
packet_right = 64.0
set!(model, N = (x, y, kx, ky) -> packet_left <= x < packet_right ? 1.0 : 0.0);

# ## Time stepping
#
# 360 RK3 steps of ``\Delta t = 0.25\,\mathrm{s}``. Snapshots are stored
# along the way for a space-time (Hovmöller) plot of ``m_0`` and the
# 2-D ``x``-``\kappa`` action density.

x_nodes = collect(xnodes(grid))

m0_profile()    = vec(interior(m0(model.action))[:, 1, 1])
x_kappa_frame() = [model.action[i, 1, m, 1] for i in 1:Nx, m in 1:Nk]

dt              = 0.25
step_count      = 120
sample_interval = max(1, step_count ÷ 11)

times        = [model.clock.time]
profiles     = [m0_profile()]
phase_frames = [x_kappa_frame()]

for step in 1:step_count
    time_step!(model, dt)
    if step == step_count || step % sample_interval == 0
        push!(times,        model.clock.time)
        push!(profiles,     m0_profile())
        push!(phase_frames, x_kappa_frame())
    end
end

# ## Hovmöller of m₀

hovmoller = reduce(vcat, transpose.(profiles))

fig1 = Figure(size = (720, 360))
ax1  = Axis(fig1[1, 1]; title  = "Bounded-domain wave-packet transport",
                         xlabel = "x cell",
                         ylabel = "time sample")
hm1  = heatmap!(ax1, hovmoller; colormap = :viridis)
Colorbar(fig1[1, 2], hm1)
fig1

# ## Final 2-D ``x``-``\kappa`` density

fig2 = Figure(size = (720, 360))
ax2  = Axis(fig2[1, 1]; title  = "Final x-κ action density",
                         xlabel = "x cell",
                         ylabel = "κ bin")
hm2  = heatmap!(ax2, last(phase_frames); colormap = :viridis)
Colorbar(fig2[1, 2], hm2)
fig2

# ## Frequency-dependent dispersion animation
#
# The packet spreads in ``\kappa`` even as it advects in ``x``: the
# fastest (largest ``\kappa`` for deep water → no, smallest ``\kappa``)
# group-velocity component pulls ahead.

fig3 = Figure(size = (720, 360))
ax3  = Axis(fig3[1, 1]; title  = "Frequency-dependent wave-packet transport",
                         xlabel = "x cell",
                         ylabel = "κ bin")
frame_obs = Observable(first(phase_frames))
hm3 = heatmap!(ax3, frame_obs;
               colormap   = :viridis,
               colorrange = (0, maximum(maximum.(phase_frames))))
Colorbar(fig3[1, 2], hm3)

record(fig3, "x_kappa_packet_dispersion.mp4", eachindex(phase_frames); framerate = 5) do idx
    frame_obs[] = phase_frames[idx]
end
nothing #hide

# ![](x_kappa_packet_dispersion.mp4)
