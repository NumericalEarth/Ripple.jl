# # Narrow-Band Wave Scattering by a Barotropic Vortex
#
# A plane wave of carrier wavenumber ``\kappa`` enters a prescribed, barotropic
# Gaussian vortex. Because `NarrowBandWaveModel` resolves the carrier in space
# (no WKB scale separation), it captures current-induced **refraction and
# multidirectional scattering**: the amplitude ``A`` develops a focusing /
# defocusing pattern in the wake of the vortex, driven by the Doppler transport
# ``-\nabla\cdot(\mathbf{v}^{\mathrm{eff}} A)`` and the refraction operator
# ``R``. This is the flagship non-WKB example — the regime where ray theory
# (Ripple's `SpectralWaveModel`) breaks down because the current varies on the
# wavelength scale.

using Oceananigans, Ripple
using CairoMakie
CairoMakie.activate!(type = "png")

# ## Grid and prescribed vortex
#
# A periodic box with a vertically resolved (deep) column so the projection has
# a current profile to integrate. The vortex is barotropic (``z``-independent)
# and divergence-free.

Nx = 192
L  = 200.0

grid = RectilinearGrid(CPU();
                       size     = (Nx, Nx, 8),
                       halo     = (3, 3, 3),
                       x        = (0, L),
                       y        = (0, L),
                       z        = (-100, 0),
                       topology = (Periodic, Periodic, Bounded))

Γ  = 0.6                          # peak swirl speed (m/s)
x₀ = y₀ = L / 2
r₀ = 18.0                          # vortex core radius (m)
gaussian(x, y) = exp(-((x - x₀)^2 + (y - y₀)^2) / (2r₀^2))
u_vortex(x, y, z) = -Γ * (y - y₀) / r₀ * gaussian(x, y)
v_vortex(x, y, z) =  Γ * (x - x₀) / r₀ * gaussian(x, y)

# ## Model
#
# Carrier wavelength ``2\pi/\kappa \approx 10`` m — several times smaller than
# the vortex, so scale separation fails and ``R`` matters. `WENO` transport of
# the carrier; the current is passed as `velocities`.

κ = 0.6
model = NarrowBandWaveModel(grid; κ,
                            velocities = (u = u_vortex, v = v_vortex),
                            advection  = WENO())

# Incident rightward plane wave.
set!(model; A = (x, y) -> cis(κ * x))

# ## Time stepping
#
# Record the wave-height envelope ``|A|`` as the scattered pattern develops.

envelope() = abs.(amplitude(model)[:, :, 1])

Δt         = 0.1 / model.dispersion.ω
step_count = 280
sample     = 8

times  = [model.clock.time]
frames = [envelope()]
for step in 1:step_count
    time_step!(model, Δt)
    if step == step_count || step % sample == 0
        push!(times, model.clock.time)
        push!(frames, envelope())
    end
end

x_nodes = collect(xnodes(grid))
y_nodes = collect(ynodes(grid))

focusing = maximum(last(frames)) / maximum(first(frames))
@info "wave-height focusing" ratio = focusing

# ## Scattered wave-height field
#
# The steady envelope shows the focusing lobe downstream of the vortex and the
# defocused shadow to the sides — the signature of current-induced refraction.

fig1 = Figure(size = (620, 540))
ax1  = Axis(fig1[1, 1]; title = "Wave-height envelope |A| (final)",
            xlabel = "x (m)", ylabel = "y (m)", aspect = 1)
hm1  = heatmap!(ax1, x_nodes, y_nodes, last(frames); colormap = :magma)
Colorbar(fig1[1, 2], hm1)
fig1

# ## Animation of the scattering

fig2 = Figure(size = (620, 540))
ax2  = Axis(fig2[1, 1]; title = "Narrow-band scattering by a vortex",
            xlabel = "x (m)", ylabel = "y (m)", aspect = 1)
frame_obs = Observable(first(frames))
hm2 = heatmap!(ax2, x_nodes, y_nodes, frame_obs;
               colormap = :magma, colorrange = (0, maximum(maximum.(frames))))
Colorbar(fig2[1, 2], hm2)

record(fig2, "narrow_band_vortex_scattering.mp4", eachindex(frames); framerate = 12) do idx
    frame_obs[] = frames[idx]
end
nothing #hide

# ![](narrow_band_vortex_scattering.mp4)
