# # Narrow-Band Packet Dispersion
#
# The "hello world" of Ripple's narrow-band amplitude model
# (`NarrowBandWaveModel`, Onuki & Fujiwara 2026). Unlike the spectral action
# model, here we evolve a complex wave *amplitude* ``A(x, y, t)`` whose carrier
# ``e^{i\kappa x}`` lives in space; only the fast time oscillation
# ``e^{-i\omega t}`` is factored out. A compact Gaussian packet built on the
# carrier therefore translates at the group velocity ``\omega_\kappa`` and
# spreads dispersively — with the spreading rate set by the curvature
# ``\omega_{\kappa\kappa}`` of the reconstituted dispersion relation.
#
# The model prognoses the reconstituted amplitude ``G = [1 + \alpha(\nabla_h^2 +
# \kappa^2)]A`` and diagnoses ``A`` each stage by a screened-Poisson solve, so
# the dispersion enters entirely through that solve and is *bounded* — plain
# RK3 is stable at ``\Delta t \sim 1/\omega``.

using Oceananigans, Ripple
using CairoMakie
CairoMakie.activate!(type = "png")

# ## Grid and carrier
#
# Periodic in ``x`` and ``y`` (required by the FFT-based amplitude solve), with
# a deep vertical extent so the carrier is deep-water. The physics is
# one-dimensional, so ``N_y`` is small.

Nx = 192
Ny = 8
Lx = 100.0

grid = RectilinearGrid(CPU();
                       size     = (Nx, Ny, 4),
                       halo     = (3, 3, 3),
                       x        = (0, Lx),
                       y        = (0, Lx * Ny / Nx),
                       z        = (-400, 0),
                       topology = (Periodic, Periodic, Bounded))

κ = 1.0                                   # carrier wavenumber (wavelength 2π ≈ 6.3 m)
model = NarrowBandWaveModel(grid; κ, depth = InfiniteDepth())

ω_κ  = model.dispersion.ω_κ               # group velocity
ω_κκ = model.dispersion.ω_κκ              # dispersion curvature (sets spreading)

# ## Initial packet
#
# A Gaussian envelope of width ``\sigma`` riding the carrier ``e^{i\kappa x}``.
# The envelope is a few wavelengths wide, so the packet is narrow-band.

x₀ = 25.0
σ  = 8.0
packet(x, y) = exp(-(x - x₀)^2 / (2σ^2)) * cis(κ * x)
set!(model; A = packet)

# ## Time stepping
#
# Record the amplitude envelope ``|A|(x)`` along a row as the packet moves.

x_nodes = collect(xnodes(grid))
envelope() = abs.(amplitude(model)[:, 1, 1])

Δt         = 0.15 / model.dispersion.ω
step_count = 320
sample     = 4

times     = [model.clock.time]
envelopes = [envelope()]
for step in 1:step_count
    time_step!(model, Δt)
    if step == step_count || step % sample == 0
        push!(times, model.clock.time)
        push!(envelopes, envelope())
    end
end

# Measured envelope centroid speed vs the analytic group velocity ``\omega_\kappa``.

centroid(env) = sum(x_nodes .* env) / sum(env)
c₀ = centroid(first(envelopes))
c₁ = centroid(last(envelopes))
measured_group_speed = (c₁ - c₀) / (times[end] - times[1])
@info "group velocity" analytic = ω_κ measured = measured_group_speed

# ## Hovmöller of the envelope
#
# The packet centroid tracks the dashed group-velocity ray ``x_0 +
# \omega_\kappa t`` while the envelope broadens — dispersive spreading set by
# ``\omega_{\kappa\kappa} = `` $(round(ω_κκ; digits = 3)).

hovmoller = reduce(hcat, envelopes)

fig1 = Figure(size = (760, 380))
ax1  = Axis(fig1[1, 1]; title = "|A|(x, t)", xlabel = "x (m)", ylabel = "t (s)")
hm1  = heatmap!(ax1, x_nodes, times, hovmoller; colormap = :magma)
lines!(ax1, x₀ .+ ω_κ .* times, times; color = :cyan, linestyle = :dash,
       label = "group-velocity ray x₀ + ω_κ t")
axislegend(ax1; position = :rb, framevisible = false, labelcolor = :white)
Colorbar(fig1[1, 2], hm1)
fig1

# ## What reconstitution buys
#
# The reconstitution parameter ``\alpha`` repairs the narrow-band Taylor
# expansion so the model's frequency offset ``s(k) = \omega(k) - \omega(\kappa)``
# tracks the exact deep-water dispersion to third order in ``k - \kappa``. The
# bare (``\alpha = 0``) quadratic truncation drifts away much sooner.

d = model.dispersion
ks      = range(0.6κ, 1.4κ; length = 121)
s_exact = [sqrt(d.gravity * k) - sqrt(d.gravity * κ) for k in ks]
s_recon = [(d.ω_κ / 2κ) * (k^2 - κ^2) / (1 + d.α * (κ^2 - k^2)) for k in ks]
s_bare  = [(d.ω_κ / 2κ) * (k^2 - κ^2) for k in ks]

fig2 = Figure(size = (720, 360))
ax2  = Axis(fig2[1, 1]; title = "Frequency offset s(k) = ω(k) − ω(κ)",
            xlabel = "k / κ", ylabel = "s (rad/s)")
lines!(ax2, ks ./ κ, s_exact; label = "exact ω(k) − ω(κ)", linewidth = 3)
lines!(ax2, ks ./ κ, s_recon; label = "reconstituted (α ≠ 0)", linestyle = :dash)
lines!(ax2, ks ./ κ, s_bare;  label = "bare Taylor (α = 0)", linestyle = :dot)
vlines!(ax2, [1.0]; color = :gray, linestyle = :dot)
axislegend(ax2; position = :lt, framevisible = false)
fig2

# ## Animation of the moving packet

fig3 = Figure(size = (720, 320))
ax3  = Axis(fig3[1, 1]; title = "Amplitude envelope |A|(x, t)",
            xlabel = "x (m)", ylabel = "|A|")
env_obs = Observable(first(envelopes))
lines!(ax3, x_nodes, env_obs; color = :dodgerblue)
ylims!(ax3, 0, 1.05 * maximum(maximum.(envelopes)))

record(fig3, "narrow_band_packet.mp4", eachindex(envelopes); framerate = 12) do idx
    env_obs[] = envelopes[idx]
end
nothing #hide

# ![](narrow_band_packet.mp4)
