# # Spectral Refraction by a Sheared Current
#
# Ripple's wave-action model carries advection both in physical space
# ``\partial_t N + (c_g + u^L) \cdot \nabla N`` *and* in spectral space
# ``\partial_t N + \nabla_k \cdot (c_k N)``. The spectral fluxes
# ``c_\kappa, c_\varphi`` are driven by gradients of the
# Lagrangian-mean current ``u^L``. For a horizontal shear ``\partial u
# / \partial y`` acting on waves whose group velocity points in ``+x``,
# the direction tendency reduces (to leading order in
# ``\partial u / \partial y``) to
#
# ```math
# \frac{\mathrm{d}\varphi}{\mathrm{d}t} = -\cos^2(\varphi)\,\frac{\partial u}{\partial y}.
# ```
#
# This page exercises that prediction in a clean, doubly-periodic
# setting: a uniform initial spectrum peaked at ``\varphi = 0`` is
# advanced by ``c_\varphi`` alone, and the mean direction at each
# ``y`` is compared to the linearized analytic estimate.

using Oceananigans, Ripple
using CairoMakie
CairoMakie.activate!(type = "png")

# ## Grids
#
# Doubly periodic in ``(x, y)``. Two ``z`` cells (model needs a vertical
# discretisation; the dynamics here is two-dimensional). One ``\kappa``
# ring and a well-resolved direction axis.

Nx = 16
Ny = 64
Lx = Ly = 100.0

grid = RectilinearGrid(CPU();
                       size     = (Nx, Ny, 2),
                       halo     = (3, 3, 3),
                       x        = (0, Lx),
                       y        = (0, Ly),
                       z        = (-1, 0),
                       topology = (Periodic, Periodic, Bounded))

Nφ = 64
κ0 = 0.4
spectral_grid = PolarWaveVectorGrid(Float64;
                                    κ = [κ0],
                                    φ = range(0, 2pi; length = Nφ + 1)[1:Nφ])

# ## Sinusoidal-shear current
#
# Choose ``u(y) = A \sin(\omega y)`` with ``\omega = 2\pi / L_y`` so the
# velocity is exactly periodic on the grid. The shear is
# ``\partial u / \partial y = A \omega \cos(\omega y)``, sinusoidal in
# ``y`` and zero in mean.

A = 0.02
ω = 2pi / Ly

u_field = Field{Face,   Center, Center}(grid)
v_field = Field{Center, Face,   Center}(grid)
set!(u_field, (x, y, z) -> A * sin(ω * y))
fill_halo_regions!(u_field)
fill_halo_regions!(v_field);

let
    fig = Figure(size = (640, 300))
    ax  = Axis(fig[1, 1]; title  = "Sheared current u(y)",
                          xlabel = "y (m)", ylabel = "u (m/s)")
    lines!(ax, ynodes(grid), [A * sin(ω * y) for y in ynodes(grid)])
    fig
end

# ## Model and initial spectrum
#
# Action density is uniform in ``(x, y)`` and a narrow Gaussian peaked
# at ``\varphi = 0``. Because ``N`` is spatially uniform and ``c_g
# \cdot \nabla N = 0``, *all* spatial structure that appears later
# comes from spectral refraction.

model = SpectralWaveModel(grid, spectral_grid;
                          velocities  = (; u = u_field, v = v_field),
                          sources     = nothing,
                          timestepper = :RK3);

σφ = 0.15
set!(model, N = (x, y, kx, ky) -> exp(-(atan(ky, kx))^2 / (2 * σφ^2)));

# ## Time integration
#
# 10 s of model time at ``\Delta t = 0.1\,\mathrm{s}`` via SSP-RK3
# through the fused refraction kernel.

dt    = 0.1
steps = 100
sim   = Simulation(model; Δt = dt, stop_iteration = steps, verbose = false)
run!(sim);

T = model.clock.time

# ## Mean direction vs linearized prediction
#
# At each column ``y``, integrate over ``\kappa`` and ``\varphi`` to get
# the mean direction. The linearised prediction is obtained by treating
# ``\mathrm{d}\varphi / \mathrm{d}t \approx -\partial u / \partial y``
# for ``\varphi \ll 1``, so after time ``T``,
#
# ```math
# \overline{\varphi}(y) \approx -T\,A\,\omega\,\cos(\omega y).
# ```

φ̄_model = Array(interior(mean_direction(model.action)))[1, :, 1]
ys      = collect(ynodes(grid))
φ̄_pred  = [-T * A * ω * cos(ω * y) for y in ys]

fig = Figure(size = (720, 360))
ax  = Axis(fig[1, 1];
           title  = "Mean direction at t = $(round(T; digits = 1)) s",
           xlabel = "y (m)",
           ylabel = "φ̄ (rad)")
lines!(ax, ys, φ̄_model; label = "model")
lines!(ax, ys, φ̄_pred;  label = "linearised: −T A ω cos(ω y)", linestyle = :dash)
axislegend(ax; position = :rb)
fig

# The two curves are in phase and within ~10 % in amplitude. The residual
# is intermediate-depth ``Q``-transform damping (``\kappa H = 0.4`` here,
# not the deep-water limit) plus second-order ``\cos^2 \varphi`` terms
# the linearised prediction omits.
