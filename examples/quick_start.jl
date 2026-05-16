# # Quick Start
#
# This page walks through the smallest end-to-end Ripple simulation we run
# to ship a paper-quality figure: wave action refracted through a barotropic
# Gaussian vortex. Wave action lives on the product of a 3-D physical
# `RectilinearGrid` and a 2-D `PolarWaveVectorGrid`. The fused refraction
# kernel applies Doppler-shifted physical transport at ``c_g + u^L``
# together with kinematic spectral refraction ``\nabla_k \cdot (c_k N)`` in
# a single pass, advanced with SSP-RK3 via an `Oceananigans.Simulation`.
#
# This example targets clarity over performance: small grid, small spectrum,
# short trajectory. The
# [Wave Refraction Through A Barotropic Vortex](@ref) page goes through the
# same physics at production resolution and with multi-panel diagnostics.

using Oceananigans, Ripple
using CairoMakie
CairoMakie.activate!(type = "png")

# ## Grids
#
# A 32²×4 periodic physical grid with halo `(3, 3, 3)` (required by the
# default `WENO()` advection that backs each spectral bin) and a spectral
# grid with four ``\kappa`` rings and sixteen direction bins.

Nx = Ny = 32
Nz = 4
Lx = Ly = 80.0
Lz = 1.0

grid = RectilinearGrid(CPU();
                       size     = (Nx, Ny, Nz),
                       halo     = (3, 3, 3),
                       x        = (0, Lx),
                       y        = (0, Ly),
                       z        = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

spectral_grid = PolarWaveVectorGrid(; κ = range(0.30, 0.50; length = 4),
                                      φ = range(0, 2pi; length = 17)[1:16])

# ## Prescribed vortex velocity
#
# Set the velocity from a streamfunction so the flow is exactly
# nondivergent. For a Gaussian-cored vortex with peak azimuthal speed
# ``U_0`` at radius ``a``,
#
# ```math
# \psi(x, y) = U_0\, a\, \sqrt{e} \left( 1 - \exp\!\left( -\frac{(x - x_c)^2 + (y - y_c)^2}{2 a^2} \right) \right),
# ```
#
# whose gradient gives ``u = -\partial_y \psi`` and ``v = +\partial_x \psi``.

xc, yc = Lx / 2, Ly / 2
a      = Lx / 8
U0     = 0.4

ψ_vortex(x, y) = U0 * a * sqrt(exp(1)) *
                 (1 - exp(-((x - xc)^2 + (y - yc)^2) / (2 * a^2)))

ψ = Field{Face, Face, Center}(grid)
set!(ψ, (x, y, z) -> ψ_vortex(x, y))
fill_halo_regions!(ψ)

u_field = Field(-∂y(ψ))   # (Face,   Center, Center)
v_field = Field(+∂x(ψ))   # (Center, Face,   Center)
compute!(u_field)
compute!(v_field);

xs = xnodes(grid)
ys = ynodes(grid)

# ## Model
#
# `velocities = (; u, v)` builds the wave-current coupling internally;
# `physics = nothing` disables all forcing so we're seeing pure
# Doppler-shifted transport plus kinematic refraction. The fused kernel
# handles physical and spectral transport together when both CWCM coupling
# and `spectral_advection` are set (both defaults).

model = SpectralWaveModel(grid, spectral_grid;
                          velocities  = (; u = u_field, v = v_field),
                          physics     = nothing,
                          timestepper = :RK3);

# Narrow-banded Gaussian initial condition, uniform in ``(x, y)``, peaking
# at ``\kappa \approx 0.4`` and direction ``\varphi \approx 0`` (waves
# travelling in ``+x``).

κ0 = 0.4
σκ = 0.05
σφ = 0.30

set!(model, N = (x, y, kx, ky) -> begin
    κ = hypot(kx, ky)
    φ = atan(ky, kx)
    exp(-((κ - κ0) / σκ)^2 - (sin(φ / 2)^2) / σφ^2)
end);

# ## Time stepping
#
# 200 RK3 steps of ``\Delta t = 0.05\,\mathrm{s}``.

simulation = Simulation(model; Δt = 0.05, stop_iteration = 200)
run!(simulation);

# ## Diagnostics
#
# Each call below returns a `Field` whose underlying kernel runs on the
# same architecture as `model.action`. Indexing is 3-D `[i, j, Nz]` for a
# surface slab; `interior(field)[:, :, 1]` materializes a CPU `Matrix`.

m0_field   = m0(model.action)
κrms_field = root_mean_square_wavenumber(model.action)
mean_dir   = mean_direction(model.action)

m0_arr   = Array(interior(m0_field))[:, :, 1]
κrms_arr = Array(interior(κrms_field))[:, :, 1]
dir_arr  = Array(interior(mean_dir))[:, :, 1]

# ## Plot

fig = Figure(size = (1320, 360))
ax1 = Axis(fig[1, 1]; title = "m₀ (total action)",       xlabel = "x", ylabel = "y", aspect = 1)
ax2 = Axis(fig[1, 3]; title = "κᵣₘₛ (rad/m)",            xlabel = "x", ylabel = "y", aspect = 1)
ax3 = Axis(fig[1, 5]; title = "mean direction (rad)",    xlabel = "x", ylabel = "y", aspect = 1)
hm1 = heatmap!(ax1, xs, ys, m0_arr;   colormap = :viridis)
hm2 = heatmap!(ax2, xs, ys, κrms_arr; colormap = :magma)
hm3 = heatmap!(ax3, xs, ys, dir_arr;  colormap = :balance)
Colorbar(fig[1, 2], hm1); Colorbar(fig[1, 4], hm2); Colorbar(fig[1, 6], hm3)
fig
