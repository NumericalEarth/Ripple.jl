# Ripple.jl

Ripple.jl is an Oceananigans-style spectral wave-action model prototype. It
stores wave action on a product space of three-dimensional physical
`RectilinearGrid` cells and two-dimensional spectral coordinates using exact
finite-volume spectral cell measures.

Physical transport uses Oceananigans tracer advection schemes for each
spectral bin (`horizontal_advection`), while kinematic refraction is enabled
via `spectral_advection` for `CWCMPrescribedCurrentCoupling`. Either can be
disabled with `nothing`. Oceananigans is a hard dependency, and Ripple does
not define private advection schemes, simulation drivers, or output writers.

```@contents
Pages = ["model_api.md", "finite_volume_integration.md", "api_reference.md", "examples.md", "validation.md", "publication.md"]
Depth = 2
```

## Quick Start

Set up an initially uniform, narrow-banded wave-action field and refract it
through a barotropic Gaussian vortex. The vortex velocity is the
Lagrangian-mean current `uᴸ`; Ripple's fused refraction kernel applies
Doppler-shifted physical transport at `c_g + uᴸ` together with kinematic
spectral refraction `∇_k·(c_k N)` in a single pass, advanced with SSP-RK3.

```julia
using Oceananigans, Ripple

Nx = Ny = 64
Nz = 16
Lx = Ly = 80.0

grid = RectilinearGrid(CPU();
                       size = (Nx, Ny, Nz),
                       halo = (3, 3, 3),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-1, 0),
                       topology = (Periodic, Periodic, Bounded))

spectral_grid = PolarWaveVectorGrid(; κ = range(0.30, 0.50; length = 4),
                                      φ = range(0, 2pi; length = 17)[1:16])

# Barotropic Gaussian-cored vortex: peak azimuthal speed U₀ at radius a.
xc, yc, a, U0 = Lx/2, Ly/2, Lx/8, 0.4

@inline function vortex_uθ(x, y)
    r = hypot(x - xc, y - yc)
    return r == 0 ? zero(r) : U0 * (r/a) * exp(0.5 - 0.5*(r/a)^2)
end

u = CenterField(grid)
v = CenterField(grid)
set!(u, (x, y, z) -> -vortex_uθ(x, y) * (y - yc) / max(hypot(x - xc, y - yc), eps()))
set!(v, (x, y, z) -> +vortex_uθ(x, y) * (x - xc) / max(hypot(x - xc, y - yc), eps()))

model = SpectralWaveModel(grid, spectral_grid;
                          velocities = (; u, v),
                          horizontal_advection = nothing,  # fused kernel drives transport
                          sources = nothing,
                          timestepper = :RK3)

# Narrow-banded Gaussian initial condition: uniform in (x, y), centred on
# κ₀ = 0.4 and direction φ = 0. set!(::ProductField, fun) tracks the
# spectral grid — for PolarWaveVectorGrid `fun` takes (x, y, κ, φ).
κ0, σκ, σφ = 0.4, 0.05, 0.30
function initial_action(x, y, κ, φ)
    return exp(-((κ - κ0)/σκ)^2 - (sin(φ/2)^2)/σφ^2)
end
set!(model, N = initial_action)

simulation = Simulation(model; Δt = 0.05, stop_iteration = 400)
run!(simulation)

m0_field   = m0(model.action)
κrms_field = root_mean_square_wavenumber(model.action)
mean_dir   = mean_direction(model.action)
```

See `examples/vortex_refraction.jl` for the full literate tutorial that
produces a multi-panel movie of `m₀`, `κᵣₘₛ`, and the mean direction.
