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
Pages = ["notation.md", "theory.md", "model_api.md", "finite_volume_integration.md", "api_reference.md", "examples.md"]
Depth = 2
```

## Where to start

The shortest end-to-end Ripple simulation is laid out as a runnable
literate page: [Quick Start](@ref). It refracts a narrow-banded
wave-action field through a barotropic Gaussian vortex via the fused
Doppler-plus-refraction kernel, advanced with SSP-RK3. The example uses
the major Ripple constructs (`RectilinearGrid`, `PolarWaveVectorGrid`,
`SpectralWaveModel`, the `velocities` kwarg, `Simulation`, and the
`m0` / `root_mean_square_wavenumber` / `mean_direction` diagnostics) in
about fifty lines.

For a longer tour at production resolution with an animated three-panel
movie of `m₀`, `κᵣₘₛ`, and the mean direction, follow on to
[Wave Refraction Through A Barotropic Vortex](@ref).

For notation, continuum equations, and the numerical implementation behind the
model, start with [Notation](@ref) and [Theory](@ref).
