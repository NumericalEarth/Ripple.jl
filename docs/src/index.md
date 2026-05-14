# Ripple.jl

Ripple.jl is an Oceananigans-style spectral wave-action model prototype. It
stores wave action on a product space of three-dimensional physical
`RectilinearGrid` cells and two-dimensional spectral coordinates using exact
finite-volume spectral cell measures.

Physical transport uses Oceananigans tracer advection schemes for each spectral
bin, while `advection=nothing` means no phase-space transport is applied.
Oceananigans is a hard dependency, and Ripple does not define private
advection schemes, simulation drivers, or output writers.

```@contents
Pages = ["model_api.md", "finite_volume_integration.md", "api_reference.md", "examples.md", "validation.md", "publication.md"]
Depth = 2
```

## Quick Start

```julia
using Ripple

grid = RectilinearGrid(CPU();
                       size=(16, 8, 4),
                       x=(0, 16),
                       y=(0, 8),
                       z=(-1, 0))

spectral_grid = PolarWaveVectorGrid(Float64;
                                    kappa=range(0.3, 1.2; length=6),
                                    theta=range(0, 2pi; length=9)[1:8])

sources = SourceTermSet(
    ExponentialWindInput(rate=0.04, direction=0.0, spreading_power=2),
    WhitecappingDissipation(rate=0.02, saturation_threshold=1.0),
)

model = SpectralWaveModel(; grid,
                            spectral_grid,
                            advection=nothing,
                            sources,
                            timestepper=:SemiImplicitEuler)

set!(model, N=1.0)
time_step!(model, 0.1)

Hs = significant_wave_height(model.action)
total = total_action(model.action)
```
