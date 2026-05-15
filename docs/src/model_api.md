# Model API

Ripple depends directly on Oceananigans and follows Oceananigans and Breeze
conventions for optional model components: absent advection, sources, and
coupling are represented by `nothing`.

## Semantics Contract

- `SpectralWaveModel(grid, spectral_grid; ...)` takes both grids as positional
  arguments, matching Oceananigans's convention.
- `horizontal_advection=nothing` means no physical transport is applied.
- `horizontal_advection=Centered()`, `horizontal_advection=UpwindBiased()`,
  `horizontal_advection=WENO()` (default), or
  `horizontal_advection=FluxFormAdvection(...)` uses Oceananigans tracer
  advection for horizontal transport of every spectral bin.
- `spectral_advection=nothing` disables kinematic refraction.
- `spectral_advection=WENO()` (default) enables the fused refraction kernel
  when the coupling is a `CWCMPrescribedCurrentCoupling`. When both CWCM
  coupling and `spectral_advection` are set, the fused kernel handles physical
  transport too, and `horizontal_advection` is ignored.
- `sources=nothing` means no source tendency is applied.
- `coupling=nothing` means no current-coupling update is applied.
- `SourceTermSet()`, `NoSource()`, and `NoCurrentCoupling()` are compatibility
  inputs and normalize to `nothing`.
- With all optional dynamics absent, `time_step!` advances the clock and leaves
  the action field unchanged.
- The CFL diagnostic is zero when `horizontal_advection=nothing` and uses the
  active transport velocities otherwise.
- `advection=` is a convenience shortcut that sets both `horizontal_advection`
  and `spectral_advection` to the same scheme.

Ripple no longer provides `HamiltonianFiniteVolume`, Hamiltonian velocity
operators, a `Simulation` type, or diagnostic/output writer types. Transport
is routed through Oceananigans advection machinery.

## Product Fields

`ProductField` stores data over horizontal physical space and coordinate space
without flattening the spectrum. A wave-action field is indexed as
`N[i, j, m, n]`, where `i, j` address physical cells and `m, n` address
spectral cells.

Primary constructors and helpers:

- `RectilinearGrid`
- `WaveActionField(grid, spectral_grid)`
- `ProductField`
- `physical_grid(field)`
- `coordinate_grid(field)`
- `product_grid(field)`

## Physical Grid

`RectilinearGrid` carries `x`, `y`, and `z` coordinates. The vertical coordinate
is part of the physical grid and is used directly by CWCM Q transforms:

```julia
grid = RectilinearGrid(CPU();
                       size=(8, 4, 16),
                       x=(0, 8),
                       y=(0, 4),
                       z=(-1, 0))
qtransform = QTransform(QKernel(Float64), grid)
```

## Spectral Grids

Available spectral grids:

- `CartesianWaveVectorGrid`
- `PolarWaveVectorGrid`
- `FrequencyDirectionGrid`

Spectral integrals use exact finite-volume cell measures through
`spectral_cell_measure`, `spectral_cell_measures`, and `integrate_spectrum`.

## Coupling

CWCM coupling uses matrix-free `QTransform` operators based on the physical
grid's vertical faces. Use `QKernel`, `QTransform`,
`CWCMPrescribedCurrentCoupling`, `compute_doppler_velocity!`, and
pseudomomentum helpers to connect wave action to current-coupling diagnostics.
