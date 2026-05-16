# Model API

Ripple depends directly on Oceananigans and follows Oceananigans and Breeze
conventions for optional model components: absent advection, sources, and
coupling are represented by `nothing`.

The `SpectralWaveModel` API exposes wave-action transport in physical and
wavevector phase space. This is the same organizing principle used by
[VannesteYoung2026](@citet), who combine Doppler-shifted action transport with
wave pseudomomentum forcing of the current equations.

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
  when the coupling is a CWCM coupling. When both CWCM
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

`RectilinearGrid` carries `x`, `y`, and optionally `z` coordinates. Pure
wave-action runs may use a vertically `Flat` grid. The model-level `depth`
kwarg sets the intrinsic dispersion depth and may be `InfiniteDepth()`, a
positive scalar, a function of horizontal position, or an Oceananigans `Field`
on the horizontal wave grid. Scalars are materialized as Oceananigans
`ConstantField`s and functions are materialized to horizontal `Field`s. Raw
arrays are intentionally not part of the public depth interface.

A CWCM Q transform also needs a resolved vertical coordinate. When
`velocities=(; u, v)` passes Oceananigans `Field`s, Ripple infers the Q grid
from those fields and, if `depth=InfiniteDepth()`, derives the finite Q
projection depth from that grid. Array-valued velocities and
`PseudomomentumVelocities` can either pass an explicit `q_grid` or let Ripple
build one from finite model `depth`:

```julia
wave_grid = RectilinearGrid(CPU();
                            size=(8, 4),
                            x=(0, 8),
                            y=(0, 4),
                            topology=(Periodic, Periodic, Flat))

q_grid = RectilinearGrid(CPU();
                         size=(8, 4, 16),
                         x=(0, 8),
                         y=(0, 4),
                         z=(-1, 0),
                         topology=(Periodic, Periodic, Bounded))

model = SpectralWaveModel(wave_grid, spectral_grid;
                          velocities=PseudomomentumVelocities(),
                          depth=1.0)
```

## Spectral Grids

Available spectral grids:

- `CartesianWaveVectorGrid`
- `PolarWaveVectorGrid`
- `FrequencyDirectionGrid`

Spectral integrals use exact finite-volume cell measures through
`spectral_cell_measure`, `spectral_cell_measures`, and `integrate_spectrum`.

## Coupling

CWCM coupling uses matrix-free `QTransform` operators based on the Q grid's
vertical faces. Use `QKernel`, `QTransform`, `CWCMPrescribedCurrentCoupling`,
`CWCMPseudomomentumCoupling`, `compute_doppler_velocity!`, and pseudomomentum
helpers to connect wave action to current-coupling diagnostics.
The vertical projection used by `QTransform` keeps the Doppler velocity and
pseudomomentum tendencies on the same discrete geometry, matching the
consistency requirement emphasized by [VannesteYoung2026](@citet). The
available inertial-oscillation example follows the wave-driven current problem
posed by [Hasselmann1970](@citet).

`PseudomomentumVelocities(; q_grid=nothing)` builds the Lagrangian-mean
velocity from the wave pseudomomentum itself. If `q_grid` is omitted on a Flat
wave grid, finite model `depth` is required and Ripple chooses a stretched
vertical grid whose top-cell spacing is set by the largest spectral wavenumber.
Ripple precomputes the finite-volume vertical overlap between source and
target wavenumber rings and refreshes the Doppler velocity caches from
`model.action` before each tendency evaluation. Use `depth=InfiniteDepth()` to
keep deep-water intrinsic dispersion while deriving Q-projection depth from a
finite-depth velocity grid.
