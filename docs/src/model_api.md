# Model API

Ripple depends directly on Oceananigans and follows Oceananigans and Breeze
conventions for optional model components: absent advection, sources, and
coupling are represented by `nothing`.

The `SpectralWaveModel` API exposes wave-action transport in physical and
wavevector phase space. This is the same organizing principle used by
[VannesteYoung2026](@citet), who combine Doppler-shifted action transport with
wave pseudomomentum forcing of the current equations.
See [Notation](@ref) and [Theory](@ref) for the continuum equations, numerical
methods, and how they map to model kwargs.

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

## Monobanded Model

`MonobandedWaveModel(grid; ...)` implements the single-wave-train reduction
with prognostic fields `A`, `AKx`, and `AKy` on the physical grid's top
surface. The diagnosed wavevector is `K = AK / A`, and the diagnostic fields
include `κ`, the Q-projected Doppler velocity `uᴰ`, its `κ` derivative `H`,
the ray velocity `C`, absolute frequency `Ω`, and the refraction tensor `Γ`.
See [Monobanded Wave Model](@ref) for the full equations, constructor contract,
diagnostics, and numerical constraints.

The constructor follows Oceananigans and Breeze model conventions: boundary
conditions are consumed while constructing the prognostic `Field`s and are not
stored as a model-level slot. `velocities=PrescribedVelocities(...)` or a
bare `velocities=(; u, v)` builds the monobanded prescribed-current coupling;
`velocities=PseudomomentumVelocities()` uses the monobanded pseudomomentum as
the Lagrangian velocity; and `velocities=nothing` gives intrinsic deep-water
propagation. A Flat monobanded grid requires an explicit Q grid for
`PseudomomentumVelocities`.
`pseudomomentum_fields(model)` Q-projects the monobanded moments `AKx` and
`AKy` onto the model grid or the prescribed-current Q grid, and its vertical
integral recovers the horizontal pseudomomentum. `MonobandedWaveModel`
currently supports scalar action-only `LinearWindInput`, scalar action-only
`BottomFriction`, and `SourceTermSet` combinations of those. These sources add
`S_A` to `A` and `K S_A` to the moments, preserving local `K` under pure
growth or decay.
`advection=nothing` disables physical transport but leaves refraction and
sources active. The default `advection=WENO()` uses Ripple's monobanded
conservative transport kernel with WENO5 face reconstruction, while other
accepted Oceananigans advection schemes currently fall back to conservative
upwind reconstruction. The monobanded kernels require uniform horizontal
spacing and currently support only default NoFlux/Periodic prognostic boundary
conditions.

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

```@example model_api
using Oceananigans, Ripple

wave_grid = RectilinearGrid(CPU();
                            size=(8, 4),
                            halo=(3, 3),
                            x=(0, 8),
                            y=(0, 4),
                            topology=(Periodic, Periodic, Flat))

q_grid = RectilinearGrid(CPU();
                         size=(8, 4, 16),
                         x=(0, 8),
                         y=(0, 4),
                         z=(-1, 0),
                         topology=(Periodic, Periodic, Bounded))

spectral_grid = PolarWaveVectorGrid(; κ=[0.5], φ=[0.0])

model = SpectralWaveModel(wave_grid, spectral_grid;
                          velocities=PseudomomentumVelocities(; q_grid),
                          depth=1.0,
                          advection=nothing)

model isa SpectralWaveModel
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
`model.action` before each tendency evaluation. In the equations this
Q-projected Doppler velocity is denoted ``\boldsymbol{u}^{D}``; internal cache
names are implementation details. Use `depth=InfiniteDepth()` to keep
deep-water intrinsic dispersion while deriving Q-projection depth from a
finite-depth velocity grid.
