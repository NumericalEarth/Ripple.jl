# Ripple.jl — Agent Rules

## Project Overview

Ripple.jl is a spectral wave-action model that runs on Oceananigans grids and
shares Oceananigans' time-stepping conventions. Wave action `N(x, y, k)` is
stored on a product space of three-dimensional physical `RectilinearGrid`
cells and two-dimensional spectral coordinates (`PolarWaveVectorGrid`,
`FrequencyDirectionGrid`, or `CartesianWaveVectorGrid`). Physical transport
goes through Oceananigans tracer advection schemes; spectral physics
(sources, refraction) is implemented inside Ripple. The package targets
research wave-current coupling and is intended to interoperate with
Oceananigans and Breeze ecosystems.

## Language & Environment

- **Julia 1.10+** | CPU and GPU (via `Oceananigans.Architectures`)
- **Key packages**: Oceananigans.jl, KernelAbstractions.jl, OffsetArrays.jl,
  CairoMakie.jl (examples + docs only)
- **Style**: explicit imports inside `src/`; `using Ripple` in examples and
  tests

## Critical Rules

### Kernel Functions (GPU compatibility)

- Use `@kernel` / `@index` (KernelAbstractions.jl)
- Kernels must be **type-stable** and **allocation-free**
- Use `ifelse` — never short-circuiting `if`/`else` inside kernels
- No error messages, no `Models` or `Field` objects inside kernels
- Mark functions called inside kernels with `@inline`
- **Never loop over grid points outside kernels** — KA kernels are the only
  loop primitive. Plain `for i in 1:Nx` host loops are not GPU-portable.
- Refraction tendency uses a single fused 4D kernel
  (`_wave_current_refraction_tendency!`) operating on `flat_data`; follow
  that pattern when adding similar bulk operations
- RK3 stages, current-gradient computation, and Doppler-shift velocity
  setup are all individual KA kernels by design — keep them that way

### Type Stability & Memory

- All structs concretely typed; rely on Julia's parametric machinery
- `ProductField` is backed by a single contiguous 5D array `flat_data`
  `(x_with_halo, y_with_halo, z_slab, κ, φ)` shared with per-bin Field views.
  **Never** allocate one Field per spectral bin at construction time —
  `physical_field(f, m, n)` rebuilds a Field on demand using a cached
  metadata stencil
- Type annotations are for **dispatch**, not documentation
- Allocation-heavy paths (`compute_wave_current_refraction_tendency!` and
  friends) lazily cache scratch on the coupling struct; reuse the existing
  pattern when adding new tendency work
- **Never hardcode Float64**: no literal `0.0` or `1.0` in kernels or
  constructors. Use `zero(grid)`, `one(grid)`, `zero(eltype(...))`, etc.

### Imports

- Source code: explicit imports
- Examples/docs/tests: rely on `using Ripple, Oceananigans`; do not
  explicitly import exported names

### Docstrings

- Use DocStringExtensions.jl with `$(SIGNATURES)` where applicable
- Prefer `jldoctest` blocks over plain `julia` blocks so examples are
  exercised by docs CI
- Math should use unicode (`κ`, `φ`, `Δt`), not LaTeX-in-comments

### Model Constructors

- `SpectralWaveModel(grid, spectral_grid; ...)` — both grids positional,
  matching Oceananigans's convention. A keyword form
  `SpectralWaveModel(; grid, spectral_grid, ...)` is also accepted.
- The `velocities` kwarg builds the wave-current coupling internally. Accepted
  values: `nothing` (no coupling), `ZeroVelocities()`, `PrescribedVelocities`,
  `PseudomomentumVelocities()`, or a bare NamedTuple `(; u, v)`
- `coupling` and `velocities` are mutually exclusive
- Default `horizontal_advection=WENO()` and `spectral_advection=WENO()` —
  column-style tests/validation opt out with `horizontal_advection=nothing`.
  Bigger physical halos (`halo=(3, 3, 3)`) are required for default WENO5.
- With a `CWCMPrescribedCurrentCoupling` and `spectral_advection !== nothing`,
  `compute_tendencies!` uses the fused refraction kernel that includes
  Doppler-shifted physical transport, so `horizontal_advection` is ignored
  in that path.
- `advection=` is a convenience shortcut that sets both `horizontal_advection`
  and `spectral_advection` to the same scheme.

## Naming Conventions

- **Files**: snake_case matching the type they define
- **Types/Constructors**: PascalCase only for true constructors
- **Functions**: snake_case; mutating functions end with `!`
- **Kernels**: prefix with underscore — `_wave_current_refraction_tendency!`
- **Unicode coordinates**: spectral grids use `κ` (radial wavenumber) and
  `φ` (direction). The `θ` symbol is intentionally avoided because Breeze
  uses it for potential temperature; `φ` is the standard alternate. Drop
  English aliases (`theta=`, `kappa=`); only unicode kwargs are accepted

## Module Structure

```
src/
├── Ripple.jl                   # Main module, exports
├── Architectures.jl            # Re-export Oceananigans CPU/GPU
├── Grids.jl                    # Physical-grid helpers
├── Locations.jl                # Location markers
├── ProductFields/              # Product (physical × spectral) fields
├── CoordinateGrids/            # Polar, frequency-direction, Cartesian
├── Diagnostics/                # m0, mean direction, peak, etc.
├── InitialConditions/          # JONSWAP, Gaussian wave packets
├── Coupling/                   # Q-transform, current coupling, refraction
├── Forcing/                    # Idealized winds, hurricanes
├── Sources/                    # Source-term zoo (wind, whitecapping, ...)
├── Models/                     # SpectralWaveModel, time-step, tendencies
├── OceananigansIntegration.jl  # Hooks into Oceananigans
└── Validation/                 # Validation cases + harness
```

## Common Pitfalls

1. **Type instability** in KA kernels ruins GPU performance
2. **Missing halos for WENO**: with default `advection=WENO()` the physical
   grid needs `halo=(3, 3, 3)`; smaller halos are valid only with
   `advection=nothing`
3. **Action-tendency confusion**: the fused refraction kernel writes the
   *tendency* into a separate `WaveActionField`, not into `model.action`.
   Use `rk3_step!` (or write your own stage update) to apply tendencies
4. **Negative `N`**: WENO5 is not strictly positivity-preserving. The RK3
   stage kernels clamp `N ≥ 0`; preserve that clamp when adding new
   integrators
5. **Scope creep in PRs**: keep changes focused; cleanup goes in a
   separate PR
6. **Hardcoded depth**: water depth comes from the grid via
   `Oceananigans.Grids.column_depthᶜᶜᵃ(i, j, grid)`. Don't pass `depth`
   through the velocity API
7. **`coupling.Ux` vs grid `flat_data`**: the coupling caches are
   `(Nx, Ny, Nκ)` arrays of Doppler velocity per κ ring; do not confuse
   them with the 5D `flat_data` of `ProductField`
8. **Modifying `Project.toml` dependencies**: deps changes have CI and
   downstream consequences. Touch them only when the task requires it,
   and call it out in the PR

## Git Workflow

Feature branches, descriptive commits, update tests and docs with code
changes, check CI before merging. Squash trivial fixup commits.

## Design Principles

- **Dispatch over conditionals**: backend differences live in extensions
  (`ext/`), not in `if` branches in `src/`
- **Use `on_architecture` for data placement** — never manual `Array()` /
  `CuArray()` calls
- **Defaults serve the common case**: `advection=WENO()` is the default
  because most uses care about transport; `velocities=nothing` is the
  default because most simulations don't run a coupled current yet
- **Velocity paradigms are reified as types**, not flags:
  `AbstractLagrangianVelocities` with `ZeroVelocities`,
  `PrescribedVelocities`, `PseudomomentumVelocities`
- **Examples drive design**: every public surface change should leave an
  updated literate example or test

## Agent Behavior

- Prioritize type stability and GPU compatibility
- Follow established patterns in `src/Coupling/wave_current_refraction.jl`
  for new fused tendency kernels
- Add tests for new functionality; update exports in `src/Ripple.jl` when
  adding public API
- Reference the wave-action physics (group velocity, refraction velocity,
  Q-projection) in comments when implementing dynamics
