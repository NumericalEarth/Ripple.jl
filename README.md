# Ripple.jl

[![CI](https://github.com/NumericalEarth/Ripple.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/NumericalEarth/Ripple.jl/actions/workflows/ci.yml)
[![Docs Build](https://github.com/NumericalEarth/Ripple.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/NumericalEarth/Ripple.jl/actions/workflows/documentation.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://NumericalEarth.github.io/Ripple.jl/dev/)

Ripple.jl is an Oceananigans-style spectral wave-action model prototype based on
the consistent wave-current coupled model derived by [Vanneste and Young (2026)](https://arxiv.org/abs/2602.21976).
It stores wave action on a product space of three-dimensional physical
`RectilinearGrid` cells and two-dimensional spectral coordinates, with logical
indexing `N[i, j, m, n]` for horizontally varying wave action.

Ripple uses Oceananigans tracer advection schemes for horizontal physical
transport of each spectral bin. Ripple intentionally does not define its own
advection schemes, simulation driver, or output writers; absent transport is
represented as `advection=nothing`, and native simulation/output workflows
belong to Oceananigans.

## Quick Start

Set up an initially uniform, narrow-banded wave-action field and refract it
through a barotropic Gaussian vortex. The vortex velocity is the
Lagrangian-mean current `uᴸ`; Ripple's fused refraction kernel applies
Doppler-shifted physical transport at `c_g + uᴸ` together with kinematic
spectral refraction `∇_k·(c_k N)` in a single pass, driven by the model's
SSP-RK3 time-stepper through `Oceananigans.Simulation`.

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

https://github.com/user-attachments/assets/fe1716be-50c4-475b-8662-d736fb57301c

See `examples/vortex_refraction.jl` (a literate tutorial that produces a
multi-panel movie of `m₀`, `κᵣₘₛ`, and the mean direction) for the full
visualization.

## Design Shape

- `RectilinearGrid` is the physical grid, including `z` and `z_faces`.
- `ProductField` stores horizontally varying wave-action fields over physical
  space x spectral coordinate space without flattening the spectrum.
- Spectral integrals treat entries as finite-volume cell averages and multiply
  by exact spectral cell measures, not point-sample quadrature weights.
- Physical transport is computed with Oceananigans tracer advection operators.
  The default `horizontal_advection=WENO()` matches Oceananigans' fifth-order
  WENO; other schemes like `Centered()`, `UpwindBiased()`, and
  `FluxFormAdvection(...)` can be passed instead. Pass
  `horizontal_advection=nothing` to disable transport (e.g. in source-only
  column tests).
- Spectral kinematic refraction is controlled by `spectral_advection`, which
  defaults to `WENO()`. Pass `spectral_advection=nothing` to disable
  refraction.
- Absent optional model components follow Oceananigans/Breeze-style `nothing`
  semantics: `sources=nothing` and `coupling=nothing` opt out cleanly.
- Spectral grids use unicode coordinates: `κ` for radial wavenumber and `φ` for
  the directional/azimuthal angle (chosen so the symbol does not clash with
  Breeze's `θ` for potential temperature). Wave grids match Oceananigans
  signatures, e.g. `PolarWaveVectorGrid(CPU(), Float64; κ=..., φ=...)`.
- `QTransform` takes the physical `RectilinearGrid` directly and uses its
  vertical faces for perfect finite-volume integration.
- Oceananigans is a hard dependency. CairoMakie is used by the literate
  examples for plots and MP4 movies, while CUDA- and Metal-backed storage are
  exercised through Oceananigans' native GPU architecture in optional runtime
  smoke paths.

## Examples

The `examples/` directory is a literate tutorial sequence included in the
documentation. Every checked-in example writes at least one CairoMakie plot and
one CairoMakie-recorded MP4 animation.

- `examples/product_field_basics.jl`
- `examples/source_only_fetch_limited_growth.jl`
- `examples/bounded_wave_packet_dispersion.jl`
- `examples/hasselmann_inertial_oscillation.jl`
- `examples/cwcm_q_transform_sheared_current.jl`
- `examples/frequency_direction_source_package.jl`
- `examples/exact_finite_volume_source_rates.jl`
- `examples/vortex_refraction.jl`

Run the smoke harness with:

```bash
julia --startup-file=no --project=. test/examples_smoke/run_examples.jl
```

## Tests

In restricted environments, use a writable depot:

```bash
JULIA_DEPOT_PATH=/private/tmp/ripple-julia-depot \
/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia \
  --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'

JULIA_DEPOT_PATH=/private/tmp/ripple-julia-depot \
/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia \
  --startup-file=no --project=. scripts/validation/patch_oceananigans_manifest_triggers.jl .

JULIA_DEPOT_PATH=/private/tmp/ripple-julia-depot \
/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia \
  --startup-file=no --project=. test/runtests.jl
```

Set `RIPPLE_TEST_SUMMARY=default_suite.tsv` on the same command to write a
machine-readable pass-count summary.

## Documentation

Documentation is built with Documenter.jl:

```bash
julia --startup-file=no --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --startup-file=no --project=docs docs/make.jl
```

The documentation workflow deploys to
`https://NumericalEarth.github.io/Ripple.jl/dev/` after the repository is
published at `NumericalEarth/Ripple.jl`.

## Publication Wiring

Repository hygiene is tracked in `CONTRIBUTING.md` and these validation
scripts:

- `scripts/validation/check_publication_readiness.jl`
- `scripts/validation/write_goal_completion_checklist.jl`
- `scripts/validation/create_publication_bundle.jl`
- `scripts/validation/test_publication_bundle.jl`
- `scripts/validation/patch_oceananigans_manifest_triggers.jl`
- `scripts/validation/publish_to_numericalearth.jl`

## Optional Runtime Smokes

Optional gates cover Oceananigans hard-dependency grid/field integration, CUDA
and Metal storage through Oceananigans' GPU architecture, and external
wave-model comparison harnesses:

- `scripts/validation/check_optional_runtime_gates.jl`
- `scripts/validation/run_available_optional_gates.jl`
- `scripts/oceananigans/run_oceananigans_smoke.jl`
- `scripts/gpu/run_cuda_smoke.jl`
- `scripts/gpu/run_metal_smoke.jl`
- `scripts/external_models/run_swan_fetch_limited.jl`
- `scripts/external_models/run_wam_fetch_limited.jl`
- `scripts/external_models/run_ww3_fetch_limited.jl`
- `scripts/external_models/run_ecwam_fetch_limited.jl`
- `scripts/external_models/run_picles_vortex_wind.jl`

## Status

Ripple.jl is not yet a production wave model. The default suite verifies the
source, transport, and coupling core; transport uses Oceananigans advection
machinery rather than private Ripple schemes.
