# Ripple.jl

[![CI](https://github.com/NumericalEarth/Ripple.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/NumericalEarth/Ripple.jl/actions/workflows/ci.yml)
[![Docs Build](https://github.com/NumericalEarth/Ripple.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/NumericalEarth/Ripple.jl/actions/workflows/documentation.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://NumericalEarth.github.io/Ripple.jl/stable/)

Ripple.jl is an Oceananigans-style spectral wave-action model prototype. It
stores wave action on a product space of three-dimensional physical
`RectilinearGrid` cells and two-dimensional spectral coordinates, with logical
indexing `N[i, j, m, n]` for horizontally varying wave action.

Ripple uses Oceananigans tracer advection schemes for horizontal physical
transport of each spectral bin. Ripple intentionally does not define its own
advection schemes, simulation driver, or output writers; absent transport is
represented as `advection=nothing`, and native simulation/output workflows
belong to Oceananigans.

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

## Design Shape

- `RectilinearGrid` is the physical grid, including `z` and `z_faces`.
- `ProductField` stores horizontally varying wave-action fields over physical
  space x spectral coordinate space without flattening the spectrum.
- Spectral integrals treat entries as finite-volume cell averages and multiply
  by exact spectral cell measures, not point-sample quadrature weights.
- Physical transport is computed with Oceananigans tracer advection operators,
  so schemes like `Centered()`, `UpwindBiased()`, `WENO()`, and
  `FluxFormAdvection(...)` can be passed directly as `advection`.
- Absent optional model components follow Oceananigans/Breeze-style `nothing`
  semantics: `advection=nothing`, `sources=nothing`, and `coupling=nothing`.
- `QTransform` takes the physical `RectilinearGrid` directly and uses its
  vertical faces for perfect finite-volume integration.
- Oceananigans is a hard dependency. CairoMakie is used by the literate
  examples for plots and MP4 movies, while CUDA-backed storage is exercised through
  Oceananigans' native `GPU()` architecture in the optional runtime smoke path.

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
`https://NumericalEarth.github.io/Ripple.jl/stable/` after the repository is
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
storage through Oceananigans' `GPU()` architecture, and external wave-model
comparison harnesses:

- `scripts/validation/check_optional_runtime_gates.jl`
- `scripts/validation/run_available_optional_gates.jl`
- `scripts/oceananigans/run_oceananigans_smoke.jl`
- `scripts/gpu/run_cuda_smoke.jl`
- `scripts/external_models/run_swan_fetch_limited.jl`
- `scripts/external_models/run_wam_fetch_limited.jl`
- `scripts/external_models/run_ww3_fetch_limited.jl`
- `scripts/external_models/run_ecwam_fetch_limited.jl`
- `scripts/external_models/run_picles_vortex_wind.jl`

## Status

Ripple.jl is not yet a production wave model. The default suite verifies the
source, transport, and coupling core; transport uses Oceananigans advection
machinery rather than private Ripple schemes.
