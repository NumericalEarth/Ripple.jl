# Examples

The `examples/` directory is a literate tutorial sequence. Every checked-in
example is executable Julia, every example is included in this documentation, and
every example writes at least one CairoMakie plot and one CairoMakie-recorded
MP4 animation.

Run any example from the repository root:

```bash
julia --startup-file=no --project=. examples/hasselmann_inertial_oscillation.jl
```

Use small mode for CI-speed runs:

```bash
RIPPLE_EXAMPLE_MODE=small julia --startup-file=no --project=. examples/hasselmann_inertial_oscillation.jl
```

By default plots and animations are written under
`joinpath(tempdir(), "ripple_example_outputs", example_name)`. Set
`RIPPLE_EXAMPLE_OUTPUT_DIR` to collect them somewhere specific.

The smoke harness runs every checked-in example and verifies finite model state,
validation results, plot files, animation files, and the literate docs manifest:

```bash
julia --startup-file=no --project=. test/examples_smoke/run_examples.jl
```

The tutorial order starts with field construction, then adds source-only
columns, bounded-domain WENO transport, finite-volume source semantics, and
current-coupling projections.

```@contents
Pages = [
    "generated/examples/product_field_basics.md",
    "generated/examples/source_only_fetch_limited_growth.md",
    "generated/examples/bounded_wave_packet_dispersion.md",
    "generated/examples/hasselmann_inertial_oscillation.md",
    "generated/examples/cwcm_q_transform_sheared_current.md",
    "generated/examples/frequency_direction_source_package.md",
    "generated/examples/exact_finite_volume_source_rates.md",
]
Depth = 1
```
