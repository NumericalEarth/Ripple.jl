# External comparison harness

Ripple.jl can write validation summaries as tab-separated scalar metrics:

```julia
results = run_validation()
write_validation_summary("ripple_validation.tsv", results)
```

The same format can be produced by optional external-model scripts for SWAN,
WAM, WAVEWATCH III, ecWAM, or PiCLES-derived cases. Once a reference file is
available, compare it with Ripple's output:

```julia
comparison = compare_validation_summaries("reference.tsv", "ripple_validation.tsv";
                                          atol = 1e-10,
                                          rtol = 1e-8)

validation_passed(comparison)
```

Required columns:

```text
case    passed    metric    value    tolerance    description
```

Only scalar metrics are compared. Missing candidate metrics fail the comparison.
Extra candidate metrics are ignored so references can remain stable while Ripple
adds additional diagnostics.

## Runner Scripts

The scripts in `scripts/external_models/` wrap external commands that emit
scalar metrics to stdout, or ingest scalar metrics from a file. They do not
require the external model in normal CI. Set the corresponding environment
variable when the model or a post-processing wrapper is available:

```bash
SWAN_METRICS_COMMAND="julia emit_swan_metrics.jl" \
  julia --startup-file=no --project=. scripts/external_models/run_swan_fetch_limited.jl swan.tsv

WAM_METRICS_COMMAND="julia emit_wam_metrics.jl" \
  julia --startup-file=no --project=. scripts/external_models/run_wam_fetch_limited.jl wam.tsv

WW3_METRICS_COMMAND="julia emit_ww3_metrics.jl" \
  julia --startup-file=no --project=. scripts/external_models/run_ww3_fetch_limited.jl ww3.tsv

ECWAM_METRICS_COMMAND="julia emit_ecwam_metrics.jl" \
  julia --startup-file=no --project=. scripts/external_models/run_ecwam_fetch_limited.jl ecwam.tsv

PICLES_METRICS_COMMAND="julia emit_picles_metrics.jl" \
  julia --startup-file=no --project=. scripts/external_models/run_picles_vortex_wind.jl picles.tsv
```

To inspect all optional runtime gates in one pass, run:

```bash
julia --startup-file=no --project=. scripts/validation/check_optional_runtime_gates.jl \
  optional_runtime_gates.tsv
```

To run every optional gate that is available in the current environment, use the
consolidated runner:

```bash
julia --startup-file=no --project=. scripts/validation/run_available_optional_gates.jl \
  optional_gate_outputs
```

The runner writes `optional_runtime_gates.tsv` and
`optional_gate_run_summary.tsv` in the output directory. Missing gates are
recorded as skipped. Gates that run must emit a validation-summary TSV whose
`passed` column is true for every metric; otherwise the runner records the gate
as failed and exits with an error. Use `--dry-run` to verify command selection
without executing available gates. Use `--require-all` when the environment is
intended to close every optional gate and skipped gates should fail the command.

The readiness checker follows the same precedence as the external metric
runner for each model: `*_METRICS_COMMAND`, then `*_METRICS_FILE`, then
`*_BULK_FILE`, then `*_EXECUTABLE`, then common executable names found on
`PATH`. A configured file must exist, and a configured executable must resolve
through `PATH`; an empty or invalid higher-priority variable is reported as a
missing gate because it would shadow lower-priority fallbacks at runtime.

If the external model has already produced a scalar metrics file in the same
format, use the matching `*_METRICS_FILE` variable instead:

```bash
SWAN_METRICS_FILE=swan_metrics.txt \
  julia --startup-file=no --project=. scripts/external_models/run_swan_fetch_limited.jl swan.tsv
```

If the external model produces a columnar bulk-output table, use the matching
`*_BULK_FILE` variable. Ripple will reduce the table to scalar validation
metrics:

```bash
SWAN_BULK_FILE=swan_bulk_output.txt \
  julia --startup-file=no --project=. scripts/external_models/run_swan_fetch_limited.jl swan.tsv
```

The same conversion is available directly:

```bash
julia --startup-file=no --project=. scripts/postprocess/external_bulk_to_metrics.jl \
  swan_bulk_output.txt swan.tsv swan
```

Supported columns include `x`, `y`, `dx`, `dy`, `area`, `Hs`/`Hm0`/`SWH`,
`m0`, `energy_density`, `mean_direction`, `peak_direction`, `mean_period`,
`peak_period`, `mean_frequency`, `peak_frequency`, `peak_wavenumber`,
`peak_phase_speed`, `group_velocity`, and Ripple's
`mean_deep_water_group_speed`, plus common model aliases such as
SWAN `XP`, `YP`, `DIR`, and `RTP`, WAM/WW3/ecWAM `Hsig`, `Tm02`, `MWD`, `Dp`,
and `fp`, and PiCLES-style `cg`. Direction columns are reduced with circular
means in degrees by default, so wraparound at 0/360 is handled correctly.
Tables may be comma-separated or whitespace-separated.

If a model executable is available, the same per-model scripts can generate a
small input deck, run the executable in a working directory, and post-process
the scalar metrics or bulk table it writes. Set `<MODEL>_EXECUTABLE`; optionally
set `<MODEL>_WORKDIR` to keep the generated deck and `<MODEL>_LAUNCH_PROFILE`
to select a command adapter profile:

```bash
SWAN_EXECUTABLE=/path/to/swanrun \
SWAN_WORKDIR=swan_fetch \
SWAN_LAUNCH_PROFILE=swanrun \
  julia --startup-file=no --project=. scripts/external_models/run_swan_fetch_limited.jl swan.tsv
```

The planning API exposes the same path:

```julia
profile = external_model_launch_profile(:swan; profile = :swanrun,
                                         executable = "/path/to/swanrun")

plan = external_model_launch_plan("swan_fetch", :swan;
                                  profile = :swanrun,
                                  executable = "/path/to/swanrun")

run_external_model_launch_plan!(plan, "swan.tsv")
```

Each command should write lines such as:

```text
fetch_limited.hm0_error=0.0
fetch_limited.mean_direction_error	0.01	0.05	external model comparison
```

## Input Deck Templates

Ripple.jl can also generate small external-model input deck templates and a
manifest for the currently supported comparison cases:

```julia
deck = external_model_input_deck(:swan; fetch_length = 100_000.0,
                                 wind_speed = 12.0,
                                 depth = 50.0)

write_external_model_input_deck("swan_fetch", :swan)
write_external_model_input_deck("picles_vortex", :picles)
```

The same path is available from the command line:

```bash
julia --startup-file=no --project=. scripts/external_models/write_external_input_deck.jl \
  swan_fetch swan

julia --startup-file=no --project=. scripts/external_models/write_external_input_deck.jl \
  picles_vortex picles stationary_vortex
```

`swan`, `wam`, `ww3`, and `ecwam` default to a `fetch_limited` case.
`picles` defaults to a `stationary_vortex` case. These templates are meant to
pin down shared case parameters and output file names for external comparison
workflows; model-version-specific executables can be connected through launch
plans, the `<MODEL>_EXECUTABLE` script path, and launch profiles including
`swanrun`, `wam_cycle7`, `ww3_shel`, `ecwam_standalone`, and `picles_julia`.
