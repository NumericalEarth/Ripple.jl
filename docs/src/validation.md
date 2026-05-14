# Validation

The default test command uses the package environment and does not require
optional runtimes:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. scripts/validation/patch_oceananigans_manifest_triggers.jl .
julia --startup-file=no --project=. test/runtests.jl
```

It covers product fields, coordinate grids, diagnostics, forcing, coupling,
sources, model API semantics, validation cases, and example smoke tests.

Optional runtime gates are reproducible but require extra packages, hardware,
or external model executables:

```bash
export RIPPLE_OPTIONAL_RUNTIME_ENV=/tmp/ripple-optional-runtime
julia --startup-file=no --project="$RIPPLE_OPTIONAL_RUNTIME_ENV" -e 'using Pkg; Pkg.add(["Oceananigans", "GPUArraysCore", "KernelAbstractions", "CUDA"])'
julia --startup-file=no --project=. scripts/validation/patch_oceananigans_manifest_triggers.jl "$RIPPLE_OPTIONAL_RUNTIME_ENV"
export JULIA_LOAD_PATH="@:$RIPPLE_OPTIONAL_RUNTIME_ENV:@stdlib"
julia --startup-file=no --project=. scripts/validation/check_optional_runtime_gates.jl optional_runtime_gates.tsv
julia --startup-file=no --project=. scripts/validation/run_available_optional_gates.jl optional_gate_outputs
julia --startup-file=no --project=. scripts/validation/check_publication_readiness.jl publication_readiness.tsv
julia --startup-file=no --project=. scripts/validation/write_goal_completion_checklist.jl goal_completion_checklist.tsv --default-suite-summary default_suite.tsv
```

The optional gates cover Oceananigans integration, CUDA, and external model harnesses.
JLD2 and NetCDF output are no longer Ripple weak-dependency gates because output
writers should be used directly from Oceananigans.

For publication or full cross-runtime completion, set
`RIPPLE_PUBLICATION_GIT_DIR` when this checkout uses a separate git directory,
then run `scripts/validation/run_available_optional_gates.jl` in a provisioned
environment. The helper `scripts/validation/patch_oceananigans_manifest_triggers.jl`
records `GPUArraysCore` and `KernelAbstractions` trigger packages in the
throwaway optional runtime environment.
