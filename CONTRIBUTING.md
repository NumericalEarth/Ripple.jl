# Contributing To Ripple.jl

Ripple.jl is validation-first. Changes should keep implementation, examples,
documentation, and audit artifacts aligned.

## Local Checks

Run the dependency-free suite before opening a pull request:

```bash
julia --startup-file=no --project=. test/runtests.jl
```

To write the audit summary consumed by CI and readiness scripts:

```bash
RIPPLE_TEST_SUMMARY=default_suite.tsv julia --startup-file=no --project=. test/runtests.jl
```

When publication wiring changes, also run:

```bash
julia --startup-file=no --project=. scripts/validation/check_publication_readiness.jl publication_readiness.tsv
julia --startup-file=no --project=. scripts/validation/write_goal_completion_checklist.jl goal_completion_checklist.tsv --default-suite-summary default_suite.tsv
```

## Design Rules

- Treat spectral values as finite-volume cell averages. Do not replace exact
  spectral cell integration with midpoint quadrature.
- Use Oceananigans/Breeze-style absent-component semantics:
  `horizontal_advection=nothing`, `spectral_advection=nothing`,
  `sources=nothing`, and `coupling=nothing`.
- Keep optional runtimes behind weak dependencies, extension files, or smoke
  scripts. The default suite must remain self-contained.
- Add or update validation cases when behavior changes.
- Keep examples runnable in `RIPPLE_EXAMPLE_MODE=small`.

## Optional Gates

Oceananigans, CUDA, Metal, and external model comparisons are validated by
optional gates on provisioned machines:

```bash
julia --startup-file=no --project=. scripts/validation/run_available_optional_gates.jl --require-all optional_gate_outputs
```

The manual `Optional Runtime Gates` workflow collects the corresponding
readiness, publication, and goal-completion audit artifacts.
