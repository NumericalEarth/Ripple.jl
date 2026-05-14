# Goal Completion Audit

The old completion audit claimed support for private Ripple transport,
simulation, and output-writer subsystems. Those subsystems have now been
removed.

## Completed In This Cut

- Removed bespoke Hamiltonian finite-volume advection code.
- Removed Ripple-owned `Simulation`, callback, diagnostic writer, checkpoint,
  and dataset writer code.
- Removed JLD2/NCDatasets weak-dependency output extensions.
- Moved vertical Q-transform geometry into the physical `RectilinearGrid`.
- Reduced examples to the source-only and coupling tutorial path.
- Added direct `Oceananigans.Simulation` and `JLD2Writer` coverage.
- Added horizontal physical transport through Oceananigans tracer advection
  schemes, with WENO integration coverage.

## Open

- Transport validation still needs current-coupled and non-periodic cases.
- Publication readiness should be re-audited after those optional gates pass.
