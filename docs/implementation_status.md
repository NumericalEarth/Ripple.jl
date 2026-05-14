# Implementation Status

This document reflects the breaking Oceananigans-alignment refactor.

## Current Core

- `RectilinearGrid` is now a three-dimensional physical grid with `x`, `y`,
  and `z` faces.
- `QTransform` uses the physical grid's vertical faces directly.
- `SpectralWaveModel` defaults to `advection=nothing`, and accepts
  Oceananigans tracer advection schemes for horizontal physical transport.
- Ripple-owned advection schemes, Hamiltonian transport operators,
  `Simulation`, diagnostic writers, dataset writers, and JLD2/NetCDF output
  backends have been removed.
- Oceananigans is a hard dependency. CairoMakie is used by the literate examples
  for plots and MP4 movies.
- `SpectralWaveModel` can be advanced by `Oceananigans.Simulation`, and examples
  use `JLD2Writer` directly for output.
- WENO transport is covered by the integration suite.

## Remaining Design Work

- Expand transport validation beyond periodic deep-water propagation.
- Couple physical transport velocities to CWCM/current fields.
