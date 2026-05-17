# Duration-limited growth — Ripple vs WW3

Single-point, spatially homogeneous test. Constant wind blows over deep water
from rest. The model spectrum grows, the peak downshifts in frequency, and the
significant wave height `Hs(t)` approaches a fully-developed equilibrium.

## What we test

- Ripple `PrecomputedSources` (ST3-equivalent: pressure-correlation input +
  mean-spectrum whitecapping) without nonlinear interactions.
- Compared against:
  1. Analytic PM equilibrium (`Hs_PM = 0.0246 · U10² / g`, ≈ 6.6 m at 17 m/s).
  2. WW3 v6 reference run with `ST4` + `NL1` (DIA) over the same point.

## Running

```bash
# 1. Ripple
julia --project=. validation/duration_limited_ww3/run_ripple.jl

# 2. WW3 (binaries already built at /tmp/ww3-build/build/bin)
bash validation/duration_limited_ww3/run_ww3.sh

# 3. Compare
julia --project=. validation/duration_limited_ww3/compare.jl
```

Outputs land under `validation/duration_limited_ww3/output/`.
