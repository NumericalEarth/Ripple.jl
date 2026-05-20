# Rectangular-wind WW3 comparison

Seven 2-D test cases with prescribed spatially varying wind fields on a
flat, deep-water rectangular domain. Each case drives Ripple with the same
wind forcing used to produce the WW3 v7.14 reference output.

## Test cases

| ID | Name | Domain | dx | Duration |
|----|------|--------|----|----------|
| Test01_2D | Horizontal Static Wind   | 200 × 200 km | 4 km | 24 h |
| Test02_2D | Diagonal Static Wind     | 100 × 100 km | 2 km | 48 h |
| Test03_2D | Halfdomain Static Wind   | 200 × 200 km | 4 km | 24 h |
| Test04_2D | Halfdomain V sinusoidal  | 200 × 200 km | 4 km | 48 h |
| Test05_2D | Halfdomain growing       | 200 × 200 km | 4 km | 24 h |
| Test06_2D | Halfdomain decaying      | 200 × 200 km | 4 km | 24 h |
| Test07_2D | Gaussian blob            | 500 × 500 km | 12.5 km | 240 h |

All cases use U10 ≈ 10 m/s, deep water, cold start from near-zero action.

## What we test

- Ripple `PrecomputedSources` (ST3-equivalent: `PressureCorrelationInput` +
  `LocalSaturationDissipation` + `DiscreteInteractionApproximation`) driven
  by spatially and temporally varying prescribed wind fields.
- Physical-space advection with `WENO` + `LowStorageRK3` time stepping.
- Compared against WW3 v7.14 reference Hs(x, y, t) at hourly output intervals.

## Data

```
data/
├── wind/      — prescribed U10 wind fields (NetCDF + JSON metadata)
└── reference/ — WW3 reference Hs(x,y,t) output (NetCDF) + run logs
```

The wind NetCDF files contain `u10m(x, y, time)` and `v10m(x, y, time)` on
the same 51×51 grid used by WW3. A bilinear-in-space, linear-in-time
interpolator (`gridded_wind.jl`) feeds the wind into `PressureCorrelationInput`.

## Running

```bash
# 1. Run all 7 Ripple cases (writes output/TestXX_2D_ripple.nc)
julia --project=. validation/rectangular_wind_ww3/run_ripple.jl

# 2. Compare and plot (writes output/compare/*.png)
julia --project=. validation/rectangular_wind_ww3/compare.jl
```

Outputs land under `validation/rectangular_wind_ww3/output/`.
