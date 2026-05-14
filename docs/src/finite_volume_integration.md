# Finite-Volume Integration

Ripple treats spectral values as finite-volume cell averages. A value stored at
`N[i, j, m, n]` represents the average over physical cell `(i, j)` and
spectral control volume `(m, n)`.

Spectral integrals are therefore sums of cell averages multiplied by exact cell
measures:

```julia
integral = integrate_spectrum(cell_average_values, spectral_grid)
```

This is not a point-sample quadrature rule. For a scalar quantity `q` over a
spectral cell `C[m, n]`, the value supplied to `integrate_spectrum` should be
the cell average

```text
qbar[m, n] = (1 / measure(C[m, n])) * integral_C q dC
```

and the integral uses

```text
sum(qbar[m, n] * measure(C[m, n]))
```

## Spectral Measures

`spectral_cell_measure` and `spectral_cell_measures` provide the exact
coordinate-space control-volume measures for Cartesian, polar, and
frequency-direction spectral grids. For polar and frequency-direction grids,
the measures include the radial Jacobian.

Power-law averages use analytic finite-volume formulas:

- `spectral_radial_power_average`
- `spectral_frequency_power_average`
- `spectral_first_moment_measures`
- `spectral_second_moment_measures`

These helpers are used by diagnostics and source terms so moments, energy
diagnostics, source rates, and dissipation rates are consistent with the same
cell geometry.

Directional wind-input and swell-dissipation spreading on polar and
frequency-direction grids also uses angular finite-volume cell averages for
nonnegative integer spreading powers. This avoids replacing a directional bin
with its center angle when a wind direction cuts through the cell.

## Q Transform Geometry

CWCM `QTransform` operators use exact vertical finite-volume cell integrals
from CDF differences. Doppler velocity and pseudomomentum paths share the same
cell geometry, which keeps the discrete wave-current coupling diagnostics
consistent.

## Practical Rule

When constructing analytic initial conditions or source rates, either provide a
quantity that is constant over each spectral cell or use the finite-volume
average helpers. Avoid replacing cell averages with midpoint samples on coarse
spectral grids.
