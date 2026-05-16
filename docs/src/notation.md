# Notation

This page fixes the symbols used in Ripple's theory notes, examples, and code.
The package uses Unicode for spectral coordinates where that keeps formulas and
source code close, but public Julia API names stay descriptive when a symbol
would be ambiguous.

## Coordinates

Ripple's physical horizontal coordinates are
``\boldsymbol{x} = (x, y)``. The vertical coordinate is ``z``, with ``z = 0``
at the mean free surface and ``z < 0`` below it. Water depth is ``d > 0``, so a
flat bottom is at ``z = -d``.

Wavevector coordinates are either Cartesian,
``\boldsymbol{k} = (k_x, k_y)``, or polar,
``(\kappa, \phi)``. Ripple uses ``\kappa`` for radial wavenumber and ``\phi``
for direction. The symbol ``\theta`` is intentionally avoided because Breeze
uses ``\theta`` for potential temperature.

```math
\kappa = \lVert \boldsymbol{k} \rVert,
\qquad
\boldsymbol{k}
= \kappa \, (\cos \phi, \sin \phi),
\qquad
d^2\boldsymbol{k} = d k_x \, d k_y = \kappa \, d\kappa \, d\phi .
```

The action balance is most compact in canonical ``(x, y, k_x, k_y)``
coordinates. Ripple often discretizes the same spectrum on polar
``(\kappa, \phi)`` cells; polar cell measures include the Jacobian ``\kappa``.

## Symbols

| Math | Code | Meaning |
|:-----|:-----|:--------|
| ``\boldsymbol{x}`` | `x, y` | Horizontal position |
| ``z`` | `z` | Vertical coordinate, positive upward |
| ``d`` | `depth` | Water depth; `InfiniteDepth()` selects deep-water dispersion |
| ``\boldsymbol{k}`` | `kx, ky` | Horizontal wavevector |
| ``\kappa`` | `κ` | Radial wavenumber, ``\lVert \boldsymbol{k} \rVert`` |
| ``\phi`` | `φ` | Wave direction |
| ``N`` | `model.action`, `N` | Wave action density / finite-volume cell average |
| ``G`` | `model.tendencies`, `G` | Tendency of `N`, including transport and sources |
| ``S_N`` | `sources` | Source contribution to the action equation |
| ``\sigma`` | `σ` in text | Intrinsic wave frequency |
| ``\Omega`` | `Ω` in text | Absolute frequency, including Doppler shift |
| ``c_g`` | `cg` | Intrinsic group speed |
| ``\dot{\boldsymbol{x}}`` | transport velocity | Ray velocity in physical space |
| ``\dot{\boldsymbol{k}}`` | refraction velocity | Ray velocity in wavevector space |
| ``\boldsymbol{u}^{L}`` | `u`, `v` velocity fields | Lagrangian-mean horizontal velocity |
| ``\boldsymbol{U}`` | `Ux`, `Uy` caches | Q-projected Doppler velocity for each ``\kappa`` |
| ``Q`` | `QKernel`, `QTransform` | Vertical weighting kernel for wave-current coupling |
| ``\boldsymbol{p}`` | `pseudomomentum_fields` | Wave pseudomomentum |
| ``m_0`` | `m0` | Zeroth spectral moment, total action over the spectrum |
| ``\Delta t`` | `dt`, `Δt` | Time step |
| ``L_s, L_n`` | internal GSE lengths | Tolman spatial-averaging extents |

## Discrete Conventions

`WaveActionField(grid, spectral_grid)` stores cell averages, not point samples.
For a physical cell ``V_{ij}`` and spectral cell ``C_{mn}``,

```math
N_{ijmn}
= \frac{1}{|V_{ij}|\, |C_{mn}|}
  \int_{V_{ij}} \int_{C_{mn}}
  N(\boldsymbol{x}, \boldsymbol{k}, t)
  \, d^2\boldsymbol{k} \, d^2\boldsymbol{x}.
```

Spectral integrals multiply those averages by exact cell measures. For polar
grids, ``|C_{mn}|`` includes the ``\kappa`` Jacobian.

The Q transform is also finite-volume in ``z``: Ripple integrates ``Q`` across
vertical cells rather than sampling it at cell centers. This keeps Doppler
velocity, ``\partial \boldsymbol{U} / \partial \kappa``, and pseudomomentum on
the same vertical geometry.
