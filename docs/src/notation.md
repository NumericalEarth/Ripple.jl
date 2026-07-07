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

## Narrow-band amplitude model (Onuki & Fujiwara 2026)

Ripple is growing a second model type, `NarrowBandWaveModel`, that evolves a
complex wave *amplitude* rather than a wave-action spectrum (see the design plan
in `docs/design/narrow_band_model_plan.md`). It uses its own set of symbols,
fixed here; the `Code` column names the current or planned Julia API, and the
guided derivation will live in the theory documentation.

| Math | Code | Meaning |
|:-----|:-----|:--------|
| ``A(x, y, t)`` | `amplitude(model)` | Complex wave amplitude; the carrier ``e^{i\boldsymbol{k}\cdot\boldsymbol{x}}`` lives *inside* ``A`` (only ``e^{-i\omega t}`` is factored out) |
| ``G`` | prognostic ``(G^r, G^i)`` | Reconstituted amplitude ``G \equiv [1 + \alpha(\nabla_h^2 + \kappa^2)] A``; prognosed, with ``A`` diagnosed per stage |
| ``\kappa`` | `κ`, `dispersion.κ` | **Carrier wavenumber** — a fixed scalar parameter (see the collision note below) |
| ``\omega`` | `carrier_frequency` | Carrier frequency, ``\omega^2 = g\kappa\tanh\kappa h`` |
| ``\omega_\kappa`` | `carrier_group_velocity` | Group velocity ``d\omega/d\kappa`` |
| ``\omega_{\kappa\kappa}`` | `carrier_frequency_curvature` | Dispersion curvature ``d^2\omega/d\kappa^2`` |
| ``\alpha`` | `reconstitution_parameter` | Reconstitution parameter ``(\kappa\omega_{\kappa\kappa} - \omega_\kappa)/(4\kappa^2\omega_\kappa) < 0`` |
| ``\Phi(z), \Phi_z`` | `vertical_structure` | Carrier vertical structure, ``\int_{-h}^0 \Phi^2\,dz = 1`` |
| ``C`` | `vertical_structure_constant` | Normalization of ``\Phi``, ``C^2 = g\kappa/(\omega\omega_\kappa)`` |
| ``H`` | — | Helmholtz "bandwidth" operator ``\nabla_h^2 + \kappa^2`` (``HA \approx 0`` for narrow-band ``A``) |
| ``m`` | `screened_poisson_symbol` | Screened-Poisson symbol ``(1 + \alpha\kappa^2)/\alpha < 0`` used to diagnose ``A`` from ``G`` |
| ``\bar{\boldsymbol{U}}, \tilde{\boldsymbol{U}}`` | depth-weighted velocities | ``\Phi^2``- and ``\Phi_z^2``-weighted vertical integrals of ``\boldsymbol{U}^L`` |
| ``\boldsymbol{v}^{\mathrm{eff}}`` | `effective_transport_velocity` | Effective Doppler-transport velocity ``(\kappa\omega_\kappa/\omega)(\bar{\boldsymbol{U}} + \tilde{\boldsymbol{U}})`` |
| ``\boldsymbol{U}^s`` | `NarrowBandStokesDrift` | Stokes drift diagnosed from ``A``, separable as ``\Phi^2 S_i + \Phi_z^2 T_i`` |
| ``\mathcal{A}, \mathcal{E}, \mathcal{P}^w`` | `wave_action`, `coupled_energy`, `pseudomomentum` | Wave action, coupled wave–current energy, wave pseudomomentum |

!!! warning "Two symbol collisions to keep straight"
    The narrow-band model reuses two symbols that mean something different in
    Ripple's spectral action model:

    - **``\kappa``** is the model's *fixed carrier wavenumber* here, whereas in
      the spectral action model ``\kappa`` is a *spectral coordinate* (radial
      wavenumber) that indexes the spectrum. The two models do not share fields,
      so the clash is cognitive, not programmatic — but read ``\kappa`` as a
      scalar parameter throughout the narrow-band pages.
    - **``G``** is the *reconstituted amplitude* ``[1 + \alpha H]A`` here,
      whereas elsewhere on this page ``G`` denotes the *tendency of the action*
      ``N``. Context (amplitude model vs. action model) disambiguates.

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
