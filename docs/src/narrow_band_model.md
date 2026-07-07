# Narrow-band amplitude wave–current model

Ripple has two wave models. The `SpectralWaveModel` evolves a
wave-action spectrum ``N(x, y, \kappa, \varphi)`` under WKB ray dynamics — the
right tool when the current varies slowly compared with the wavelength. The
`NarrowBandWaveModel` evolves a complex wave *amplitude* ``A(x, y, t)`` and is
the complementary **no-scale-separation** model: it resolves the carrier in
space, so it captures current-induced advection, refraction, and
multidirectional scattering for currents at wavelength-comparable scales —
Langmuir cells — where ray theory breaks down. It implements the reduced
wave–current model of Onuki & Fujiwara (2026).

!!! note "Which model should I use?"
    Use `SpectralWaveModel` for basin- and storm-scale forecasting where the
    spectrum is broad and currents are slowly varying. Use `NarrowBandWaveModel`
    for process studies of wave–current interaction at the wavelength scale
    (Langmuir turbulence, wave scattering by submesoscale currents), where the
    waves are narrow-band in frequency and the current cannot be treated as
    locally uniform.

## The model

The waves are described by a complex amplitude ``A`` from which the fast
oscillation ``e^{-i\omega t}`` at carrier frequency ``\omega`` has been factored
out; the horizontal carrier ``e^{i\boldsymbol{k}\cdot\boldsymbol{x}}`` with
``|\boldsymbol{k}| \approx \kappa`` lives *inside* ``A``. The carrier wavenumber
``\kappa`` is fixed and sets every coefficient through the linear dispersion
relation ``\omega^2 = g\kappa\tanh\kappa h``:

- the group velocity ``\omega_\kappa = d\omega/d\kappa`` and curvature
  ``\omega_{\kappa\kappa}``,
- the reconstitution parameter ``\alpha = (\kappa\omega_{\kappa\kappa} -
  \omega_\kappa)/(4\kappa^2\omega_\kappa) < 0`` (Thomas & Yamada 2018), which
  repairs the narrow-band Taylor expansion so the reconstituted dispersion
  matches the exact ``\omega(|\boldsymbol{k}|)`` to third order,
- the vertical structure ``\Phi(z) = C\cosh\kappa(z+h)/\cosh\kappa h``,
  normalized so ``\int_{-h}^0 \Phi^2\,dz = 1``.

The amplitude obeys (Onuki & Fujiwara 2026, eq. 2.3)

```math
[1 + \alpha(\nabla_h^2 + \kappa^2)]\,\partial_t A
  - \frac{i\omega_\kappa}{2\kappa}(\nabla_h^2 + \kappa^2) A
  = \frac{\omega_\kappa}{2\kappa\omega}\,\mathcal{L}(\boldsymbol{U}^L, A),
```

coupled to an Oceananigans `NonhydrostaticModel` for the Lagrangian-mean flow
through a Stokes drift diagnosed from ``A``.

## The flux-divergence rewrite

Expanding the wave–current operator ``\mathcal{L}`` and folding in the carrier
gives the exact identity that Ripple discretizes:

```math
[1 + \alpha H]\,\partial_t A
  = \underbrace{\tfrac{i\omega_\kappa}{2\kappa} H A}_{\text{dispersion}}
  \;\underbrace{- \nabla\cdot(\boldsymbol{v}^{\mathrm{eff}} A)}_{\text{Doppler transport}}
  \;+\;\underbrace{\tfrac{\kappa\omega_\kappa}{\omega}\!\left[\nabla\cdot\bar{\boldsymbol{U}} + \tfrac12\nabla\cdot\tilde{\boldsymbol{U}}\right] A}_{\text{compressibility source}}
  \;+\;\underbrace{\tfrac{\omega_\kappa}{2\kappa\omega} R}_{\text{refraction/scattering}},
```

with ``H = \nabla_h^2 + \kappa^2`` and the effective transport velocity
``\boldsymbol{v}^{\mathrm{eff}} = (\kappa\omega_\kappa/\omega)(\bar{\boldsymbol{U}}
+ \tilde{\boldsymbol{U}})``. The depth-weighted velocities

```math
\bar{U}_i = \int_{-h}^0 \Phi^2 U_i^L\,dz,
\qquad
\tilde{U}_i = \frac{1}{\kappa^2}\int_{-h}^0 \Phi_z^2 U_i^L\,dz,
```

are computed by exact finite-volume cell integrals of ``\Phi^2`` and
``\Phi_z^2``. In deep water ``\boldsymbol{v}^{\mathrm{eff}} = \bar{\boldsymbol{U}}
= 2\kappa\int e^{2\kappa z}\boldsymbol{U}^L\,dz`` — the classical Stewart & Joy
(1974) / Kirby & Chen (1989) depth-weighted advection velocity — and a plane
wave on a uniform current recovers the Doppler shift
``\boldsymbol{k}\cdot\bar{\boldsymbol{U}}``. The residual

```math
R = 2\bar{\boldsymbol{U}}\cdot\nabla(HA) + 2\bar{U}_{i,j}A_{,ij}
    + (\nabla\cdot\bar{\boldsymbol{U}})\nabla^2 A + \bar{U}_{i,ij}A_{,j}
```

is the scattering operator that ray theory linearizes, not an error term.

## The (G, A) formulation

The mass operator ``M = 1 + \alpha H`` is nonlocal, so Ripple prognoses the
**reconstituted amplitude** ``G \equiv M A`` and diagnoses ``A`` from it. This
makes the dispersion term local, ``\alpha H A = G - A``, so the time stepper
never sees a mass matrix — exactly the prognostic-momentum / diagnostic-pressure
pattern of an Oceananigans model:

```math
\partial_t G = \frac{i\omega_\kappa}{2\kappa\alpha}(G - A)
  - \nabla\cdot(\boldsymbol{v}^{\mathrm{eff}} A) + \text{source}
  + \frac{\omega_\kappa}{2\kappa\omega} R,
\qquad
[1 + \alpha(\nabla_h^2 + \kappa^2)] A = G.
```

The diagnostic solve is the generalized ("screened") Poisson equation
``(\nabla_h^2 + m) A = G/\alpha`` with ``m = (1 + \alpha\kappa^2)/\alpha < 0``,
solved by the stock `Oceananigans.Solvers.FFTBasedPoissonSolver` on a
``(\texttt{Periodic}, \texttt{Periodic}, \texttt{Flat})`` companion grid. Because
``\alpha < 0`` and ``1 + \alpha\kappa^2 > 0``, the operator is symmetric positive
definite and nonsingular. The wave action becomes the pairing ``\mathcal{A} =
\mathrm{Re}\iint A^\dagger G\,dx\,dy``.

The fields are stored as real pairs ``(G^r, G^i)`` (prognostic) and ``(A^r,
A^i)`` (diagnostic) so that Oceananigans tracer advection, halo filling, and
output writers work unmodified. Only the prognostic ``G`` is needed to restart.

## Numerics

- **Dispersion** enters entirely through the elliptic solve, so the
  semi-discrete dispersion relation ``\Lambda_d(\lambda) =
  \tfrac{\omega_\kappa}{2\kappa}(\kappa^2 - \lambda)/(1 + \alpha(\kappa^2 -
  \lambda))`` (with ``\lambda`` the discrete Laplacian eigenvalue) is automatic
  and **bounded** — plain SSP-RK3 is stable at ``\Delta t \sim 1/\omega``.
- **Doppler transport** ``-\nabla\cdot(\boldsymbol{v}^{\mathrm{eff}} A)`` uses
  Oceananigans tracer advection (`WENO()` by default, `Centered()` for
  conservation studies), applied to ``A^r`` and ``A^i`` separately. WENO's
  upwind dissipation acts on the carrier but scales with the weak current speed
  ``|\boldsymbol{v}^{\mathrm{eff}}|``, not the phase speed.
- **Refraction** ``R`` is a fused centered finite-difference kernel.
- **Conservation**: with `Centered` transport and a divergence-free current the
  wave action ``\mathcal{A} = \mathrm{Re}\langle A, G\rangle`` is conserved to
  discretization error; the `WENO` default trades exact conservation for
  monotone, resolution-convergent action decay.

## API

```julia
grid = RectilinearGrid(size=(Nx, Ny, Nz), x=..., y=..., z=(-h, 0),
                       topology=(Periodic, Periodic, Bounded), halo=(3, 3, 3))

model = NarrowBandWaveModel(grid; κ,                    # carrier wavenumber (required)
                            depth = nothing,            # from the grid, or InfiniteDepth()
                            velocities = nothing,        # nothing | (u, v) prescribed current
                            advection = WENO())          # Doppler-transport scheme

set!(model; A = (x, y) -> cis(κ * x))                   # or G = ...
time_step!(model, Δt)
```

Diagnostics: `amplitude`, `reconstituted_amplitude`,
`stokes_functionals`, `surface_elevation_amplitude`, and the
carrier utilities `carrier_frequency`,
`reconstitution_parameter`.

## References

Onuki, Y. & Fujiwara, Y. (2026). *A reduced model for surface wave–current
interactions without spatial scale separation.* arXiv:2606.03231.
See also the Notation page for the symbol conventions and the two
symbol collisions (``\kappa`` and ``G``) between the two wave models.
```
