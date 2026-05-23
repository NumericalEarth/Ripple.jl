# Monobanded Wave Model

`MonobandedWaveModel` is Ripple's single-wave-train model. It keeps the
horizontal phase-space geometry of the wave-action equations, but replaces the
resolved spectral distribution ``N(x, y, \boldsymbol{k}, t)`` with one local
action density and one local wavevector:

```math
A(x, y, t) = \int N \, d^2\boldsymbol{k},
\qquad
\boldsymbol{M}(x, y, t) = A \boldsymbol{K}
  = \int \boldsymbol{k} N \, d^2\boldsymbol{k}.
```

The prognostic fields are stored as surface-slab Oceananigans `Field`s:

- `model.action`: ``A``
- `model.wavenumber_moment.x`: ``AK_x``
- `model.wavenumber_moment.y`: ``AK_y``

The diagnosed wavevector is

```math
\boldsymbol{K}
= (K_x, K_y)
= \frac{(AK_x, AK_y)}{A}.
```

This model is appropriate when a wave field is narrow enough in wavenumber and
direction that a single local wavevector is the intended state variable. Use
`SpectralWaveModel` when directional spreading, frequency spreading, or
spectral source terms are part of the question.

## Equations

The monobanded prognostic equations are the action equation and the first
wavevector-moment equations:

```math
\frac{\partial A}{\partial t}
+ \nabla_{\boldsymbol{x}} \cdot (\boldsymbol{C} A)
= S_A,
```

and

```math
\frac{\partial (A K_\alpha)}{\partial t}
+ \nabla_{\boldsymbol{x}} \cdot (\boldsymbol{C} A K_\alpha)
= - A K_\beta \Gamma_{\beta\alpha} + K_\alpha S_A,
\qquad \alpha,\beta \in \{x, y\}.
```

The moment source contribution ``K_\alpha S_A`` preserves the local wavevector
under pure action growth or decay. The refraction force
``- A K_\beta \Gamma_{\beta\alpha}`` bends the wavevector without directly
changing action. Repeated ``\beta`` indices are summed, so the two moment
refraction tendencies are

```math
\left.\frac{\partial (A K_x)}{\partial t}\right|_{\mathrm{refraction}}
= -A (K_x \Gamma_{xx} + K_y \Gamma_{yx}),
```

and

```math
\left.\frac{\partial (A K_y)}{\partial t}\right|_{\mathrm{refraction}}
= -A (K_x \Gamma_{xy} + K_y \Gamma_{yy}).
```

For no current coupling, the intrinsic deep-water frequency and group velocity
are

```math
\Omega = \sqrt{g\kappa},
\qquad
\boldsymbol{C} = \frac{\partial \Omega}{\partial \boldsymbol{K}}
               = \sqrt{\frac{g}{4\kappa^3}} \, \boldsymbol{K},
```

where ``\kappa = |\boldsymbol{K}|``.

With a Lagrangian-mean current, Ripple computes the Q-projected Doppler velocity

```math
\boldsymbol{u}^{D}(x, y, \kappa, t)
= \int_{-d}^{0} Q(z; \kappa, d)
  \boldsymbol{u}^{L}(x, y, z, t) \, dz
```

and its radial derivative
``\boldsymbol{H} = \partial\boldsymbol{u}^{D}/\partial\kappa``. The absolute
frequency and ray velocity are

```math
\Omega = \sqrt{g\kappa} + \boldsymbol{K}\cdot\boldsymbol{u}^{D},
```

```math
\boldsymbol{C}
= \sqrt{\frac{g}{4\kappa^3}} \, \boldsymbol{K}
  + \boldsymbol{u}^{D}
  + \frac{\boldsymbol{K}\cdot\boldsymbol{H}}{\kappa}\boldsymbol{K}.
```

The intrinsic part of this formula is currently the deep-water relation
``\sqrt{g\kappa}``. A finite-depth Q grid controls the vertical projection of
the current; it does not change the monobanded intrinsic dispersion relation.

The fixed-``\kappa`` current-gradient tensor used by refraction is

```math
\Gamma_{\beta\alpha}
= \partial_\alpha u^{D}_{\beta}
  - H_{\beta} \, \partial_\alpha \kappa.
```

`model.diagnostics.Γxy`, for example, is ``\Gamma_{xy}``, the ``y`` gradient
of the ``x`` component of ``\boldsymbol{u}^{D}`` at fixed ``\kappa``.

## Constructor

The basic constructor takes only the physical grid as a positional argument:

`MonobandedWaveModel(grid; kwargs...)`

Common keywords:

| Keyword | Meaning |
|:--------|:--------|
| `action` | Optional initial `Field` for ``A``. If omitted, Ripple allocates one. |
| `wavenumber_moment` | Optional `(; x, y)` fields for ``AK_x`` and ``AK_y``. |
| `advection` | Physical transport scheme. Default is `WENO()`. Use `Centered()` for conservative centered fluxes or `nothing` to disable physical transport only. |
| `sources` | Supported action-only source term or `SourceTermSet`; default `nothing`. |
| `velocities` | `nothing`, `ZeroVelocities()`, `PrescribedVelocities`, `PseudomomentumVelocities()`, or `(; u, v)`. |
| `coupling` | Explicit monobanded coupling object. Mutually exclusive with `velocities`. |
| `boundary_conditions` | NamedTuple keyed by `A`, `AKx`, and `AKy`. Only default NoFlux/Periodic behavior is currently supported. |
| `timestepper` | `:RungeKutta3` default, materialized as Oceananigans' `RungeKutta3TimeStepper`; `:RK3`, `:SSPRungeKutta3`, `:LowStorageRK3`, and `:LSRK3` are aliases. Also accepts `:ForwardEuler` and `:AB2`. |
| `gravitational_acceleration` | Defaults to `9.81`. |
| `minimum_action` | Floor used when diagnosing ``K = AK/A``. |
| `minimum_wavenumber` | Floor used when diagnosing ``\kappa``. |

The model follows Oceananigans conventions, so `time_step!(model, Δt)` advances
the model and `fields(model)` returns prognostic and diagnostic fields.
Initialize a prescribed wavevector by setting the moments consistently:
`AKx = A * Kx` and `AKy = A * Ky`.

## Minimal Setup

```@example monobanded_model
using Oceananigans, Ripple

grid = RectilinearGrid(CPU();
                       size = (64, 32, 1),
                       halo = (3, 3, 3),
                       x = (0, 64),
                       y = (0, 32),
                       z = (-1, 0),
                       topology = (Periodic, Periodic, Bounded))

model = MonobandedWaveModel(grid; timestepper = :RungeKutta3)

packet(x, y, z) = exp(-((x - 16)^2 + (y - 16)^2) / 16)

set!(model;
     A   = packet,
     AKx = (x, y, z) -> 0.5 * packet(x, y, z),
     AKy = 0)

time_step!(model, 0.05)

model
```

Here ``K_x = AK_x/A \approx 0.5`` inside the packet and ``K_y = 0``.

## Current Coupling

`velocities=nothing` and `velocities=ZeroVelocities()` both run intrinsic
deep-water propagation with no current coupling.

For prescribed currents, pass C-grid Oceananigans velocity fields:

```@example monobanded_model
u = Field{Face, Center, Center}(grid)
v = Field{Center, Face, Center}(grid)
set!(u, (x, y, z) -> 0.1 * sin(2π * y / 32))
set!(v, 0)

current_model = MonobandedWaveModel(grid;
                                    velocities = PrescribedVelocities(; u, v),
                                    advection = WENO())
```

A bare `velocities=(; u, v)` is accepted and is converted to
`PrescribedVelocities`. The Q grid is inferred from the velocity fields unless
you pass it explicitly through `PrescribedVelocities`.

`velocities=PseudomomentumVelocities()` feeds the monobanded pseudomomentum
back as the Lagrangian-mean velocity. If the monobanded model grid has `Flat`
vertical topology, pass an explicit finite-depth `q_grid`.

## Diagnostics

`prognostic_fields(model)` returns `(; A, AKx, AKy)`.
`fields(model)` also includes the diagnostic fields:

| Diagnostic | Meaning |
|:-----------|:--------|
| `Kx`, `Ky` | Diagnosed wavevector components. |
| `κ` | Diagnosed radial wavenumber. |
| `uᴰx`, `uᴰy` | Q-projected Doppler velocity. |
| `Hx`, `Hy` | ``\partial\boldsymbol{u}^{D}/\partial\kappa``. |
| `Cx`, `Cy` | Full ray velocity ``\boldsymbol{C}``. |
| `Ĉx`, `Ĉy` | Intrinsic group velocity plus ``\boldsymbol{u}^{D}``, before the ``\boldsymbol{H}`` correction. |
| `Ω` | Absolute frequency. |
| `Γxx`, `Γyx`, `Γxy`, `Γyy` | Fixed-``\kappa`` current-gradient tensor for refraction. |
| `ZK` | Horizontal curl of ``\boldsymbol{K}``, ``\partial_x K_y - \partial_y K_x``. |

`pseudomomentum_fields(model)` returns Q-projected pseudomomentum fields
``p_x`` and ``p_y`` on the model's Q geometry. Their vertical integral recovers
the horizontal monobanded moments.

`pseudomomentum_tendency_fields(model)` returns the analytic time derivative
``\partial_t p_x, \partial_t p_y`` evaluated from the wave-model tendency
``G^n``. For the monobanded model this includes both
``Q(\kappa, z)\, \partial_t (A K_\alpha)`` from the analytic action and
moment tendencies *and* the
``\partial_\kappa Q(\kappa, z)\, A K_\alpha\, \partial_t \kappa`` correction
that captures how the Q-shape changes when refraction or sources move
``\kappa``. These fields are exactly the ``\partial_t u^s, \partial_t v^s``
that Oceananigans' Craik-Leibovich `StokesDrift` wants as input to evaluate
the Stokes-acceleration term ``+\partial_t u^s`` in the momentum tendency.
The caller is expected to have run `compute_tendencies!(model)` for the state
of interest. After `time_step!`, `G^n` is the last stage tendency, not a fresh
tendency recomputed at the final state.

## Transport And Refraction

`advection=nothing` disables physical transport but does not disable
refraction or source terms. This is useful for column-style checks of
``d(AK_\alpha)/dt = -A K_\beta \Gamma_{\beta\alpha}``.

With `advection=WENO()`, Ripple uses the monobanded conservative transport
kernel with fifth-order WENO face reconstruction. With `advection=Centered()`,
the same flux form uses centered face values. Other Oceananigans advection
schemes are rejected until they are implemented with their corresponding
reconstruction.

The current implementation requires uniform horizontal grid spacing. Periodic
directions wrap; bounded directions use no-flux edge fluxes. Non-default
prognostic boundary conditions are rejected because the monobanded transport
and gradient kernels do not yet apply user-specified boundary fluxes or values.

## Sources

`MonobandedWaveModel` supports action-only source terms:

- `LinearWindInput` with scalar `rate`
- `BottomFriction` with scalar parameters
- `SourceTermSet` combinations of the above

For a source rate ``r``, the model applies

```math
S_A = r A,
\qquad
S_{AK_x} = K_x S_A,
\qquad
S_{AK_y} = K_y S_A.
```

This preserves the diagnosed wavevector under pure source growth or decay.
Spectral source terms that depend on frequency or direction, such as
whitecapping formulations for a resolved spectrum, belong in
`SpectralWaveModel`.

## Numerical Safeguards

The diagnosed wavevector uses
``K = AK / \max(A, \mathrm{minimum\_action})`` and
``\kappa \ge \mathrm{minimum\_wavenumber}``. After each time-step update, cells
with action below `minimum_action` have `A`, `AKx`, and `AKy` zeroed together.
This prevents a vanishing-action cell from retaining finite moments and
creating an unbounded diagnostic group velocity.

The action update is clamped nonnegative in the built-in timesteppers. The
moment fields are not individually positivity-limited, because their signs
encode the wavevector direction.

## Example Validation

The `examples/monobanded_linear_shear_refraction.jl` example checks a
nontrivial quasi-analytic solution. For a barotropic linear shear
``u_x^D(y)=U_0+S(y-y_c)``, ``u_y^D=0``, the monobanded refraction law predicts

```math
K_x(t) = K_{x0},
\qquad
K_y(t) = K_{y0} - S K_{x0} t.
```

The example compares the model to this wavevector solution and to an analytic
ray-map prediction for a compact Gaussian action packet.
