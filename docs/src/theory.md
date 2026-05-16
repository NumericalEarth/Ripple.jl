# Theory

Ripple is built around the wave-action balance used in phase-averaged
wave-current theory. The core prognostic quantity is the wave action density
``N(\boldsymbol{x}, \boldsymbol{k}, t)``, represented on polar grids as
``N(x, y, \kappa, \phi, t)`` and stored as finite-volume cell averages over
physical cells and spectral control volumes. This page describes both the
continuum equations and the numerical implementation used to approximate
them.
See [Notation](@ref) for the coordinate and symbol conventions used below.

## Action Balance

In canonical wavevector coordinates ``\boldsymbol{k} = (k_x, k_y)``, wave
action is transported by Hamiltonian ray velocities in physical and wavevector
space [BrethertonGarrett1968](@citep), [Whitham1974](@citep). With source
terms, the continuum balance is

```math
\frac{\partial N}{\partial t}
+ \nabla_{\mathbf{x}} \cdot (\dot{\mathbf{x}} N)
+ \nabla_{\mathbf{k}} \cdot (\dot{\mathbf{k}} N)
= S_N ,
```

where

```math
\dot{\mathbf{x}} = \nabla_{\mathbf{k}} \Omega,
\qquad
\dot{\mathbf{k}} = -\nabla_{\mathbf{x}} \Omega.
```

For a smooth Hamiltonian ``\Omega``, the phase-space ray velocity is
divergence-free in the canonical coordinates, so the same equation can be
written in advective form as

```math
\frac{\partial N}{\partial t}
+ \nabla_{\mathbf{k}} \Omega \cdot \nabla_{\mathbf{x}} N
- \nabla_{\mathbf{x}} \Omega \cdot \nabla_{\mathbf{k}} N
= S_N .
```

Ripple stores many spectra in polar cells ``(\kappa, \phi)``. The conservative
equation is still the guiding balance, but polar finite volumes carry the cell
measure ``d^2\boldsymbol{k} = \kappa\, d\kappa\, d\phi``. This is why Ripple's
spectral diagnostics and sources use exact cell measures rather than midpoint
quadrature.

This action balance organizes Ripple's split between physical transport,
spectral refraction, and source tendencies. `horizontal_advection` approximates
the physical flux divergence, `spectral_advection` enables spectral refraction
for CWCM coupling, and `sources` contributes `S_N`.

## Dispersion And Currents

For finite depth ``d``, the intrinsic surface-gravity-wave frequency is

```math
\sigma^2 = g \kappa \tanh(\kappa d),
```

with deep-water limit ``\sigma^2 = g\kappa``. The absolute frequency used by wave-current
ray theory is

```math
\Omega(\mathbf{x}, \mathbf{k}, t)
= \sigma(\kappa, d) + \mathbf{k} \cdot \mathbf{U}(\mathbf{x}, \kappa, t),
```

where ``\boldsymbol{U}`` is the vertically projected Lagrangian-mean velocity.
This is the Doppler-shifted action transport used in the consistent
wave-current model of [VannesteYoung2026](@citet). Ripple's `InfiniteDepth()`
selects the deep-water dispersion relation even when velocities are supplied on
a finite-depth Q grid.

In finite depth, the intrinsic group velocity magnitude is

```math
c_g
= \frac{1}{2}
  \left( 1 + \frac{2 \kappa d}{\sinh(2 \kappa d)} \right)
  \sqrt{\frac{g \tanh(\kappa d)}{\kappa}} .
```

Ripple precomputes the intrinsic group-velocity components per spectral cell
for the transport kernels. If `depth` varies horizontally, those tables are
rebuilt from the materialized depth field.

## Q Projection And Pseudomomentum

The CWCM coupling follows the vertical structure emphasized by
[VannesteYoung2026](@citet). For finite depth, Ripple uses the normalized
kernel

```math
Q(z; \kappa, d)
= \frac{2\kappa \cosh(2\kappa[z+d])}{\sinh(2\kappa d)},
\qquad
\int_{-d}^{0} Q(z; \kappa, d)\, dz = 1 .
```

The Doppler velocity for each wavenumber ring is the Q-weighted current

```math
\mathbf{U}(\mathbf{x}, \kappa, t)
= \int_{-d}^{0} Q(z; \kappa, d)\,
  \mathbf{u}^{L}(\mathbf{x}, z, t)\, dz .
```

Ripple computes this integral with finite-volume vertical cell integrals from
CDF differences, so the same discrete Q geometry is used for Doppler velocity,
its κ derivative, and pseudomomentum diagnostics.

Wave pseudomomentum is represented as a Q-projected spectral moment,

```math
\mathbf{p}(\mathbf{x}, z, t)
= \int Q(z; \kappa, d)\,
  \mathbf{k}\, N(\mathbf{x}, \mathbf{k}, t)\, d^2\mathbf{k}.
```

In code, the spectral integral is a finite-volume sum over the model's spectral
cells. `PrescribedVelocities` supplies `uᴸ` externally; `PseudomomentumVelocities`
uses the model action to refresh a self-coupled pseudomomentum velocity before
each CWCM tendency evaluation.

## Refraction

Current gradients bend rays through

```math
\dot{\boldsymbol{k}} = -\nabla_{\boldsymbol{x}} \Omega .
```

In Ripple's polar coordinates this becomes advection in ``\kappa`` and
``\phi``. Let

```math
\boldsymbol{e}_{\kappa} = (\cos\phi, \sin\phi),
\qquad
\boldsymbol{e}_{\phi} = (-\sin\phi, \cos\phi).
```

For the Doppler part ``\boldsymbol{k}\cdot\boldsymbol{U}``, the implemented
current-gradient refraction velocities are

```math
\dot{\kappa}
= -\kappa \,
  \boldsymbol{e}_{\kappa}
  \cdot \nabla_{\boldsymbol{x}}\boldsymbol{U}
  \cdot \boldsymbol{e}_{\kappa},
```

and

```math
\dot{\phi}
= \boldsymbol{e}_{\phi}
  \cdot
  \left[-(\nabla_{\boldsymbol{x}}\boldsymbol{U})^{\mathsf{T}}
  \boldsymbol{e}_{\kappa}\right].
```

The fused CWCM kernel advects action in physical space and spectral space in a
single pass using these current-gradient velocities.

In component form, with ``c = \cos\phi`` and ``s = \sin\phi``,

```math
\dot{\kappa}
= -\kappa
\left[
c^2 U_{x,x}
+ cs (U_{y,x} + U_{x,y})
+ s^2 U_{y,y}
\right],
```

and

```math
\dot{\phi}
= cs (U_{x,x} - U_{y,y})
+ s^2 U_{y,x}
- c^2 U_{x,y}.
```

These are the quantities read by the fused refraction kernel.

The full continuum equation also includes bathymetric refraction through
``\nabla_{\boldsymbol{x}}\sigma(\kappa, d)``. Ripple currently uses `depth` for
intrinsic dispersion, automatic Q-grid construction, and Q projection. A
complete conservative bathymetric refraction term is not yet part of the
spectral-refraction kernel.

## Source Terms

Operational spectral wave models usually write the right-hand side as a sum of
wind input, nonlinear transfer, whitecapping, bottom friction, depth-limited
breaking, ice or swell dissipation, and optional relaxation terms. Ripple's
source API is deliberately modular: each source returns a local tendency or an
implicit rate, and `SourceTermSet` combines them into `S_N`.

This layout follows the action-balance organization used by third-generation
models such as WAVEWATCH III [WAVEWATCHIII2019](@citep), with source-term
background covered by standard references including [Komen1994](@citep),
[Janssen2004](@citep), and [Holthuijsen2007](@citep). Ripple leaves the choice
of source-term physics explicit in user code.

## Numerical Implementation

Ripple discretizes the action balance as a method-of-lines finite-volume model
on the product of a horizontal Oceananigans grid and a spectral grid. The stored
unknown is the cell average

```math
\bar{N}_{ijmn}(t)
= \frac{1}{|\mathcal{V}_{ij}|\,|\mathcal{C}_{mn}|}
  \int_{\mathcal{V}_{ij}}
  \int_{\mathcal{C}_{mn}}
  N(x, y, \boldsymbol{k}, t)\,
  d^2\boldsymbol{k}\, dx\,dy ,
```

where ``\mathcal{V}_{ij}`` is the horizontal physical cell and
``\mathcal{C}_{mn}`` is the spectral control volume. For polar coordinates,
``|\mathcal{C}_{mn}|`` includes the Jacobian ``\kappa``. The same measures are
used by diagnostics, source terms, and Q-projected pseudomomentum integrals; see
[Finite-Volume Integration](@ref) for the standalone integration conventions.

In code, `model.action` stores ``\bar{N}_{ijmn}`` in a `WaveActionField`.
The backing array is a contiguous product-field storage with horizontal halo
cells, one physical vertical slab for wave action, and the spectral dimensions
``(\kappa, \phi)``. `physical_field(model.action, m, n)` reconstructs an
Oceananigans `Field` view for one spectral bin when a per-bin operation needs
Oceananigans tracer machinery. The matching `model.tendencies` field stores the
semi-discrete right-hand side

```math
\frac{d\bar{N}_{ijmn}}{dt}
= \mathcal{T}^{x,y}_{ijmn}
 + \mathcal{T}^{\kappa,\phi}_{ijmn}
 + \bar{S}_{ijmn}.
```

### Tendency Assembly

Without current coupling, Ripple either loops over spectral bins and delegates
physical transport to Oceananigans tracer advection, or uses a fused intrinsic
transport kernel for the common multi-bin `WENO()` case on periodic horizontal
grids. The per-bin path is the most general path and supports the configured
`horizontal_advection` scheme. The fused intrinsic path reads the product-field
backing directly and applies fifth-order WENO in ``x`` and ``y`` with the
cell-averaged intrinsic group velocity for each spectral bin. Source tendencies
are added after transport.

With CWCM coupling and `spectral_advection !== nothing`, Ripple uses a fused
KernelAbstractions kernel over ``(i, j, m, n)``. This kernel computes the
Doppler-shifted physical transport velocity

```math
\dot{\boldsymbol{x}}
= c_g \boldsymbol{e}_{\kappa}
  + \boldsymbol{U}(\boldsymbol{x}, \kappa, t)
```

and the current-gradient spectral velocities ``\dot{\kappa}`` and
``\dot{\phi}`` described above. It then applies fifth-order WENO to the
physical and spectral advection terms in one pass through memory. In this mode,
the fused kernel handles physical transport too, so `horizontal_advection` is
ignored. The ``\kappa`` domain has no flux through the two outer faces, while
``\phi`` and the currently implemented CWCM horizontal stencils are periodic.

The no-coupling fused intrinsic kernel and the CWCM fused refraction kernel are
written as explicit KernelAbstractions kernels so the same implementation can
run on CPU and GPU architectures. They operate on `flat_data` and cache
wavenumber-dependent tables lazily on the model or coupling object.

### Depth, Q Grids, And Coupling Caches

The model-level `depth` kwarg controls intrinsic dispersion and group velocity.
`InfiniteDepth()` selects ``\sigma^2 = g\kappa``. A positive number is
materialized as an Oceananigans `ConstantField`, and a function or
Oceananigans `Field` is materialized on the horizontal wave grid. Raw arrays
are deliberately excluded from the public `depth` interface.

The Q transform needs a resolved vertical grid even when the wave-action grid is
vertically `Flat`. For prescribed velocity `Field`s, Ripple uses the fields'
grid as the Q grid. If prescribed velocities are arrays, or if
`PseudomomentumVelocities()` is used on a Flat wave grid, Ripple can build an
automatic stretched Q grid from finite model `depth`. When `depth` is
`InfiniteDepth()` but velocity fields live on a finite-depth grid, the
intrinsic waves use the deep-water dispersion relation while the Q projection
depth is derived from the velocity grid, including local bottom height on an
`ImmersedBoundaryGrid`.

`PrescribedVelocities` caches the Q-projected Doppler velocity
``\boldsymbol{U}(\boldsymbol{x}, \kappa, t)`` and its ``\kappa`` derivative.
`PseudomomentumVelocities` builds the Lagrangian-mean velocity from
`model.action`; before each CWCM tendency evaluation, the coupling refreshes
the pseudomomentum fields and the Doppler-velocity caches. In pseudomomentum
mode, Ripple precomputes vertical overlap tables for each source and target
wavenumber ring so repeated Q-projected spectral moments do not rebuild the
same vertical integrals.

### Time Integration And Smoothing

`time_step!(model, Δt)` advances the semi-discrete ODE with the selected
time-stepper: forward Euler, semi-implicit Euler for split source damping, AB2,
SSP-RK3, or low-storage RK3. Each explicit update kernel clamps action to
``\bar{N} \ge 0`` after a stage update. This clamp prevents negative action
values produced by high-order advection or explicit source updates from
propagating to diagnostics and subsequent stages; it is not a replacement for a
positivity-preserving flux scheme.

Propagation smoothing is applied as a fractional step after each completed
model time step, not after every RK substage. This ordering matches the
interpretation of Tolman's GSE smoothing as a post-propagation spatial average
over the distance traveled during ``\Delta t``.

### Current Limitations

The implemented CWCM spectral refraction is the current-gradient part of
``-\nabla_{\boldsymbol{x}}\Omega``. The continuum finite-depth Hamiltonian also
contains bathymetric refraction from ``\nabla_{\boldsymbol{x}}\sigma(\kappa,d)``.
Ripple uses `depth` for dispersion, group-velocity tables, Q-grid construction,
and Q projection, but a complete conservative bathymetric refraction flux is
still future work.

The fused WENO kernels currently assume uniform horizontal grid spacing. The
generic per-bin path should be preferred when a run needs boundary handling or
advection behavior outside the assumptions of the fused kernels.

## Garden-Sprinkler-Effect Smoothing

Discrete spectral propagation can create artificial ray-like beams from a
localized source, the Garden Sprinkler Effect (GSE). The issue was analyzed for
discrete spectral wave models by [BooijHolthuijsen1987](@citet) and treated in
WAVEWATCH III with additional smoothing options by [Tolman2002](@citet).

Ripple implements Tolman's spatial-averaging strategy as
`SpatialAveraging(; αs, αn)`. After each full `time_step!`, each spectral bin is
averaged over a small rectangle aligned with the bin's propagation direction:

```math
L_s = \alpha_s |\Delta c_g| \Delta t,
\qquad
L_n = \alpha_n c_g \Delta \phi \Delta t .
```

`L_s` smooths along the propagation direction using neighboring radial group
velocities, while `L_n` smooths across the beam using the directional bin width.
The default `αs = αn = 0.5` matches the discrete-bin extent used for the
idealized estimate in [Tolman2002](@citet). Larger values suppress GSE more
strongly but also smear physically sharp gradients.

## Implementation Map

- `model.action` stores finite-volume wave-action averages.
- `model.tendencies` stores the semi-discrete right-hand side.
- `horizontal_advection` applies Oceananigans tracer advection to physical
  transport.
- `spectral_advection` enables the fused CWCM spectral-refraction kernel.
- `velocities` chooses no coupling, prescribed Lagrangian velocities, or
  pseudomomentum velocities.
- `depth` chooses deep-water or finite-depth intrinsic dispersion and Q
  projection.
- `propagation_smoothing=SpatialAveraging(...)` applies the Tolman GSE
  alleviation step after each model time step.
