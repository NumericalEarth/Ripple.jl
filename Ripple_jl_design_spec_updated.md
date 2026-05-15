# Ripple.jl design specification

**Working title:** Ripple.jl  
**Purpose:** an Oceananigans-native spectral wave-action model for surface gravity waves and wave-current interaction.  
**Primary equation:** the action equation and coupled wave-current structure described in Vanneste & Young, *A consistent phase-averaged model of the interactions between surface gravity waves and currents*, arXiv:2602.21976.  
**Design status:** v0.2, based on the design iteration in this chat.  
**Central implementation choice:** build a new `ProductField` abstraction for fields over physical space × auxiliary coordinate space, rather than modifying Oceananigans' existing `Field` or flattening spectral dimensions.

Relevant external sources:

- CWCM paper: <https://arxiv.org/pdf/2602.21976>
- Oceananigans.jl: <https://github.com/CliMA/Oceananigans.jl>
- Oceananigans docs: <https://clima.github.io/OceananigansDocumentation/stable/>
- WAVEWATCH III: <https://github.com/NOAA-EMC/WW3>
- WAVEWATCH III docs: <https://noaa-emc.github.io/WW3/>
- WAM Cycle 7: <https://github.com/myWAveModel/WAM>
- WAM information page: <https://mywave.github.io/WAM/>
- natESM WAM page: <https://www.nat-esm.de/services/models/wam>
- ecWAM: <https://github.com/ecmwf-ifs/ecwam>
- SWAN GitLab: <https://gitlab.tudelft.nl/citg/wavemodels/swan>
- TU Delft SWAN description: <https://research.tudelft.nl/en/datasets/swan/>
- PiCLES.jl: <https://github.com/mochell/PiCLES.jl>
- PiCLES paper: <https://doi.org/10.1029/2025MS005221>
- PiCLES article record: <https://impacts.ucar.edu/en/publications/a-particle-in-cell-wave-model-for-efficient-sea-state-estimates-i/>

---

## 1. Executive summary

Ripple.jl should be a Julia spectral wave model built in the style of Oceananigans.jl. It should solve a conservative wave-action equation in horizontal position-wavevector phase space, support operationally useful spectral-wave source terms, and eventually couple consistently to Oceananigans ocean models through the CWCM coupling operators.

The model should **not** be a Fortran port of WAVEWATCH III, WAM, ecWAM, or SWAN. Those models should be cloned and audited deeply, but used as sources of architecture, physics organization, source-term structure, validation cases, and performance lessons. Ripple.jl should be a new implementation whose data structures, kernel organization, and user-facing API feel native to Oceananigans and Julia.

The core state is wave action density,

```math
N = N(x, y, \xi, \eta),
```

where `(ξ, η)` may be `(κ, φ)` for a polar wavenumber-direction grid or `(kx, ky)` for a Cartesian wavevector grid. In code, the state should be logically indexed as

```julia
N[i, j, m, n]
```

where `i, j` are physical horizontal indices and `m, n` are spectral or auxiliary-coordinate indices.

A key conclusion from the design discussion is that the spectrum should **not** be flattened into a single `q` dimension in the model's numerical method. Flattening makes spectral differencing and boundary conditions more complicated, obscures the physical-spectral symmetry of the Hamiltonian action equation, and becomes less compelling once we accept that a custom field abstraction is needed anyway.

Instead, Ripple.jl should introduce a new Oceananigans-style `ProductField`:

```julia
ProductField{LX, LY, LZ, Lξ, Lη, ...}
```

`ProductField` preserves Oceananigans' existing physical location semantics, for example

```julia
location(N) = (Center, Center, Nothing)
```

and adds coordinate-space locations,

```julia
coordinate_location(N) = (Center, Center)
```

so physical and spectral fluxes can be represented symmetrically:

```julia
N   :: ProductField{Center, Center, Nothing, Center, Center}
Fx  :: ProductField{Face,   Center, Nothing, Center, Center}
Fy  :: ProductField{Center, Face,   Nothing, Center, Center}
Fξ  :: ProductField{Center, Center, Nothing, Face,   Center}
Fη  :: ProductField{Center, Center, Nothing, Center, Face}
```

Ripple.jl should keep Oceananigans' `Field` untouched. Generalizing `Field` itself to arbitrary location tuples would be elegant, but it would require invasive changes to `Field`, locations, grid metrics, boundary conditions, halo filling, and downstream dispatch. `ProductField` localizes the new machinery.

The logical indexing should remain physical-first:

```julia
N[i, j, nκ, nθ]
```

but the likely production memory layout should put spectral dimensions inner, for example

```julia
data[nθ, nκ, i, j]
```

so that each spatial cell's full local spectrum is contiguous. This is expected to help source terms, spectral reductions, pseudomomentum reductions, nonlinear interactions, and GPU kernels. The logical API and storage order must be separated by a layout type.

Parallelization should be **spatial only**. Each MPI rank / GPU owns a local physical tile and the complete local spectrum:

```text
rank owns: local x-y tile × all spectral coordinates
rank does not own: only a frequency band or direction sector
```

This diverges deliberately from any architecture that distributes spectral space. The full spectrum is needed locally for source terms, nonlinear interactions, moments, diagnostics, and CWCM coupling.

---

## 2. Target equation and CWCM physics

### 2.1 Wave-action equation

Ripple.jl targets the action equation written on horizontal position-wavevector phase space:

```math
\partial_t N
+ \nabla_k \Omega \cdot \nabla_x N
- \nabla_x \Omega \cdot \nabla_k N
= S_N.
```

The conservative finite-volume form is

```math
\partial_t N
+ \nabla_x \cdot (\dot{x} N)
+ \nabla_k \cdot (\dot{k} N)
= S_N,
```

with Hamiltonian phase-space velocity

```math
\dot{x} = \nabla_k \Omega,
\qquad
\dot{k} = -\nabla_x \Omega.
```

The Hamiltonian / dispersion relation is

```math
\Omega(x, k) = \sigma(x, \kappa) + k \cdot U(x, \kappa),
\qquad
\kappa = |k|,
```

with intrinsic frequency

```math
\sigma(x, \kappa) = \sqrt{g \kappa \tanh(\kappa d(x))}.
```

The CWCM Doppler velocity is

```math
U(x, \kappa) = \int_{-d(x)}^0 u^L(x,z) Q(x,z,\kappa) \, dz,
```

where `uᴸ` is the Lagrangian-mean current velocity and

```math
Q(x,z,\kappa)
= \frac{2\kappa\cosh(2\kappa(z+d(x)))}{\sinh(2\kappa d(x))},
\qquad
\int_{-d(x)}^0 Q(x,z,\kappa)\,dz = 1.
```

The wave pseudomomentum is

```math
p(x,z) = \iint Q(x,z,\kappa) N(x,k) k \, dk.
```

Integrating vertically gives the identity

```math
\int_{-d(x)}^0 p(x,z)\,dz = \iint N(x,k) k \, dk.
```

This identity is important for diagnostics, tests, and coupled momentum budgets.

### 2.2 Group velocity and the unfamiliar `∂κU` term

Because `U(x,κ)` depends on wavenumber magnitude, the group velocity is not simply the usual intrinsic group velocity plus a barotropic current. The CWCM gives

```math
C(x,k) = \nabla_k \Omega
       = G(x,k) k + U(x,\kappa),
```

where

```math
G(x,k)
= \frac{1}{\kappa}\frac{\partial \sigma}{\partial \kappa}
+ \frac{k}{\kappa} \cdot \frac{\partial U}{\partial \kappa}.
```

Design implication: Ripple.jl should not hard-code only the conventional group velocity formula. The numerical core should preferably derive phase-space velocities from discrete derivatives of a single discrete Hamiltonian `Ω`. This reduces the chance of omitting the `∂κU` term and supports discrete Hamiltonian consistency.

### 2.3 Coupling design implications

The same kernel `Q` appears in both:

```text
U(x,κ):  vertical transform of uᴸ using Q
p(x,z):  spectral transform of N k using Q
```

The discrete implementation should preserve this pairing. The `Q`-weighted transform used for `U` and the transform used for `p` should use the same vertical finite-volume cell geometry, exact `Q` cell integrals, and stable `Q` evaluation. This is the discrete path toward the conservation structure described in the CWCM paper.

---

## 3. Core design principles

### 3.1 Field over a product space

Ripple.jl's central object is a field over a product space:

```math
\text{physical space} \times \text{coordinate space}.
```

For the wave model:

```text
physical space:    x, y
coordinate space:  κ, θ      or      kx, ky
```

The state is logically

```julia
N[i, j, m, n]
```

not

```julia
N[i, j, q]
```

and not a disguised Oceananigans pseudo-vertical dimension.

### 3.2 Preserve physical-spectral symmetry

The action equation is a transport equation in phase space. Waves move in physical space because `Ω` varies in spectral space, and they refract in spectral space because `Ω` varies in physical space:

```math
\dot{x} = \nabla_k \Omega,
\qquad
\dot{k} = -\nabla_x \Omega.
```

This symmetry should be visible in the code. A tendency should look conceptually like

```julia
G = - divergence_x(Fx)
    - divergence_y(Fy)
    - divergence_ξ(Fξ)
    - divergence_η(Fη)
    + sources
```

rather than graph-neighbor operations over a flattened spectral index.

### 3.3 Oceananigans-native, but not constrained to Oceananigans `Field`

Ripple.jl should reuse Oceananigans ideas and conventions:

- `CPU()` / `GPU()` architecture dispatch.
- Grid-based allocation.
- Halo regions.
- Explicit locations (`Center`, `Face`, `Nothing`).
- Finite-volume operators.
- KernelAbstractions-style GPU kernels where appropriate.
- `Simulation`-like stepping, callbacks, diagnostics, and output.

But Ripple.jl should not force its action field into Oceananigans' existing 3D `Field`. Oceananigans' `AbstractField` is rank-generic, but its location semantics are still physical `(LX, LY, LZ)`. The product-space wave field needs auxiliary-coordinate locations as well. Thus, `ProductField` is the clean local extension.

### 3.4 Do not allocate derived fields unless they pay for themselves

The high dimensionality is the central performance challenge. Ripple.jl should avoid storing full product-space arrays for quantities that can be computed on the fly:

Avoid by default:

```julia
Ω[i,j,m,n]
Cx[i,j,m,n]
Cy[i,j,m,n]
Aξ[i,j,m,n]
Aη[i,j,m,n]
Q[i,j,z,m,n]
source_component_1[i,j,m,n]
source_component_2[i,j,m,n]
```

Store only what is needed:

```text
N                      prognostic action density
G or stage tendency     time integration
U[i,j,m]                Doppler velocity cache, if coupled
∂κU[i,j,m]              optional cache
moments[i,j,*]          only when sources/diagnostics need them
```

### 3.5 Spatial decomposition only

Every rank owns all spectral indices for its local spatial tile. This is non-negotiable for the initial design:

```text
partition x: yes
partition y: yes
partition κ/θ or kx/ky: no
```

Spectral operations are local. Physical halos communicate full spectral columns.

---

## 4. Data model

## 4.1 `ProductField`

### 4.1.1 Purpose

`ProductField` is an Oceananigans-style field over a physical grid and an auxiliary coordinate grid.

For Ripple.jl, the primary `ProductField` is the wave-action density:

```julia
N[i, j, m, n]
```

where

```text
i, j  -> physical horizontal grid
m, n  -> spectral coordinate grid
```

### 4.1.2 Why not call it `SpectralField`?

`SpectralField` is too narrow. Ripple.jl's action field is spectral, but the abstraction can cover other auxiliary-coordinate fields:

```text
f(x, y, particle_radius)
I(x, y, wavelength, direction)
c(x, y, z, species)
a(x, y, vertical_mode)
```

`ProductField` names the mathematical concept: a field on a product of coordinate spaces. Wave-action fields can then be constructed by a wave-specific alias or constructor:

```julia
WaveActionField(grid, spectral_grid; kwargs...)
```

Internally, this returns a centered `ProductField`.

### 4.1.3 Proposed type sketch

A concrete first implementation for two physical horizontal dimensions and two coordinate dimensions:

```julia
struct ProductField{LX, LY, LZ,
                    Lξ, Lη,
                    PG, CG,
                    D, T,
                    Layout,
                    PI, CI,
                    BC,
                    Status,
                    Buffers} <: AbstractField{LX, LY, LZ, PG, T, 4}

    grid :: PG                  # physical Oceananigans grid
    coordinate_grid :: CG       # spectral / auxiliary coordinate grid

    data :: D                   # rank-4 array-like storage
    layout :: Layout            # maps logical to raw storage order

    physical_indices :: PI      # physical interior / view indices
    coordinate_indices :: CI    # auxiliary-coordinate interior / view indices

    boundary_conditions :: BC
    status :: Status
    communication_buffers :: Buffers
end
```

`ProductField` should subtype Oceananigans' `AbstractField` if that enables useful dispatch, but it should define its own key methods explicitly rather than relying on defaults that assume 3D physical fields.

### 4.1.4 Location semantics

Do not change the meaning of Oceananigans' `location(field)`. For a `ProductField`,

```julia
location(N)
```

returns the **physical** location only:

```julia
(Center, Center, Nothing)
```

The auxiliary coordinate locations are queried separately:

```julia
coordinate_location(N) = (Center, Center)
```

Combined location can be exposed with

```julia
product_location(N) = (Center, Center, Nothing, Center, Center)
```

or, after omitting inactive physical `z`,

```julia
active_product_location(N) = (Center, Center, Center, Center)
```

The physical-spectral flux fields then have natural locations:

```julia
N   :: ProductField{Center, Center, Nothing, Center, Center}
Fx  :: ProductField{Face,   Center, Nothing, Center, Center}
Fy  :: ProductField{Center, Face,   Nothing, Center, Center}
Fξ  :: ProductField{Center, Center, Nothing, Face,   Center}
Fη  :: ProductField{Center, Center, Nothing, Center, Face}
```

For a polar grid:

```text
ξ = κ, η = θ
```

For a Cartesian wavevector grid:

```text
ξ = kx, η = ky
```

### 4.1.5 Required interface

Define at minimum:

```julia
Base.size(f::ProductField)
Base.axes(f::ProductField)
Base.getindex(f::ProductField, i, j, m, n)
Base.setindex!(f::ProductField, value, i, j, m, n)
Base.parent(f::ProductField)
Base.eltype(f::ProductField)

architecture(f::ProductField)
grid(f::ProductField)                 # physical grid
physical_grid(f::ProductField)
coordinate_grid(f::ProductField)
location(f::ProductField)             # physical location
coordinate_location(f::ProductField)
product_location(f::ProductField)
interior(f::ProductField)
boundary_conditions(f::ProductField)
fill_halo_regions!(f::ProductField)
```

`grid(f)` should return the physical grid, matching Oceananigans expectations. If a combined grid is needed, provide:

```julia
product_grid(f) = ProductGrid(grid(f), coordinate_grid(f))
```

as a lightweight view.

---

## 4.2 Coordinate grids

### 4.2.1 General concept

The auxiliary coordinate grid is not merely metadata. It has centers, faces, topology, metrics, finite-volume cell measures, and boundary conditions.

```julia
abstract type AbstractCoordinateGrid end
abstract type AbstractSpectralGrid <: AbstractCoordinateGrid end
```

### 4.2.2 Polar wavevector grid

Production wave modeling will likely use a polar wavenumber-direction grid:

```julia
struct PolarWaveVectorGrid{FT, K, Θ, W, Topo} <: AbstractSpectralGrid
    κᶜ :: K
    κᶠ :: K
    θᶜ :: Θ
    θᶠ :: Θ
    weights :: W
    topology :: Topo       # usually (Bounded, Periodic)
end
```

The logical state is

```julia
N[i, j, nκ, nθ]
```

Polar grids are attractive for coupled CWCM runs because `Q` and `U` depend on `κ`, not `θ`.

### 4.2.3 Cartesian wavevector grid

A Cartesian grid in wavevector space is the cleanest grid for developing and verifying the Hamiltonian transport core:

```julia
struct CartesianWaveVectorGrid{FT, KX, KY, W, Topo} <: AbstractSpectralGrid
    kxᶜ :: KX
    kxᶠ :: KX
    kyᶜ :: KY
    kyᶠ :: KY
    weights :: W
    topology :: Topo       # usually (Bounded, Bounded)
end
```

The logical state is

```julia
N[i, j, nkx, nky]
```

This grid makes the canonical phase-space equation easiest to reason about:

```math
(x,y,k_x,k_y).
```

### 4.2.4 Frequency-direction grid

A frequency-direction grid is useful for compatibility with existing wave models and source packages:

```julia
struct FrequencyDirectionGrid{FT, F, Θ, K, W, Topo} <: AbstractSpectralGrid
    fᶜ :: F
    fᶠ :: F
    θᶜ :: Θ
    θᶠ :: Θ
    κᶜ :: K             # derived from dispersion relation / depth policy
    weights :: W
    topology :: Topo
end
```

This should not be the first transport grid for CWCM because the governing equation is naturally in `k` and because `U(x,κ)` is radial in wavevector space.

---

## 4.3 Layout: logical order vs memory order

### 4.3.1 Logical order

The model should expose physical-first indexing:

```julia
N[i, j, m, n]
```

This is the natural user-facing order:

```text
where are we physically?    i, j
which wave component?       m, n
```

### 4.3.2 Storage order

For performance, the default storage should likely put coordinate dimensions inner:

```julia
data[nθ, nκ, i, j]
```

for polar grids, while preserving logical access

```julia
N[i, j, nκ, nθ]
```

Reason: many important operations need the full local spectrum at one spatial cell:

```text
source terms
spectral moments
significant wave height
mean direction
peak diagnostics
pseudomomentum reductions
nonlinear interactions
DIA / exact quadruplet packages
CFL reductions over spectral velocities
semi-implicit source updates
```

With `data[nθ,nκ,i,j]`, `N[i,j,:,:]` is contiguous.

### 4.3.3 Layout types

```julia
abstract type AbstractProductLayout end

struct PhysicalFirstLayout <: AbstractProductLayout end
struct CoordinateFirstLayout <: AbstractProductLayout end
struct PolarCoordinateFirstLayout <: AbstractProductLayout end
struct TiledProductLayout{Bξ, Bη, Bx, By} <: AbstractProductLayout end
```

Meanings:

```julia
PhysicalFirstLayout()
# logical N[i, j, m, n]
# storage data[i, j, m, n]

CoordinateFirstLayout()
# logical N[i, j, m, n]
# storage data[m, n, i, j]

PolarCoordinateFirstLayout()
# logical N[i, j, nκ, nθ]
# storage data[nθ, nκ, i, j]

TiledProductLayout(...)
# future blocked / tiled storage for GPU and cache optimization
```

### 4.3.4 Accessors

All numerical code should access through layout-dispatched accessors:

```julia
@inline function getnode(N, i, j, m, n)
    return getnode(N.layout, N.data, i, j, m, n)
end
```

Example for polar coordinate-first storage:

```julia
@inline function getnode(::PolarCoordinateFirstLayout, data, i, j, m, n)
    @inbounds return data[n, m, i, j]
end
```

The numerical kernels should not use raw `data[...]` unless they are layout-specific performance kernels.

### 4.3.5 Loop and kernel ordering

For CPU loops and GPU thread organization, follow storage order. With `data[nθ,nκ,i,j]`, the innermost loop or fastest GPU thread dimension should usually be `nθ`.

Potential GPU mapping:

```text
threadIdx.x -> θ or flattened local spectral index
threadIdx.y -> κ or small spectral/physical tile
blockIdx    -> spatial tile
```

For physical advection, neighboring physical cells are separated by a full spectrum. This is acceptable because each neighbor's full spectral slab is contiguous. A kernel can load contiguous spectra for `(i,j)`, `(i+1,j)`, `(i-1,j)`, etc.

---

## 4.4 Boundary conditions and halos

### 4.4.1 Product boundary conditions

Use product boundary conditions rather than forcing spectral boundaries into Oceananigans' existing `FieldBoundaryConditions`:

```julia
struct ProductBoundaryConditions{PBC, CBC}
    physical :: PBC
    coordinate :: CBC
end
```

For polar spectra:

```julia
bcs = ProductBoundaryConditions(
    physical = (x = Periodic(), y = Periodic()),
    coordinate = (κ = NoFlux(), θ = Periodic()),
)
```

For Cartesian wavevector spectra:

```julia
bcs = ProductBoundaryConditions(
    physical = (x = Periodic(), y = Periodic()),
    coordinate = (kx = NoFlux(), ky = NoFlux()),
)
```

### 4.4.2 Physical vs coordinate halos

Physical halos:

```text
x/y halos may require MPI communication.
```

Coordinate halos:

```text
κ/θ or kx/ky halos are local boundary-condition halos only.
```

Halo fill should separate these actions:

```julia
fill_physical_halos!(N)      # may communicate
fill_coordinate_halos!(N)    # local only
fill_halo_regions!(N) = (fill_physical_halos!(N);
                         fill_coordinate_halos!(N))
```

### 4.4.3 Spatial decomposition rule

The partition must enforce:

```julia
partition.coordinate == Serial()
```

or equivalently:

```julia
@assert !is_distributed(coordinate_grid)
```

Each physical halo exchange sends complete spectral columns:

```text
x-halo: Hx × Ny_local × Nξ × Nη
y-halo: Nx_local × Hy × Nξ × Nη
```

This is expensive, but source terms and nonlinear interactions remain local.

---

## 5. Numerical method

## 5.1 Conservative finite-volume transport

The source-free equation should be implemented in conservative flux form:

```math
\partial_t N
+ \partial_x(F^x)
+ \partial_y(F^y)
+ \partial_\xi(F^\xi)
+ \partial_\eta(F^\eta)
= 0.
```

For Cartesian wavevector coordinates:

```text
ξ = kx, η = ky
```

and

```math
F^x = \dot{x} N,
\quad
F^y = \dot{y} N,
\quad
F^{k_x} = \dot{k}_x N,
\quad
F^{k_y} = \dot{k}_y N.
```

For polar coordinates:

```text
ξ = κ, η = θ
```

and the metric must be included. A conservative polar update should be written for metric-weighted cell content:

```math
\partial_t (J N)
+ \partial_x(J \dot{x} N)
+ \partial_y(J \dot{y} N)
+ \partial_\kappa(J \dot{\kappa} N)
+ \partial_\varphi(J \dot{\varphi} N)
= J S_N,
```

with `J = κ` for polar wavevector space.

## 5.2 Hamiltonian velocity construction

For Cartesian wavevector grids, prefer computing phase-space velocities from a single discrete Hamiltonian:

```math
\dot{x} = \partial_{k_x}\Omega,
\qquad
\dot{y} = \partial_{k_y}\Omega,
\qquad
\dot{k}_x = -\partial_x\Omega,
\qquad
\dot{k}_y = -\partial_y\Omega.
```

Discrete face velocities should use compatible finite-volume differences of the same `Ω`. This helps preserve the continuous cancellation

```math
\nabla_x \cdot \nabla_k \Omega
-
\nabla_k \cdot \nabla_x \Omega
= 0.
```

This is the right way to make the Hamiltonian structure visible in the numerical method and reduce spurious phase-space divergence.

## 5.3 Flux reconstruction

Initial milestones:

1. First-order upwind.
2. Flux-limited MUSCL / TVD.
3. Positivity-preserving high-order option.
4. WENO-like options only after the data layout and basic operators are settled.

First-order upwind should be the debugging scheme. The first success criteria are:

```text
constant N remains constant
total action is conserved in source-free periodic tests
phase-space pulse advection works without indexing errors
spectral boundary conditions behave as intended
```

## 5.4 Timestepping

Initial options:

```text
Forward Euler        debugging only
RK3                  robust explicit transport tests
AB2 / quasi-AB2       memory-lean production candidate
low-storage RK       production candidate if source coupling permits
```

Avoid timesteppers that require many full 4D field copies unless there is a strong accuracy or stability reason.

## 5.5 Positivity

Wave action should remain nonnegative. The source and transport updates should include positivity controls:

```julia
Nnew = max(0, Nnew)
```

as a last-resort guard, with better positivity-preserving fluxes and semi-implicit sinks developed later.

---

## 6. Dispersion, `QTransform`, and avoiding 5D `Q`

## 6.1 The challenge

The CWCM dispersion relation depends on

```math
U(x,\kappa) = \int_{-d(x)}^0 u^L(x,z) Q(x,z,\kappa)\,dz.
```

A naive implementation might allocate

```julia
Q[i, j, z, m, n]
```

where `m,n` are spectral indices. This must be avoided.

Reasons:

1. `Q` does **not** depend on direction `θ`, so `Q[i,j,z,κ,θ]` duplicates data across `θ`.
2. Even nonredundant `Q[i,j,z,κ]` can be larger than the action field for typical `Nz > Nθ`.
3. `Q` is a known analytic kernel. It should be an operator, not model state.
4. The main 4D action kernel should not repeat a vertical integral for every direction.

## 6.2 `QKernel` and `QTransform`

Introduce a matrix-free transform layer:

```julia
struct QKernel{FT}
    # empty or contains numerical-stability options
end

struct QTransform{Q, VG, Policy}
    kernel :: Q
    vertical_grid :: VG
    cache_policy :: Policy
end
```

The core operations are:

```julia
compute_doppler_velocity!(U, uᴸ, qtransform)
compute_pseudomomentum!(p, N, qtransform)
```

Both should evaluate the same `Q` kernel and use the same vertical finite-volume
cell geometry. The vertical transform should integrate the analytic `Q` kernel
exactly across each cell with CDF differences, not midpoint quadrature.

## 6.3 Stable evaluation of `Q`

Use nondimensional variables

```math
\mu = \kappa d,
\qquad
s = \frac{z+d}{d}.
```

Then

```math
Q = \frac{1}{d}\frac{2\mu\cosh(2\mu s)}{\sinh(2\mu)}.
```

For small `μ`, use the limit

```math
Q \to \frac{1}{d}.
```

For large `μ`, avoid overflow by using exponential forms rather than direct `cosh/sinh`:

```math
\frac{\cosh(a)}{\sinh(b)}
=
\frac{\exp(a-b) + \exp(-a-b)}{1 - \exp(-2b)},
```

with

```math
a = 2\kappa(z+d),
\qquad
b = 2\kappa d.
```

In deep water the kernel tends to

```math
Q = 2\kappa e^{2\kappa z}.
```

## 6.4 What to cache

Default coupled runs should cache the Doppler velocity:

```julia
Ux[i, j, m]
Uy[i, j, m]
```

where `m` is the radial `κ` index.

Optionally cache

```julia
∂κUx[i, j, m]
∂κUy[i, j, m]
```

or compute `∂κU` by finite differences from cached `U`.

Cache-size argument:

```text
N size        ~ Nx × Ny × Nκ × Nθ
U size        ~ 2 × Nx × Ny × Nκ
U/N ratio     ~ 2 / Nθ
```

For `Nθ = 36`, caching both components of `U` costs roughly 5.6% of the action field. This is a good trade. Caching `Q[i,j,z,κ]` is not generally a good trade.

## 6.5 Main action kernel

The main 4D action kernel should see only low-dimensional cached fields:

```julia
Ω(i,j,m,n) = σ(depth[i,j], κ[m])
           + kx[m,n] * Ux[i,j,m]
           + ky[m,n] * Uy[i,j,m]
```

No vertical integral and no `Q` evaluation should occur in the main `N[i,j,m,n]` transport kernel for coupled runs.

## 6.6 Computing pseudomomentum without storing `Q`

Use a two-stage reduction that exploits the independence of `Q` from direction.

First, reduce over direction:

```julia
Mˣ[i,j,m] = sum_n N[i,j,m,n] * kx[m,n] * wθ[n]
Mʸ[i,j,m] = sum_n N[i,j,m,n] * ky[m,n] * wθ[n]
```

Then transform radially and vertically:

```julia
pˣ[i,j,k] = sum_m Q(i,j,k,m) * Mˣ[i,j,m] * wκ[m]
pʸ[i,j,k] = sum_m Q(i,j,k,m) * Mʸ[i,j,m] * wκ[m]
```

`Q` is evaluated on the fly inside the second kernel. The result `p(x,z)` is an Oceananigans field on the ocean grid or a compatible surface-current coupling grid.

## 6.7 Cache policies

```julia
abstract type AbstractQStoragePolicy end

struct OnTheFlyQ <: AbstractQStoragePolicy end
struct CacheDopplerVelocity <: AbstractQStoragePolicy end
struct CacheDopplerVelocityAndDerivative <: AbstractQStoragePolicy end
struct PrecomputeQWeights <: AbstractQStoragePolicy end
```

Recommended default:

```julia
CacheDopplerVelocityAndDerivative()
```

Meaning:

```text
evaluate Q matrix-free when computing U
store U(x,κ)
store or derive ∂κU
compute Ω on the fly
do not store Q
```

Use `PrecomputeQWeights` only for small static-depth tests or flat-bottom benchmarks where the memory/runtime trade is demonstrably favorable.

---

## 7. Source terms

## 7.1 Source-term architecture

Source terms should be composable and concrete:

```julia
abstract type AbstractWaveSourceTerm end

struct SourceTermSet{T}
    terms :: T   # tuple, not Vector
end
```

Using a tuple lets the compiler specialize and unroll:

```julia
sources = SourceTermSet((
    LinearWindInput(...),
    Whitecapping(...),
    BottomFriction(...),
))
```

Each source term should expose a positive / damping split when possible:

```julia
source_split(i, j, m, n, model, source) -> (S⁺, D)
```

Then a semi-implicit source update can be

```math
N^{n+1} = \frac{N^* + \Delta t S^+}{1 + \Delta t D}.
```

This helps with stiff sink terms and positivity.

## 7.2 Source categories

### Pointwise sources

Examples:

```text
linear wind input
simple exponential growth
simple whitecapping sink
bottom friction with local depth
ice damping
```

These may be fused into the main update kernel.

### Moment-dependent sources

Examples:

```text
saturation-based whitecapping
mean-frequency-dependent dissipation
wind-sea / swell separation logic
source packages requiring m0, m1, mean direction, peak proxy
```

These require a spectral reduction kernel before source evaluation.

### Spectrum-column sources

Examples:

```text
DIA nonlinear quadruplet interactions
exact or approximate Boltzmann integral packages
triad interactions for shallow water
```

These likely require separate kernels and may strongly prefer coordinate-inner storage.

## 7.3 Clean-room implementation policy

Existing models contain source packages with long histories. Ripple.jl should use them to learn:

```text
which source packages matter
what inputs and moments they need
how diagnostics are organized
which validation cases reveal bugs
```

But GPL-licensed source code should not be translated line-by-line into Ripple.jl unless Ripple.jl's license strategy explicitly permits that. Treat WAM and SWAN as sources of scientific and validation insight, not as copy-paste sources. ecWAM's Apache-2.0 license makes it more permissive for architectural inspiration, but clean implementation is still preferable.

---

## 8. Parallelism and performance

## 8.1 Spatial-only decomposition

Parallelization rule:

```text
Decompose physical x and y.
Do not decompose spectral coordinates.
```

Every rank/GPU owns:

```text
local physical tile × complete spectrum
```

This is essential because source terms, nonlinear interactions, moments, and CWCM coupling need a complete local spectrum.

## 8.2 Halo exchange

With coordinate-inner storage such as

```julia
data[nθ, nκ, i, j]
```

physical halo packing sends contiguous spectral chunks for each boundary cell.

Future optimization:

```text
1. start asynchronous halo exchange
2. compute interior physical cells
3. wait for halos
4. compute boundary cells
```

## 8.3 Kernel fusion

Organize kernels by dependency pattern, not by Fortran subroutine tradition.

### Transport tendency kernel

A fused transport kernel should compute:

```text
Ω or discrete Ω differences
phase-space face velocities
spatial fluxes
spectral fluxes
transport tendency
optional simple explicit sources
```

### Spectral reduction kernels

Needed for:

```text
m0, m1, m2
significant wave height
mean direction
mean/peak frequency proxies
source-term moments
pseudomomentum moments Mx, My
CFL maxima
```

### Source update kernels

Needed when sources depend on moments or spectrum-column physics.

### Q-transform kernels

Needed only when coupled to sheared currents:

```text
compute U(x,κ)
compute ∂κU
compute p(x,z)
```

### Halo pack/unpack kernels

Needed if Oceananigans' distributed halo machinery cannot directly handle `ProductField`.

## 8.4 Memory policy

Minimum full-size fields:

```text
N                     action density
G or previous tendency
stage field if required by timestepper
```

Avoid full-size product-space caches unless profiling proves they pay.

For local tile size `Nx × Ny × Nκ × Nθ`, a single `Float32` action field costs:

```math
4 N_x N_y N_\kappa N_\varphi \text{ bytes}
```

For `Nκ = 48`, `Nθ = 36`, `Nspec = 1728`. Memory growth is brutal; do not casually allocate multiple product-space temporaries.

## 8.5 Precision

Default production action fields may use `Float32` on GPUs, but all literal constants and source-term parameters must be carefully typed. Mixed precision should be explicit. Summations for moments and pseudomomentum may require compensated or higher-precision accumulation in some regimes.

---

## 9. Comparison with existing models and source-code audit plan

## 9.1 Summary table

| Model | Repository | License / availability | What Ripple.jl should learn | What Ripple.jl should avoid |
|---|---|---:|---|---|
| WAVEWATCH III | <https://github.com/NOAA-EMC/WW3> | NOAA public GitHub repository; review repo license before reuse | Operational source-term modularity, action-balance workflow, propagation/source splitting, regression tests, output/restart conventions, unstructured-grid ideas | Reproducing gather/scatter or legacy Fortran decomposition patterns that conflict with spatial-only decomposition |
| WAM Cycle 7 | <https://github.com/myWAveModel/WAM> | WAM is described by natESM as GPLv3; verify repo license files | Third-generation wave-model organization, SWAMP tests, source-term output, OASIS coupling, MPI domain decomposition, ST6/BYDBR physics | Copying GPL code into a permissive Julia package; line-by-line translations |
| ecWAM | <https://github.com/ecmwf-ifs/ecwam> | Apache-2.0 according to repo README | GPU offload strategy, standalone/coupled architecture, Fortran performance organization, coupling to IFS/NEMO, Loki/OpenACC transformation strategy | Blindly reproducing Fortran loop structure if Julia kernels can fuse more naturally |
| SWAN | <https://gitlab.tudelft.nl/citg/wavemodels/swan> | GPLv3 according to TU Delft SWAN page | Nearshore physics, coastal boundary conditions, shallow-water source terms, triads, depth-induced breaking, validation cases | Copying GPL source code; making coastal complexity block the first deep-water core |
| PiCLES.jl | <https://github.com/mochell/PiCLES.jl> | Apache-2.0 according to repo `LICENSE` | Julia wave-model structure, Oceananigans-inspired simulation workflow, particle-in-cell gridding/remeshing ideas, idealized wind-forcing examples, bulk sea-state comparisons against WW3 | Confusing PiCLES's parametric/PIC state with Ripple.jl's Eulerian spectral action density; using it as a spectral reference model |
| Oceananigans.jl | <https://github.com/CliMA/Oceananigans.jl> | MIT-style Julia package; verify repo license | Architecture dispatch, fields, halos, finite-volume operators, GPU kernels, diagnostics, `Simulation` workflow | Forcing wave action into existing 3D `Field` when product-coordinate locations require a new abstraction |

## 9.2 WAVEWATCH III audit

Clone:

```bash
git clone https://github.com/NOAA-EMC/WW3.git
cd WW3
git rev-parse HEAD
```

Initial files and directories to inspect:

```text
model/src/w3wavemd.F90       high-level wave-model driver
model/src/w3srcemd.F90       source-term integration wrapper
model/src/w3src*.F90         source-term packages
model/src/w3pro*.F90         propagation-related modules, if present
model/src/w3gath*.F90        gather/scatter patterns, if present
model/src/w3iogrmd.F90       grid / restart / I/O patterns, if relevant
regtests/                    validation and regression tests
manual/ docs/                equations, switches, source packages
```

Useful commands:

```bash
rg -n "SUBROUTINE W3WAVE|SUBROUTINE W3SRCE|W3GATH|W3SCAT|NSPEC|NK|NTH" model/src
rg -n "ST4|ST6|DIA|NL[0-9]|WHITE|FRIC|ICE|SOURCE" model/src
rg -n "MPI|DOMAIN|HALO|UNST|SMC|PDLIB|ParMetis" model/src regtests manual docs
```

Deliverables from WW3 audit:

1. Call graph of one full time step.
2. Source-term package dependency table.
3. List of diagnostic moments and output fields.
4. Validation cases relevant to Ripple.jl milestones.
5. Notes on where WW3 decomposes, gathers, or scatters spectra.
6. Performance lessons, especially where source terms dominate runtime.

Value to Ripple.jl:

```text
Use WW3 to understand operational completeness and validation.
Do not inherit its data layout or decomposition if they conflict with ProductField and spatial-only decomposition.
```

## 9.3 WAM Cycle 7 audit

Clone:

```bash
git clone https://github.com/myWAveModel/WAM.git
cd WAM
git rev-parse HEAD
git branch -a
```

Initial files and directories to inspect:

```text
README.md
src/chief/wamodel.f90        model supervisor / integration flow
src/chief/initmdl.f90        initialization
src/mod/wam_timopt_module.f90 time options
src/mod/wam_user_module.f90  user configuration
src/mod/*source*             source-term infrastructure, if present
src/print/*netcdf*           output conventions
SWAMPtest/                   SWAMP validation tests
const/WAM_User               user configuration examples
```

Useful commands:

```bash
rg -n "CHIEF|PROPAG|SOURCE|SNET|DIA|ST6|BYDBR|OASIS|MPI|HALO|DOMAIN" src
rg -n "SWAMP|fetch|duration|JONSWAP|swell|current" SWAMPtest const
rg -n "NETCDF|SOURCE OUTPUT|PARAMETER|SPECTR" src const
```

Deliverables:

1. WAM time-step workflow summary.
2. MPI domain-decomposition notes.
3. Source-term package dependency table.
4. SWAMP test mapping to Ripple.jl validation cases.
5. Coupling notes: OASIS inputs/outputs and flux conventions.
6. Legal notes: GPL contamination risks.

Value to Ripple.jl:

```text
Use WAM to learn mature wave-model organization, source tests, SWAMP cases, coupling workflows.
Do not line-by-line translate GPL code unless Ripple.jl chooses a GPL-compatible path.
```

## 9.4 ecWAM audit

Clone:

```bash
git clone https://github.com/ecmwf-ifs/ecwam.git
cd ecwam
git rev-parse HEAD
```

Initial focus:

```text
README.md                     installation, license, GPU notes
source / propagation modules  exact paths to be identified after clone
gpu / OpenACC / Loki hooks    transformation and kernel-offload patterns
tests                         standalone validation
coupling interfaces           IFS/NEMO/FESOM-related code paths
```

Useful commands:

```bash
rg -n "Loki|OpenACC|GPU|scc|hoist|stack|source-term|propagation|NEMO|FESOM|coupl" .
rg -n "DIA|ST4|ST6|SNL|whitecapping|input|dissipation" .
rg -n "MPI|domain|decomposition|halo|exchange" .
```

Deliverables:

1. GPU offload strategy notes.
2. Single-column-coalesced / source-term performance lessons.
3. Coupled-mode interface summary.
4. Source/propagation decomposition summary.
5. License compatibility note.

Value to Ripple.jl:

```text
ecWAM is especially valuable for performance and GPU strategy.
Its Apache-2.0 license makes architectural learning less legally fraught than GPL code.
```

## 9.5 SWAN audit

Clone:

```bash
git clone https://gitlab.tudelft.nl/citg/wavemodels/swan.git
cd swan
git rev-parse HEAD
```

Initial focus:

```text
README / documentation
source-term routines
nearshore boundary-condition routines
triads, depth-induced breaking, bottom friction
input command parser
test cases
```

Useful commands:

```bash
rg -n "TRIAD|BREAK|BOTTOM|FRICTION|GEN3|whitecap|quadruplet|DIA|boundary" .
rg -n "current|refraction|depth|stationary|nonstationary" .
rg -n "TEST|CASE|validation|benchmark" .
```

Deliverables:

1. Coastal physics map.
2. Boundary-condition taxonomy.
3. Shallow-water validation cases.
4. Source-term dependency table.
5. GPL risk notes.

Value to Ripple.jl:

```text
Use SWAN for eventual coastal and shallow-water physics.
Do not let SWAN-level coastal complexity delay the first Hamiltonian deep/open-water core.
```


## 9.6 PiCLES.jl audit

Repository:

```bash
git clone https://github.com/mochell/PiCLES.jl.git
cd PiCLES.jl
git rev-parse HEAD
```

PiCLES is not a spectral action-density model like Ripple.jl. It is a Particle-in-Cell wave model for efficient sea-state estimates in Earth-system models. Its state is intentionally much smaller than a spectral wave model: the JAMES paper describes evolving a parametric spectrum's peak wavenumber vector and total wave energy, reducing state-vector size by roughly 50--200 relative to standard spectral models, and comparing idealized cases to WW3.

Initial files and directories to inspect:

```text
README.md                         installation, minimal example, Oceananigans-inspired structure
LICENSE                           Apache-2.0 license
examples/                         examples, especially homogeneous wind boxes
benchmark/                        throughput and allocation lessons
src/Models/                       WaveGrowth2D and model organization
src/ParticleSystems/              particle wave ODE systems
src/ParticleMesh/                 particle-to-grid projection and gridded diagnostics
src/FetchRelations/               fetch-limited wind-sea relations and initialization
src/Simulations/                  Simulation-style workflow
```

Useful commands:

```bash
rg -n "WaveGrowth2D|Simulation|ParticleDefaults|MinimalWindsea|FetchRelations" src examples benchmark test
rg -n "wind|vortex|hurricane|stationary|WW3|SWAMP|homogenous|homogeneous" .
rg -n "threads|distributed|process|allocation|BenchmarkTools|Profile" examples benchmark src
```

Deliverables:

1. Identify which PiCLES examples can be ported as Ripple.jl examples using full spectra rather than particles.
2. Extract idealized wind-forcing patterns: homogeneous wind, half-domain wind, changing wind direction, stationary rotating vortex, and moving severe-storm / hurricane-like winds.
3. Compare PiCLES's Julia/Oceananigans-inspired simulation organization with Ripple.jl's proposed `SpectralWaveModel` and `Simulation` integration.
4. Build a table of PiCLES bulk diagnostics (`Hs`, peak direction, group velocity, energy) that Ripple.jl should output for cross-comparison.
5. Identify source/dissipation parameterizations used in PiCLES examples and decide which minimal clean-room versions Ripple.jl needs for the examples suite.
6. Record performance lessons: allocation control, gridded output, forcing-field interpolation, threading, and benchmark design.

Value to Ripple.jl:

```text
PiCLES is the most readily runnable wave-model comparison source because it is Julia, public, Apache-2.0, and already uses an Oceananigans-like simulation structure.
Use it for example design, bulk sea-state comparison, forcing-pattern tests, and Julia performance lessons.
Do not use it as a spectral action-equation reference; it is a different reduced-order/PIC model.
```

## 9.7 Oceananigans audit

Clone:

```bash
git clone https://github.com/CliMA/Oceananigans.jl.git
cd Oceananigans.jl
git rev-parse HEAD
```

Initial files and directories to inspect:

```text
src/Fields/abstract_field.jl
src/Fields/field.jl
src/Fields/
src/Grids/
src/BoundaryConditions/
src/DistributedComputations/
src/Operators/ or src/AbstractOperations/
src/TimeSteppers/
src/Simulations/
src/OutputWriters/
```

Useful commands:

```bash
rg -n "abstract type AbstractField|struct Field|Abstract4DField|location\(" src/Fields src/AbstractOperations
rg -n "fill_halo_regions|halo|communication|Distributed" src
rg -n "@kernel|launch!|KernelAbstractions|architecture" src
rg -n "Simulation|time_step!|compute_tendencies!|prognostic_fields" src
```

Deliverables:

1. Exact interface needed for `ProductField` to feel field-like.
2. Which `AbstractField` methods can be reused safely.
3. Which methods assume 3D physical locations.
4. Halo-fill extension plan.
5. Output and diagnostics integration plan.
6. Kernel-launch style guide for Ripple.jl.

Value to Ripple.jl:

```text
Oceananigans is the style guide and integration target.
Ripple.jl should feel like an Oceananigans model even though its action field is a ProductField.
```

---

## 10. Ripple.jl package structure

Recommended source tree:

```text
Ripple.jl/
  Project.toml
  src/
    Ripple.jl

    ProductFields/
      ProductFields.jl
      product_field.jl
      layouts.jl
      boundary_conditions.jl
      halos.jl
      set.jl

    CoordinateGrids/
      CoordinateGrids.jl
      spectral_grids.jl
      polar_wavevector_grid.jl
      cartesian_wavevector_grid.jl
      frequency_direction_grid.jl
      finite_volume_integration.jl

    Operators/
      Operators.jl
      differences.jl
      interpolations.jl
      metrics.jl
      fluxes.jl
      hamiltonian_velocities.jl

    Models/
      Models.jl
      spectral_wave_model.jl
      time_step.jl
      tendencies.jl
      clocks.jl

    Coupling/
      Coupling.jl
      q_kernel.jl
      q_transform.jl
      doppler_velocity.jl
      pseudomomentum.jl
      oceananigans_coupling.jl

    Sources/
      Sources.jl
      source_term_set.jl
      linear_wind_input.jl
      whitecapping.jl
      bottom_friction.jl
      nonlinear_interactions.jl

    Diagnostics/
      Diagnostics.jl
      moments.jl
      bulk_statistics.jl
      cfl.jl

    InitialConditions/
      InitialConditions.jl
      jonswap.jl
      gaussian_wave_packet.jl

    Output/
      Output.jl
      netcdf.jl
      jld2.jl

  test/
    product_fields/
      indexing.jl
      layouts.jl
      halos.jl
      boundary_conditions.jl
      cpu_gpu_parity.jl
    coordinate_grids/
      polar_wavevector_grid.jl
      cartesian_wavevector_grid.jl
      finite_volume_integration.jl
    operators/
      differences.jl
      finite_volume_fluxes.jl
      hamiltonian_velocities.jl
    transport/
      constant_action.jl
      analytic_translation.jl
      cartesian_hamiltonian.jl
      polar_metric.jl
      manufactured_solutions.jl
    coupling/
      q_kernel_limits.jl
      doppler_velocity.jl
      pseudomomentum.jl
      hasselmann_column.jl
      oceananigans_fields.jl
    sources/
      relaxation_to_spectrum.jl
      wind_input.jl
      whitecapping.jl
      bottom_friction.jl
      positivity.jl
    integration/
      simulation_interface.jl
      oceananigans_prescribed_current.jl
      oceananigans_coupled_column.jl
      distributed_spatial_halos.jl
    validation/
      external_models/
        swan/
        picles/
        wam/
        ww3/
        ecwam/
      regression_data/
    examples_smoke/
    performance/

  examples/
    00_product_field_basics.jl
    01_free_swell_packet_cartesian.jl
    02_free_swell_packet_polar.jl
    03_prescribed_current_refraction.jl
    04_source_only_fetch_limited_growth.jl
    05_hasselmann_inertial_oscillation.jl
    06_cwcm_q_transform_sheared_current.jl
    07_oceananigans_prescribed_current_coupling.jl
    08_stationary_vortex_wind_picles_inspired.jl
    09_moving_hurricane_wind_forcing.jl
    10_coupled_oceananigans_demo.jl

  docs/
```

---

## 11. Public API sketch

Absent optional model components should use `nothing`, matching Oceananigans
and Breeze conventions. In particular, `advection = nothing` means no
transport/source-only evolution, while omitted `sources` and `coupling` default
to `nothing`. Legacy no-op sentinels may be accepted as compatibility inputs,
but model state should canonicalize them to `nothing`.

## 11.1 Uncoupled polar wave model

```julia
using Oceananigans
using Ripple

arch = GPU()

grid = RectilinearGrid(arch;
                       size = (Nx, Ny),
                       x = (0, Lx),
                       y = (0, Ly),
                       topology = (Periodic, Periodic, Flat))

spectral_grid = PolarWaveVectorGrid(Float32;
                                    κ = exponential_range(κmin, κmax, Nκ),
                                    θ = range(0, 2π; length = Nθ + 1)[1:end-1],
                                    topology = (Bounded, Periodic))

N = WaveActionField(grid, spectral_grid;
                    layout = PolarCoordinateFirstLayout(),
                    boundary_conditions = default_wave_action_bcs(grid, spectral_grid),
                    halo = (Hx, Hy, Hκ, Hθ))

sources = SourceTermSet((
    LinearWindInput(),
    WhitecappingKomen(),
    BottomFriction(),
))

model = SpectralWaveModel(; grid,
                            spectral_grid,
                            action = N,
                            advection = HamiltonianFiniteVolume(order = 1),
                            sources,
                            coupling = nothing,
                            timestepper = :RK3)

set!(model, N = JONSWAPSpectrum(Hs = 2, Tp = 8, direction = π/4))

simulation = Simulation(model; Δt = 60, stop_time = 6hours)
run!(simulation)
```

## 11.2 Source-free Cartesian Hamiltonian test

```julia
spectral_grid = CartesianWaveVectorGrid(Float64;
                                        kx = range(-kmax, kmax, Nkx),
                                        ky = range(-kmax, kmax, Nky),
                                        topology = (Bounded, Bounded))

model = SpectralWaveModel(; grid,
                            spectral_grid,
                            advection = HamiltonianFiniteVolume(order = 1),
                            timestepper = :RK3)
```

## 11.3 Prescribed sheared-current CWCM test

```julia
current = PrescribedLagrangianMeanCurrent(u = ShearProfile(...),
                                          v = zero,
                                          interpretation = LagrangianMeanVelocity())

qtransform = QTransform(QKernel(Float64),
                        VerticalFiniteVolumeGrid(; faces=z_faces),
                        CacheDopplerVelocityAndDerivative())

coupling = CWCMPrescribedCurrentCoupling(current, qtransform)

model = SpectralWaveModel(; grid,
                            spectral_grid,
                            coupling,
                            advection = HamiltonianFiniteVolume(order = 1),
                            timestepper = :RK3)
```

## 11.4 Future coupled Oceananigans model

```julia
ocean = NonhydrostaticModel(; grid = ocean_grid,
                              closure = ...,
                              buoyancy = ...)

waves = SpectralWaveModel(; grid = surface_grid(ocean.grid),
                            spectral_grid,
                            sources,
                            coupling = OceananigansLagrangianMeanCoupling(ocean),
                            advection = HamiltonianFiniteVolume())

coupled = CoupledWaveCurrentModel(; ocean, waves,
                                    coupling_scheme = SplitRK3())

simulation = Simulation(coupled; Δt = 30, stop_time = 1day)
run!(simulation)
```

---

## 12. Implementation roadmap

## Milestone A: `ProductField` and coordinate grids

Deliver:

```text
ProductField rank-4 storage
logical N[i,j,m,n] indexing
physical and coordinate locations
coordinate-inner layout
physical-first debug layout
polar and Cartesian spectral grids
coordinate boundary conditions
local coordinate halo fill
basic set! support
CPU/GPU allocation
```

Tests:

```text
indexing correctness for both layouts
location and coordinate_location tests
halo fill for periodic θ
NoFlux κ boundary behavior
CPU/GPU parity
```

## Milestone B: spectral finite-volume integration and diagnostics

Deliver:

```text
spectral weights
m0, m1, mean direction
significant wave height
JONSWAP initialization
Gaussian wave packet initialization
```

Tests:

```text
analytic finite-volume spectrum integration
directional moment symmetry
isotropic spectrum has zero mean direction vector
```

## Milestone C: source-free Cartesian Hamiltonian transport

Deliver:

```text
CartesianWaveVectorGrid
first-order upwind finite-volume transport
periodic physical boundaries
bounded spectral boundaries
RK3 / Forward Euler
CFL diagnostic
```

Tests:

```text
constant N preserved
total action conserved
phase-space advection with constant velocities
refraction by prescribed Ω gradient
```

## Milestone D: polar transport

Deliver:

```text
PolarWaveVectorGrid conservative metric update
κ metric J = κ
directional periodicity
κ boundary policies
```

Tests:

```text
polar finite-volume consistency
action conservation in source-free periodic θ tests
agreement with Cartesian test in isotropic/simple cases
```

## Milestone E: `QTransform`, `U`, and `p`

Deliver:

```text
stable QKernel
matrix-free U(x,κ) computation
optional ∂κU
matrix-free pseudomomentum p(x,z)
two-stage directional moment reduction for p
```

Tests:

```text
∫Q dz = 1
small-κd limit Q -> 1/d
deep-water limit Q -> 2κ exp(2κz)
barotropic current gives U = uᴸ
∫p dz = ∫∫ N k dk
U and p use matching finite-volume cell geometry
```

## Milestone F: source terms

Deliver:

```text
SourceTermSet
semi-implicit source update
RelaxationToSpectrum for analytic Hasselmann tests
linear / exponential wind input
simple whitecapping or saturation dissipation
bottom friction
moment-dependent source infrastructure
minimal wind-input + dissipation package sufficient for fetch, vortex, and hurricane examples
```

Tests:

```text
source-only single-column growth/decay
positivity under stiff sinks
moment dependency correctness
comparison to clean analytic solutions
```

## Milestone G: Oceananigans-style simulation integration

Deliver:

```text
SpectralWaveModel time_step!
compute_tendencies!
fields / prognostic_fields interfaces
callbacks
output writers for bulk diagnostics
Simulation compatibility
```

Tests:

```text
minimal Simulation run
diagnostic scheduling
restart-like save/load prototype
```

## Milestone H: distributed spatial decomposition

Deliver:

```text
spatial-only distributed ProductField
physical halo exchange of full spectral slabs
blocking halo fill
CFL global reduction
```

Future:

```text
async halo exchange + interior/boundary split
multi-GPU benchmarks
```

## Milestone I: test, validation, and examples suite

Deliver:

```text
comprehensive unit tests for ProductField, coordinate grids, operators, and QTransform
analytic transport and manufactured-solution tests
Hasselmann inertial-oscillation integration test from the CWCM paper
Oceananigans prescribed-current and coupled-column tests
example smoke tests for all examples
PiCLES-inspired bulk wind-forcing examples
SWAN/WAM/WW3/ecWAM optional external-comparison scripts
performance regression harness for layouts, reductions, QTransform, and transport
```

---


## 13. Test suite, validation suite, and examples suite

Ripple.jl should have an unusually strong test suite from the beginning. The action equation is high-dimensional, coupled, and easy to get subtly wrong: a sign error in spectral refraction, an inconsistent metric factor, a silently incorrect halo fill, or an omitted `∂κU` term can produce plausible-looking waves while violating the actual dynamics. The test suite should therefore be layered: small unit tests, property/invariant tests, analytic integration tests, Oceananigans integration tests, optional external-model comparisons, example smoke tests, and performance/regression tests.

The test suite should be organized so that ordinary CI is fast, deterministic, and self-contained, while slower external-model comparisons and GPU/distributed runs are opt-in.

Recommended test tags:

```text
unit             fast, pure Julia, always in CI
integration      model-level, still self-contained
examples         smoke tests for examples
coupling         Oceananigans coupling tests
validation       analytic and reference-result tests
external         optional tests requiring SWAN/WAM/WW3/ecWAM/PiCLES
slow             long tests, not default CI
gpu              GPU tests, CI only when a GPU runner is available
distributed      MPI / multi-rank tests, optional CI
performance      benchmark/regression tests, not pass/fail physics tests
```

### 13.1 Test-suite philosophy

The suite should test four different kinds of truth:

1. **Indexing truth:** `ProductField`, coordinate grids, layouts, halos, and locations mean exactly what they say.
2. **Discrete conservation truth:** source-free transport conserves action and preserves constants to machine precision where the discrete scheme should do so.
3. **Analytic-solution truth:** special cases of the action equation and CWCM coupling match closed-form or manufactured solutions.
4. **Model-comparison truth:** in cases that existing models can run, Ripple.jl agrees on bulk outputs and qualitative structures within expected tolerance, without relying on external models for every CI run.

External wave models should be used mainly to build confidence and discover missing physics. They should not be the only definition of correctness, because they have different numerics, source packages, grids, conventions, and coupling assumptions.

### 13.2 Unit tests

#### ProductField tests

Required tests:

```text
logical indexing N[i,j,m,n] works for all supported layouts
physical-first and coordinate-first storage return identical logical values
`location(N)` returns only physical location
`coordinate_location(N)` returns auxiliary-coordinate location
`product_location(N)` combines both without changing Oceananigans semantics
`interior(N)` excludes physical and coordinate halos correctly
CPU and GPU allocations have identical logical axes
set! works with functions of x, y, ξ, η
```

Specific layout tests:

```julia
Nphys = WaveActionField(grid, spectral_grid; layout = PhysicalFirstLayout())
Nspec = WaveActionField(grid, spectral_grid; layout = PolarCoordinateFirstLayout())

set!(Nphys) do x, y, κ, θ
    f(x, y, κ, θ)
end

set!(Nspec) do x, y, κ, θ
    f(x, y, κ, θ)
end

@test all(Nphys[i,j,m,n] ≈ Nspec[i,j,m,n] for i,j,m,n in interior_indices(Nphys))
```

#### Coordinate-grid tests

Required tests:

```text
κ/θ centers and faces have expected monotonicity and topology
θ is periodic and κ is bounded for PolarWaveVectorGrid
kx/ky topology is bounded for CartesianWaveVectorGrid
finite-volume cell measures integrate cell-average analytic functions
polar weights include the κ metric where appropriate
frequency-direction grids produce consistent κ values for flat-bottom no-current tests
```

#### Operator tests

Required tests:

```text
δx, δy, δξ, δη return expected finite differences on linear and quadratic test fields
face interpolation preserves constants
finite-volume divergence of a constant flux is zero
periodic coordinate differences wrap correctly
NoFlux coordinate boundaries produce zero boundary flux
polar metric divergence is conservative for constant N and zero source
```

### 13.3 Matrix-free QTransform tests

The CWCM kernel tests should be a separate group because they are foundational for coupling.

Required tests:

```text
∫ Q dz = 1 for many κd values and vertical grids
small-κd limit Q -> 1/d
large-κd / deep-water limit Q -> 2κ exp(2κz)
Q evaluation avoids overflow for large κd
barotropic current gives U(x,κ) = uᴸ(x) for every κ
piecewise-constant vertical current gives U from exact Q cell integrals
∫ p dz = ∫∫ N k dk for arbitrary positive spectra
U and p transforms use matching finite-volume cell geometry
```

For regression stability, include edge-case grids:

```text
very small κd
very large κd
coarse Nz
nonuniform vertical grid
shallow flat bottom
deep flat bottom
variable depth d(x,y)
```

### 13.4 Source-free action-transport correctness tests

The first correctness tests should not require wind input, whitecapping, or nonlinear interactions. They should isolate the transport core.

#### Constant-action preservation

With periodic physical boundaries, no source, no current, and flat depth:

```math
N(x,y,k,t) = N_0(k)
```

or a fully constant field should remain unchanged.

Acceptance criteria:

```text
max norm of N(t)-N(0) is exactly zero or roundoff-level for constant N
integrated action is conserved to timestepper tolerance
no spectral boundary contamination occurs
```

#### Exact translation test

Choose a dispersion/velocity configuration that produces constant phase-space velocity, for example by using a manufactured Hamiltonian

```math
	ilde{\Omega}(x,y,k_x,k_y) = c_x k_x + c_y k_y - a_x x - a_y y,
```

so that

```math
rac{dx}{dt} = (c_x,c_y),
rac{dk}{dt} = (a_x,a_y).
```

On periodic domains in all active dimensions, a Gaussian packet should translate exactly up to numerical diffusion:

```math
N(x,y,k_x,k_y,t) = N_0(x-c_x t, y-c_y t, k_x-a_x t, k_y-a_y t).
```

This is the cleanest 4D integration test of the transport solver.

#### Ray-equation comparison

Initialize a narrow packet in phase space and compare the packet centroid to the ODE ray equations:

```math
rac{dx}{dt} = 
abla_k \\Omega,
rac{dk}{dt} = -
abla_x \\Omega.
```

Use simple prescribed depth or current gradients where the ODE can be integrated accurately with `OrdinaryDiffEq.jl`. This verifies refraction signs and the Hamiltonian construction.

#### Polar-vs-Cartesian consistency

For isotropic or near-isotropic spectra over a narrow radial band, compare polar and Cartesian solvers after transforming diagnostics to bulk quantities:

```text
total action
mean wavevector
bulk energy
centroid trajectory
```

This should not be a bitwise test. It is a convergence and consistency test.

### 13.5 Hasselmann / CWCM analytic integration test

The Hasselmann problem from the CWCM paper should become one of Ripple.jl's flagship examples and a required integration test.

The setup is horizontally uniform, flat-bottom, initially motionless ocean with no waves and no currents. The wave action is forced by relaxation to an equilibrium spectrum:

```math
N^\circ = \alpha (N_\star(k) - N).
```

Because the setup is horizontally uniform, the transport operator vanishes and the exact solution is

```math
N(k,t) = (1 - e^{-\alpha t}) N_\star(k).
```

The pseudomomentum is

```math
p(z,t) = (1 - e^{-\alpha t}) p_\star(z)\,\hat{x},
```

where

```math
p_\star(z) = \iint Q(z,\kappa) N_\star(k) k \, dk.
```

With traditional Coriolis `f \hat{z}`, the Lagrangian current obeys

```math
\partial_t u^L + f \hat{z} \times u^L = \partial_t p,
```

and the analytic solution is

```math
u^L + i v^L =
\frac{\alpha p_\star}{\alpha - i f}
\left(e^{-i f t} - e^{-\alpha t}\right).
```

Long-time Lagrangian kinetic energy is

```math
\frac{1}{2}|u^L|^2 =
\frac{1}{2}\frac{\alpha^2 p_\star^2}{\alpha^2 + f^2}.
```

The test should verify:

```text
N(t) matches (1-exp(-αt))N⋆
p(z,t) matches (1-exp(-αt))p⋆(z)
uᴸ(z,t), vᴸ(z,t) match the analytic inertial-oscillation solution
Lagrangian kinetic energy approaches the analytic long-time value
energy accounting separates wave energy and inertial-oscillation work as in the paper
```

This test is also the natural bridge to Oceananigans integration: represent the current with Oceananigans fields, drive it with `p_t`, and compare the vertical column solution against the analytic formula. This exercises `QTransform`, `p(x,z)`, Coriolis integration, and wave-current coupling without needing a full turbulent ocean simulation.

Recommended implementation objects:

```julia
source = RelaxationToSpectrum(α, Nstar)
coupling = HasselmannColumnCoupling(f, qtransform)
example = examples/05_hasselmann_inertial_oscillation.jl
```

### 13.6 Source-term tests

The hurricane/vortex examples require at least minimal wind input and dissipation. The first source suite should be deliberately simple, then extended toward operational packages.

#### Minimal source terms required for examples

```julia
RelaxationToSpectrum(α, Nstar)
LinearWindInput(...)
ExponentialWindInput(...)
SimpleWhitecapping(...)
SaturationDissipation(...)
BottomFriction(...)
```

`RelaxationToSpectrum` is mandatory for the Hasselmann test. `LinearWindInput` or `ExponentialWindInput` plus `SimpleWhitecapping` is enough for early fetch-limited, vortex, and hurricane examples. More sophisticated ST4/ST6-style source terms can come later.

Required source tests:

```text
RelaxationToSpectrum matches exact exponential relaxation
pure damping decays exponentially and preserves positivity
semi-implicit source update remains positive under stiff sinks
source-only single-column tests produce monotone energy growth toward equilibrium
moment-dependent sources read exactly the moments computed by Diagnostics
wind input vanishes or changes sign appropriately for opposing swell, depending on formulation
bottom friction strengthens in shallow water and vanishes when disabled
```

#### Fetch-limited growth tests

Use one-dimensional or two-dimensional uniform wind over initially calm water. Compare bulk growth curves against:

```text
analytic behavior of the chosen simple source model
PiCLES bulk Hs / energy curves for similar homogeneous wind tests, where appropriate
WW3/WAM/SWAN optional external runs once source packages are comparable
```

These should be treated as source-package validation, not as pure action-transport tests.

### 13.7 Oceananigans integration tests

Oceananigans integration should be tested at several levels.

#### Interface smoke tests

```text
construct SpectralWaveModel on an Oceananigans RectilinearGrid
use CPU and GPU architectures
run through Oceananigans-style Simulation
callbacks and diagnostics execute on schedule
output writers can write bulk diagnostics
```

#### Prescribed-current tests

Use an Oceananigans `Field` or field-like object for a prescribed current:

```text
barotropic current: U(x,κ) = u(x)
sheared current: U(x,κ) from QTransform
spatially uniform current: no spectral refraction
spatial gradient in current: spectral refraction agrees with ray ODE
```

#### Pseudomomentum field tests

Compute `p(x,z)` from a wave spectrum and write it into Oceananigans-compatible `Field`s:

```julia
pˣ, pʸ = pseudomomentum_fields(waves, ocean.grid)
```

Verify:

```text
p fields have correct Oceananigans locations
vertical integral identity holds
CPU/GPU values agree
p can be consumed by a coupling tendency kernel
```

#### Coupled-column Hasselmann test

Use Oceananigans fields for `uᴸ(z,t), vᴸ(z,t)` and Ripple.jl for `N(k,t)` in a horizontally uniform column. This is the first real coupled integration test and should be part of regular CI in a small configuration.

#### Split-coupling tests

For later milestones:

```text
wave step produces p and U
Oceananigans step consumes p_t or vortex/Stokes-Coriolis forcing
coupled split step preserves analytic solution in the Hasselmann limit
coupled source-free tests conserve total energy/momentum to expected splitting error
```

### 13.8 External-model comparison strategy

External comparisons should be optional and scripted. They should not require GPL code to be linked into Ripple.jl, and they should not run in ordinary CI unless a special environment is configured.

Recommended structure:

```text
test/validation/external_models/
  picles/
    README.md
    run_picles_case.jl
    parse_picles_output.jl
  swan/
    README.md
    Dockerfile or docker command notes
    run_swan_case.sh
    parse_swan_output.jl
  wam/
    README.md
    run_swamp_case.sh
    parse_wam_output.jl
  ww3/
    README.md
    run_ww3_regtest.sh
    parse_ww3_output.jl
  ecwam/
    README.md
    run_ecwam_test.sh
    parse_ecwam_output.jl
```

External comparison outputs should be normalized into a small common format:

```text
time
x, y or station id
Hs
mean direction
peak frequency / peak period
mean frequency / mean period
total action / energy if available
optional spectra at selected points
```

#### Readily runnable comparison models

Priority order for early Ripple.jl development:

1. **PiCLES.jl** — easiest to run because it is Julia, public, Apache-2.0, and has a minimal example and test instructions. It is not a spectral model, so compare bulk fields and forcing-case behavior rather than spectra.
2. **SWAN via Docker or official test cases** — likely the easiest traditional spectral model for external smoke tests because Docker images and small test cases exist. Use it for simple propagation, shoaling/refraction, and wind-growth examples.
3. **WAM Cycle 7 SWAMPtest** — useful because it includes automated SWAMP test scripts and reference outputs, but it requires MPI/NetCDF setup and possibly HPC-script adaptation.
4. **ecWAM tests** — valuable because it is Apache-2.0, has CMake/ctest infrastructure, standalone mode, validation norms, and GPU-performance lessons, but dependencies are heavier.
5. **WAVEWATCH III regression tests** — scientifically central and operationally authoritative, but more cumbersome for routine comparison because the README notes a separate binary data bundle is needed in addition to the GitHub repository.

#### Best early comparison opportunities

```text
PiCLES homogeneous wind box:
    compare Hs / energy growth and gridded bulk fields.

PiCLES stationary vortex wind:
    compare qualitative Hs pattern and timing in a rotating-vortex wind field.

SWAN refraction / shoal test cases:
    compare propagation, refraction, and depth effects in cases with small input files.

SWAN or WAM simple fetch-limited growth:
    compare bulk Hs and mean period after implementing comparable source terms.

WAM SWAMPtest:
    compare selected SWAMP cases once source terms and output diagnostics are mature.

WW3 regular-grid propagation/source cases:
    use for focused validation against a modern third-generation reference, not as daily CI.

ecWAM standalone tests:
    use for output conventions, validation norms, and GPU/source-term performance comparisons.
```

Important caveat: existing models generally do not implement the CWCM `uᴸ`, `Q`, and `p` coupling exactly. Comparisons with WW3/WAM/SWAN/ecWAM should focus first on uncoupled action transport, standard source terms, and bulk spectral-wave diagnostics. CWCM-specific behavior should be verified primarily by analytic tests and Oceananigans-coupled tests.

### 13.9 Examples suite

Examples should be curated as teaching material and as smoke tests. Each example should run in a small mode quickly, and selected examples should have a larger/high-resolution mode for science figures.

#### Example 00: ProductField basics

Purpose:

```text
show logical indexing N[i,j,m,n]
show physical vs coordinate locations
show coordinate-inner storage does not affect logical indexing
plot a simple spectrum at one spatial point
```

#### Example 01: free swell packet, Cartesian wavevector grid

Purpose:

```text
demonstrate source-free Hamiltonian transport in (x,y,kx,ky)
compare packet centroid to ray ODE
verify total action conservation
```

#### Example 02: free swell packet, polar grid

Purpose:

```text
demonstrate practical (κ, φ) grid
show metric-aware polar finite-volume update
compare to Cartesian result in a simple case
```

#### Example 03: prescribed current refraction

Purpose:

```text
show spectral refraction by a spatially varying current or depth
illustrate ∂xΩ-driven k-space transport
compare packet centroid to ray equations
```

#### Example 04: source-only fetch-limited growth

Purpose:

```text
demonstrate wind input + dissipation with transport disabled via `advection = nothing`
plot Hs, mean period, and spectrum growth
provide a first comparison target for PiCLES, SWAN, WAM, and WW3
```

Required source terms:

```text
LinearWindInput or ExponentialWindInput
SimpleWhitecapping or SaturationDissipation
optional BottomFriction
```

#### Example 05: Hasselmann inertial oscillation

Purpose:

```text
reproduce the paper's Hasselmann problem
verify relaxation-to-spectrum source
verify p(z,t), U(κ,t), and Oceananigans current coupling
plot uᴸ/p⋆, vᴸ/p⋆, and Lagrangian kinetic energy
```

This should be both an example and an integration test.

#### Example 06: CWCM Q-transform with sheared current

Purpose:

```text
show U(x,κ) computed by matrix-free QTransform
show ∂κU effect on group velocity or discrete Ω derivatives
show p(x,z) from a prescribed spectrum
verify vertical-integral identities visually and numerically
```

#### Example 07: Oceananigans prescribed-current coupling

Purpose:

```text
use an Oceananigans grid and fields for a prescribed barotropic or sheared current
run Ripple.jl wave action on the surface grid
write bulk diagnostics with Oceananigans-style output
```

#### Example 08: PiCLES-inspired stationary vortex wind

Purpose:

```text
create a stationary rotating vortex wind field
run wave growth + propagation with simple source/dissipation terms
plot Hs evolution and compare qualitatively with PiCLES / WW3 idealized vortex figures
```

Initial simple wind field:

```julia
wind = StationaryVortexWind(; center = (Lx/2, Ly/2),
                              diameter = 600e3,
                              speed = 20,
                              rotation = Clockwise())
```

This example is inspired by PiCLES idealized forcing cases, including a stationary vortex with constant rotating winds. It is a good stress test because waves generated in different quadrants propagate, overlap, and refract through a strongly nonuniform forcing pattern.

Required source terms:

```text
wind input
whitecapping / saturation dissipation
optional swell decay
```

#### Example 09: moving hurricane wind forcing

Purpose:

```text
demonstrate a severe moving storm / hurricane-like wind field
exercise wind-input and dissipation functions under strong spatially varying forcing
produce Hs, peak direction, and wave-age diagnostics
compare qualitatively with PiCLES severe-storm examples and later WW3 runs
```

Initial idealized wind model:

```julia
wind = IdealizedHurricaneWind(; center = t -> (x0 + Ustorm*t, y0 + Vstorm*t),
                                vmax = 45,          # m/s, example only
                                rmax = 50e3,
                                radius = 400e3,
                                inflow_angle = 20degrees,
                                background = (0, 0))
```

Early implementation can use a Rankine-like profile. A later implementation can add a Holland-type pressure/wind model or read best-track data. This example should not be added until the minimal source/dissipation suite is in place; without dissipation, the strong wind case will be physically and numerically misleading.

#### Example 10: coupled Oceananigans demo

Purpose:

```text
show the intended endgame: Ripple.jl waves coupled to an Oceananigans model
start with a simple prescribed or weakly coupled current
write both ocean and wave diagnostics
show p(x,z), U(x,κ), Hs, and mean direction
```

This should remain a demonstration until the coupled conservation tests are mature.

### 13.10 Example smoke tests

Every example should provide a small mode:

```julia
run_example(:small)
```

or an environment-variable-controlled mode:

```bash
RIPPLE_EXAMPLE_MODE=small julia examples/05_hasselmann_inertial_oscillation.jl
```

Smoke-test acceptance criteria:

```text
example runs without error
output diagnostics have expected shape and finite values
basic invariant or analytic check passes
runtime is short enough for CI small mode
```

### 13.11 Performance and regression tests

Performance tests should track:

```text
ProductField indexing overhead
layout-dependent source-column throughput
transport-kernel throughput
spectral-reduction throughput
QTransform throughput
halo-packing bandwidth
GPU allocation count
time per step for selected Nx × Ny × Nκ × Nθ
```

Avoid treating benchmark numbers as hard pass/fail in ordinary CI. Instead, store benchmark history and fail only on severe allocation regressions or obviously accidental slowdowns.

### 13.12 Updated validation hierarchy

1. ProductField indexing and layout correctness.
2. Coordinate-grid topology, finite-volume integration, and metrics.
3. Boundary conditions and coordinate halos.
4. Intrinsic dispersion relation.
5. Matrix-free `Q` normalization, limits, and finite-volume cell integration.
6. Barotropic-current Doppler velocity.
7. Sheared-current `U(x,κ)` transform.
8. Pseudomomentum identities.
9. Source-free Cartesian Hamiltonian transport.
10. Source-free polar metric transport.
11. Analytic/manufactured phase-space transport solutions.
12. Relaxation-to-spectrum source solution.
13. Hasselmann inertial-oscillation coupled-column solution.
14. Source-only column growth/decay tests.
15. Fetch-limited growth with simple source/dissipation packages.
16. PiCLES-inspired bulk comparisons.
17. SWAN/WAM/WW3/ecWAM optional external comparisons.
18. Oceananigans prescribed-current integration.
19. Oceananigans coupled-column integration.
20. Multi-rank spatial halo and conservation tests.


## 14. Open design questions

1. Should `ProductField` support `2D physical × 2D coordinate` only at first, or try to support arbitrary physical × coordinate rank?
2. Should coordinate-inner storage be the default immediately, or should physical-first be the first debug implementation?
3. Should the first transport solver be Cartesian `(kx,ky)` for Hamiltonian clarity, or polar `(κ, φ)` for practical relevance?
4. How much of Oceananigans' `AbstractField` interface should `ProductField` subtype and reuse?
5. What is the right default timestepper once source terms are included?
6. Should `∂κU` be cached, computed by finite differences from `U`, or avoided by constructing velocities from discrete `Ω` differences?
7. What is Ripple.jl's license target, and how does that constrain use of WAM/SWAN-derived algorithms?
8. What are the first validation cases that should block merging the dynamical core?
9. Which minimal source/dissipation package is sufficient for the first vortex and hurricane examples before implementing ST4/ST6-level physics?
10. Which external-model comparison should be automated first: PiCLES homogeneous wind, SWAN Docker test case, WAM SWAMPtest, or WW3 regtest?
11. Should the Hasselmann coupled-column test use a minimal custom current integrator first, or immediately use Oceananigans fields and timestep machinery?

---

## 15. Non-goals for the first version

The first version should not attempt to implement everything in WW3, WAM, ecWAM, or SWAN.

Explicit non-goals:

```text
full operational source-term suite
unstructured grids
nested grids
ice physics
coastal triads and depth-induced breaking
OASIS/ESMF coupling
exact nonlinear interaction solver
full restart/output parity with WW3/WAM
production multi-GPU scaling
```

The first real target is much narrower:

```text
A correct, conservative, Oceananigans-native, product-space action-transport core with clean ProductField storage, spatial-only decomposition, and matrix-free CWCM coupling operators.
```

---

## 16. Short design mantra

```text
Ripple.jl is not a Fortran wave model translated into Julia.
It is an Oceananigans-native phase-space transport model.

The state is N[i, j, m, n].
The field is a ProductField.
The spectrum is explicit, not flattened.
The storage is layout-parametric and probably coordinate-inner.
The decomposition is spatial only.
Q is a matrix-free transform, not a 5D field.
The Hamiltonian Ω is the center of the transport discretization.
Existing wave models are sources of wisdom, tests, and caution—not templates to copy blindly.
The test suite is part of the model design, not an afterthought.
Examples should double as small, runnable validation stories.
```
