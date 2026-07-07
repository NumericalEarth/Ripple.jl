# Implementing the Onuki & Fujiwara (2026) reduced wave–current model in Ripple.jl

**Paper:** Y. Onuki & Y. Fujiwara (2026), *A reduced model for surface wave–current
interactions without spatial scale separation*, arXiv:2606.03231.

**Target:** Ripple.jl, currently a finite-volume spectral wave-action model
(`SpectralWaveModel`) implementing the Vanneste & Young (2026) consistent
wave–current model (CWCM) on Oceananigans grids.

**Guiding constraint:** implement the amplitude equation by **reusing Oceananigans
operators and solvers to the maximum extent** — grids, Fields, tracer-advection
schemes (WENO), derivative operators, the FFT-based generalized Poisson solver,
and time-stepping conventions. This requires rewriting the wave equation in a
more familiar form; §3 does that algebra explicitly and compares candidate
formulations before recommending one.

---

## 1. What the paper proposes

A bidirectional closure of the Craik–Leibovich (CL) framework in which the Stokes
drift is **not prescribed** but computed from a prognostic complex wave-amplitude
field. Two coupled equation sets (dimensional form, paper §2):

**Mean flow — CL momentum for the Lagrangian-mean velocity** (2.1)–(2.2):

    ∇·Uᴸ = 0,   Wᴸ = 0 at z = −h, 0
    ∂ₜUᴸ + Uᴸ·∇Uᴸ + (f ẑ − ∇×Uˢ)×Uᴸ = −∇Π + ∂ₜUˢ

with constant f (traditional approximation), flat bottom, rigid lid. This is the
Suzuki & Fox-Kemper (2016) variant of CL — exactly the formulation Oceananigans'
`NonhydrostaticModel` solves when `stokes_drift` is provided.

**Waves — complex amplitude A(x₁, x₂, t)** on the horizontal plane (2.3)–(2.5):

    [1 + α(∇ₕ² + κ²)] ∂ₜA − (iω_κ/2κ)(∇ₕ² + κ²) A = (ω_κ/2κω) 𝓛(Uᴸ, A)

    𝓛(Uᴸ, A) = (Ūᴸᵢ A,ᵢⱼ),ⱼ + (Ūᴸᵢ A,ⱼ),ᵢⱼ − κ² Ũᵢ A,ᵢ − κ² (Ũᵢ A),ᵢ

    Ūᴸᵢ = ∫₋ₕ⁰ Φ² Uᴸᵢ dz,    Ũᵢ = (1/κ²) ∫₋ₕ⁰ Φ_z² Uᴸᵢ dz

    α = (κ ω_κκ − ω_κ)/(4κ² ω_κ)      (reconstitution parameter, Thomas & Yamada 2018)
    ω² = gκ tanh(κh),   Φ(z) = C cosh(κ(z+h))/cosh(κh),  ∫Φ² dz = 1,  C² = gκ/(ω ω_κ)

**Stokes drift diagnosed from A** (2.8):

    Uˢᵢ = (−i/ω)(Φ² A†,ⱼ A,ᵢⱼ + Φ_z² A† A,ᵢ) + c.c.,   Uˢ = (Uˢ₁, Uˢ₂, 0)

Key structural facts that shape the implementation:

- **A resolves the carrier in space, not in time.** Only e^{−iωt} is factored out;
  the horizontal oscillation e^{i k·x} with |k| ≈ κ lives inside A. The grid must
  resolve the wavelength 2π/κ. The model is narrow-band in *frequency* but
  isotropic in *direction* (spectrum concentrated on the circle |k| = κ) — this is
  what buys multidirectional scattering with no WKB scale separation.
- **𝓛 uses the Lagrangian-mean velocity** (Stokes correction, §5.3): the
  wave–current interaction operator is evaluated with Uᴸ = U + Uˢ, which recovers
  the deep-water third-order Stokes frequency correction phenomenologically.
- **Everything on the wave side is linear in A** (quasi-linear closure — no
  quartic wave–wave interactions) and bilinear in (Uᴸ, A). The Stokes drift and
  all conservation densities are quadratic in A.
- **The Uˢ profile is separable**: Uˢᵢ(x,y,z) = Φ²(z) Sᵢ(x,y) + Φ_z²(z) Tᵢ(x,y)
  with Sᵢ = (−i/ω) A†,ⱼA,ᵢⱼ + c.c. and Tᵢ = (−i/ω) A†A,ᵢ + c.c. Two 2-D fields
  and two 1-D profiles fully determine the 3-D Stokes drift, all its gradients,
  and (via the product rule with ∂ₜA from the RHS) ∂ₜUˢ.

**Conservation laws** (§6, periodic domain) — the consistency-test targets:

- Wave action:      𝒜 = ∬ [ |A|² − α(|∇A|² − κ²|A|²) ] dx dy                    (6.2)
- Coupled energy:   ℰ = ∬ (|∇A|² − κ²|A|²) dx dy + ∭ |Uᴸ|²/2 dV                (6.5)
  via the exchange identities d/dt ∬(|∇A|²−κ²|A|²) = −∭ Uᴸ·∂ₜUˢ (6.3) and
  d/dt ∭ |Uᴸ|²/2 = +∭ Uᴸ·∂ₜUˢ (6.4)
- Momentum (f = 0): 𝒫ᵢ = 𝒫ʷᵢ + ∭ Uᵢ dV with pseudomomentum
  𝒫ʷᵢ = (κ/iω_κ) ∬ [ (1+ακ²)(A†A,ᵢ − A A†,ᵢ) − α(A†,ⱼ A,ᵢⱼ − A,ⱼ A†,ᵢⱼ) ] dx dy  (6.7)

Assumptions/limits to document prominently: weak nonlinearity (no breaking),
narrow frequency band, flat bottom, no buoyancy/viscosity in the base derivation
(§7 says viscous Boussinesq extension is direct), quasi-linear (no wave–wave
transfer), no mean sea-surface elevation.

---

## 2. Where this fits in Ripple.jl

Ripple currently has one model: `SpectralWaveModel`, a WKB action model
N(x, y, κ, φ) with VY-2026 Q-transform coupling. The OF-2026 model is the
**complementary non-WKB regime**: currents with horizontal scales comparable to
the wavelength (Langmuir cells!), where ray theory breaks. Adding it gives Ripple
two ends of the same physics with an overlap regime for cross-validation —
a genuinely strong internal validation axis (see §8).

Proposed addition: a second model type, **`NarrowBandWaveModel`**, plus a coupled
driver **`WaveCurrentModel`** pairing it with an Oceananigans
`NonhydrostaticModel`. New submodule `src/NarrowBand/`, mirroring existing
structure and AGENTS.md conventions (KA kernels, `nothing` semantics, positional
grid, unicode κ/φ, no hardcoded Float64, depth from grid).

### Reuse inventory

| Need | Reuse |
|---|---|
| Mean-flow solver (2.2) with vortex force, ∂ₜUˢ, pressure Π, rigid lid | Oceananigans `NonhydrostaticModel` + `FPlane` + Stokes-drift interface, unchanged |
| Stokes-drift injection | Custom type implementing the 5 kernel functions in `Oceananigans.StokesDrifts` (`∂t_uˢ`, `∂t_vˢ`, `x/y/z_curl_Uˢ_cross_U`) — the interface already supports array-backed entries |
| Storage for wave fields | Real Oceananigans `Field{Center, Center, Nothing}`s on the shared grid (§3.4) — halos, `set!`, output writers all work unmodified |
| Doppler transport of A | Oceananigans **tracer-advection schemes** (`WENO()` default, `Centered` for conservation studies) via the flux-divergence term identified in §3.1 |
| Mass-matrix inversion [1 + α(∇²+κ²)]⁻¹ | `Oceananigans.Solvers.FFTBasedPoissonSolver`: `solve!(ϕ, solver, b, m)` already solves the **generalized ("screened") Poisson equation (∇² + m)ϕ = b for arbitrary scalar m** — our solve is the stock call with m = (1+ακ²)/α (§3.3) |
| Refraction operators | `Oceananigans.Operators`: first/second derivatives and interpolants, inside Ripple KA kernels |
| Exact finite-volume vertical integrals for Ū, Ũ | Pattern from Ripple's Q-transform machinery (`src/Coupling/q_transform.jl`); cell integrals of cosh²/sinh² are analytic |
| Grids, architectures, halos, `on_architecture` | Oceananigans / existing Ripple infrastructure |
| Simulation driver, output, callbacks | `Oceananigans.Simulation` (Ripple convention) |
| Docs/tests/validation harness | Literate + Documenter pipeline, `src/Validation/` case framework, `test/` layout |

### What is genuinely new

1. The **flux-divergence rewrite** of the amplitude equation (§3.1) and its
   Oceananigans-operator discretization.
2. The **two-field (G, A) formulation** (§3.2) pairing a mass-matrix-free
   prognostic equation with a diagnostic screened-Poisson solve.
3. The bilinear refraction/scattering operator R, the Stokes functionals (S, T),
   and the conservation diagnostics.
4. A **stage-synchronized coupled time-stepper**.

---

## 3. Formulation and discretization

### 3.1 The flux-divergence rewrite

Expand 𝓛 (eq. 2.4a) and add/subtract the carrier contribution ∇²A → −κ²A. With
the Helmholtz "bandwidth" operator **H ≡ ∇ₕ² + κ²** (H A ≈ 0 for narrow-band A),
the expansion is the exact identity

    𝓛(Uᴸ, A) = −2κ² (Ū + Ũ)·∇A − κ² (∇·Ũ) A + R(Ū, A)

    R(Ū, A) = 2 Ū·∇(H A) + 2 Ūᵢ,ⱼ A,ᵢⱼ + (∇·Ū) ∇²A + Ūᵢ,ᵢⱼ A,ⱼ

(derivation: (ŪᵢA,ᵢⱼ),ⱼ + (ŪᵢA,ⱼ),ᵢⱼ = 2Ūᵢ∂ᵢ∇²A + 2Ūᵢ,ⱼA,ᵢⱼ + (∇·Ū)∇²A
+ Ūᵢ,ᵢⱼA,ⱼ, then 2Ūᵢ∂ᵢ∇²A = 2Ū·∇(HA) − 2κ²Ū·∇A; the Ũ terms contribute
−2κ²Ũ·∇A − κ²(∇·Ũ)A.)

The transport must be a **flux divergence** to be discretized with WENO
(Oceananigans tracer advection computes ∇·(vc), not v·∇c). Converting with
−2κ²(Ū+Ũ)·∇A = −2κ²∇·[(Ū+Ũ)A] + 2κ²(∇·Ū)A + 2κ²(∇·Ũ)A and merging the (∇·Ũ)A
terms, then multiplying by the prefactor ω_κ/(2κω), the wave equation becomes

    [1 + αH] ∂ₜA  =  (iω_κ/2κ) H A                          (isotropic dispersion)
                   −  ∇·(vᵉᶠᶠ A)                             (Doppler flux divergence)
                   +  (κω_κ/ω) [ (∇·Ū) + ½ (∇·Ũ) ] A         (compressibility source)
                   +  (ω_κ/2κω) R(Ū, A)                      (refraction/scattering)

with the **effective transport velocity**

    vᵉᶠᶠ ≡ (κω_κ/ω) (Ū + Ũ).

This is the "familiar form": a Schrödinger-type dispersive term, a conservative
flux-divergence transport term in exactly the shape Oceananigans tracer
advection expects, a pointwise source from the non-solenoidality of the
depth-weighted velocities (Ū, Ũ are *not* horizontally divergence-free — they
carry the vertical-velocity contribution through the weighted integrals; the
source vanishes for barotropic currents), and a bilinear refraction/scattering
residual.

Physical checks that make this rewrite trustworthy (and pedagogically valuable):

- **Deep water:** κω_κ/ω = 1/2 and Ũ = Ū (since Φ_z² = κ²Φ²), so
  vᵉᶠᶠ = Ū = 2κ∫ e^{2κz} Uᴸ dz — precisely the classical depth-weighted
  effective advection velocity for deep-water waves over sheared currents
  (Stewart & Joy 1974 / Kirby & Chen 1989 weighting). The model's Doppler term
  literally *is* the textbook one.
- **Plane wave** A ∝ e^{ik·x}, |k| = κ, uniform current: the equation reduces to
  the Doppler shift k·Ū in deep water. ✓
- R contains the physics that ray theory linearizes: current-gradient
  refraction (Ūᵢ,ⱼA,ᵢⱼ terms) and multidirectional scattering. Only 2Ū·∇(HA) is
  narrow-band-small; the rest is leading-order whenever ∇Ū varies on the
  wavelength scale. R is *not* an error term — it is the scattering operator,
  and its symmetrized form is what the conservation laws rest on.
- The flux-divergence transport conserves ∬A discretely by construction
  (telescoping fluxes), for any advection scheme.

### 3.2 Two-field formulation: prognose G, diagnose A

Can the mass matrix be traded for a second evolution equation, e.g. for ∇²A? A
literal prognostic pair (A, ∇²A) does **not** close: applying ∇² to the equation
produces α ∂ₜ∇⁴A, then ∇⁶A, ... — an infinite hierarchy, because [1 + αH] is
irreducibly nonlocal (the 2-D analogue of the BBM / regularized-long-wave
structure). What *does* close exactly is the pair (G, A) with the
**reconstituted amplitude**

    G ≡ [1 + αH] A = (1 + ακ²) A + α ∇²A

prognostic and A diagnostic. Since αHA = G − A, the dispersive term becomes
*local in (G, A)*, and the system is exactly two equations:

    ∂ₜG = (iω_κ/2κα)(G − A) − ∇·(vᵉᶠᶠ A)                    (prognostic; no mass
        + (κω_κ/ω)[(∇·Ū) + ½(∇·Ũ)] A + (ω_κ/2κω) R(Ū, A)    matrix, no Laplacian
                                                             stencil in dispersion)

    [1 + α(∇ₕ² + κ²)] A = G                                 (diagnostic elliptic
                                                             solve, once per stage)

Why this is the preferred formulation:

- Structurally identical to Oceananigans' prognostic-momentum / diagnostic-
  pressure pattern: the time stepper advances G with plain stage updates and
  never sees a mass matrix; one constant-coefficient screened-Poisson solve
  recovers A per stage.
- The dispersion enters *entirely through the elliptic solve*: the semi-discrete
  dispersion relation Λ_d(λ) = (ω_κ/2κ)(κ² − λ)/(1 + α(κ² − λ)) (−λ = discrete
  Laplacian eigenvalue) is automatic, and the stencil-mismatch hazard between a
  mass matrix and a separate dispersion operator disappears by construction.
- The wave action becomes a bilinear pairing: 𝒜 = ∬[|A|² − α(|∇A|² − κ²|A|²)]
  = Re ∬ A†G dx dy (integrate by parts) — the primary invariant is a one-line
  diagnostic of the two stored fields.
- Exact equivalence: because M = 1 + αH is linear with constant coefficients,
  advancing G and diagnosing A = M⁻¹G generates *identical* discrete
  trajectories to advancing A with ∂ₜA = M⁻¹(RHS). The G-form is purely better
  bookkeeping; a unit test asserts the equivalence to roundoff.
- In coupled mode, ∂ₜA (needed for the analytic ∂ₜUˢ) is one extra solve,
  ∂ₜA = M⁻¹ ∂ₜG — cheap, and only needed when the Stokes feedback is on.

### 3.3 Candidate approaches, pros and cons, recommendation

**Formulation of the wave solver:**

| Approach | Pros | Cons |
|---|---|---|
| (a) Fourier pseudospectral A-solver | Exact derivatives at the carrier; exact spatial conservation | Parallel/GPU story, bounded-domain path, and time-stepping all bespoke; near-zero reuse of Oceananigans/Ripple machinery; a second numerical dialect in one package |
| (b) Oceananigans operators, prognose A, apply M⁻¹ to the RHS | Full operator/advection reuse; A directly stored | Mass matrix lives inside `compute_tendencies!`; dispersion needs an explicit ∇² stencil that must match the solver's eigenvalues |
| (c) Oceananigans operators, **prognose G, diagnose A** (§3.2) | All of (b), plus: stepper is mass-matrix-free (stock stage updates), dispersion term is local (no Laplacian stencil), action = Re⟨A,G⟩, pressure-solve-like structure familiar to every Oceananigans reader | A is diagnostic (must document that restarts need only G); one extra solve for ∂ₜA in coupled mode |
| (d) Explicit expansion M⁻¹ ≈ 1 − αH (no solve) | No elliptic solve at all | Reintroduces an unbounded ~|k|⁴ biharmonic term (stiff), and breaks the Thomas–Yamada invariant structure — rejected |
| (e) Prognostic pair (A, ∇²A) | — | Does not close (infinite hierarchy, §3.2) — rejected |

**Inversion of [1 + αH]** (verified against Oceananigans source):

`Oceananigans.Solvers.FFTBasedPoissonSolver` solves the *generalized* Poisson
equation **(∇² + m)ϕ = b for arbitrary scalar m** — the screened-Poisson case
m < 0 is explicitly supported (`solve!(ϕ, solver, b, m)`;
`src/Solvers/fft_based_poisson_solver.jl`). Our solve is

    (∇ₕ² + m) A = G/α,    m = (1 + ακ²)/α.

Since α < 0 and 1 + ακ² > 0 (deep water 5/8, shallow limit 3/4 — assert in the
constructor), m < 0 strictly: every discrete mode has λ − m > 0, the operator is
SPD, and the k = 0 null-mode special-casing in the solver is never triggered.
Options:

| Solver | Pros | Cons |
|---|---|---|
| `FFTBasedPoissonSolver` with m = (1+ακ²)/α | **Stock call, zero new code**; exact one-shot solve; eigenvalues are those of the discrete FD Laplacian (spectral–stencil consistency built in); CPU+GPU via existing transform plans; supports `Periodic`, `Bounded` (DCT/staggered-Neumann), and `Flat` topologies | Requires uniform spacing in the transform directions (assert regular rectilinear horizontal grid in the constructor) |
| `FourierTridiagonalPoissonSolver` | Handles one stretched/bounded direction | Irrelevant for a uniform 2-D horizontal solve; keep in mind for exotic grids |
| `ConjugateGradientPoissonSolver` / `KrylovSolver` | General topologies, immersed boundaries, variable coefficients | Iterative cost + tolerance where an exact solve exists — future option for bounded/immersed domains, not v1 |

**Recommendation: (c) + `FFTBasedPoissonSolver`.** Prognose G on a 2-D companion
grid (same x, y as the mean-flow grid, `Flat` z — `poisson_eigenvalues` returns
zeros for `Flat`, so the 3-D solver degenerates correctly to the 2-D solve);
diagnose A once per stage with the stock generalized-Poisson call; discretize
transport with Oceananigans tracer advection and R with centered operators. The
wave model then has *exactly* the anatomy of an Oceananigans model: prognostic
fields, flux-divergence + source tendencies, and one elliptic solve per stage.

### 3.4 Representation: real Fields

Store the prognostic pair **(Gʳ, Gⁱ)** and diagnostic **(Aʳ, Aⁱ)** as real
Oceananigans `Field{Center, Center, Nothing}`s on the shared grid. Decisive
reasons for real pairs over a complex-valued field:

- WENO smoothness indicators are sums of squares — meaningless on complex data.
  With the real pair, each component is advected as an ordinary real tracer
  through the stock advection schemes, zero modification.
- `solve!` for the FFT solver writes the real part of its complex storage into a
  real output field — the real-pair layout matches the solver's native calling
  convention (two real solves per stage).
- Halo fill, `set!`, `interior`, reductions, and the NetCDF/JLD2 output writers
  all work unmodified on real fields (NetCDF has no complex type anyway).
- The i in the dispersive term just swaps components:
  ∂ₜGʳ ⊃ −(ω_κ/2κα)(Gⁱ − Aⁱ),  ∂ₜGⁱ ⊃ +(ω_κ/2κα)(Gʳ − Aʳ).
  All other operators (advection, R with real Ū) act identically and
  independently on both components.

Provide `amplitude(model)` returning a lazy complex view for diagnostics and
user code, so the physics-facing API still speaks complex A. Restarts need only
G (A is diagnostic) — checkpoint accordingly.

### 3.5 Term-by-term discretization

**Doppler flux divergence (WENO).** −∇·(vᵉᶠᶠA) is natively what Oceananigans
tracer advection computes: `div_Uc` with the model's `advection` scheme (default
`WENO()`, Ripple convention; `Centered` selectable) applied separately to Aʳ and
Aⁱ. Face velocities come for free from staggering: computing the Ū, Ũ vertical
projections from the mean-flow u (at Face, Center, Center) and v (at Center,
Face, Center) columns yields vᵉᶠᶠ components natively at the (Face, Center) and
(Center, Face) locations `div_Uc` expects — no horizontal interpolation. The
compressibility source (κω_κ/ω)[(∇·Ū) + ½(∇·Ũ)]A is pointwise. Two numerical
notes, both to be documented and tested:

- *WENO dissipation acts on the carrier*, since A oscillates at wavelength 2π/κ.
  But the upwind dissipation scales with the transport speed, which here is the
  **current speed** |vᵉᶠᶠ| = O(ε) — not the phase speed. Damping rate
  ~ |vᵉᶠᶠ| κ (κΔx)⁵: weak for weak currents and well-resolved carriers. Quantify
  with a required carrier-damping test (§6) and publish points-per-wavelength
  guidance (expect ≥ 8–12 ppw).
- *WENO breaks exact action conservation by construction* — upwinding is
  sign-definite dissipation of the |A|² part. This is a feature (grid-scale
  regularization) but changes what the conservation tests assert: the `Centered`
  configuration is the conservation-study configuration (§7); the WENO default
  must show monotone, resolution-convergent action decay.

**Dispersion (local).** (iω_κ/2κα)(G − A): pointwise, no stencil. The discrete
dispersion relation is set by the elliptic solve's eigenvalues (§3.2/§3.3) and
is bounded — deep-water large-k limit 2ω/3 — because the same m appears in the
solve and the (G − A) term by construction.

**Refraction/scattering R (centered).** All terms built from
`Oceananigans.Operators` first/second derivatives and interpolants in one fused
KA kernel over (i, j), following the `_wave_current_refraction_tendency!`
fused-kernel pattern from AGENTS.md. Keep the operator in the symmetrized
grouping of (2.4a) when assembling fluxes so the discrete integrals mimic the
continuous integration-by-parts structure as closely as the FD stencils allow.

**Elliptic solve.** One `solve!(A, solver, G/α, m)` per component per stage on
the `Flat`-z companion grid; `FFTW.PATIENT`-planned on CPU, CUFFT on GPU,
through the existing transform-plan machinery.

### 3.6 Time stepping

Match Oceananigans/Ripple convention: **SSP-RK3** stages over (Gʳ, Gⁱ) (no
positivity clamp — that is an action-model concern; amplitudes are signed), with
the diagnostic solve for (Aʳ, Aⁱ) at the top of each stage's tendency
evaluation. Stability: the dispersive eigenvalues iΛ_d sit on the imaginary
axis, inside the RK3 stability region for Δt |Λ_d|max ≤ √3; since |Λ_d| is
bounded (~2ω/3 deep water, thanks to reconstitution + solve-consistent
dispersion), the wave step is Δt ≲ 2.6/ω — a carrier period, comparable to the
LES time step of the coupled mean flow in Langmuir configurations, so a shared
Δt is expected to work without substepping. Advective CFL from vᵉᶠᶠ is O(ε)
weaker. An exponential-integrator variant is a possible later optimization,
*not* in scope for v1.

### 3.7 Coupled stepping (conservation-critical)

Naive Lie splitting (step waves, then currents) degrades the energy identity
(6.3)+(6.4) to O(Δt) regardless of the inner schemes. Design the coupled driver
so that **within each RK3 stage** both tendencies are evaluated from the same
state:

1. Diagnose A = M⁻¹G; from current-model velocities compute Ūᴸ, Ũ (adding the
   wave model's own Uˢ contribution to form Uᴸ = U + Uˢ, per paper §5.3).
2. Evaluate the wave RHS → ∂ₜG (explicit; no circularity: it depends on Uᴸ, A
   only), and ∂ₜA = M⁻¹∂ₜG.
3. Form ∂ₜUˢ from (A, ∂ₜA) via the quadratic product rule — analytic, no time
   finite-differencing.
4. Evaluate the mean-flow tendency with the vortex force and ∂ₜUˢ; project
   pressure.
5. Advance both models' fields with the shared stage weights.

Practical shape: `WaveCurrentModel` wraps `(wave_model, current_model,
stokes_drift, projections)` and owns `time_step!`, driving both models' RK3
stage primitives. A simpler callback-coupled mode (update coupling once per Δt
via `Simulation` callbacks) ships first in Phase 3a for bring-up, with the
stage-synchronized integrator as Phase 3b; the conservation suite quantifies the
difference.

### 3.8 Vertical projections and Stokes structure

- Ūᴸᵢ, Ũᵢ: KA kernel computing per-column sums of Uᴸᵢ against **analytically
  cell-integrated** Φ² and Φ_z² weights over the current model's z-faces
  (∫cosh²(κ(z+h)) dz is closed-form; mirrors the exact-FV-integral pattern in
  `q_transform.jl`). Evaluate on u- and v-columns so the results live at face
  locations (§3.5). Precompute the two weight vectors at construction (they
  depend only on κ, h, z-faces). Deep-water limit: Φ_z² = κ²Φ² ⇒ Ũ = Ū — unit
  test and a fast path.
- `NarrowBandStokesDrift`: stores S₁, S₂, T₁, T₂ (2-D fields, assembled from
  Oceananigans derivative operators on Aʳ, Aⁱ), their horizontal derivatives as
  needed by the curl terms, ∂ₜS, ∂ₜT, and the two 1-D vertical profiles Φ², Φ_z²
  sampled at the current grid's z-centers/faces. Overload the five
  `Oceananigans.StokesDrifts` kernel functions to evaluate separably, e.g.
  ∂z_uˢ(i,j,k) = (Φ²)'(z_k)·S₁(i,j) + (Φ_z²)'(z_k)·T₁(i,j). No 3-D Stokes
  storage.

### 3.9 Constructor sketch

```julia
grid = RectilinearGrid(size=(Nx, Ny, Nz), x=..., y=..., z=(-h, 0),
                       topology=(Periodic, Periodic, Bounded))

waves = NarrowBandWaveModel(grid;
    κ,                          # carrier wavenumber (required; ω, Φ, α derived)
    velocities = nothing,       # nothing | PrescribedVelocities | CoupledVelocities
    advection = WENO(),         # Doppler-transport scheme; Centered() for
                                # conservation studies
    timestepper = :RK3)

ocean = NonhydrostaticModel(; grid, coriolis=FPlane(f),
    stokes_drift = NarrowBandStokesDrift(waves),
    advection = WENO(), timestepper = :RungeKutta3)

model = WaveCurrentModel(waves, ocean)
simulation = Simulation(model, Δt=..., stop_time=...)
```

Depth h comes from the grid (AGENTS.md rule); constructor asserts uniform
horizontal spacing (FFT solve) and 1 + ακ² > 0. `InfiniteDepth` supported via a
2-D-horizontal + deep-water code path where Φ² is analytic and Ū uses e^{2κz}
weights against a user-supplied or coupled 3-D current.

### 3.10 Module layout

```
src/NarrowBand/
├── NarrowBand.jl                 # submodule, includes, explicit imports
├── dispersion.jl                 # ω(κ,h), ω_κ, ω_κκ, α, Φ, C; deep-water limits
├── helmholtz_solve.jl            # companion Flat-z grid + FFTBasedPoissonSolver
│                                 # wrapper: A = M⁻¹G via solve!(A, s, G/α, m)
├── wave_operators.jl             # Doppler flux divergence (div_Uc), local
│                                 # dispersion, compressibility source, fused R
│                                 # kernel; RHS assembly
├── vertical_projection.jl        # exact-FV Φ², Φ_z² weights; Ū, Ũ at faces
├── stokes_drift.jl               # S, T functionals; NarrowBandStokesDrift +
│                                 # Oceananigans.StokesDrifts overloads; ∂ₜUˢ
├── narrow_band_wave_model.jl     # NarrowBandWaveModel struct + constructor
│                                 # (Gʳ, Gⁱ prognostic; Aʳ, Aⁱ diagnostic;
│                                 # complex amplitude view)
├── time_step.jl                  # RK3 stages over the G pair (KA kernels)
├── coupled_model.jl              # WaveCurrentModel, stage-synchronized stepping
└── diagnostics.jl                # 𝒜 = Re⟨A,G⟩, wave energy, 𝒫ʷ, ℰ, bandwidth
                                  # monitor, significant wave height, Uˢ fields
```

Exports (add to `src/Ripple.jl`): `NarrowBandWaveModel`, `WaveCurrentModel`,
`NarrowBandStokesDrift`, `amplitude`, `wave_action`, `wave_energy`,
`pseudomomentum`, `coupled_energy`, `stokes_drift_fields`, `spectral_bandwidth`,
plus dispersion utilities (`carrier_frequency`, `reconstitution_parameter`,
`effective_transport_velocity`, ...).

The **amplitude ↔ observables dictionary** deserves first-class treatment: from
(4.8), p₀|_{z=0} = iωC A e^{−iωt} + c.c. and ζ′ = p₀|_{z=0}/g, so the surface
elevation envelope is a = 2ωC|A|/g. Implement `surface_elevation_amplitude(A)`
etc., and verify against the classical Stokes drift (see tests). Also
`set!(waves, ...)` initial conditions: plane wave, Gaussian packet (reuse
`GaussianWavePacket` conventions), isotropic narrow-band ring spectrum with
random phases (the paper's natural IC: Â supported near |k| = κ); `set!` accepts
A and computes G = [1+αH]A with centered operators.

---

## 4. Pedagogical documentation plan

New docs section "Narrow-band wave–current model" (Documenter pages +
DocumenterCitations; add to `docs/make.jl` pages and `refs.bib`):

1. **`narrow_band_theory.md` — Physical background and derivation.**
   Written as a tutorial, not a paper recap:
   - Langmuir circulation and CL theory in three paragraphs; what "prescribed
     Stokes drift" misses (waves as a *dynamical* energy reservoir; cite
     Fujiwara & Yoshikawa 2020, Scully & Zippel 2024).
   - The regime map: where WKB/ray theory (Ripple's `SpectralWaveModel`, VY 2026)
     applies vs where scale separation fails (currents ~ wavelength: Langmuir
     cells) — one figure, two columns of assumptions side by side. This page is
     also the "which Ripple model should I use?" guide.
   - Guided derivation at the level of the paper's §3–5 but with the steps
     unpacked: nondimensionalization and the ε-ladder (steepness ε, current
     O(ε), slow time ε²t, f = O(ε²)); the leading-order potential flow and Φ(z);
     the solvability condition at O(ε²); reconstitution (why α, what it buys —
     with the dispersion-accuracy figure from the validation suite); the two
     phenomenological repairs (∇·Uᴸ = 0 and the 𝓛(Uᴸ,·) Stokes correction) and
     *why* each is needed for conservation.
   - **The flux-divergence rewrite and the (G, A) system** (§3.1–3.2 here) as
     their own docs subsections: deriving vᵉᶠᶠ, the deep-water reduction to the
     Stewart–Joy weighted current, R as the scattering operator, and why (A, ∇²A)
     doesn't close but (G, A) does. This is the bridge between the paper's
     compact 𝓛 and what the code actually discretizes — the most pedagogically
     load-bearing page.
   - Boxed final equation set = paper §2, in Ripple notation, alongside the
     rewritten transport–dispersion–refraction form.
2. **`narrow_band_numerics.md` — Discrete formulation.**
   The (G, A) real-pair representation; term-by-term mapping onto Oceananigans
   operators and the screened-Poisson solve (which stencil/solver discretizes
   what); the solve symbol and its positivity; the semi-discrete dispersion
   relation Λ_d and its boundedness; RK3 stability; WENO-on-the-carrier —
   measured damping vs points-per-wavelength and the |vᵉᶠᶠ|-scaling argument for
   why it's weak; the stage-synchronized coupled step and why splitting order
   matters for energy (show the measured drift for callback-coupling vs
   stage-coupling — an honest numerics page).
3. **`narrow_band_conservation.md` — Invariants and their discrete fate.**
   Derive 𝒜 = Re⟨A,G⟩, ℰ, 𝒫 and state exactly what is conserved in which
   configuration: `Centered` transport (conservation to FD-stencil +
   time-discretization error) vs `WENO` (sign-definite action dissipation,
   resolution-convergent); document the paper's caveat that ℰ is the O(ε²)
   *exchanged* energy, not total energy, and that total energy ≈ linear
   combination of 𝒜 and ℰ.
4. **`narrow_band_model_api.md`** — constructor, coupling modes
   (`nothing` / prescribed / coupled), advection-scheme choice, diagnostics, IC
   helpers, output/restart (G only); mirrors the existing `model_api.md` style.
5. **Notation page update** — A, G, κ (carrier) vs Ripple's spectral-grid κ
   (coordinate!) — flag this collision explicitly; α, Φ, Ū, Ũ, vᵉᶠᶠ, H, R,
   S, T, 𝒜, ℰ, 𝒫ʷ.
6. **refs.bib additions**: Onuki & Fujiwara 2026; Craik & Leibovich 1976;
   Leibovich 1977, 1980; Thomas & Yamada 2018; Suzuki & Fox-Kemper 2016;
   McWilliams, Sullivan & Moeng 1997; Fujiwara & Yoshikawa 2020; Xie & Vanneste
   2015; Wagner, Ferrando & Young 2017; Stewart & Joy 1974; Kirby & Chen 1989;
   Vergeles & Vointsev 2024, 2026 (VY 2026 likely already present).

Docstrings: DocStringExtensions `$(SIGNATURES)`, jldoctest blocks (AGENTS.md).

---

## 5. Illustrative examples (Literate.jl, added to `EXAMPLE_TUTORIALS`)

Keep to four, each ≤ a few minutes on CPU, each with a Makie figure/movie:

1. **`narrow_band_packet_dispersion.jl`** — wave-only. A Gaussian packet with
   carrier κ: propagation at the group speed, dispersive spreading; overlay the
   analytic reconstituted dispersion; toggle α = 0 to *show* what reconstitution
   buys. The "hello world" of the model; twin of the existing
   `bounded_wave_packet_dispersion.jl` action-model example.
2. **`narrow_band_vortex_scattering.jl`** — prescribed barotropic Gaussian
   vortex (same current as the existing `vortex_refraction.jl`). Plane wave in →
   scattered A field, wave-height focusing/defocusing pattern. Side panel:
   Ripple's own `SpectralWaveModel` on the identical current — the WKB vs
   non-WKB comparison in one movie. Run twice: current scale ≫ 2π/κ (agreement)
   and ~ 2π/κ (departure). Flagship pedagogical example.
3. **`langmuir_circulation_two_way.jl`** — the paper's raison d'être. McWilliams
   et al. (1997)-style setup (λ = 60 m, 150×150×90 m box, aligned wind stress),
   but with A prognostic: CL2 instability, Langmuir cells, and the energy
   exchange time series d/dt(wave energy) = −∭ Uᴸ·∂ₜUˢ; compare against a frozen-A
   (prescribed-Stokes) twin run to show what bidirectionality changes — exactly
   the numerical experiment the paper's §7 calls for.
4. **`narrow_band_ring_spectrum.jl`** — isotropic narrow-band ring IC with random
   phases over a prescribed mesoscale strain field: directional redistribution
   and statistical wave–current scattering; demonstrates the "narrow-band in
   frequency, isotropic in direction" design point and the bandwidth-monitor
   diagnostic.

---

## 6. Unit tests (`test/narrow_band/`, wired into `test/runtests.jl`)

Conventions: `using Test, Ripple`, nested `@testset`s, `@test_throws
ArgumentError` for constructor validation, no explicit imports.

**`dispersion.jl`**
- ω, ω_κ, ω_κκ vs central finite differences of ω(κ) across κh ∈ [10⁻², 10²].
- Deep-water limits: ω_κ → ω/2κ, α → −3/(8κ²); shallow limit α → −1/(4κ²).
- 1 + ακ² > 0 and screened-Poisson well-posedness (λ − m > 0) scan over κh.
- Φ normalization: analytic cell-integrated weights sum to 1 exactly (not
  quadrature-approximately) for uniform and stretched z-spacing.
- Effective velocity: vᵉᶠᶠ → Ū in deep water; reconstituted dispersion
  ω + s(|k|) matches exact ω(|k|) to third order in (|k| − κ) (continuous
  symbol), and the discrete Λ_d matches with the FD modified wavenumber.

**`helmholtz_solve.jl`**
- Applying (1 + ακ²) + α∇²ᶜᶜᶜ (centered operators) to the output of
  `solve!(A, solver, G/α, m)` recovers G to solver tolerance on random fields —
  the operator/eigenvalue consistency check, both components.
- Plane-wave eigenvalues; `Flat`-z companion-grid degeneration; GPU smoke.

**`wave_operators.jl`**
- The rewrite is exact: assemble 𝓛 (a) in the paper's symmetrized form with
  centered operators and (b) as flux divergence + compressibility source + R;
  the two agree to roundoff for random smooth (A, Ū, Ũ) when the transport
  scheme is `Centered` and stencils match.
- **G-form/A-form equivalence**: stepping G and diagnosing A reproduces stepping
  A with M⁻¹-applied tendencies to roundoff over many steps.
- R on manufactured (A, Ū) vs an independent high-order-FD/symbolic reference.
- Discrete action identity with `Centered`: the quadratic form
  Re ∬ A† 𝓛(U, A) combination vanishes to stencil-consistent tolerance.
- WENO carrier-damping characterization: plane wave transported by uniform vᵉᶠᶠ;
  measured decay rate vs points-per-wavelength; assert the |vᵉᶠᶠ|-scaling and
  the expected high-order Δx convergence. (Doubles as the docs figure.)
- Discrete energy-exchange identity: the wave-side quadratic form equals
  −∭ Uᴸ·∂ₜUˢ term-by-term (spatial half of 6.3), `Centered` configuration.

**`vertical_projection.jl`**
- Ū, Ũ against analytic integrals for polynomial and exponential Uᴸ(z);
  face-location evaluation.
- Deep-water Ũ = Ū; convergence of the deep-water path to the finite-depth path
  as κh grows.

**`stokes_drift.jl`**
- Plane wave A = a e^{iκx₁}: Uˢ matches the classical finite-depth Stokes
  profile, and the amplitude dictionary maps a → surface-elevation amplitude a₀
  with Uˢ(0) = a₀²ωκ·coth-form (deep water: a₀²ωκ e^{2κz}).
- ∂ₜUˢ from the (A, ∂ₜA) quadratic form vs time finite-differencing of Uˢ across
  a short integration.
- `NarrowBandStokesDrift` kernel functions vs directly assembled 3-D fields on a
  small grid (validates the separable evaluation and the Oceananigans interface
  overloads, including the curl terms).

**`model_api.jl` (integration)**
- Constructor validation (missing κ, non-periodic or stretched-horizontal
  topology, bad advection option); exports present; `fields`/`prognostic_fields`
  (G pair prognostic, A pair diagnostic); the complex `amplitude` view
  round-trips `set!` with complex data; `time_step!` smoke for all velocity
  modes and both advection schemes; `WaveCurrentModel` +
  `Oceananigans.Simulation` smoke.
- GPU smoke additions to `scripts/gpu/run_cuda_smoke.jl`.

---

## 7. Consistency tests (nontrivial IVPs; conservation + structure)

Live in `test/narrow_band/conservation.jl` with tight-tolerance small cases, and
as `default_validation_cases()` entries (`src/Validation/`) with production-size
versions: `narrow_band_action`, `narrow_band_coupled_energy`,
`narrow_band_coupled_momentum`, `narrow_band_reversibility`.

Standard nontrivial IVP kit (fixed seeds):
- **A₀**: ring spectrum |k| ≈ κ, Gaussian radial profile (bandwidth Δk/κ ≈ 0.1),
  uniform random phases, isotropic in direction.
- **U₀**: random smooth divergence-free 3-D field (solenoidal projection of
  band-limited noise, surface-intensified) + a barotropic dipole — no symmetry,
  all terms of R and the vortex force active.

Unless stated otherwise, conservation assertions use the **`Centered`
configuration** (the conservative discretization); each test also runs the
default WENO configuration and asserts *sign-definite, resolution-convergent*
action decay instead of conservation.

Tests:

1. **Wave action, prescribed frozen current.** d𝒜/dt = 0 with 𝒜 = Re⟨A, G⟩.
   With `Centered` operators the spatial residual is stencil-consistent, so
   assert (a) |Δ𝒜|/𝒜 < tol over ≥ 50 carrier periods at working Δt and
   resolution, with joint (Δt, Δx) convergence at the expected orders, (b)
   third-order convergence of the drift in Δt at fixed resolution, (c)
   invariance holds also with *time-dependent* prescribed U (action is conserved
   for any U — the sharper statement), (d) WENO: 𝒜 decays monotonically, decay
   → 0 with resolution.
2. **Coupled energy, f = 0 and f ≠ 0.** Run `WaveCurrentModel`; track ℰ (6.5) and
   the two exchange identities (6.3), (6.4) separately — testing the pieces
   catches sign errors the total hides. Assert drift ≪ the integrated exchange
   |∫∭ Uᴸ·∂ₜUˢ| (the meaningful yardstick, since the wave energy term is
   sign-indefinite and small for narrow-band A), plus convergence order in Δt.
   Run both coupling modes: callback-coupled (expect O(Δt) drift — documented,
   not asserted tight) and stage-synchronized (assert order ≥ 2).
3. **Momentum, f = 0.** 𝒫 = 𝒫ʷ + ∭U (6.10): drift and Δt-convergence as above;
   verify 𝒫ʷ reduces to ∭ Uˢ when A is Helmholtz-constrained ((∇²+κ²)A = 0),
   the paper's consistency remark after (6.10).
4. **Divergence and boundary structure.** max|∇·Uᴸ| at pressure-solver tolerance
   each step; Wᴸ = 0 at z = 0, −h.
5. **Time reversibility** (inviscid, `Centered`, f = 0 and f ≠ 0): integrate the
   coupled nontrivial IVP forward N steps, reverse (t → −t, conjugate A, negate
   U, f), integrate back; recover the IC to a tolerance that converges with Δt.
   A stringent whole-system test that catches dissipative bugs conservation
   sums can miss (and, run with WENO, *demonstrates* the upwind irreversibility).
6. **Narrow-band monitor**: the bandwidth diagnostic stays within its initial
   ballpark over the test window (guards against silent spectral spreading of
   energy to poorly resolved scales).

---

## 8. Validation cases

In-package (`src/Validation/` cases) and external (`validation/` directory
scripts + comparison plots), following the existing two-tier pattern.

1. **Linear dispersion accuracy** (in-package). Single-mode frequencies vs exact
   ω(|k|) across the resolved annulus, at several points-per-wavelength;
   reproduce the Thomas & Yamada (2018) reconstitution accuracy claim
   quantitatively (error curves with/without α) and separate the FD
   modified-wavenumber contribution. Doubles as the figure for the theory docs.
2. **WKB cross-validation vs `SpectralWaveModel`** (in-package; flagship).
   Prescribed slowly varying current (scale L ≫ 2π/κ): steady wave statistics
   (energy pattern, mean wavenumber vector) from `NarrowBandWaveModel` vs
   Ripple's action model with CWCM prescribed coupling on the same current.
   Quantify agreement → then shrink L/λ and document the WKB departure. This
   validates *both* Ripple models against each other in their overlap regime,
   and — since vᵉᶠᶠ is the same depth-weighted current that enters the action
   model's Doppler shift — checks the effective-transport identification
   directly.
3. **Amplification by an opposing jet** (in-package). Steady wave height change
   across a prescribed jet vs wave-action conservation / ray theory prediction
   (classic 1-D-in-x benchmark with an analytic answer in the WKB limit).
4. **Prescribed-Stokes CL regression** (external tier). Freeze A (drop the wave
   step): the mean-flow side must reproduce a reference Oceananigans Langmuir
   LES (McWilliams et al. 1997 setup with equivalent uniform Uˢ) — near
   machine-precision agreement of the vortex-force terms, since it's the same
   solver. Validates the `NarrowBandStokesDrift` plumbing in isolation.
5. **CL2 / Langmuir instability growth rates** (external tier). Linearized
   regime: monochromatic waves + small shear perturbation; measured growth rates
   vs classical CL2 theory (Leibovich 1977, 1980) with prescribed Uˢ, then with
   live A vs the wave-scattering-modified rates of Vergeles & Vointsev (2024,
   2026) — the paper's own positioning targets.
6. **Energy-exchange vs wave-resolving reference** (external tier, stretch).
   Qualitative/semi-quantitative comparison of the wave–current energy exchange
   against Fujiwara & Yoshikawa (2020) wave-resolving simulations — the
   phenomenon this model exists to capture. Success = right sign, magnitude, and
   spatial structure of the exchange; document honestly.
7. **Conservation suite** = §7 cases registered in `default_validation_cases()`.

---

## 9. Phasing and acceptance criteria

**Phase 0 — Scaffolding (small PR).** `src/NarrowBand/` skeleton, `dispersion.jl`
complete with unit tests, refs.bib entries, notation-page update. Spike: the
screened-Poisson solve on the `Flat`-z companion grid via
`solve!(A, solver, G/α, m)`, CPU+GPU. *Accept:* dispersion + helmholtz_solve
tests green on CPU/GPU.

**Phase 1 — Linear wave model (U = 0).** (G, A) real pairs + complex `amplitude`
view, local dispersion tendency + diagnostic solve, RK3, `NarrowBandWaveModel`
with `velocities = nothing`, packet example, dispersion validation case.
*Accept:* single-mode frequencies match Λ_d to roundoff; packet example renders;
𝒜 = Re⟨A,G⟩ conserved to time-discretization error (U = 0 ⇒ no transport,
purely dispersive).

**Phase 2 — Prescribed current.** Vertical projections at faces, vᵉᶠᶠ +
`div_Uc`/WENO transport, compressibility source, fused R kernel,
`PrescribedVelocities` mode, vortex-scattering example, action-conservation
consistency tests (Centered + WENO variants), carrier-damping characterization,
WKB cross-validation + jet-amplification validation cases. *Accept:* consistency
test 1 and validation cases 1–3 pass; rewrite-equivalence and G/A-form
equivalence tests pass.

**Phase 3 — Two-way coupling.** Stokes functionals + `NarrowBandStokesDrift` +
∂ₜUˢ; (3a) callback-coupled `WaveCurrentModel`, prescribed-Stokes regression;
(3b) stage-synchronized stepper; energy/momentum/reversibility consistency
tests; Langmuir example. *Accept:* consistency tests 2–5 pass; validation case 4
at machine precision; Langmuir two-way example produces cells + exchange
diagnostics.

**Phase 4 — Docs, validation hardening, performance.** Theory/numerics/
conservation/API doc pages with generated figures; ring-spectrum example;
CL2-growth-rate and (stretch) Fujiwara–Yoshikawa validation; GPU smoke +
performance entry via `run_performance_smoke`; README/AGENTS.md updates
describing the second model type. *Accept:* docs build with executed examples;
`run_validation` green including new cases.

Per AGENTS.md: feature branches per phase, tests+docs move with code, exports
updated in `src/Ripple.jl`, no new `Project.toml` dependencies expected — the
elliptic solve, transforms, advection, and operators all come through
Oceananigans.

---

## 10. Risks and open questions

1. **WENO dissipation of the carrier.** The default transport scheme damps the
   resolved carrier wave; mitigations are (i) the damping scales with the weak
   current speed |vᵉᶠᶠ|, not the phase speed, (ii) the required
   points-per-wavelength characterization test, (iii) `Centered` escape hatch.
   If damping proves material in the Langmuir regime, a different scheme is a
   one-kwarg change — that flexibility is the payoff of reusing Oceananigans
   advection.
2. **Stencil consistency between `div_Uc`, the R kernel, and the solve.** The
   conservation structure relies on discrete operators mimicking the continuous
   integration-by-parts. The (G, A) formulation removes the mass-matrix/
   dispersion mismatch by construction; the remaining hazard is face
   interpolants in `div_Uc` vs centered derivatives in R — the
   rewrite-equivalence and discrete quadratic-form unit tests (§6) are designed
   to catch exactly this.
3. **Symbol collision**: κ is Ripple's spectral *coordinate* name and OF's fixed
   *carrier* parameter. Keep `κ` for the carrier (paper-faithful) but document
   the distinction loudly; the two models don't share fields so the collision is
   cognitive, not programmatic.
4. **Splitting error vs conservation targets** — mitigated by designing the
   stage-synchronized stepper up front (3b); the callback mode remains for
   robustness/bring-up.
5. **Finite-depth Stokes correction caveat** (paper §5.3): the 𝓛(Uᴸ,·) Stokes
   correction is justified for deep water; for κh = O(1) document that the
   third-order dispersion correction is not asymptotically controlled (the
   paper says so explicitly).
6. **Rigid lid**: (2.2) has no free surface; consistent with
   `NonhydrostaticModel`. Any future free-surface/HSFS pairing is out of scope.
7. **Buoyancy/viscosity**: paper §7 sanctions adding them to the CL equation
   directly (Oceananigans gives them for free via closures/tracers), possibly
   with linear damping on A for consistency — expose `wave_damping` as an
   optional linear term but default to inviscid; flag as physics-validation TODO.
8. **Uniform horizontal spacing** is required by the FFT solve (assert in the
   constructor). Stretched or bounded horizontal grids would move the solve to
   `FourierTridiagonalPoissonSolver` or the CG/Krylov solvers — deferred.

Deferred by design: bounded domains (the solver supports `Bounded` topologies
via DCT/staggered-Neumann when the physics BCs for A are worked out), immersed
boundaries (CG path), exponential integrators, wave–wave interactions (paper
§7), bathymetry.
