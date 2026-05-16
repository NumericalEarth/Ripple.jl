#####
##### Physics bundle interface.
#####
##### A `PhysicsPackage` (= `AbstractPhysicsBundle`) groups wind input,
##### dissipation, nonlinear-interaction, and other terms whose evaluation can
##### share per-grid-point state (bulk moments, drag, wave-supported stress, ...).
#####
##### Two-level evaluation:
#####
#####     prepare_physics(bundle, model) -> state   # 2D physical-grid fields
#####     source_split(bundle, state, model, i, j, m, n) -> (S⁺, λ)
#####
##### The state is computed once per time step via `Reduction` / dedicated kernels
##### (Oceananigans pattern: closures-with-state). The fused 4D tendency kernel
##### reads from `state` per spectral cell.
#####
##### `GenericPhysics` (tuple wrapper) and any non-bundle term return `nothing`
##### from `prepare_physics`; the state-free fallbacks below dispatch back to
##### the per-term `source_split(::T, model, ...)` methods. Compiler
##### specialization elides the precompute pass on this branch.

# Default: no shared state.
prepare_physics(::AbstractPhysicsTerm, model) = nothing
prepare_physics(::Nothing, model) = nothing

# State-aware fallbacks. Any term + `nothing`-state falls through to the
# state-free signature. Bundles that need state override these.
source_split(t::AbstractPhysicsTerm, ::Nothing, model, i, j, m, n) =
    source_split(t, model, i, j, m, n)
source_tendency(t::AbstractPhysicsTerm, ::Nothing, model, i, j, m, n) =
    source_tendency(t, model, i, j, m, n)

source_split(::Nothing, ::Nothing, model, i, j, m, n) =
    (zero(eltype(model.action)), zero(eltype(model.action)))
source_tendency(::Nothing, ::Nothing, model, i, j, m, n) =
    zero(eltype(model.action))

implicit_source_rate(t::AbstractPhysicsTerm, ::Nothing, model, i, j, m, n) =
    implicit_source_rate(t, model, i, j, m, n)
implicit_source_rate(::Nothing, ::Nothing, model, i, j, m, n) =
    zero(eltype(model.action))
