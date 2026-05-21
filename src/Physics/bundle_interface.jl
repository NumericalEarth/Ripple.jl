#####
##### Bundle interface — minimal extension to the source-term framework.
#####
##### A bundle (e.g. `PrecomputedSources`) groups wind input, dissipation,
##### and nonlinear-interaction terms whose evaluation can share precomputed
##### per-grid-point state (bulk moments, wave-supported stress cap, the
##### DIA transfer field). Two-level evaluation:
#####
#####     state = prepare_sources(bundle, model)                # KA kernels
#####     source_tendency(bundle, state, model, i, j, m, n)     # per-cell read
#####
##### `prepare_sources(::Any, model) = nothing` and the state-aware
##### `source_tendency(..., ::Nothing, ...)` fallback are declared in
##### `src/Models/tendencies.jl` so the framework can reference them without a
##### Physics/ load-order dependency.
#####
##### Bundles that compose existing source terms loop through their members
##### with the matching per-term state slot. Members without precomputed state
##### (slot = `nothing`) fall through to the state-free `source_split(t, model,
##### …)` / `source_tendency(t, model, …)` via these fallbacks.

source_split(t::AbstractSourceTerm, ::Nothing, model, i, j, m, n) =
    source_split(t, model, i, j, m, n)
source_tendency(t::AbstractSourceTerm, ::Nothing, model, i, j, m, n) =
    source_tendency(t, model, i, j, m, n)
implicit_source_rate(t::AbstractSourceTerm, ::Nothing, model, i, j, m, n) =
    implicit_source_rate(t, model, i, j, m, n)
