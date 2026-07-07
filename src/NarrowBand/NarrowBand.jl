# Narrow-band amplitude wave–current model (Onuki & Fujiwara 2026).
#
# Phase 0 (scaffolding): the carrier dispersion relation and its derivatives,
# and the diagnostic screened-Poisson solve that inverts the reconstitution
# operator [1 + α(∇ₕ² + κ²)]. Later phases add the (G, A) prognostic pair, the
# Doppler/refraction operators, the Stokes-drift functionals, and the coupled
# `WaveCurrentModel`. See docs/design/narrow_band_model_plan.md.

include("dispersion.jl")
include("helmholtz_solve.jl")
include("vertical_projection.jl")
include("current_coupling.jl")
include("narrow_band_wave_model.jl")
include("wave_operators.jl")
include("stokes_drift.jl")
include("time_step.jl")
include("coupled_model.jl")
