#####
##### Air-sea drag and friction velocity.
#####
##### Three flavors of drag → u* mapping:
#####
#####   `BulkWindDrag{:linear}`  — Wu (1982): Cd × 10⁻³ = 0.8 + 0.065·U10
#####   `BulkWindDrag{:capped}`  — Hwang (2001): Cd × 10⁻⁴ = 8.058 + 0.967·U10 − 0.016·U10²,
#####                              floored to prevent u* → 0 at hurricane winds
#####   `WaveSupportedDrag`      — Janssen quasi-linear: Cd depends on wave-supported stress
#####                              fraction τ_w/τ_total via a precomputed lookup table
#####
##### The interface is `friction_velocity(drag, U10[, τ_w_fraction])`. Bulk forms
##### ignore the wave-stress fraction; the wave-supported form needs it for the
##### iterative closure described in WW3 §2.3.9.

abstract type AbstractDrag end

#####
##### Bulk (wave-state-independent) drag.
#####
struct BulkWindDrag{Form} <: AbstractDrag end

BulkWindDrag(form::Symbol=:linear) = BulkWindDrag{form}()

drag_coefficient(::BulkWindDrag{:linear}, U10) =
    (0.8 + 0.065 * U10) * 1e-3

function drag_coefficient(::BulkWindDrag{:capped}, U10)
    Cd = (8.058 + 0.967 * U10 - 0.016 * U10^2) * 1e-4
    return max(Cd, 4e-4)        # floor: Hwang's curve undershoots at U10 ~ 50 m/s
end

friction_velocity(drag::AbstractDrag, U10) =
    U10 * sqrt(max(drag_coefficient(drag, U10), zero(U10)))

#####
##### Wave-supported (quasi-linear) drag.
#####
##### z₁ = α₀ τ / sqrt(1 - τ_w/τ),  U10 = (u*/κ) log(z_u/z₁).
##### Stored as a precomputed table over (u*, τ_w/τ_total). At runtime the
##### wind-input kernel queries this table; calibration of α₀ (ALPHA0) and the
##### Charnock-like coefficient is per-package.
#####
struct WaveSupportedDrag{FT, T} <: AbstractDrag
    alpha0 :: FT
    z_u    :: FT       # reference height for U10 (10 m by default)
    table  :: T        # 2D lookup over (u*, τ_w/τ_total) → Cd
end

WaveSupportedDrag(; alpha0=0.0095, z_u=10.0, table=nothing) =
    WaveSupportedDrag(float(alpha0), float(z_u), table)

# Placeholder: implementation builds the τ_w lookup at init when ST3/ST4 lands.
drag_coefficient(::WaveSupportedDrag, U10) =
    error("WaveSupportedDrag requires τ_w/τ_total; use friction_velocity(drag, U10, τ_w_fraction)")

function friction_velocity(::WaveSupportedDrag, U10, τ_w_fraction)
    error("WaveSupportedDrag iteration not yet implemented; lands with ST3 wind input")
end
