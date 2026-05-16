#####
##### Pressure-correlation wind input (Janssen quasi-linear theory).
#####
##### Source term for wave growth driven by wave-induced air-pressure perturbations
##### in phase with the wave slope. WW3 ST3 (WAM4/BJA, manual eq. 2.99) uses this
##### kernel; ST4 adds a wave-supported-stress sheltering reduction on top.
#####
##### Per-cell formula:
#####
#####     S_in(k,θ) = (ρa/ρw) (β_max / κ²) e^Z Z⁴ (u*/C + zα)² cos^pin(θ - θ_u) σ N
#####
##### where
#####
#####     Z   = log(k·z₁) + κ_vk / [(u*/C + zα) · cos(θ - θ_u)]
#####     z₁  = α₀ · u*² / g       (ST3-lite: τ_w-independent Charnock roughness)
#####     κ_vk= von-Kármán constant ≈ 0.4
#####
##### BJA defaults: β_max = 1.2, zα = 0.011, p_in = 2, α₀ = 0.0095, ZWND = 10 m.
#####
##### This implementation is "ST3-lite": uses bulk drag (BulkWindDrag) for u*
##### rather than iterative wave-supported drag. ST4 sheltering (s_u > 0)
##### requires a precomputed Sin/c integral and lives in the bundle path.

struct PressureCorrelationInput{Drag, Wind, Dir, FT} <: AbstractWindInput
    drag       :: Drag        # AbstractDrag for u* (typically BulkWindDrag)
    wind       :: Wind        # scalar speed OR wind-field struct (Vortex/Hurricane/...)
    direction  :: Dir         # scalar angle [rad] used only when `wind` is a scalar speed;
                              # ignored if `wind` is a struct with its own `wind_angle`
    β_max      :: FT          # BJA: 1.2
    z_α        :: FT          # BJA: 0.011 (wave-age tuning shift)
    p_in       :: FT          # BJA: 2  (cos^p directional)
    α₀         :: FT          # BJA: 0.0095 (Charnock-like)
    ρ_air      :: FT
    ρ_water    :: FT
    von_karman :: FT          # κ_vk ≈ 0.4
    gravity    :: FT
end

function PressureCorrelationInput(; drag, wind, direction=0.0,
                                    β_max=1.2,
                                    z_α=0.011,
                                    p_in=2.0,
                                    α₀=0.0095,
                                    ρ_air=1.225,
                                    ρ_water=1025.0,
                                    von_karman=0.4,
                                    gravity=9.81)
    PressureCorrelationInput(drag, wind, float(direction),
                             float(β_max), float(z_α), float(p_in),
                             float(α₀), float(ρ_air), float(ρ_water),
                             float(von_karman), float(gravity))
end

# Pull U10 magnitude from a wind specification at grid point (i, j).
pressure_correlation_wind_speed(w::Number, model, i, j) = w
pressure_correlation_wind_speed(w, model, i, j) =
    wind_speed(w, xnodes(model.grid)[i], ynodes(model.grid)[j], model.clock.time)

# Pull wind direction (radians). Scalar wind needs its `direction` field;
# wind-field structs (Hurricane / Vortex) provide their own `wind_angle`.
pressure_correlation_wind_dir(w::Number, dir, model, i, j) = dir
pressure_correlation_wind_dir(w, dir, model, i, j) =
    wind_angle(w, xnodes(model.grid)[i], ynodes(model.grid)[j], model.clock.time)

function source_split(s::PressureCorrelationInput, model, i, j, m, n)
    FT = eltype(model.action)
    U10 = pressure_correlation_wind_speed(s.wind, model, i, j)
    U10 > 0 || return (zero(FT), zero(FT))

    θ_u = pressure_correlation_wind_dir(s.wind, s.direction, model, i, j)
    u_star = friction_velocity(s.drag, U10)
    u_star > 0 || return (zero(FT), zero(FT))

    k_x, k_y = k_components(model.spectral_grid, m, n)
    k = hypot(k_x, k_y)
    k > 0 || return (zero(FT), zero(FT))
    σ = sqrt(s.gravity * k)
    C = σ / k
    θ = atan(k_y, k_x)

    cos_θu = cos(θ - θ_u)
    cos_θu > 0 || return (zero(FT), zero(FT))

    # Charnock-style roughness without τ_w feedback (ST3-lite).
    z₁ = s.α₀ * u_star^2 / s.gravity
    z₁ > 0 || return (zero(FT), zero(FT))

    inv_age = u_star / C + s.z_α
    Z = log(k * z₁) + s.von_karman / (inv_age * cos_θu)
    Z >= 0 && return (zero(FT), zero(FT))    # only Z < 0 gives a positive growth rate

    rate = (s.ρ_air / s.ρ_water) * (s.β_max / s.von_karman^2) *
           exp(Z) * Z^4 * inv_age^2 *
           cos_θu^s.p_in * σ

    return split_growth_rate(rate, model.action[i, j, m, n])
end

function source_tendency(s::PressureCorrelationInput, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

#####
##### Per-cell raw growth rate, without state. Used by the bundle to integrate
##### the wave-supported stress τ_w and apply the stress-cap. Returns 0 when
##### the cell is outside the growth window.
#####
function pressure_correlation_raw_rate(s::PressureCorrelationInput, model, i, j, m, n)
    FT = eltype(model.action)
    U10 = pressure_correlation_wind_speed(s.wind, model, i, j)
    U10 > 0 || return zero(FT)

    θ_u = pressure_correlation_wind_dir(s.wind, s.direction, model, i, j)
    u_star = friction_velocity(s.drag, U10)
    u_star > 0 || return zero(FT)

    k_x, k_y = k_components(model.spectral_grid, m, n)
    k = hypot(k_x, k_y)
    k > 0 || return zero(FT)
    σ = sqrt(s.gravity * k)
    C = σ / k
    θ = atan(k_y, k_x)

    cos_θu = cos(θ - θ_u)
    cos_θu > 0 || return zero(FT)
    z₁ = s.α₀ * u_star^2 / s.gravity
    z₁ > 0 || return zero(FT)

    inv_age = u_star / C + s.z_α
    Z = log(k * z₁) + s.von_karman / (inv_age * cos_θu)
    Z >= 0 && return zero(FT)

    return (s.ρ_air / s.ρ_water) * (s.β_max / s.von_karman^2) *
           exp(Z) * Z^4 * inv_age^2 * cos_θu^s.p_in * σ
end

#####
##### State-aware source_split with stress cap. `state.stress_factor[i,j]` is a
##### scalar in [0, 1] that scales the raw rate down to satisfy τ_w ≤ τ_max =
##### ρ_a u*². Computed once per (i,j) in `prepare_physics`. This is a one-pass
##### approximation of WW3's iterative τ_w feedback — enough to suppress the
##### bistability that makes ST3-lite overshoot at moderate winds.
#####
function source_split(s::PressureCorrelationInput, state::NamedTuple, model, i, j, m, n)
    FT = eltype(model.action)
    raw = pressure_correlation_raw_rate(s, model, i, j, m, n)
    raw > 0 || return (zero(FT), zero(FT))
    factor = state.stress_factor[i, j]
    return split_growth_rate(raw * factor, model.action[i, j, m, n])
end

function source_tendency(s::PressureCorrelationInput, state::NamedTuple, model, i, j, m, n)
    positive, damping = source_split(s, state, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end
