struct JONSWAPSpectrum{FT}
    Hs :: FT
    Tp :: FT
    direction :: FT
    spread :: FT
    gamma :: FT
end

function JONSWAPSpectrum(; Hs=2.0, Tp=8.0, direction=0.0, spread=0.35, gamma=3.3)
    values = promote(float(Hs), float(Tp), float(direction), float(spread), float(gamma))
    Hs_value, Tp_value, _, spread_value, gamma_value = values
    Hs_value >= 0 || throw(ArgumentError("JONSWAP significant wave height must be nonnegative"))
    Tp_value > 0 || throw(ArgumentError("JONSWAP peak period must be positive"))
    spread_value > 0 || throw(ArgumentError("JONSWAP directional spread must be positive"))
    gamma_value > 0 || throw(ArgumentError("JONSWAP peak-enhancement factor must be positive"))
    return JONSWAPSpectrum(values...)
end

function (s::JONSWAPSpectrum)(x, y, kx, ky)
    g = 9.81
    k = hypot(kx, ky)
    k == 0 && return zero(k)
    omega = sqrt(g * k)
    omega_p = 2pi / s.Tp
    sigma = omega <= omega_p ? 0.07 : 0.09
    r = exp(-((omega / omega_p - 1)^2) / (2sigma^2))
    alpha = 0.076 * (s.Hs^2 * omega_p^4 / g^2)^0.22
    one_d = alpha * g^2 * omega^-5 * exp(-1.25 * (omega_p / omega)^4) * s.gamma^r
    φ = atan(ky, kx)
    Δφ = atan(sin(φ - s.direction), cos(φ - s.direction))
    directional = exp(-0.5 * (Δφ / s.spread)^2) / (sqrt(2pi) * s.spread)
    return max(one_d * directional, zero(one_d))
end
