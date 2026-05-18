#####
##### Local-saturation whitecapping (precursor to Ardhuin's ST4 directional Sds).
#####
##### Per-cell dissipation rate depends on the LOCAL saturation
##### B(k,θ) = N(k,θ) · σ · k³ rather than a mean-spectrum measure. When B
##### exceeds a threshold B_r, the cell dissipates proportional to the overshoot.
##### Cells with B < B_r are not dissipated.
#####
##### This is structurally closer to WW3's ST4 than the WAM4 form: it dissipates
##### where the spectrum is "too steep", not uniformly. Result is a smoother
##### tail and a single (rather than spiky) spectral peak.
#####
#####     S_ds(k,θ) = -C_ds · σ · [max(0, B(k,θ)/B_r − 1)]^p · N(k,θ)
#####
##### Simplification vs full ST4: no directional sector integral (no `cos^sB`
##### projection), no cumulative breaking, no swell boundary-layer damping.

struct LocalSaturationDissipation{FT} <: AbstractSourceTerm
    C_ds    :: FT      # WW3 ST4 SDSC2 default: -2.2e-5 (negative for dissipation)
    B_r     :: FT      # saturation threshold
    p       :: FT      # exponent on the overshoot
    σ_power :: FT      # σ^q multiplier (q=1 in WW3; q>1 dampens high-f tail more)
    gravity :: FT
end

LocalSaturationDissipation(; C_ds=-2.2e-5, B_r=9.0e-4, p=2.0, σ_power=1.0, gravity=9.81) =
    LocalSaturationDissipation(float(C_ds), float(B_r), float(p),
                                float(σ_power), float(gravity))

function source_split(s::LocalSaturationDissipation, model, i, j, m, n)
    FT = eltype(model.action)
    N_a = model.action[i, j, m, n]
    N_a > 0 || return (zero(FT), zero(FT))

    kx, ky = k_components(model.spectral_grid, m, n)
    k = hypot(kx, ky)
    k > 0 || return (zero(FT), zero(FT))
    σ = sqrt(s.gravity * k)

    B = N_a * σ * k^3
    overshoot = max(B / s.B_r - 1, zero(FT))
    overshoot > 0 || return (zero(FT), zero(FT))
    overshoot = min(overshoot, FT(1e6))

    damping_rate = -s.C_ds * σ^s.σ_power * overshoot^s.p
    return split_damping_rate(damping_rate, N_a)
end

function source_tendency(s::LocalSaturationDissipation, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end
