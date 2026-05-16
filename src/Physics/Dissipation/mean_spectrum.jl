#####
##### Mean-spectrum whitecapping dissipation (WAM4 / Komen-style).
#####
##### Manual eq. 2.107: rate keyed off the mean wavenumber k̄ and mean steepness
##### α̅ = E·k̄² of the local spectrum, not on the local saturation. This is the
##### ST3 dissipation partner; ST4 replaces it with directional saturation.
#####
#####     S_ds(k,θ) = C_ds · α̅² · σ̄ · [δ₁ k/k̄ + δ₂ (k/k̄)²] · N(k,θ)
#####
##### where
#####
#####     E   = m₀ = ∫ N(k,θ) dk dθ
#####     k̄   = (∫ k^p N / ∫ N)^(1/p)
#####     σ̄   = (∫ σ^p N / ∫ N)^(1/p)
#####     α̅   = E · k̄²
#####
##### BJA defaults: C_ds = -2.1, δ₁ = 0.4, δ₂ = 0.6, p = 0.5.
##### ST3 historical defaults (WAM4): C_ds = -4.5, δ₁ = 0.5, δ₂ = 0.5.

struct MeanSpectrumWhitecapping{FT} <: AbstractDissipation
    C_ds    :: FT          # dimensionless dissipation strength (negative)
    δ₁      :: FT          # weight on linear k/k̄ term
    δ₂      :: FT          # weight on quadratic (k/k̄)² term
    p       :: FT          # exponent for generalized mean
    gravity :: FT
end

function MeanSpectrumWhitecapping(; C_ds=-2.1,
                                    δ₁=0.4,
                                    δ₂=0.6,
                                    p=0.5,
                                    gravity=9.81)
    MeanSpectrumWhitecapping(float(C_ds), float(δ₁), float(δ₂),
                              float(p), float(gravity))
end

#####
##### Per-cell bulk-moment helpers. These recompute the (i,j) integrals on every
##### source_split call — O(Nκ Nφ) work per spectral cell, so O((NκNφ)²) per (i,j).
##### Bundle path (MeanSpectrumPhysics) will precompute these once per (i,j) into
##### state fields.
#####
function local_pth_mean_wavenumber(model, i, j, p)
    N = model.action
    cgrid = model.spectral_grid
    _, _, Nxi, Neta = size(N)
    num = zero(eltype(N))
    den = zero(eltype(N))
    for n in 1:Neta, m in 1:Nxi
        action = N[i, j, m, n]
        action <= 0 && continue
        w = spectral_weight(cgrid, m, n)
        kx, ky = k_components(cgrid, m, n)
        k = hypot(kx, ky)
        k <= 0 && continue
        num += action * w * k^p
        den += action * w
    end
    den > 0 || return zero(eltype(N))
    return (num / den)^(1 / p)
end

function local_pth_mean_frequency(model, i, j, p, gravity)
    N = model.action
    cgrid = model.spectral_grid
    _, _, Nxi, Neta = size(N)
    num = zero(eltype(N))
    den = zero(eltype(N))
    for n in 1:Neta, m in 1:Nxi
        action = N[i, j, m, n]
        action <= 0 && continue
        w = spectral_weight(cgrid, m, n)
        kx, ky = k_components(cgrid, m, n)
        k = hypot(kx, ky)
        k <= 0 && continue
        σ = sqrt(gravity * k)
        num += action * w * σ^p
        den += action * w
    end
    den > 0 || return zero(eltype(N))
    return (num / den)^(1 / p)
end

function source_split(s::MeanSpectrumWhitecapping, model, i, j, m, n)
    FT = eltype(model.action)
    E = local_zeroth_moment(model, i, j)
    E > 0 || return (zero(FT), zero(FT))

    k̄ = local_pth_mean_wavenumber(model, i, j, s.p)
    k̄ > 0 || return (zero(FT), zero(FT))
    σ̄ = local_pth_mean_frequency(model, i, j, s.p, s.gravity)

    α̅ = E * k̄^2

    kx, ky = k_components(model.spectral_grid, m, n)
    k = hypot(kx, ky)
    k > 0 || return (zero(FT), zero(FT))

    ratio = k / k̄
    bracket = s.δ₁ * ratio + s.δ₂ * ratio^2
    # S_ds = C_ds·α̅²·σ̄·bracket·N. C_ds < 0 by convention for dissipation, so
    # the damping rate λ = -S_ds/N = -C_ds·α̅²·σ̄·bracket (positive).
    damping_rate = -s.C_ds * α̅^2 * σ̄ * bracket
    return split_damping_rate(damping_rate, model.action[i, j, m, n])
end

function source_tendency(s::MeanSpectrumWhitecapping, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

#####
##### State-aware path: reads precomputed (m₀, k̄, σ̄) from the bundle state
##### rather than re-integrating over the spectral domain on every cell. The
##### per-cell cost drops from O(Nκ Nφ) to O(1).
#####
function source_split(s::MeanSpectrumWhitecapping, state::NamedTuple, model, i, j, m, n)
    FT = eltype(model.action)
    m₀ = state.m₀[i, j]
    m₀ > 0 || return (zero(FT), zero(FT))
    k̄ = state.k̄[i, j]
    k̄ > 0 || return (zero(FT), zero(FT))
    σ̄ = state.σ̄[i, j]

    α̅ = m₀ * k̄^2

    kx, ky = k_components(model.spectral_grid, m, n)
    k = hypot(kx, ky)
    k > 0 || return (zero(FT), zero(FT))

    ratio = k / k̄
    bracket = s.δ₁ * ratio + s.δ₂ * ratio^2
    damping_rate = -s.C_ds * α̅^2 * σ̄ * bracket
    return split_damping_rate(damping_rate, model.action[i, j, m, n])
end

function source_tendency(s::MeanSpectrumWhitecapping, state::NamedTuple, model, i, j, m, n)
    positive, damping = source_split(s, state, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end
