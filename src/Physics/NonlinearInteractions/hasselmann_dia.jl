#####
##### Hasselmann discrete interaction approximation (Snl1) — GPU-compatible.
#####
##### Single-λ quadruplet truncation of the 4-wave Boltzmann integral
##### (Hasselmann et al. 1985, WW3 `w3snl1md.F90`). For a donor at (k_a, θ_a):
#####
#####     σ_± = (1 ± λ) σ_a,   |k_±| = (1 ± λ)² |k_a|
#####     θ_+ = θ_a ± Δθ_+,    θ_- = θ_a ∓ Δθ_-
#####
##### **Unit convention (matches WW3 W3SNL1):** energy density E(f,θ) per direction,
##### `E = (2π σ / c_g) · N` = `4π σ²/g · N` for deep water. Kernel:
#####
#####     T_E(k_a) = C/g⁴ · f_a^11 · E_a · [E_a·(E_+·DAL1 + E_-·DAL2) − E_+·E_-·DAL3]
#####
##### with DAL1 = (1+λ)⁻⁴, DAL2 = (1-λ)⁻⁴, DAL3 = 2·DAL1·DAL2, f = σ/(2π).
##### Bilinear 4-corner spread of receivers matching WW3's IP/IM index pattern.
#####
##### Implementation notes:
#####   - Frequency lookup is analytical (log-XFR formula) — no `argmin`/allocation.
#####   - Direction lookup is analytical (uniform Δφ).
#####   - Donor depletion uses an atomic add (multiple `(m_a, n_a, orientation)`
#####     threads can target the same receiver cell), via `Atomix.@atomic`.
#####   - Receiver gains use atomic adds too.
#####
##### Partial-quadruplet edge handling: if σ_+ falls off the top of the grid but
##### σ_- is in range (or vice versa), the donor still depletes at half rate and
##### the in-range receiver gains. Lets energy flow down off the diagnostic
##### tail without pooling at f_max.

import KernelAbstractions
import KernelAbstractions: @kernel, @index
import Oceananigans.Architectures: architecture, device
using Atomix: @atomic

struct HasselmannDIA{FT} <: AbstractSourceTerm
    C        :: FT
    λ        :: FT
    Δθ_plus  :: FT
    Δθ_minus :: FT
    gravity  :: FT
end

function HasselmannDIA(; C=2.78e7, λ=0.25, gravity=9.81)
    a = (1 + λ)^2
    b = (1 - λ)^2
    cosp = (a^2 - b^2 + 4) / (4a)
    cosm = (b^2 - a^2 + 4) / (4b)
    Δθ_plus  = acos(clamp(cosp, -1.0, 1.0))
    Δθ_minus = acos(clamp(cosm, -1.0, 1.0))
    HasselmannDIA(float(C), float(λ), float(Δθ_plus), float(Δθ_minus), float(gravity))
end

# Energy ↔ action conversion at angular frequency σ (deep water): E = (4πσ²/g)·N.
@inline _energy_factor(σ, g) = 4 * Float64(pi) * σ^2 / g

#####
##### Analytical bilinear-interpolation index lookup on a geometric f-grid +
##### uniform-Δφ periodic direction grid.
#####
##### Returns the 4 bilinear-corner indices/weights (m_lo, n_lo, w00, etc.) and
##### a flag `in_range` (true if σ_target ∈ [σ_min, σ_max]). All scalar, no
##### allocations — GPU-callable.
#####
@inline function _bilinear_lookup(σ_target, θ_target, σ_min, σ_max,
                                   log_xfr, φ_min, Δφ, Nκ, Nφ)
    if σ_target < σ_min || σ_target > σ_max
        return false, 0, 0, 0, 0, zero(σ_target), zero(σ_target), zero(σ_target), zero(σ_target)
    end
    m_real = one(σ_target) + log(σ_target / σ_min) / log_xfr
    m_lo   = clamp(unsafe_trunc(Int, m_real), 1, Nκ - 1)
    w_f    = m_real - m_lo
    m_hi   = m_lo + 1

    n_real_frac = mod(θ_target - φ_min, 2 * pi) / Δφ
    n_lo_raw    = unsafe_trunc(Int, n_real_frac)
    n_lo        = mod(n_lo_raw, Nφ) + 1
    w_θ         = n_real_frac - n_lo_raw
    n_hi        = n_lo == Nφ ? 1 : n_lo + 1

    w00 = (1 - w_f) * (1 - w_θ)
    w10 =      w_f  * (1 - w_θ)
    w01 = (1 - w_f) *      w_θ
    w11 =      w_f  *      w_θ
    return true, m_lo, m_hi, n_lo, n_hi, w00, w10, w01, w11
end

#####
##### Per-cell donor kernel.
#####
##### Each thread visits one (i, j, m_a, n_a) donor cell, computes the DIA
##### transfer T_E for both quadruplet orientations, and atomically updates
##### the donor cell and the 4 bilinear-corner cells of each receiver.
#####
@kernel function _dia_transfer_kernel!(
        transfer, N_data,
        @Const(σ_centers), @Const(k_centers), @Const(φ_centers),
        σ_min, σ_max, log_xfr, φ_min, Δφ,
        Hx, Hy, iz, Nκ, Nφ,
        C_eff, λ, Δθ_plus, Δθ_minus,
        DAL1, DAL2, DAL3, g)
    i, j, m_a, n_a = @index(Global, NTuple)
    ix = i + Hx; jy = j + Hy

    @inbounds N_a = N_data[ix, jy, iz, m_a, n_a]
    if N_a > 0
        @inbounds k_a = k_centers[m_a]
        σ_a  = sqrt(g * k_a)
        f_a  = σ_a / (2 * Float64(pi))
        @inbounds θ_a = φ_centers[n_a]
        Ef_a = _energy_factor(σ_a, g)
        E_a  = Ef_a * N_a

        σ_plus  = (1 + λ) * σ_a
        σ_minus = (1 - λ) * σ_a

        for orientation in (1, -1)
            θ_plus  = θ_a + orientation * Δθ_plus
            θ_minus = θ_a - orientation * Δθ_minus

            in_p, mlp, mhp, nlp, nhp, wp00, wp10, wp01, wp11 =
                _bilinear_lookup(σ_plus,  θ_plus,  σ_min, σ_max, log_xfr, φ_min, Δφ, Nκ, Nφ)
            in_m, mlm, mhm, nlm, nhm, wm00, wm10, wm01, wm11 =
                _bilinear_lookup(σ_minus, θ_minus, σ_min, σ_max, log_xfr, φ_min, Δφ, Nκ, Nφ)

            if in_p || in_m
                E_p = zero(N_a)
                if in_p
                    @inbounds begin
                        σ_p1 = sqrt(g * k_centers[mlp]); σ_p2 = sqrt(g * k_centers[mhp])
                        Efp1 = _energy_factor(σ_p1, g);  Efp2 = _energy_factor(σ_p2, g)
                        E_p = wp00 * Efp1 * max(N_data[ix, jy, iz, mlp, nlp], zero(N_a)) +
                              wp10 * Efp2 * max(N_data[ix, jy, iz, mhp, nlp], zero(N_a)) +
                              wp01 * Efp1 * max(N_data[ix, jy, iz, mlp, nhp], zero(N_a)) +
                              wp11 * Efp2 * max(N_data[ix, jy, iz, mhp, nhp], zero(N_a))
                    end
                end
                E_m = zero(N_a)
                if in_m
                    @inbounds begin
                        σ_m1 = sqrt(g * k_centers[mlm]); σ_m2 = sqrt(g * k_centers[mhm])
                        Efm1 = _energy_factor(σ_m1, g);  Efm2 = _energy_factor(σ_m2, g)
                        E_m = wm00 * Efm1 * max(N_data[ix, jy, iz, mlm, nlm], zero(N_a)) +
                              wm10 * Efm2 * max(N_data[ix, jy, iz, mhm, nlm], zero(N_a)) +
                              wm01 * Efm1 * max(N_data[ix, jy, iz, mlm, nhm], zero(N_a)) +
                              wm11 * Efm2 * max(N_data[ix, jy, iz, mhm, nhm], zero(N_a))
                    end
                end

                sa  = E_a * (E_p * DAL1 + E_m * DAL2) - E_p * E_m * DAL3
                T_E = C_eff * f_a^11 * sa / 2

                donor_factor = (in_p && in_m) ? 2 * one(T_E) : one(T_E)
                @atomic transfer[i, j, m_a, n_a] -= donor_factor * T_E / Ef_a

                if in_p
                    @inbounds begin
                        σ_p1 = sqrt(g * k_centers[mlp]); σ_p2 = sqrt(g * k_centers[mhp])
                        @atomic transfer[i, j, mlp, nlp] += wp00 * T_E / _energy_factor(σ_p1, g)
                        @atomic transfer[i, j, mhp, nlp] += wp10 * T_E / _energy_factor(σ_p2, g)
                        @atomic transfer[i, j, mlp, nhp] += wp01 * T_E / _energy_factor(σ_p1, g)
                        @atomic transfer[i, j, mhp, nhp] += wp11 * T_E / _energy_factor(σ_p2, g)
                    end
                end
                if in_m
                    @inbounds begin
                        σ_m1 = sqrt(g * k_centers[mlm]); σ_m2 = sqrt(g * k_centers[mhm])
                        @atomic transfer[i, j, mlm, nlm] += wm00 * T_E / _energy_factor(σ_m1, g)
                        @atomic transfer[i, j, mhm, nlm] += wm10 * T_E / _energy_factor(σ_m2, g)
                        @atomic transfer[i, j, mlm, nhm] += wm01 * T_E / _energy_factor(σ_m1, g)
                        @atomic transfer[i, j, mhm, nhm] += wm11 * T_E / _energy_factor(σ_m2, g)
                    end
                end
            end
        end
    end
end

function _compute_dia_transfer(s::HasselmannDIA, model)
    cgrid = model.spectral_grid
    cgrid isa FrequencyDirectionGrid || return zeros(eltype(model.action), 0, 0, 0, 0)
    grid  = model.grid
    arch  = architecture(grid)
    N     = model.action
    N_data = flat_data(N)
    Nx, Ny, Nκ, Nφ = size(N)
    FT = eltype(N)

    # 4D transfer field in *logical* (no-halo) layout. Source_split reads it
    # with logical (i, j, m, n) indices straight through.
    transfer = KernelAbstractions.zeros(KernelAbstractions.get_backend(N_data), FT, Nx, Ny, Nκ, Nφ)

    g       = FT(s.gravity)
    DAL1    = FT(1 / (1 + s.λ)^4)
    DAL2    = FT(1 / (1 - s.λ)^4)
    DAL3    = FT(2 * DAL1 * DAL2)
    C_eff   = FT(s.C / g^4)
    λ_FT    = FT(s.λ)
    Δθ_p    = FT(s.Δθ_plus)
    Δθ_m    = FT(s.Δθ_minus)

    σ_centers = sqrt.(g .* cgrid.κ)
    k_centers = cgrid.κ
    φ_centers = cgrid.φ
    σ_min     = first(σ_centers)
    σ_max     = last(σ_centers)
    log_xfr   = log(σ_centers[2] / σ_centers[1])
    φ_min     = first(φ_centers)
    Δφ        = FT(2 * pi / Nφ)

    Hx, Hy, _ = halo_size_3d(grid)
    iz        = data_z_index(N)

    workgroup = (1, 1, min(Nκ, 8), min(Nφ, 8))
    kernel = _dia_transfer_kernel!(device(arch), workgroup, (Nx, Ny, Nκ, Nφ))
    kernel(transfer, N_data,
           σ_centers, k_centers, φ_centers,
           σ_min, σ_max, log_xfr, φ_min, Δφ,
           Hx, Hy, iz, Nκ, Nφ,
           C_eff, λ_FT, Δθ_p, Δθ_m, DAL1, DAL2, DAL3, g)
    KernelAbstractions.synchronize(device(arch))

    return transfer
end

function source_split(s::HasselmannDIA, state::NamedTuple, model, i, j, m, n)
    FT = eltype(model.action)
    @inbounds T = state.transfer[i, j, m, n]
    @inbounds N_a = model.action[i, j, m, n]
    if T >= 0
        return (T, zero(FT))
    else
        damping = -T / max(N_a, eps(FT))
        damping = min(damping, one(FT))
        return (zero(FT), damping)
    end
end

function source_tendency(s::HasselmannDIA, state::NamedTuple, model, i, j, m, n)
    positive, damping = source_split(s, state, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function source_split(s::HasselmannDIA, model, i, j, m, n)
    state = (transfer = _compute_dia_transfer(s, model),)
    return source_split(s, state, model, i, j, m, n)
end

function source_tendency(s::HasselmannDIA, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end
