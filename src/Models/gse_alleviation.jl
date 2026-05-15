import KernelAbstractions
import KernelAbstractions: @kernel, @index
import Oceananigans.Architectures: architecture, device

"""
    AbstractGSEAlleviation

Marker for the Garden-Sprinkler-Effect (GSE) alleviation strategies of
Tolman (2002, *Ocean Modelling* 4, 269–289). Concrete subtypes plug into
`SpectralWaveModel(...; gse_alleviation=...)` and are applied as a
fractional step at the end of `time_step!`. Currently:

  * [`SpatialAveraging`](@ref) — Tolman 2002 §2 (the recommended
    operational method, replaced BH87 diffusion in WAVEWATCH III).

The §3 divergent-advection method has its own slot reserved for a future
`DivergentAdvection` subtype.
"""
abstract type AbstractGSEAlleviation end

"""
    SpatialAveraging(; αs=0.5, αn=0.5)

Tolman 2002 spatial-averaging GSE alleviation. After each full
`time_step!`, each spectral cell's physical field is replaced by a local
average over a rectangle of half-widths

    L_s = αs · |Δc_g| · Δt    (along the bin's propagation direction)
    L_n = αn · c_g · Δθ · Δt  (perpendicular)

implemented as the mean of the four corner values, each obtained by
bilinear interpolation from the nine-point physical stencil around the
target cell. The averaging is decoupled from the propagation, so the
time step is unaffected — this is what makes it cheaper than BH87
diffusion at high resolution.

`αs = αn = 0.5` matches the discrete-cell extent (the "ideal" choice in
Eq. 15 of the paper); `αs = αn ≈ 0.75–1.5` gives stronger GSE removal at
the cost of additional smearing.
"""
mutable struct SpatialAveraging{FT} <: AbstractGSEAlleviation
    αs :: FT
    αn :: FT
    cg_table :: Any
    Δcg_table :: Any
    cos_table :: Any
    sin_table :: Any
    Δφ :: Any
    scratch :: Any
end

SpatialAveraging(; αs = 0.5, αn = 0.5) =
    SpatialAveraging(promote(float(αs), float(αn))..., nothing, nothing, nothing, nothing, nothing, nothing)

@inline _gse_periodic(i, N) = i < 1 ? i + N : (i > N ? i - N : i)
@inline _gse_clamp(i, N) = i < 1 ? 1 : (i > N ? N : i)

# Tolman 2002 Eq. 15 averaging kernel. For each (i, j, m, n) target cell, the
# four corners of the averaging rectangle (aligned with the bin's propagation
# axis e_s and the perpendicular e_n) are bilinearly interpolated from the
# nine-point physical stencil and the mean is written to `N_out`.
@kernel function _spatial_averaging_kernel!(
    N_out, N_in,
    cg_table, Δcg_table, cos_table, sin_table, Δφ,
    αs, αn, dt, Δx_inv, Δy_inv,
    Hx, Hy, iz, Nx, Ny,
    periodic_x, periodic_y)
    i, j, m, n = @index(Global, NTuple)

    cg = cg_table[m]
    Δcg = Δcg_table[m]
    cosθ = cos_table[n]
    sinθ = sin_table[n]

    Ls = αs * Δcg * dt
    Ln = αn * cg * Δφ * dt

    Ls_x = Ls * cosθ * Δx_inv
    Ls_y = Ls * sinθ * Δy_inv
    Ln_x = -Ln * sinθ * Δx_inv
    Ln_y = Ln * cosθ * Δy_inv

    acc = zero(eltype(N_out))
    @inbounds for s1 in (-1, +1), s2 in (-1, +1)
        di = s1 * Ls_x + s2 * Ln_x
        dj = s1 * Ls_y + s2 * Ln_y

        ireal = i + di
        jreal = j + dj
        i0 = floor(Int, ireal)
        j0 = floor(Int, jreal)
        wi = ireal - i0
        wj = jreal - j0
        i1 = i0 + 1
        j1 = j0 + 1

        i0c = periodic_x ? _gse_periodic(i0, Nx) : _gse_clamp(i0, Nx)
        i1c = periodic_x ? _gse_periodic(i1, Nx) : _gse_clamp(i1, Nx)
        j0c = periodic_y ? _gse_periodic(j0, Ny) : _gse_clamp(j0, Ny)
        j1c = periodic_y ? _gse_periodic(j1, Ny) : _gse_clamp(j1, Ny)

        f00 = N_in[i0c + Hx, j0c + Hy, iz, m, n]
        f10 = N_in[i1c + Hx, j0c + Hy, iz, m, n]
        f01 = N_in[i0c + Hx, j1c + Hy, iz, m, n]
        f11 = N_in[i1c + Hx, j1c + Hy, iz, m, n]

        acc += (1 - wi) * (1 - wj) * f00 + wi * (1 - wj) * f10 +
               (1 - wi) * wj * f01 + wi * wj * f11
    end

    @inbounds N_out[i + Hx, j + Hy, iz, m, n] = acc / 4
end

# Lazily populate the per-bin tables that `_spatial_averaging_kernel!` reads.
function ensure_spatial_averaging_tables!(averaging::SpatialAveraging, model)
    cgrid = model.spectral_grid
    Nx, Ny, Nκ, Nφ = size(model.action)
    FT = eltype(model.action)
    gravity = FT(9.81)

    if averaging.cg_table === nothing || length(averaging.cg_table) != Nκ
        cg = zeros(FT, Nκ)
        Δcg = zeros(FT, Nκ)
        for m in 1:Nκ
            cg[m] = 0.5 * sqrt(gravity / cgrid.κ[m])
        end
        for m in 1:Nκ
            cg_lo = m == 1  ? cg[1]  : 0.5 * sqrt(gravity / cgrid.κ[m - 1])
            cg_hi = m == Nκ ? cg[Nκ] : 0.5 * sqrt(gravity / cgrid.κ[m + 1])
            Δcg[m] = abs(cg_lo - cg_hi) / 2
        end
        averaging.cg_table = cg
        averaging.Δcg_table = Δcg
    end

    if averaging.cos_table === nothing || length(averaging.cos_table) != Nφ
        cs = zeros(FT, Nφ)
        sn = zeros(FT, Nφ)
        for n in 1:Nφ
            cs[n] = cos(cgrid.φ[n])
            sn[n] = sin(cgrid.φ[n])
        end
        averaging.cos_table = cs
        averaging.sin_table = sn
        averaging.Δφ = FT(first(coordinate_spacings(cgrid, 2)))
    end

    if averaging.scratch === nothing || size(averaging.scratch) != size(model.action)
        averaging.scratch = similar(model.action)
    end

    return averaging
end

# Apply the chosen GSE-alleviation method as a post-step fractional step. The
# default no-op covers `nothing`; concrete strategies override.
apply_gse_alleviation!(model, ::Nothing, dt) = model

function apply_gse_alleviation!(model, averaging::SpatialAveraging, dt)
    ensure_spatial_averaging_tables!(averaging, model)

    grid = model.grid
    cgrid = model.spectral_grid
    Nx, Ny, Nκ, Nφ = size(model.action)
    Hx, Hy = halo_size_3d(grid)[1:2]
    iz = data_z_index(model.action)

    Δx = first(xspacings(grid))
    Δy = first(yspacings(grid))

    topology = Oceananigans.Grids.topology(grid)
    periodic_x = topology[1] === Oceananigans.Grids.Periodic
    periodic_y = topology[2] === Oceananigans.Grids.Periodic

    arch = architecture(grid)
    kernel = _spatial_averaging_kernel!(device(arch), (8, 8, 1, 1), (Nx, Ny, Nκ, Nφ))
    kernel(flat_data(averaging.scratch), flat_data(model.action),
           averaging.cg_table, averaging.Δcg_table,
           averaging.cos_table, averaging.sin_table, averaging.Δφ,
           convert(eltype(model.action), averaging.αs),
           convert(eltype(model.action), averaging.αn),
           convert(eltype(model.action), dt),
           1 / Δx, 1 / Δy,
           Hx, Hy, iz, Nx, Ny,
           periodic_x, periodic_y)
    KernelAbstractions.synchronize(device(arch))

    # Swap the freshly-averaged values back into model.action. Use a KA kernel
    # to stay GPU-friendly.
    Hx_, Hy_, _ = halo_size_3d(grid)
    swap = _spatial_averaging_copy!(device(arch), (8, 8, 1, 1), (Nx, Ny, Nκ, Nφ))
    swap(flat_data(model.action), flat_data(averaging.scratch), Hx_, Hy_, iz)
    KernelAbstractions.synchronize(device(arch))

    return model
end

@kernel function _spatial_averaging_copy!(N_dest, N_src, Hx, Hy, iz)
    i, j, m, n = @index(Global, NTuple)
    ix = i + Hx
    jy = j + Hy
    @inbounds N_dest[ix, jy, iz, m, n] = N_src[ix, jy, iz, m, n]
end
