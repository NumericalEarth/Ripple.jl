import KernelAbstractions
import KernelAbstractions: @kernel, @index
import Oceananigans.Architectures: architecture, device

# Compute spatial gradients of the Doppler velocity caches with a KA kernel
# so the path is GPU-compatible. Run once per coupling update.
@kernel function _current_gradients!(Ux_x, Ux_y, Uy_x, Uy_y, Ux, Uy, Δx_inv, Δy_inv, Nx, Ny)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ip = i == Nx ? 1 : i + 1
        im = i == 1 ? Nx : i - 1
        jp = j == Ny ? 1 : j + 1
        jm = j == 1 ? Ny : j - 1
        Ux_x[i, j, k] = (Ux[ip, j, k] - Ux[im, j, k]) * (Δx_inv * 0.5)
        Ux_y[i, j, k] = (Ux[i, jp, k] - Ux[i, jm, k]) * (Δy_inv * 0.5)
        Uy_x[i, j, k] = (Uy[ip, j, k] - Uy[im, j, k]) * (Δx_inv * 0.5)
        Uy_y[i, j, k] = (Uy[i, jp, k] - Uy[i, jm, k]) * (Δy_inv * 0.5)
    end
end

function ensure_current_gradients!(coupling::CWCMPrescribedCurrentCoupling, grid)
    Ux = coupling.Ux
    Uy = coupling.Uy
    if coupling.Ux_x === nothing || size(coupling.Ux_x) != size(Ux)
        coupling.Ux_x = similar(Ux)
        coupling.Ux_y = similar(Ux)
        coupling.Uy_x = similar(Uy)
        coupling.Uy_y = similar(Uy)
    end
    Nx, Ny, Nκ = size(Ux)
    Δx = first(xspacings(grid))
    Δy = first(yspacings(grid))
    arch = architecture(grid)
    kernel = _current_gradients!(device(arch), (8, 8, 1), (Nx, Ny, Nκ))
    kernel(coupling.Ux_x, coupling.Ux_y, coupling.Uy_x, coupling.Uy_y,
           Ux, Uy, 1 / Δx, 1 / Δy, Nx, Ny)
    KernelAbstractions.synchronize(device(arch))
    return coupling
end

@inline function weno5_reconstruct(fm2, fm1, f0, fp1, fp2)
    ε = 1.0e-12
    v0 = (1/3)  * fm2 - (7/6) * fm1 + (11/6) * f0
    v1 = -(1/6) * fm1 + (5/6) * f0  + (1/3)  * fp1
    v2 = (1/3)  * f0  + (5/6) * fp1 - (1/6)  * fp2
    β0 = (13/12) * (fm2 - 2fm1 + f0)^2 + (1/4) * (fm2 - 4fm1 + 3f0)^2
    β1 = (13/12) * (fm1 - 2f0  + fp1)^2 + (1/4) * (fm1 - fp1)^2
    β2 = (13/12) * (f0  - 2fp1 + fp2)^2 + (1/4) * (3f0 - 4fp1 + fp2)^2
    α0 = 0.1 / (ε + β0)^2
    α1 = 0.6 / (ε + β1)^2
    α2 = 0.3 / (ε + β2)^2
    Σα = α0 + α1 + α2
    return (α0 * v0 + α1 * v1 + α2 * v2) / Σα
end

@inline function weno5_face_iphalf(im2, im1, i0, ip1, ip2, ip3, vel, has_stencil)
    if has_stencil
        if vel >= 0
            return weno5_reconstruct(im2, im1, i0, ip1, ip2)
        else
            return weno5_reconstruct(ip3, ip2, ip1, i0, im1)
        end
    else
        return vel >= 0 ? i0 : ip1
    end
end

@inline _periodic(i, N) = i < 1 ? i + N : (i > N ? i - N : i)
@inline _clamp_idx(i, N) = i < 1 ? 1 : (i > N ? N : i)

# Fused KA kernel: Doppler-shifted physical transport + kinematic spectral
# refraction, both 5th-order WENO, one pass over (i, j, m, n). Reads/writes
# the contiguous 5D backing of the ProductField directly (no pack/unpack).
# Layout: data[i+Hx, j+Hy, 1, m, n] holds N at physical cell (i, j) and
# spectral cell (m, n).
@kernel function _wave_current_refraction_tendency!(
    G_data, N_data,
    Ux, Uy, Ux_x, Ux_y, Uy_x, Uy_y,
    κ_centers, cg_x_table, cg_y_table,
    cos_table, sin_table,
    Δx_inv, Δy_inv, Δκ_inv, Δφ_inv,
    Hx, Hy, iz, Nx, Ny, Nκ, Nφ)
    i, j, m, n = @index(Global, NTuple)

    κ = κ_centers[m]
    cφ_val = cos_table[n]
    sφ_val = sin_table[n]

    @inbounds Uxc = Ux[i, j, m]
    @inbounds Uyc = Uy[i, j, m]
    @inbounds Uxx = Ux_x[i, j, m]
    @inbounds Uxy = Ux_y[i, j, m]
    @inbounds Uyx = Uy_x[i, j, m]
    @inbounds Uyy = Uy_y[i, j, m]

    cκ = -κ * (cφ_val^2 * Uxx + cφ_val * sφ_val * (Uyx + Uxy) + sφ_val^2 * Uyy)
    cφ =  cφ_val * sφ_val * (Uxx - Uyy) + sφ_val^2 * Uyx - cφ_val^2 * Uxy

    ux = cg_x_table[m, n] + Uxc
    uy = cg_y_table[m, n] + Uyc

    @inbounds begin
        # x faces (periodic): use 7-cell stencil. Halo cells already filled by
        # fill_halo_regions! on the per-bin Fields, so we don't need wrap-around
        # — but staying explicit keeps the kernel robust when halos aren't current.
        ix = i + Hx
        ip1 = _periodic(i + 1, Nx) + Hx; ip2 = _periodic(i + 2, Nx) + Hx; ip3 = _periodic(i + 3, Nx) + Hx
        im1 = _periodic(i - 1, Nx) + Hx; im2 = _periodic(i - 2, Nx) + Hx; im3 = _periodic(i - 3, Nx) + Hx
        jy = j + Hy
        x_m3 = N_data[im3, jy, iz, m, n]; x_m2 = N_data[im2, jy, iz, m, n]; x_m1 = N_data[im1, jy, iz, m, n]
        x_0  = N_data[ix,  jy, iz, m, n]
        x_p1 = N_data[ip1, jy, iz, m, n]; x_p2 = N_data[ip2, jy, iz, m, n]; x_p3 = N_data[ip3, jy, iz, m, n]
        N_xp = weno5_face_iphalf(x_m2, x_m1, x_0, x_p1, x_p2, x_p3, ux, true)
        N_xm = weno5_face_iphalf(x_m3, x_m2, x_m1, x_0, x_p1, x_p2, ux, true)
        flux_x = ux * (N_xp - N_xm) * Δx_inv

        # y faces (periodic).
        jp1 = _periodic(j + 1, Ny) + Hy; jp2 = _periodic(j + 2, Ny) + Hy; jp3 = _periodic(j + 3, Ny) + Hy
        jm1 = _periodic(j - 1, Ny) + Hy; jm2 = _periodic(j - 2, Ny) + Hy; jm3 = _periodic(j - 3, Ny) + Hy
        y_m3 = N_data[ix, jm3, iz, m, n]; y_m2 = N_data[ix, jm2, iz, m, n]; y_m1 = N_data[ix, jm1, iz, m, n]
        y_0  = N_data[ix, jy,  iz, m, n]
        y_p1 = N_data[ix, jp1, iz, m, n]; y_p2 = N_data[ix, jp2, iz, m, n]; y_p3 = N_data[ix, jp3, iz, m, n]
        N_yp = weno5_face_iphalf(y_m2, y_m1, y_0, y_p1, y_p2, y_p3, uy, true)
        N_ym = weno5_face_iphalf(y_m3, y_m2, y_m1, y_0, y_p1, y_p2, uy, true)
        flux_y = uy * (N_yp - N_ym) * Δy_inv

        # κ faces (bounded). No-flux at the two outer faces.
        has_κ_stencil_p = (m - 2 >= 1) & (m + 3 <= Nκ)
        has_κ_stencil_m = (m - 3 >= 1) & (m + 2 <= Nκ)
        km3 = _clamp_idx(m - 3, Nκ); km2 = _clamp_idx(m - 2, Nκ); km1 = _clamp_idx(m - 1, Nκ)
        kp1 = _clamp_idx(m + 1, Nκ); kp2 = _clamp_idx(m + 2, Nκ); kp3 = _clamp_idx(m + 3, Nκ)
        κ_m3 = N_data[ix, jy, iz, km3, n]; κ_m2 = N_data[ix, jy, iz, km2, n]; κ_m1 = N_data[ix, jy, iz, km1, n]
        κ_0  = N_data[ix, jy, iz, m,   n]
        κ_p1 = N_data[ix, jy, iz, kp1, n]; κ_p2 = N_data[ix, jy, iz, kp2, n]; κ_p3 = N_data[ix, jy, iz, kp3, n]
        N_κp = weno5_face_iphalf(κ_m2, κ_m1, κ_0, κ_p1, κ_p2, κ_p3, cκ, has_κ_stencil_p)
        N_κm = weno5_face_iphalf(κ_m3, κ_m2, κ_m1, κ_0, κ_p1, κ_p2, cκ, has_κ_stencil_m)
        flux_κ_p = (m == Nκ) ? zero(cκ) : cκ * N_κp
        flux_κ_m = (m == 1)  ? zero(cκ) : cκ * N_κm
        flux_κ = (flux_κ_p - flux_κ_m) * Δκ_inv

        # φ faces (periodic).
        nm3 = _periodic(n - 3, Nφ); nm2 = _periodic(n - 2, Nφ); nm1 = _periodic(n - 1, Nφ)
        np1 = _periodic(n + 1, Nφ); np2 = _periodic(n + 2, Nφ); np3 = _periodic(n + 3, Nφ)
        φ_m3 = N_data[ix, jy, iz, m, nm3]; φ_m2 = N_data[ix, jy, iz, m, nm2]; φ_m1 = N_data[ix, jy, iz, m, nm1]
        φ_0  = N_data[ix, jy, iz, m, n  ]
        φ_p1 = N_data[ix, jy, iz, m, np1]; φ_p2 = N_data[ix, jy, iz, m, np2]; φ_p3 = N_data[ix, jy, iz, m, np3]
        N_φp = weno5_face_iphalf(φ_m2, φ_m1, φ_0, φ_p1, φ_p2, φ_p3, cφ, true)
        N_φm = weno5_face_iphalf(φ_m3, φ_m2, φ_m1, φ_0, φ_p1, φ_p2, cφ, true)
        flux_φ = cφ * (N_φp - N_φm) * Δφ_inv

        G_data[ix, jy, iz, m, n] = -(flux_x + flux_y + flux_κ + flux_φ)
    end
end

# Fill in lazy caches on the coupling.
function ensure_refraction_tables!(coupling::CWCMPrescribedCurrentCoupling, cgrid, Nκ, Nφ, FT)
    if coupling.cg_x_table === nothing || size(coupling.cg_x_table) != (Nκ, Nφ)
        coupling.cg_x_table = zeros(FT, Nκ, Nφ)
        coupling.cg_y_table = zeros(FT, Nκ, Nφ)
        coupling.cos_table = zeros(FT, Nφ)
        coupling.sin_table = zeros(FT, Nφ)
        @inbounds for n in 1:Nφ
            coupling.cos_table[n] = cos(cgrid.φ[n])
            coupling.sin_table[n] = sin(cgrid.φ[n])
        end
        @inbounds for n in 1:Nφ, m in 1:Nκ
            u, v = deep_water_group_velocity(cgrid, m, n)
            coupling.cg_x_table[m, n] = u
            coupling.cg_y_table[m, n] = v
        end
    end
    return coupling
end

"""
    compute_wave_current_refraction_tendency!(G, N, coupling, model)

Compute the wave-action tendency `∂N/∂t` from Doppler-shifted physical
transport *and* kinematic refraction in one fused KA kernel. Reads `N` and
writes `G` through their contiguous 5D backings (no pack/unpack).
"""
function compute_wave_current_refraction_tendency!(G, N,
                                                   coupling::CWCMPrescribedCurrentCoupling,
                                                   model)
    grid = model.grid
    cgrid = model.spectral_grid
    Nx, Ny, Nκ, Nφ = size(N)
    FT = eltype(N)

    ensure_current_gradients!(coupling, grid)
    ensure_refraction_tables!(coupling, cgrid, Nκ, Nφ, FT)

    # Refresh halos so periodic stencils see correct neighbours.
    for n in 1:Nφ, m in 1:Nκ
        fill_halo_regions!(physical_field(N, m, n))
    end

    Δx = first(xspacings(grid))
    Δy = first(yspacings(grid))
    Δκ = first(coordinate_spacings(cgrid, 1))
    Δφ = first(coordinate_spacings(cgrid, 2))
    Hx, Hy, _ = halo_size_3d(grid)
    iz = data_z_index(N)

    arch = architecture(grid)
    kernel = _wave_current_refraction_tendency!(device(arch), (8, 8, 1, 1), (Nx, Ny, Nκ, Nφ))
    kernel(flat_data(G), flat_data(N),
           coupling.Ux, coupling.Uy,
           coupling.Ux_x, coupling.Ux_y, coupling.Uy_x, coupling.Uy_y,
           cgrid.κ, coupling.cg_x_table, coupling.cg_y_table,
           coupling.cos_table, coupling.sin_table,
           1 / Δx, 1 / Δy, 1 / Δκ, 1 / Δφ,
           Hx, Hy, iz, Nx, Ny, Nκ, Nφ)
    KernelAbstractions.synchronize(device(arch))
    return G
end

# Helper: halo size in 3D, robust to slight grid API differences.
function halo_size_3d(grid)
    h = Oceananigans.Grids.halo_size(grid)
    length(h) >= 3 ? (h[1], h[2], h[3]) : (h[1], h[2], 0)
end

# Read the linear index in the contiguous backing where the surface slab lives
# (since WaveActionField stores a single z-level per spectral bin, its parent
# data has only one entry in the z-dim).
function data_z_index(N)
    f11 = physical_field(N, 1, 1)
    return only(axes(parent(f11.data), 3))
end

