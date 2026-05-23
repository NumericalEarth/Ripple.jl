import KernelAbstractions
import KernelAbstractions: @kernel, @index
import Oceananigans.Architectures: architecture, device, on_architecture
import Oceananigans.Operators: ∂xᶜᶜᶜ, ∂yᶜᶜᶜ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
import Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ

# Compute spatial gradients of the Doppler velocity caches with a KA kernel
# so the path is GPU-compatible. Run once per coupling update.
@inline _cache_index(i, N, periodic) = ifelse(periodic,
                                              mod1(i, N),
                                              ifelse(i < 1, 1, ifelse(i > N, N, i)))

@inline function _x_face_cache_value(i, j, k, grid, u, m, xperiodic, yperiodic)
    ii = _cache_index(i, size(u, 1), xperiodic)
    jj = _cache_index(j, size(u, 2), yperiodic)
    return @inbounds u[ii, jj, m]
end

@inline function _y_face_cache_value(i, j, k, grid, v, m, xperiodic, yperiodic)
    ii = _cache_index(i, size(v, 1), xperiodic)
    jj = _cache_index(j, size(v, 2), yperiodic)
    return @inbounds v[ii, jj, m]
end

@inline _x_face_cache_at_center(i, j, k, grid, u, m, xperiodic, yperiodic) =
    ℑxᶜᵃᵃ(i, j, k, grid, _x_face_cache_value, u, m, xperiodic, yperiodic)

@inline _y_face_cache_at_center(i, j, k, grid, v, m, xperiodic, yperiodic) =
    ℑyᵃᶜᵃ(i, j, k, grid, _y_face_cache_value, v, m, xperiodic, yperiodic)

@inline _∂y_x_face_cache_at_y_face(i, j, k, grid, u, m, xperiodic, yperiodic) =
    ∂yᶜᶠᶜ(i, j, k, grid, _x_face_cache_at_center, u, m, xperiodic, yperiodic)

@inline _∂x_y_face_cache_at_x_face(i, j, k, grid, v, m, xperiodic, yperiodic) =
    ∂xᶠᶜᶜ(i, j, k, grid, _y_face_cache_at_center, v, m, xperiodic, yperiodic)

@kernel function _current_gradients!(uᴰx_x, uᴰx_y, uᴰy_x, uᴰy_y, uᴰx, uᴰy,
                                     grid, xperiodic, yperiodic)
    i, j, m = @index(Global, NTuple)
    @inbounds begin
        uᴰx_x[i, j, m] = ∂xᶜᶜᶜ(i, j, 1, grid, _x_face_cache_value, uᴰx, m, xperiodic, yperiodic)
        uᴰx_y[i, j, m] = ℑyᵃᶜᵃ(i, j, 1, grid, _∂y_x_face_cache_at_y_face, uᴰx, m, xperiodic, yperiodic)
        uᴰy_x[i, j, m] = ℑxᶜᵃᵃ(i, j, 1, grid, _∂x_y_face_cache_at_x_face, uᴰy, m, xperiodic, yperiodic)
        uᴰy_y[i, j, m] = ∂yᶜᶜᶜ(i, j, 1, grid, _y_face_cache_value, uᴰy, m, xperiodic, yperiodic)
    end
end

function ensure_current_gradients!(coupling::AbstractCWCMCurrentCoupling, grid)
    uᴰx = coupling.uᴰx
    uᴰy = coupling.uᴰy
    Nx, Ny = horizontal_size(grid)
    Nκ = length(coupling.kappa)
    expected_size = (Nx, Ny, Nκ)
    if coupling.uᴰx_x === nothing || size(coupling.uᴰx_x) != expected_size
        coupling.uᴰx_x = similar(uᴰx, eltype(uᴰx), expected_size)
        coupling.uᴰx_y = similar(uᴰx, eltype(uᴰx), expected_size)
        coupling.uᴰy_x = similar(uᴰy, eltype(uᴰy), expected_size)
        coupling.uᴰy_y = similar(uᴰy, eltype(uᴰy), expected_size)
    end
    arch = architecture(grid)
    topology = Oceananigans.Grids.topology(grid)
    xperiodic = topology[1] === Oceananigans.Grids.Periodic
    yperiodic = topology[2] === Oceananigans.Grids.Periodic
    kernel = _current_gradients!(device(arch), (8, 8, 1), (Nx, Ny, Nκ))
    kernel(coupling.uᴰx_x, coupling.uᴰx_y, coupling.uᴰy_x, coupling.uᴰy_y,
           uᴰx, uᴰy, grid, xperiodic, yperiodic)
    KernelAbstractions.synchronize(device(arch))
    return coupling
end

@inline function weno5_reconstruct(fm2, fm1, f0, fp1, fp2)
    FT = typeof(fm2 + fm1 + f0 + fp1 + fp2)
    ε = FT(10)^4 * eps(one(FT))
    v0 = (FT(1)/FT(3))  * fm2 - (FT(7)/FT(6)) * fm1 + (FT(11)/FT(6)) * f0
    v1 = -(FT(1)/FT(6)) * fm1 + (FT(5)/FT(6)) * f0  + (FT(1)/FT(3))  * fp1
    v2 = (FT(1)/FT(3))  * f0  + (FT(5)/FT(6)) * fp1 - (FT(1)/FT(6))  * fp2
    β0 = (FT(13)/FT(12)) * (fm2 - 2fm1 + f0)^2 + (FT(1)/FT(4)) * (fm2 - 4fm1 + 3f0)^2
    β1 = (FT(13)/FT(12)) * (fm1 - 2f0  + fp1)^2 + (FT(1)/FT(4)) * (fm1 - fp1)^2
    β2 = (FT(13)/FT(12)) * (f0  - 2fp1 + fp2)^2 + (FT(1)/FT(4)) * (3f0 - 4fp1 + fp2)^2
    α0 = (FT(1)/FT(10)) / (ε + β0)^2
    α1 = (FT(6)/FT(10)) / (ε + β1)^2
    α2 = (FT(3)/FT(10)) / (ε + β2)^2
    Σα = α0 + α1 + α2
    return (α0 * v0 + α1 * v1 + α2 * v2) / Σα
end

@inline function weno5_face_iphalf(im2, im1, i0, ip1, ip2, ip3, vel, has_stencil)
    positive_velocity = vel >= zero(vel)
    weno_value = ifelse(positive_velocity,
                        weno5_reconstruct(im2, im1, i0, ip1, ip2),
                        weno5_reconstruct(ip3, ip2, ip1, i0, im1))
    first_order_value = ifelse(positive_velocity, i0, ip1)
    return ifelse(has_stencil, weno_value, first_order_value)
end

@inline _periodic(i, N) = mod1(i, N)
@inline _clamp_idx(i, N) = ifelse(i < 1, 1, ifelse(i > N, N, i))
@inline _stencil_index(i, N, periodic) = ifelse(periodic, _periodic(i, N), _clamp_idx(i, N))

@inline function _κ_face_gradient_value(A, i, j, q, κ_face, κ_centers, Nκ)
    m₋ = _clamp_idx(q - 1, Nκ)
    m₊ = _clamp_idx(q, Nκ)
    κ₋ = κ_centers[m₋]
    κ₊ = κ_centers[m₊]
    Δκ = κ₊ - κ₋
    safe_Δκ = ifelse(Δκ == zero(Δκ), one(Δκ), Δκ)
    r = ifelse(Δκ == zero(Δκ), zero(Δκ), (κ_face - κ₋) / safe_Δκ)
    @inbounds A₋ = A[i, j, m₋]
    @inbounds A₊ = A[i, j, m₊]
    return A₋ + r * (A₊ - A₋)
end

@inline function _refraction_cκ(κ, cosφ, sinφ, ∂x_uᴰx, ∂y_uᴰx, ∂x_uᴰy, ∂y_uᴰy)
    return -κ * (cosφ^2 * ∂x_uᴰx +
                 cosφ * sinφ * (∂x_uᴰy + ∂y_uᴰx) +
                 sinφ^2 * ∂y_uᴰy)
end

@inline function _refraction_cφ(cosφ, sinφ, ∂x_uᴰx, ∂y_uᴰx, ∂x_uᴰy, ∂y_uᴰy)
    return cosφ * sinφ * (∂x_uᴰx - ∂y_uᴰy) +
           sinφ^2 * ∂x_uᴰy -
           cosφ^2 * ∂y_uᴰx
end

@inline function _κ_face_refraction_velocity(i, j, q, n,
                                             κ_centers, κ_faces,
                                             cos_table, sin_table,
                                             uᴰx_x, uᴰx_y, uᴰy_x, uᴰy_y,
                                             Nκ)
    @inbounds κ_face = κ_faces[q]
    @inbounds cosφ = cos_table[n]
    @inbounds sinφ = sin_table[n]
    ∂x_uᴰx = _κ_face_gradient_value(uᴰx_x, i, j, q, κ_face, κ_centers, Nκ)
    ∂y_uᴰx = _κ_face_gradient_value(uᴰx_y, i, j, q, κ_face, κ_centers, Nκ)
    ∂x_uᴰy = _κ_face_gradient_value(uᴰy_x, i, j, q, κ_face, κ_centers, Nκ)
    ∂y_uᴰy = _κ_face_gradient_value(uᴰy_y, i, j, q, κ_face, κ_centers, Nκ)
    return _refraction_cκ(κ_face, cosφ, sinφ, ∂x_uᴰx, ∂y_uᴰx, ∂x_uᴰy, ∂y_uᴰy)
end

@inline function _φ_face_refraction_velocity(i, j, m, q,
                                             φ_faces,
                                             uᴰx_x, uᴰx_y, uᴰy_x, uᴰy_y)
    @inbounds φ_face = φ_faces[q]
    cosφ = cos(φ_face)
    sinφ = sin(φ_face)
    @inbounds ∂x_uᴰx = uᴰx_x[i, j, m]
    @inbounds ∂y_uᴰx = uᴰx_y[i, j, m]
    @inbounds ∂x_uᴰy = uᴰy_x[i, j, m]
    @inbounds ∂y_uᴰy = uᴰy_y[i, j, m]
    return _refraction_cφ(cosφ, sinφ, ∂x_uᴰx, ∂y_uᴰx, ∂x_uᴰy, ∂y_uᴰy)
end

@inline function _x_transport_flux(iᶠ, j, k, grid, N_data,
                                   uᴰx, duᴰxdκ, duᴰydκ,
                                   cg_x, Kx, Ky, κ,
                                   Hx, Hy, iz, Nx, Ny, m, n,
                                   xperiodic, yperiodic)
    iₗ = iᶠ - 1
    jy = _stencil_index(j, Ny, yperiodic) + Hy
    boundary_face = !xperiodic & ((iᶠ == 1) | (iᶠ == Nx + 1))

    im2 = _stencil_index(iₗ - 2, Nx, xperiodic) + Hx
    im1 = _stencil_index(iₗ - 1, Nx, xperiodic) + Hx
    i0  = _stencil_index(iₗ,     Nx, xperiodic) + Hx
    ip1 = _stencil_index(iₗ + 1, Nx, xperiodic) + Hx
    ip2 = _stencil_index(iₗ + 2, Nx, xperiodic) + Hx
    ip3 = _stencil_index(iₗ + 3, Nx, xperiodic) + Hx
    has_stencil = ifelse(xperiodic, true, (iₗ - 2 >= 1) & (iₗ + 3 <= Nx))

    @inbounds begin
        N_m2 = N_data[im2, jy, iz, m, n]
        N_m1 = N_data[im1, jy, iz, m, n]
        N_0  = N_data[i0,  jy, iz, m, n]
        N_p1 = N_data[ip1, jy, iz, m, n]
        N_p2 = N_data[ip2, jy, iz, m, n]
        N_p3 = N_data[ip3, jy, iz, m, n]
    end

    Hx_face = _x_face_cache_value(iᶠ, j, 1, grid, duᴰxdκ, m, xperiodic, yperiodic)
    Hy_face = ℑxyᶠᶜᵃ(iᶠ, j, 1, grid, _y_face_cache_value, duᴰydκ, m, xperiodic, yperiodic)
    KH_over_κ = (Kx * Hx_face + Ky * Hy_face) / κ
    ux = cg_x + _x_face_cache_value(iᶠ, j, 1, grid, uᴰx, m, xperiodic, yperiodic) + KH_over_κ * Kx
    N_face = weno5_face_iphalf(N_m2, N_m1, N_0, N_p1, N_p2, N_p3, ux, has_stencil)
    flux = ux * N_face
    return ifelse(boundary_face, zero(flux), flux)
end

@inline function _y_transport_flux(i, jᶠ, k, grid, N_data,
                                   uᴰy, duᴰxdκ, duᴰydκ,
                                   cg_y, Kx, Ky, κ,
                                   Hx, Hy, iz, Nx, Ny, m, n,
                                   xperiodic, yperiodic)
    ix = _stencil_index(i, Nx, xperiodic) + Hx
    j_d = jᶠ - 1
    boundary_face = !yperiodic & ((jᶠ == 1) | (jᶠ == Ny + 1))

    jm2 = _stencil_index(j_d - 2, Ny, yperiodic) + Hy
    jm1 = _stencil_index(j_d - 1, Ny, yperiodic) + Hy
    j0  = _stencil_index(j_d,     Ny, yperiodic) + Hy
    jp1 = _stencil_index(j_d + 1, Ny, yperiodic) + Hy
    jp2 = _stencil_index(j_d + 2, Ny, yperiodic) + Hy
    jp3 = _stencil_index(j_d + 3, Ny, yperiodic) + Hy
    has_stencil = ifelse(yperiodic, true, (j_d - 2 >= 1) & (j_d + 3 <= Ny))

    @inbounds begin
        N_m2 = N_data[ix, jm2, iz, m, n]
        N_m1 = N_data[ix, jm1, iz, m, n]
        N_0  = N_data[ix, j0,  iz, m, n]
        N_p1 = N_data[ix, jp1, iz, m, n]
        N_p2 = N_data[ix, jp2, iz, m, n]
        N_p3 = N_data[ix, jp3, iz, m, n]
    end

    Hx_face = ℑxyᶜᶠᵃ(i, jᶠ, 1, grid, _x_face_cache_value, duᴰxdκ, m, xperiodic, yperiodic)
    Hy_face = _y_face_cache_value(i, jᶠ, 1, grid, duᴰydκ, m, xperiodic, yperiodic)
    KH_over_κ = (Kx * Hx_face + Ky * Hy_face) / κ
    uy = cg_y + _y_face_cache_value(i, jᶠ, 1, grid, uᴰy, m, xperiodic, yperiodic) + KH_over_κ * Ky
    N_face = weno5_face_iphalf(N_m2, N_m1, N_0, N_p1, N_p2, N_p3, uy, has_stencil)
    flux = uy * N_face
    return ifelse(boundary_face, zero(flux), flux)
end

# Fused KA kernel: Doppler-shifted physical transport + kinematic spectral
# refraction, both 5th-order WENO, one pass over (i, j, m, n). Reads/writes
# the contiguous 5D backing of the ProductField directly (no pack/unpack).
# Layout: data[i+Hx, j+Hy, 1, m, n] holds N at physical cell (i, j) and
# spectral cell (m, n).
@kernel function _wave_current_refraction_tendency!(
    G_data, N_data,
    uᴰx, uᴰy, duᴰxdκ, duᴰydκ, uᴰx_x, uᴰx_y, uᴰy_x, uᴰy_y,
    κ_centers, κ_faces, φ_faces, cg_x_table, cg_y_table,
    cos_table, sin_table,
    grid,
    Hx, Hy, iz, Nx, Ny, Nκ, Nφ, xperiodic, yperiodic)
    i, j, m, n = @index(Global, NTuple)

    κ = κ_centers[m]
    cφ_val = cos_table[n]
    sφ_val = sin_table[n]
    Kx = κ * cφ_val
    Ky = κ * sφ_val

    cg_x = intrinsic_velocity_component(cg_x_table, i, j, m, n)
    cg_y = intrinsic_velocity_component(cg_y_table, i, j, m, n)

    @inbounds begin
        ix = i + Hx
        jy = j + Hy
        κ₋ = κ_faces[m]
        κ₊ = κ_faces[m+1]
        Rκ = (κ₊^2 - κ₋^2) / (one(κ₊) + one(κ₊))
        Δφ = φ_faces[n+1] - φ_faces[n]

        flux_x = ∂xᶜᶜᶜ(i, j, 1, grid, _x_transport_flux, N_data,
                         uᴰx, duᴰxdκ, duᴰydκ,
                         cg_x, Kx, Ky, κ,
                         Hx, Hy, iz, Nx, Ny, m, n,
                         xperiodic, yperiodic)

        flux_y = ∂yᶜᶜᶜ(i, j, 1, grid, _y_transport_flux, N_data,
                         uᴰy, duᴰxdκ, duᴰydκ,
                         cg_y, Kx, Ky, κ,
                         Hx, Hy, iz, Nx, Ny, m, n,
                         xperiodic, yperiodic)

        # Conservative polar finite-volume divergence: radial faces carry
        # κ \dot{κ} N and cell averages divide by the annular measure Rκ Δφ.
        # κ faces (bounded). No-flux at the two outer faces.
        has_κ_stencil_p = (m - 2 >= 1) & (m + 3 <= Nκ)
        has_κ_stencil_m = (m - 3 >= 1) & (m + 2 <= Nκ)
        km3 = _clamp_idx(m - 3, Nκ); km2 = _clamp_idx(m - 2, Nκ); km1 = _clamp_idx(m - 1, Nκ)
        kp1 = _clamp_idx(m + 1, Nκ); kp2 = _clamp_idx(m + 2, Nκ); kp3 = _clamp_idx(m + 3, Nκ)
        κ_m3 = N_data[ix, jy, iz, km3, n]; κ_m2 = N_data[ix, jy, iz, km2, n]; κ_m1 = N_data[ix, jy, iz, km1, n]
        κ_0  = N_data[ix, jy, iz, m,   n]
        κ_p1 = N_data[ix, jy, iz, kp1, n]; κ_p2 = N_data[ix, jy, iz, kp2, n]; κ_p3 = N_data[ix, jy, iz, kp3, n]
        cκ_p = _κ_face_refraction_velocity(i, j, m+1, n,
                                            κ_centers, κ_faces,
                                            cos_table, sin_table,
                                            uᴰx_x, uᴰx_y, uᴰy_x, uᴰy_y,
                                            Nκ)
        cκ_m = _κ_face_refraction_velocity(i, j, m, n,
                                            κ_centers, κ_faces,
                                            cos_table, sin_table,
                                            uᴰx_x, uᴰx_y, uᴰy_x, uᴰy_y,
                                            Nκ)
        N_κp = weno5_face_iphalf(κ_m2, κ_m1, κ_0, κ_p1, κ_p2, κ_p3, cκ_p, has_κ_stencil_p)
        N_κm = weno5_face_iphalf(κ_m3, κ_m2, κ_m1, κ_0, κ_p1, κ_p2, cκ_m, has_κ_stencil_m)
        flux_κ_p = ifelse(m == Nκ, zero(cκ_p), κ₊ * cκ_p * N_κp)
        flux_κ_m = ifelse(m == 1,  zero(cκ_m), κ₋ * cκ_m * N_κm)
        flux_κ = (flux_κ_p - flux_κ_m) / Rκ

        # φ faces (periodic).
        nm3 = _periodic(n - 3, Nφ); nm2 = _periodic(n - 2, Nφ); nm1 = _periodic(n - 1, Nφ)
        np1 = _periodic(n + 1, Nφ); np2 = _periodic(n + 2, Nφ); np3 = _periodic(n + 3, Nφ)
        φ_m3 = N_data[ix, jy, iz, m, nm3]; φ_m2 = N_data[ix, jy, iz, m, nm2]; φ_m1 = N_data[ix, jy, iz, m, nm1]
        φ_0  = N_data[ix, jy, iz, m, n  ]
        φ_p1 = N_data[ix, jy, iz, m, np1]; φ_p2 = N_data[ix, jy, iz, m, np2]; φ_p3 = N_data[ix, jy, iz, m, np3]
        cφ_p = _φ_face_refraction_velocity(i, j, m, n+1, φ_faces,
                                            uᴰx_x, uᴰx_y, uᴰy_x, uᴰy_y)
        cφ_m = _φ_face_refraction_velocity(i, j, m, n, φ_faces,
                                            uᴰx_x, uᴰx_y, uᴰy_x, uᴰy_y)
        N_φp = weno5_face_iphalf(φ_m2, φ_m1, φ_0, φ_p1, φ_p2, φ_p3, cφ_p, true)
        N_φm = weno5_face_iphalf(φ_m3, φ_m2, φ_m1, φ_0, φ_p1, φ_p2, cφ_m, true)
        flux_φ = (cφ_p * N_φp - cφ_m * N_φm) / Δφ

        G_data[ix, jy, iz, m, n] = -(flux_x + flux_y + flux_κ + flux_φ)
    end
end

# Fill in lazy caches on the coupling.
function ensure_refraction_tables!(coupling::AbstractCWCMCurrentCoupling, cgrid, depth, grid, Nκ, Nφ, FT)
    expected_size = is_spatially_varying_depth(depth) ? (grid.Nx, grid.Ny, Nκ, Nφ) : (Nκ, Nφ)
    if coupling.cg_x_table === nothing ||
       size(coupling.cg_x_table) != expected_size ||
       is_spatially_varying_depth(depth)
        cos_table = zeros(FT, Nφ)
        sin_table = zeros(FT, Nφ)
        φ = Array(cgrid.φ)
        @inbounds for n in 1:Nφ
            cos_table[n] = cos(φ[n])
            sin_table[n] = sin(φ[n])
        end

        cg_x_table, cg_y_table = intrinsic_group_velocity_tables(cgrid, depth, grid, FT)
        arch = architecture(grid)
        coupling.cg_x_table = on_architecture(arch, cg_x_table)
        coupling.cg_y_table = on_architecture(arch, cg_y_table)
        coupling.cos_table = on_architecture(arch, cos_table)
        coupling.sin_table = on_architecture(arch, sin_table)
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
                                                   coupling::AbstractCWCMCurrentCoupling,
                                                   model)
    grid = model.grid
    cgrid = model.spectral_grid
    Nx, Ny, Nκ, Nφ = size(N)
    FT = eltype(N)

    ensure_current_gradients!(coupling, grid)
    ensure_refraction_tables!(coupling, cgrid, model.depth, grid, Nκ, Nφ, FT)

    # Refresh halos so periodic stencils see correct neighbours.
    for n in 1:Nφ, m in 1:Nκ
        fill_halo_regions!(physical_field(N, m, n))
    end

    Hx, Hy, iz = product_field_data_indices(N)

    arch = architecture(grid)
    topology = Oceananigans.Grids.topology(grid)
    xperiodic = topology[1] === Oceananigans.Grids.Periodic
    yperiodic = topology[2] === Oceananigans.Grids.Periodic
    kernel = _wave_current_refraction_tendency!(device(arch), (8, 8, 1, 1), (Nx, Ny, Nκ, Nφ))
    kernel(flat_data(G), flat_data(N),
           coupling.uᴰx, coupling.uᴰy,
           coupling.duᴰxdκ, coupling.duᴰydκ,
           coupling.uᴰx_x, coupling.uᴰx_y, coupling.uᴰy_x, coupling.uᴰy_y,
           cgrid.κ, cgrid.κ_faces, cgrid.φ_faces,
           coupling.cg_x_table, coupling.cg_y_table,
           coupling.cos_table, coupling.sin_table,
           grid,
           Hx, Hy, iz, Nx, Ny, Nκ, Nφ, xperiodic, yperiodic)
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
