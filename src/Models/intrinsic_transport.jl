import KernelAbstractions
import KernelAbstractions: @kernel, @index
import Oceananigans.Architectures: architecture, device, on_architecture
import Oceananigans.Operators: ∂xᶜᶜᶜ, ∂yᶜᶜᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ

# Source-free, no-current transport tendency in one fused KA kernel. Replaces
# the per-bin Oceananigans `div_Uc` loop in `compute_tendencies!` when
# `model.coupling === nothing` and `model.horizontal_advection isa WENO`.
# The legacy bin loop allocated a fresh `FluxFormAdvection` and a triple of
# `ConstantField`s per spectral cell per step — for the Tolman GSE test
# (480 bins × 288 steps) that's 138k allocations and 30 min wall-clock.
# This kernel reads the contiguous `flat_data` directly and does 5th-order
# WENO in (x, y) with the bin's cell-averaged intrinsic group velocity.

@inline _intrinsic_stencil_index(i, N, periodic) =
    ifelse(periodic, _periodic(i, N), _clamp_idx(i, N))

@inline function _intrinsic_cg_value(i, j, k, grid, table, m, n, Nx, Ny, periodic_x, periodic_y)
    ii = _intrinsic_stencil_index(i, Nx, periodic_x)
    jj = _intrinsic_stencil_index(j, Ny, periodic_y)
    return intrinsic_velocity_component(table, ii, jj, m, n)
end

@inline function _intrinsic_x_flux(iᶠ, j, k, grid, N_data, cg_x_table,
                                   Hx, Hy, iz, Nx, Ny, m, n,
                                   periodic_x, periodic_y)
    iₗ = iᶠ - 1
    jy = _intrinsic_stencil_index(j, Ny, periodic_y) + Hy
    boundary_face = !periodic_x & ((iᶠ == 1) | (iᶠ == Nx + 1))

    ux = ℑxᶠᵃᵃ(iᶠ, j, k, grid, _intrinsic_cg_value,
               cg_x_table, m, n, Nx, Ny, periodic_x, periodic_y)

    im2 = _intrinsic_stencil_index(iₗ - 2, Nx, periodic_x) + Hx
    im1 = _intrinsic_stencil_index(iₗ - 1, Nx, periodic_x) + Hx
    i0  = _intrinsic_stencil_index(iₗ,     Nx, periodic_x) + Hx
    ip1 = _intrinsic_stencil_index(iₗ + 1, Nx, periodic_x) + Hx
    ip2 = _intrinsic_stencil_index(iₗ + 2, Nx, periodic_x) + Hx
    ip3 = _intrinsic_stencil_index(iₗ + 3, Nx, periodic_x) + Hx
    has_stencil = ifelse(periodic_x, true, (iₗ - 2 >= 1) & (iₗ + 3 <= Nx))

    @inbounds begin
        N_m2 = N_data[im2, jy, iz, m, n]
        N_m1 = N_data[im1, jy, iz, m, n]
        N_0  = N_data[i0,  jy, iz, m, n]
        N_p1 = N_data[ip1, jy, iz, m, n]
        N_p2 = N_data[ip2, jy, iz, m, n]
        N_p3 = N_data[ip3, jy, iz, m, n]
    end

    N_face = weno5_face_iphalf(N_m2, N_m1, N_0, N_p1, N_p2, N_p3, ux, has_stencil)
    flux = ux * N_face
    return ifelse(boundary_face, zero(flux), flux)
end

@inline function _intrinsic_y_flux(i, jᶠ, k, grid, N_data, cg_y_table,
                                   Hx, Hy, iz, Nx, Ny, m, n,
                                   periodic_x, periodic_y)
    ix = _intrinsic_stencil_index(i, Nx, periodic_x) + Hx
    jₗ = jᶠ - 1
    boundary_face = !periodic_y & ((jᶠ == 1) | (jᶠ == Ny + 1))

    uy = ℑyᵃᶠᵃ(i, jᶠ, k, grid, _intrinsic_cg_value,
               cg_y_table, m, n, Nx, Ny, periodic_x, periodic_y)

    jm2 = _intrinsic_stencil_index(jₗ - 2, Ny, periodic_y) + Hy
    jm1 = _intrinsic_stencil_index(jₗ - 1, Ny, periodic_y) + Hy
    j0  = _intrinsic_stencil_index(jₗ,     Ny, periodic_y) + Hy
    jp1 = _intrinsic_stencil_index(jₗ + 1, Ny, periodic_y) + Hy
    jp2 = _intrinsic_stencil_index(jₗ + 2, Ny, periodic_y) + Hy
    jp3 = _intrinsic_stencil_index(jₗ + 3, Ny, periodic_y) + Hy
    has_stencil = ifelse(periodic_y, true, (jₗ - 2 >= 1) & (jₗ + 3 <= Ny))

    @inbounds begin
        N_m2 = N_data[ix, jm2, iz, m, n]
        N_m1 = N_data[ix, jm1, iz, m, n]
        N_0  = N_data[ix, j0,  iz, m, n]
        N_p1 = N_data[ix, jp1, iz, m, n]
        N_p2 = N_data[ix, jp2, iz, m, n]
        N_p3 = N_data[ix, jp3, iz, m, n]
    end

    N_face = weno5_face_iphalf(N_m2, N_m1, N_0, N_p1, N_p2, N_p3, uy, has_stencil)
    flux = uy * N_face
    return ifelse(boundary_face, zero(flux), flux)
end

@kernel function _intrinsic_transport_kernel!(
    G_data, N_data,
    cg_x_table, cg_y_table,
    grid,
    Hx, Hy, iz, Nx, Ny,
    periodic_x, periodic_y)
    i, j, m, n = @index(Global, NTuple)

    @inbounds begin
        flux_x = ∂xᶜᶜᶜ(i, j, 1, grid, _intrinsic_x_flux,
                         N_data, cg_x_table, Hx, Hy, iz, Nx, Ny, m, n,
                         periodic_x, periodic_y)
        flux_y = ∂yᶜᶜᶜ(i, j, 1, grid, _intrinsic_y_flux,
                         N_data, cg_y_table, Hx, Hy, iz, Nx, Ny, m, n,
                         periodic_x, periodic_y)

        G_data[i + Hx, j + Hy, iz, m, n] = -(flux_x + flux_y)
    end
end

# Lazy workspace cache hung off the model (initialized to `nothing`,
# rebuilt when the spectral grid changes size).
mutable struct IntrinsicTransportWorkspace{T}
    cg_x_table :: Any
    cg_y_table :: Any
end

function ensure_intrinsic_transport_workspace!(model)
    cgrid = model.spectral_grid
    FT = eltype(model.action)
    Nκ, Nφ = coordinate_size(cgrid)
    ws = model.intrinsic_transport_workspace
    expected_size = is_spatially_varying_depth(model.depth) ?
                    (model.grid.Nx, model.grid.Ny, Nκ, Nφ) : (Nκ, Nφ)
    if !(ws isa IntrinsicTransportWorkspace{FT}) ||
       size(ws.cg_x_table) != expected_size ||
       is_spatially_varying_depth(model.depth)
        cg_x, cg_y = intrinsic_group_velocity_tables(cgrid, model.depth, model.grid, FT)
        ws = IntrinsicTransportWorkspace{FT}(cg_x, cg_y)
        model.intrinsic_transport_workspace = ws
    end
    return ws
end

# Driver: write the source-free transport tendency into G using the fused
# kernel. Halos are refreshed once; sources (if any) are added by the
# dispatch in `compute_tendencies!`.
function compute_intrinsic_transport_tendency!(G, N, model)
    grid = model.grid
    Nx, Ny, Nκ, Nφ = size(N)

    ws = ensure_intrinsic_transport_workspace!(model)

    fill_halo_regions!(N)

    Hx, Hy, iz = product_field_data_indices(N)

    topology = Oceananigans.Grids.topology(grid)
    periodic_x = topology[1] === Oceananigans.Grids.Periodic
    periodic_y = topology[2] === Oceananigans.Grids.Periodic

    arch = architecture(grid)
    kernel = _intrinsic_transport_kernel!(device(arch), (8, 8, 1, 1), (Nx, Ny, Nκ, Nφ))
    kernel(flat_data(G), flat_data(N),
           ws.cg_x_table, ws.cg_y_table,
           grid,
           Hx, Hy, iz, Nx, Ny,
           periodic_x, periodic_y)
    KernelAbstractions.synchronize(device(arch))
    return G
end
