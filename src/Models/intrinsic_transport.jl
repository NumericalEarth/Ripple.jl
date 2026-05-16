import KernelAbstractions
import KernelAbstractions: @kernel, @index
import Oceananigans.Architectures: architecture, device, on_architecture

# Source-free, no-current transport tendency in one fused KA kernel. Replaces
# the per-bin Oceananigans `div_Uc` loop in `compute_tendencies!` when
# `model.coupling === nothing` and `model.horizontal_advection isa WENO`.
# The legacy bin loop allocated a fresh `FluxFormAdvection` and a triple of
# `ConstantField`s per spectral cell per step — for the Tolman GSE test
# (480 bins × 288 steps) that's 138k allocations and 30 min wall-clock.
# This kernel reads the contiguous `flat_data` directly and does 5th-order
# WENO in (x, y) with the bin's cell-averaged intrinsic group velocity.

@kernel function _intrinsic_transport_kernel!(
    G_data, N_data,
    cg_x_table, cg_y_table,
    Δx_inv, Δy_inv,
    Hx, Hy, iz, Nx, Ny,
    periodic_x, periodic_y)
    i, j, m, n = @index(Global, NTuple)

    ux = intrinsic_velocity_component(cg_x_table, i, j, m, n)
    uy = intrinsic_velocity_component(cg_y_table, i, j, m, n)

    @inbounds begin
        ip1 = ifelse(periodic_x, _periodic(i + 1, Nx), _clamp_idx(i + 1, Nx))
        ip2 = ifelse(periodic_x, _periodic(i + 2, Nx), _clamp_idx(i + 2, Nx))
        ip3 = ifelse(periodic_x, _periodic(i + 3, Nx), _clamp_idx(i + 3, Nx))
        im1 = ifelse(periodic_x, _periodic(i - 1, Nx), _clamp_idx(i - 1, Nx))
        im2 = ifelse(periodic_x, _periodic(i - 2, Nx), _clamp_idx(i - 2, Nx))
        im3 = ifelse(periodic_x, _periodic(i - 3, Nx), _clamp_idx(i - 3, Nx))
        has_x_plus  = ifelse(periodic_x, true, (i - 2 >= 1) & (i + 3 <= Nx))
        has_x_minus = ifelse(periodic_x, true, (i - 3 >= 1) & (i + 2 <= Nx))

        x_m3 = N_data[im3 + Hx, j + Hy, iz, m, n]
        x_m2 = N_data[im2 + Hx, j + Hy, iz, m, n]
        x_m1 = N_data[im1 + Hx, j + Hy, iz, m, n]
        x_0  = N_data[i   + Hx, j + Hy, iz, m, n]
        x_p1 = N_data[ip1 + Hx, j + Hy, iz, m, n]
        x_p2 = N_data[ip2 + Hx, j + Hy, iz, m, n]
        x_p3 = N_data[ip3 + Hx, j + Hy, iz, m, n]
        N_xp = weno5_face_iphalf(x_m2, x_m1, x_0, x_p1, x_p2, x_p3, ux, has_x_plus)
        N_xm = weno5_face_iphalf(x_m3, x_m2, x_m1, x_0, x_p1, x_p2, ux, has_x_minus)
        flux_x = ux * (N_xp - N_xm) * Δx_inv

        jp1 = ifelse(periodic_y, _periodic(j + 1, Ny), _clamp_idx(j + 1, Ny))
        jp2 = ifelse(periodic_y, _periodic(j + 2, Ny), _clamp_idx(j + 2, Ny))
        jp3 = ifelse(periodic_y, _periodic(j + 3, Ny), _clamp_idx(j + 3, Ny))
        jm1 = ifelse(periodic_y, _periodic(j - 1, Ny), _clamp_idx(j - 1, Ny))
        jm2 = ifelse(periodic_y, _periodic(j - 2, Ny), _clamp_idx(j - 2, Ny))
        jm3 = ifelse(periodic_y, _periodic(j - 3, Ny), _clamp_idx(j - 3, Ny))
        has_y_plus  = ifelse(periodic_y, true, (j - 2 >= 1) & (j + 3 <= Ny))
        has_y_minus = ifelse(periodic_y, true, (j - 3 >= 1) & (j + 2 <= Ny))

        y_m3 = N_data[i + Hx, jm3 + Hy, iz, m, n]
        y_m2 = N_data[i + Hx, jm2 + Hy, iz, m, n]
        y_m1 = N_data[i + Hx, jm1 + Hy, iz, m, n]
        y_0  = N_data[i + Hx, j   + Hy, iz, m, n]
        y_p1 = N_data[i + Hx, jp1 + Hy, iz, m, n]
        y_p2 = N_data[i + Hx, jp2 + Hy, iz, m, n]
        y_p3 = N_data[i + Hx, jp3 + Hy, iz, m, n]
        N_yp = weno5_face_iphalf(y_m2, y_m1, y_0, y_p1, y_p2, y_p3, uy, has_y_plus)
        N_ym = weno5_face_iphalf(y_m3, y_m2, y_m1, y_0, y_p1, y_p2, uy, has_y_minus)
        flux_y = uy * (N_yp - N_ym) * Δy_inv

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

    Δx = first(xspacings(grid))
    Δy = first(yspacings(grid))
    Hx, Hy, iz = product_field_data_indices(N)

    topology = Oceananigans.Grids.topology(grid)
    periodic_x = topology[1] === Oceananigans.Grids.Periodic
    periodic_y = topology[2] === Oceananigans.Grids.Periodic

    arch = architecture(grid)
    kernel = _intrinsic_transport_kernel!(device(arch), (8, 8, 1, 1), (Nx, Ny, Nκ, Nφ))
    kernel(flat_data(G), flat_data(N),
           ws.cg_x_table, ws.cg_y_table,
           1 / Δx, 1 / Δy,
           Hx, Hy, iz, Nx, Ny,
           periodic_x, periodic_y)
    KernelAbstractions.synchronize(device(arch))
    return G
end
