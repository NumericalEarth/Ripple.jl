import Oceananigans.Fields: Field, location
import Oceananigans.Grids: Center, Face
import Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
import Oceananigans.Architectures: architecture, device, on_architecture
import KernelAbstractions
import KernelAbstractions: @kernel, @index

function pseudomomentum_field(grid;
                              location=(Center, Center, Center),
                              eltype=Float64)
    LX, LY, LZ = map(canonical_location_marker, location)
    loc = (instantiate_location_marker(LX),
           instantiate_location_marker(LY),
           instantiate_location_marker(LZ))
    return Field(loc, grid, eltype)
end

vertical_spacings(field) = abs.(zspacings(grid(field)))

function vertical_integral(field)
    data = field_storage(field)
    Nx, Ny, Nz = size(data)
    arch = architecture(field)
    dz = on_architecture(arch, vertical_spacings(field))
    out = device_zeros(arch, eltype(data), (Nx, Ny))

    kernel = _vertical_integral_kernel!(device(arch), (8, 8), (Nx, Ny))
    kernel(out, data, dz, Nz)
    KernelAbstractions.synchronize(device(arch))

    return out
end

function pseudomomentum_spectral_tables(N::ProductField)
    _, _, Nxi, Neta = size(N)
    cgrid = coordinate_grid(N)
    FT = eltype(N)

    kappa = zeros(FT, Nxi, Neta)
    kx_measure = zeros(FT, Nxi, Neta)
    ky_measure = zeros(FT, Nxi, Neta)

    @inbounds for n in 1:Neta, m in 1:Nxi
        kappa[m, n] = radial_wavenumber(cgrid, m, n)
        kx, ky = spectral_first_moment_measures(cgrid, m, n)
        kx_measure[m, n] = kx
        ky_measure[m, n] = ky
    end

    arch = architecture(N)
    return (kappa = on_architecture(arch, kappa),
            kx_measure = on_architecture(arch, kx_measure),
            ky_measure = on_architecture(arch, ky_measure))
end

function pseudomomentum_moment_measure_tables(cgrid, ::Type{FT}, arch) where FT
    Nxi, Neta = coordinate_size(cgrid)
    kx_measure = zeros(FT, Nxi, Neta)
    ky_measure = zeros(FT, Nxi, Neta)

    @inbounds for n in 1:Neta, m in 1:Nxi
        kx, ky = spectral_first_moment_measures(cgrid, m, n)
        kx_measure[m, n] = kx
        ky_measure[m, n] = ky
    end

    return on_architecture(arch, kx_measure), on_architecture(arch, ky_measure)
end

function pseudomomentum_overlap_tables(qtransform::QTransform, kappa, depth, ::Type{FT}, arch) where FT
    faces = zfaces(qtransform.grid)
    Nκ = length(kappa)
    Nz = length(faces) - 1

    if !(depth isa Number)
        depths = collect(FT, depth)
        Nx, Ny = size(depths)
        overlap = zeros(FT, Nx, Ny, Nκ, Nκ)
        derivative_overlap = zeros(FT, Nx, Ny, Nκ, Nκ)

        @inbounds for source_m in 1:Nκ, target_m in 1:Nκ, j in 1:Ny, i in 1:Nx
            total, derivative_total = pseudomomentum_overlap(qtransform, kappa, faces, Nz,
                                                             target_m, source_m, depths[i, j], FT)
            overlap[i, j, target_m, source_m] = total
            derivative_overlap[i, j, target_m, source_m] = derivative_total
        end

        return on_architecture(arch, overlap), on_architecture(arch, derivative_overlap)
    end

    d = FT(depth)
    overlap = zeros(FT, Nκ, Nκ)
    derivative_overlap = zeros(FT, Nκ, Nκ)

    @inbounds for source_m in 1:Nκ, target_m in 1:Nκ
        total, derivative_total = pseudomomentum_overlap(qtransform, kappa, faces, Nz,
                                                         target_m, source_m, d, FT)
        overlap[target_m, source_m] = total
        derivative_overlap[target_m, source_m] = derivative_total
    end

    return on_architecture(arch, overlap), on_architecture(arch, derivative_overlap)
end

function pseudomomentum_overlap(qtransform::QTransform, kappa, faces, Nz,
                                target_m, source_m, d, ::Type{FT}) where FT
    source_κ = kappa[source_m]
    target_κ = kappa[target_m]
    total = zero(FT)
    derivative_total = zero(FT)

    @inbounds for k in 1:Nz
        z₁ = FT(faces[k])
        z₂ = FT(faces[k+1])
        Δz = abs(z₂ - z₁)
        source_weight = q_cell_integral(qtransform.kernel, source_κ, z₁, z₂, d)
        target_weight = q_cell_integral(qtransform.kernel, target_κ, z₁, z₂, d)
        target_derivative = q_cell_integral_kappa_derivative(qtransform.kernel, target_κ, z₁, z₂, d)
        total += target_weight * source_weight / Δz
        derivative_total += target_derivative * source_weight / Δz
    end

    return total, derivative_total
end

function compute_pseudomomentum_doppler_velocity!(coupling, N::ProductField)
    Nx, Ny, Nκ, Nφ = size(N)
    grid = physical_grid(N)
    u_size = cgrid_velocity_cache_size(grid, :x, Nκ)
    v_size = cgrid_velocity_cache_size(grid, :y, Nκ)
    size(coupling.uᴰx) == u_size ||
        throw(ArgumentError("pseudomomentum coupling uᴰx cache has wrong size; expected $u_size, got $(size(coupling.uᴰx))"))
    size(coupling.uᴰy) == v_size ||
        throw(ArgumentError("pseudomomentum coupling uᴰy cache has wrong size; expected $v_size, got $(size(coupling.uᴰy))"))

    Hx, Hy, iz = product_field_data_indices(N)
    arch = architecture(N)
    topology = Oceananigans.Grids.topology(grid)
    xperiodic = topology[1] === Oceananigans.Grids.Periodic
    yperiodic = topology[2] === Oceananigans.Grids.Periodic
    u_kernel = _compute_x_pseudomomentum_doppler_velocity_kernel!(device(arch), (8, 8, 1), u_size)
    v_kernel = _compute_y_pseudomomentum_doppler_velocity_kernel!(device(arch), (8, 8, 1), v_size)
    u_kernel(coupling.uᴰx, coupling.duᴰxdκ,
             flat_data(N), coupling.overlap, coupling.derivative_overlap,
             coupling.kx_measure, grid, Hx, Hy, iz, Nx, Ny, Nκ, Nφ,
             xperiodic, yperiodic)
    v_kernel(coupling.uᴰy, coupling.duᴰydκ,
             flat_data(N), coupling.overlap, coupling.derivative_overlap,
             coupling.ky_measure, grid, Hx, Hy, iz, Nx, Ny, Nκ, Nφ,
             xperiodic, yperiodic)
    KernelAbstractions.synchronize(device(arch))
    return coupling
end

function compute_pseudomomentum(N::ProductField, z, depth, qtransform::QTransform)
    Nx, Ny, Nxi, Neta = size(N)
    Nz = length(z)
    arch = architecture(N)
    px = device_zeros(arch, eltype(N), (Nx, Ny, Nz))
    py = device_zeros(arch, eltype(N), (Nx, Ny, Nz))
    tables = pseudomomentum_spectral_tables(N)
    Hx, Hy, iz = product_field_data_indices(N)
    z_on_arch = on_architecture(arch, z)
    depth_on_arch = q_depth_on_architecture(arch, depth)

    kernel = _compute_pseudomomentum_kernel!(device(arch), (8, 8, 1), (Nx, Ny, Nz))
    kernel(px, py, flat_data(N), z_on_arch, depth_on_arch,
           tables.kappa, tables.kx_measure, tables.ky_measure,
           qtransform.kernel, Hx, Hy, iz, Nxi, Neta)
    KernelAbstractions.synchronize(device(arch))

    return px, py
end

function compute_pseudomomentum_cells!(px, py, N::ProductField, depth, qtransform::QTransform;
                                       cell_average::Bool)
    Nx, Ny, Nxi, Neta = size(N)
    faces = vertical_faces(qtransform)
    Nz = length(faces) - 1
    px_data = field_storage(px)
    py_data = field_storage(py)

    arch = architecture(N)
    tables = pseudomomentum_spectral_tables(N)
    Hx, Hy, iz = product_field_data_indices(N)
    faces_on_arch = on_architecture(arch, faces)
    depth_on_arch = q_depth_on_architecture(arch, depth)

    if px isa Field && py isa Field &&
       location(px) == (Face, Center, Center) &&
       location(py) == (Center, Face, Center)
        px_expected_size = cgrid_velocity_cache_size(physical_grid(N), :x, Nz)
        py_expected_size = cgrid_velocity_cache_size(physical_grid(N), :y, Nz)
        size(px_data) == px_expected_size || throw(ArgumentError("px has wrong C-grid size"))
        size(py_data) == py_expected_size || throw(ArgumentError("py has wrong C-grid size"))

        topology = Oceananigans.Grids.topology(physical_grid(N))
        xperiodic = topology[1] === Oceananigans.Grids.Periodic
        yperiodic = topology[2] === Oceananigans.Grids.Periodic
        x_kernel = _compute_x_pseudomomentum_cells_kernel!(device(arch), (8, 8, 1), size(px_data))
        y_kernel = _compute_y_pseudomomentum_cells_kernel!(device(arch), (8, 8, 1), size(py_data))
        x_kernel(px_data, flat_data(N), depth_on_arch, faces_on_arch,
                 tables.kappa, tables.kx_measure, qtransform.kernel, qtransform.cache_policy,
                 physical_grid(N), Hx, Hy, iz, Nx, Ny, Nxi, Neta, cell_average,
                 xperiodic, yperiodic)
        y_kernel(py_data, flat_data(N), depth_on_arch, faces_on_arch,
                 tables.kappa, tables.ky_measure, qtransform.kernel, qtransform.cache_policy,
                 physical_grid(N), Hx, Hy, iz, Nx, Ny, Nxi, Neta, cell_average,
                 xperiodic, yperiodic)
        KernelAbstractions.synchronize(device(arch))
        return px, py
    end

    size(px_data) == (Nx, Ny, Nz) || throw(ArgumentError("px has wrong size"))
    size(py_data) == (Nx, Ny, Nz) || throw(ArgumentError("py has wrong size"))

    kernel = _compute_pseudomomentum_cells_kernel!(device(arch), (8, 8, 1), (Nx, Ny, Nz))
    kernel(px_data, py_data, flat_data(N), depth_on_arch, faces_on_arch,
           tables.kappa, tables.kx_measure, tables.ky_measure,
           qtransform.kernel, qtransform.cache_policy,
           Hx, Hy, iz, Nxi, Neta, cell_average)
    KernelAbstractions.synchronize(device(arch))

    return px, py
end

compute_pseudomomentum_cell_integrals!(px, py, N::ProductField, depth, qtransform::QTransform) =
    compute_pseudomomentum_cells!(px, py, N, depth, qtransform; cell_average=false)

function compute_pseudomomentum_cell_integrals(N::ProductField, depth, qtransform::QTransform)
    Nx, Ny = horizontal_size(N.grid)
    Nz = vertical_size(qtransform.grid)
    px = device_zeros(architecture(N), eltype(N), (Nx, Ny, Nz))
    py = device_zeros(architecture(N), eltype(N), (Nx, Ny, Nz))
    return compute_pseudomomentum_cell_integrals!(px, py, N, depth, qtransform)
end

compute_pseudomomentum_cell_averages!(px, py, N::ProductField, depth, qtransform::QTransform) =
    compute_pseudomomentum_cells!(px, py, N, depth, qtransform; cell_average=true)

function compute_pseudomomentum_cell_averages(N::ProductField, depth, qtransform::QTransform)
    Nx, Ny = horizontal_size(N.grid)
    Nz = vertical_size(qtransform.grid)
    px = device_zeros(architecture(N), eltype(N), (Nx, Ny, Nz))
    py = device_zeros(architecture(N), eltype(N), (Nx, Ny, Nz))
    return compute_pseudomomentum_cell_averages!(px, py, N, depth, qtransform)
end

function product_fields_match_for_pseudomomentum_tendency(new_N::ProductField, old_N::ProductField)
    size(new_N) == size(old_N) ||
        throw(ArgumentError("new and old wave-action fields must have matching sizes"))
    coordinate_size(coordinate_grid(new_N)) == coordinate_size(coordinate_grid(old_N)) ||
        throw(ArgumentError("new and old wave-action fields must have matching spectral grids"))
    return true
end

function compute_pseudomomentum_tendency_cell_averages!(ptx, pty,
                                                        new_N::ProductField,
                                                        old_N::ProductField,
                                                        dt,
                                                        depth,
                                                        qtransform::QTransform)
    dt > 0 || throw(ArgumentError("pseudomomentum tendency timestep must be positive"))
    product_fields_match_for_pseudomomentum_tendency(new_N, old_N)

    Nx, Ny, Nxi, Neta = size(new_N)
    faces = vertical_faces(qtransform)
    Nz = length(faces) - 1
    ptx_data = field_storage(ptx)
    pty_data = field_storage(pty)

    arch = architecture(new_N)
    tables = pseudomomentum_spectral_tables(new_N)
    Hx, Hy, iz = product_field_data_indices(new_N)
    faces_on_arch = on_architecture(arch, faces)
    depth_on_arch = q_depth_on_architecture(arch, depth)

    if ptx isa Field && pty isa Field &&
       location(ptx) == (Face, Center, Center) &&
       location(pty) == (Center, Face, Center)
        ptx_expected_size = cgrid_velocity_cache_size(physical_grid(new_N), :x, Nz)
        pty_expected_size = cgrid_velocity_cache_size(physical_grid(new_N), :y, Nz)
        size(ptx_data) == ptx_expected_size || throw(ArgumentError("ptx has wrong C-grid size"))
        size(pty_data) == pty_expected_size || throw(ArgumentError("pty has wrong C-grid size"))

        topology = Oceananigans.Grids.topology(physical_grid(new_N))
        xperiodic = topology[1] === Oceananigans.Grids.Periodic
        yperiodic = topology[2] === Oceananigans.Grids.Periodic
        x_kernel = _compute_x_pseudomomentum_tendency_cells_kernel!(device(arch), (8, 8, 1), size(ptx_data))
        y_kernel = _compute_y_pseudomomentum_tendency_cells_kernel!(device(arch), (8, 8, 1), size(pty_data))
        dtᶠ = convert(eltype(new_N), dt)
        x_kernel(ptx_data, flat_data(new_N), flat_data(old_N), dtᶠ,
                 depth_on_arch, faces_on_arch, tables.kappa, tables.kx_measure,
                 qtransform.kernel, qtransform.cache_policy,
                 physical_grid(new_N), Hx, Hy, iz, Nx, Ny, Nxi, Neta,
                 xperiodic, yperiodic)
        y_kernel(pty_data, flat_data(new_N), flat_data(old_N), dtᶠ,
                 depth_on_arch, faces_on_arch, tables.kappa, tables.ky_measure,
                 qtransform.kernel, qtransform.cache_policy,
                 physical_grid(new_N), Hx, Hy, iz, Nx, Ny, Nxi, Neta,
                 xperiodic, yperiodic)
        KernelAbstractions.synchronize(device(arch))
        return ptx, pty
    end

    size(ptx_data) == (Nx, Ny, Nz) || throw(ArgumentError("ptx has wrong size"))
    size(pty_data) == (Nx, Ny, Nz) || throw(ArgumentError("pty has wrong size"))

    kernel = _compute_pseudomomentum_tendency_cells_kernel!(device(arch), (8, 8, 1), (Nx, Ny, Nz))
    kernel(ptx_data, pty_data, flat_data(new_N), flat_data(old_N),
           convert(eltype(new_N), dt), depth_on_arch, faces_on_arch,
           tables.kappa, tables.kx_measure, tables.ky_measure,
           qtransform.kernel, qtransform.cache_policy,
           Hx, Hy, iz, Nxi, Neta)
    KernelAbstractions.synchronize(device(arch))

    return ptx, pty
end

function cwcm_momentum_tendency_fields!(ut, vt,
                                        new_N::ProductField,
                                        old_N::ProductField,
                                        dt,
                                        depth,
                                        qtransform::QTransform;
                                        coefficient=-1)
    compute_pseudomomentum_tendency_cell_averages!(ut, vt, new_N, old_N, dt, depth, qtransform)
    ut_data = field_storage(ut)
    vt_data = field_storage(vt)
    arch = architecture(ut)
    u_kernel = _scale_field_kernel!(device(arch), (8, 8, 1), size(ut_data))
    v_kernel = _scale_field_kernel!(device(arch), (8, 8, 1), size(vt_data))
    u_kernel(ut_data, convert(eltype(new_N), coefficient))
    v_kernel(vt_data, convert(eltype(new_N), coefficient))
    KernelAbstractions.synchronize(device(arch))
    return ut, vt
end

function pseudomomentum_output_fields(pgrid, ::Type{FT}, location) where FT
    if location == (Face, Center, Center) || location == (Center, Face, Center)
        px = pseudomomentum_field(pgrid; location=(Face, Center, Center), eltype=FT)
        py = pseudomomentum_field(pgrid; location=(Center, Face, Center), eltype=FT)
    else
        px = pseudomomentum_field(pgrid; location, eltype=FT)
        py = pseudomomentum_field(pgrid; location, eltype=FT)
    end
    return px, py
end

function pseudomomentum_tendency_fields(new_N::ProductField,
                                        old_N::ProductField,
                                        dt,
                                        depth,
                                        qtransform::QTransform;
                                        location=(Center, Center, Center))
    pgrid = pseudomomentum_grid(new_N, qtransform)
    ptx, pty = pseudomomentum_output_fields(pgrid, eltype(new_N), location)
    compute_pseudomomentum_tendency_cell_averages!(ptx, pty, new_N, old_N, dt, depth, qtransform)
    return ptx, pty
end

function pseudomomentum_fields(N::ProductField, depth, qtransform::QTransform;
                               location=(Center, Center, Center))
    pgrid = pseudomomentum_grid(N, qtransform)
    px, py = pseudomomentum_output_fields(pgrid, eltype(N), location)
    compute_pseudomomentum_cell_averages!(px, py, N, depth, qtransform)
    return px, py
end

function pseudomomentum_grid(N::ProductField, qtransform::QTransform)
    pgrid = grid(N)
    zfaces(pgrid) == zfaces(qtransform.grid) && return pgrid
    TX, TY, _ = OceanGrids.topology(pgrid)
    _, _, TZ = OceanGrids.topology(qtransform.grid)
    return RectilinearGrid(architecture(pgrid);
                           size=(pgrid.Nx, pgrid.Ny, qtransform.grid.Nz),
                           x=xfaces(pgrid),
                           y=yfaces(pgrid),
                           z=zfaces(qtransform.grid),
                           topology=(TX, TY, TZ))
end

@kernel function _vertical_integral_kernel!(out, data, dz, Nz)
    i, j = @index(Global, NTuple)
    total = zero(eltype(out))
    @inbounds for k in 1:Nz
        total += data[i, j, k] * dz[k]
    end
    @inbounds out[i, j] = total
end

@inline _pseudomomentum_index(i, N, periodic) =
    ifelse(periodic, mod1(i, N), ifelse(i < 1, 1, ifelse(i > N, N, i)))

@inline function _pseudomomentum_center_projection(i, j, k, grid,
                                                   N_data, overlap, measure,
                                                   Hx, Hy, iz, Nx, Ny, Nκ, Nφ,
                                                   target_m, xperiodic, yperiodic)
    ii = _pseudomomentum_index(i, Nx, xperiodic)
    jj = _pseudomomentum_index(j, Ny, yperiodic)
    ix = ii + Hx
    jy = jj + Hy
    total = zero(eltype(N_data))

    @inbounds for n in 1:Nφ, source_m in 1:Nκ
        action = N_data[ix, jy, iz, source_m, n]
        q_overlap = pseudomomentum_overlap_at(overlap, ii, jj, target_m, source_m)
        total += q_overlap * action * measure[source_m, n]
    end

    return total
end

@inline function _pseudomomentum_cell_projection(i, j, k, grid,
                                                 N_data, depth, faces, kappa, measure,
                                                 qkernel, qpolicy, Hx, Hy, iz,
                                                 Nx, Ny, Nxi, Neta, cell_average,
                                                 xperiodic, yperiodic)
    ii = _pseudomomentum_index(i, Nx, xperiodic)
    jj = _pseudomomentum_index(j, Ny, yperiodic)
    d = q_depth_at(depth, ii, jj)
    ix = ii + Hx
    jy = jj + Hy
    z₁ = faces[k]
    z₂ = faces[k+1]
    total = zero(eltype(N_data))

    @inbounds for n in 1:Neta, m in 1:Nxi
        qΔz = q_cell_weight_kernel(qpolicy, qkernel, ii, jj, k, m, kappa[m, n], z₁, z₂, d)
        total += qΔz * N_data[ix, jy, iz, m, n] * measure[m, n]
    end

    scale = ifelse(cell_average, inv(abs(z₂ - z₁)), one(total))
    return total * scale
end

@kernel function _compute_x_pseudomomentum_cells_kernel!(px, N_data, depth, faces,
                                                         kappa, kx_measure,
                                                         qkernel, qpolicy, grid,
                                                         Hx, Hy, iz, Nx, Ny, Nxi, Neta,
                                                         cell_average, xperiodic, yperiodic)
    i, j, k = @index(Global, NTuple)
    @inbounds px[i, j, k] = ℑxᶠᵃᵃ(i, j, k, grid, _pseudomomentum_cell_projection,
                                    N_data, depth, faces, kappa, kx_measure,
                                    qkernel, qpolicy, Hx, Hy, iz,
                                    Nx, Ny, Nxi, Neta, cell_average,
                                    xperiodic, yperiodic)
end

@kernel function _compute_y_pseudomomentum_cells_kernel!(py, N_data, depth, faces,
                                                         kappa, ky_measure,
                                                         qkernel, qpolicy, grid,
                                                         Hx, Hy, iz, Nx, Ny, Nxi, Neta,
                                                         cell_average, xperiodic, yperiodic)
    i, j, k = @index(Global, NTuple)
    @inbounds py[i, j, k] = ℑyᵃᶠᵃ(i, j, k, grid, _pseudomomentum_cell_projection,
                                    N_data, depth, faces, kappa, ky_measure,
                                    qkernel, qpolicy, Hx, Hy, iz,
                                    Nx, Ny, Nxi, Neta, cell_average,
                                    xperiodic, yperiodic)
end

@inline function _pseudomomentum_tendency_cell_projection(i, j, k, grid,
                                                          new_N, old_N, dt, depth, faces,
                                                          kappa, measure, qkernel, qpolicy,
                                                          Hx, Hy, iz, Nx, Ny, Nxi, Neta,
                                                          xperiodic, yperiodic)
    ii = _pseudomomentum_index(i, Nx, xperiodic)
    jj = _pseudomomentum_index(j, Ny, yperiodic)
    d = q_depth_at(depth, ii, jj)
    ix = ii + Hx
    jy = jj + Hy
    z₁ = faces[k]
    z₂ = faces[k+1]
    total = zero(eltype(new_N))

    @inbounds for n in 1:Neta, m in 1:Nxi
        qΔz = q_cell_weight_kernel(qpolicy, qkernel, ii, jj, k, m, kappa[m, n], z₁, z₂, d)
        action_tendency = (new_N[ix, jy, iz, m, n] - old_N[ix, jy, iz, m, n]) / dt
        total += qΔz * action_tendency * measure[m, n]
    end

    return total / abs(z₂ - z₁)
end

@kernel function _compute_x_pseudomomentum_tendency_cells_kernel!(ptx, new_N, old_N, dt,
                                                                  depth, faces, kappa, kx_measure,
                                                                  qkernel, qpolicy, grid,
                                                                  Hx, Hy, iz, Nx, Ny, Nxi, Neta,
                                                                  xperiodic, yperiodic)
    i, j, k = @index(Global, NTuple)
    @inbounds ptx[i, j, k] = ℑxᶠᵃᵃ(i, j, k, grid, _pseudomomentum_tendency_cell_projection,
                                     new_N, old_N, dt, depth, faces, kappa, kx_measure,
                                     qkernel, qpolicy, Hx, Hy, iz, Nx, Ny, Nxi, Neta,
                                     xperiodic, yperiodic)
end

@kernel function _compute_y_pseudomomentum_tendency_cells_kernel!(pty, new_N, old_N, dt,
                                                                  depth, faces, kappa, ky_measure,
                                                                  qkernel, qpolicy, grid,
                                                                  Hx, Hy, iz, Nx, Ny, Nxi, Neta,
                                                                  xperiodic, yperiodic)
    i, j, k = @index(Global, NTuple)
    @inbounds pty[i, j, k] = ℑyᵃᶠᵃ(i, j, k, grid, _pseudomomentum_tendency_cell_projection,
                                     new_N, old_N, dt, depth, faces, kappa, ky_measure,
                                     qkernel, qpolicy, Hx, Hy, iz, Nx, Ny, Nxi, Neta,
                                     xperiodic, yperiodic)
end

@kernel function _compute_x_pseudomomentum_doppler_velocity_kernel!(uᴰx, duᴰxdκ,
                                                                    N_data, overlap, derivative_overlap,
                                                                    kx_measure, grid,
                                                                    Hx, Hy, iz, Nx, Ny, Nκ, Nφ,
                                                                    xperiodic, yperiodic)
    i, j, target_m = @index(Global, NTuple)
    @inbounds begin
        uᴰx[i, j, target_m] = ℑxᶠᵃᵃ(i, j, 1, grid, _pseudomomentum_center_projection,
                                      N_data, overlap, kx_measure,
                                      Hx, Hy, iz, Nx, Ny, Nκ, Nφ,
                                      target_m, xperiodic, yperiodic)
        duᴰxdκ[i, j, target_m] = ℑxᶠᵃᵃ(i, j, 1, grid, _pseudomomentum_center_projection,
                                         N_data, derivative_overlap, kx_measure,
                                         Hx, Hy, iz, Nx, Ny, Nκ, Nφ,
                                         target_m, xperiodic, yperiodic)
    end
end

@kernel function _compute_y_pseudomomentum_doppler_velocity_kernel!(uᴰy, duᴰydκ,
                                                                    N_data, overlap, derivative_overlap,
                                                                    ky_measure, grid,
                                                                    Hx, Hy, iz, Nx, Ny, Nκ, Nφ,
                                                                    xperiodic, yperiodic)
    i, j, target_m = @index(Global, NTuple)
    @inbounds begin
        uᴰy[i, j, target_m] = ℑyᵃᶠᵃ(i, j, 1, grid, _pseudomomentum_center_projection,
                                      N_data, overlap, ky_measure,
                                      Hx, Hy, iz, Nx, Ny, Nκ, Nφ,
                                      target_m, xperiodic, yperiodic)
        duᴰydκ[i, j, target_m] = ℑyᵃᶠᵃ(i, j, 1, grid, _pseudomomentum_center_projection,
                                         N_data, derivative_overlap, ky_measure,
                                         Hx, Hy, iz, Nx, Ny, Nκ, Nφ,
                                         target_m, xperiodic, yperiodic)
    end
end

@inline pseudomomentum_overlap_at(overlap::AbstractArray{T, 2}, i, j, target_m, source_m) where T =
    overlap[target_m, source_m]

@inline pseudomomentum_overlap_at(overlap::AbstractArray{T, 4}, i, j, target_m, source_m) where T =
    overlap[i, j, target_m, source_m]

@kernel function _compute_pseudomomentum_kernel!(px, py, N_data, z, depth,
                                                 kappa, kx_measure, ky_measure,
                                                 qkernel, Hx, Hy, iz, Nxi, Neta)
    i, j, k = @index(Global, NTuple)
    d = q_depth_at(depth, i, j)
    ax = zero(eltype(px))
    ay = zero(eltype(py))
    ix = i + Hx
    jy = j + Hy

    @inbounds for n in 1:Neta, m in 1:Nxi
        q = q_value_kernel(qkernel, kappa[m, n], z[k], d)
        action = N_data[ix, jy, iz, m, n]
        ax += q * action * kx_measure[m, n]
        ay += q * action * ky_measure[m, n]
    end

    @inbounds begin
        px[i, j, k] = ax
        py[i, j, k] = ay
    end
end

@kernel function _compute_pseudomomentum_cells_kernel!(px, py, N_data, depth, faces,
                                                       kappa, kx_measure, ky_measure,
                                                       qkernel, qpolicy,
                                                       Hx, Hy, iz, Nxi, Neta, cell_average)
    i, j, k = @index(Global, NTuple)
    d = q_depth_at(depth, i, j)
    ax = zero(eltype(px))
    ay = zero(eltype(py))
    ix = i + Hx
    jy = j + Hy
    z₁ = faces[k]
    z₂ = faces[k+1]

    @inbounds for n in 1:Neta, m in 1:Nxi
        qΔz = q_cell_weight_kernel(qpolicy, qkernel, i, j, k, m, kappa[m, n], z₁, z₂, d)
        action = N_data[ix, jy, iz, m, n]
        ax += qΔz * action * kx_measure[m, n]
        ay += qΔz * action * ky_measure[m, n]
    end

    scale = ifelse(cell_average, inv(abs(z₂ - z₁)), one(ax))
    @inbounds begin
        px[i, j, k] = ax * scale
        py[i, j, k] = ay * scale
    end
end

@kernel function _compute_pseudomomentum_tendency_cells_kernel!(ptx, pty, new_N, old_N,
                                                                dt, depth, faces,
                                                                kappa, kx_measure, ky_measure,
                                                                qkernel, qpolicy,
                                                                Hx, Hy, iz, Nxi, Neta)
    i, j, k = @index(Global, NTuple)
    d = q_depth_at(depth, i, j)
    ax = zero(eltype(ptx))
    ay = zero(eltype(pty))
    ix = i + Hx
    jy = j + Hy
    z₁ = faces[k]
    z₂ = faces[k+1]

    @inbounds for n in 1:Neta, m in 1:Nxi
        qΔz = q_cell_weight_kernel(qpolicy, qkernel, i, j, k, m, kappa[m, n], z₁, z₂, d)
        action_tendency = (new_N[ix, jy, iz, m, n] - old_N[ix, jy, iz, m, n]) / dt
        ax += qΔz * action_tendency * kx_measure[m, n]
        ay += qΔz * action_tendency * ky_measure[m, n]
    end

    scale = inv(abs(z₂ - z₁))
    @inbounds begin
        ptx[i, j, k] = ax * scale
        pty[i, j, k] = ay * scale
    end
end

@kernel function _scale_field_kernel!(a, coefficient)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] *= coefficient
end
