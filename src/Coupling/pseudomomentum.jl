import Oceananigans.Fields: Field

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
    dz = vertical_spacings(field)
    out = zeros(eltype(data), Nx, Ny)

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        out[i, j] += data[i, j, k] * dz[k]
    end

    return out
end

function compute_pseudomomentum(N::ProductField, z, depth, qtransform::QTransform)
    Nx, Ny, Nxi, Neta = size(N)
    cgrid = coordinate_grid(N)
    Nz = length(z)
    px = device_zeros(architecture(N), eltype(N), (Nx, Ny, Nz))
    py = device_zeros(architecture(N), eltype(N), (Nx, Ny, Nz))

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        d = depth isa Number ? depth : depth[i, j]
        ax = zero(eltype(N))
        ay = zero(eltype(N))
        for n in 1:Neta, m in 1:Nxi
            kap = radial_wavenumber(cgrid, m, n)
            q = q_value(qtransform.kernel, kap, z[k], d)
            kx_measure, ky_measure = spectral_first_moment_measures(cgrid, m, n)
            ax += q * N[i, j, m, n] * kx_measure
            ay += q * N[i, j, m, n] * ky_measure
        end
        px[i, j, k] = ax
        py[i, j, k] = ay
    end

    return px, py
end

function compute_pseudomomentum_cells!(px, py, N::ProductField, depth, qtransform::QTransform;
                                       cell_average::Bool)
    Nx, Ny, Nxi, Neta = size(N)
    cgrid = coordinate_grid(N)
    faces = vertical_faces(qtransform)
    Nz = length(faces) - 1
    px_data = field_storage(px)
    py_data = field_storage(py)

    size(px_data) == (Nx, Ny, Nz) || throw(ArgumentError("px has wrong size"))
    size(py_data) == (Nx, Ny, Nz) || throw(ArgumentError("py has wrong size"))

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        d = depth isa Number ? depth : depth[i, j]
        ax = zero(eltype(N))
        ay = zero(eltype(N))
        for n in 1:Neta, m in 1:Nxi
            kap = radial_wavenumber(cgrid, m, n)
            qΔz = q_cell_weight(qtransform, i, j, k, m, kap, faces[k], faces[k+1], d)
            kx_measure, ky_measure = spectral_first_moment_measures(cgrid, m, n)
            ax += qΔz * N[i, j, m, n] * kx_measure
            ay += qΔz * N[i, j, m, n] * ky_measure
        end
        scale = cell_average ? inv(abs(faces[k+1] - faces[k])) : one(eltype(N))
        px_data[i, j, k] = ax * scale
        py_data[i, j, k] = ay * scale
    end

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
    cgrid = coordinate_grid(new_N)
    faces = vertical_faces(qtransform)
    Nz = length(faces) - 1
    ptx_data = field_storage(ptx)
    pty_data = field_storage(pty)

    size(ptx_data) == (Nx, Ny, Nz) || throw(ArgumentError("ptx has wrong size"))
    size(pty_data) == (Nx, Ny, Nz) || throw(ArgumentError("pty has wrong size"))

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        d = depth isa Number ? depth : depth[i, j]
        ax = zero(eltype(new_N))
        ay = zero(eltype(new_N))
        for n in 1:Neta, m in 1:Nxi
            kap = radial_wavenumber(cgrid, m, n)
            qΔz = q_cell_weight(qtransform, i, j, k, m, kap, faces[k], faces[k+1], d)
            action_tendency = (new_N[i, j, m, n] - old_N[i, j, m, n]) / dt
            kx_measure, ky_measure = spectral_first_moment_measures(cgrid, m, n)
            ax += qΔz * action_tendency * kx_measure
            ay += qΔz * action_tendency * ky_measure
        end
        scale = inv(abs(faces[k+1] - faces[k]))
        ptx_data[i, j, k] = ax * scale
        pty_data[i, j, k] = ay * scale
    end

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
    field_storage(ut) .*= coefficient
    field_storage(vt) .*= coefficient
    return ut, vt
end

function pseudomomentum_tendency_fields(new_N::ProductField,
                                        old_N::ProductField,
                                        dt,
                                        depth,
                                        qtransform::QTransform;
                                        location=(Center, Center, Center))
    pgrid = pseudomomentum_grid(new_N, qtransform)
    ptx = pseudomomentum_field(pgrid; location, eltype=eltype(new_N))
    pty = pseudomomentum_field(pgrid; location, eltype=eltype(new_N))
    compute_pseudomomentum_tendency_cell_averages!(ptx, pty, new_N, old_N, dt, depth, qtransform)
    return ptx, pty
end

function pseudomomentum_fields(N::ProductField, depth, qtransform::QTransform;
                               location=(Center, Center, Center))
    pgrid = pseudomomentum_grid(N, qtransform)
    px = pseudomomentum_field(pgrid; location, eltype=eltype(N))
    py = pseudomomentum_field(pgrid; location, eltype=eltype(N))
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
