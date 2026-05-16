import Oceananigans.Architectures: architecture, device, on_architecture
import KernelAbstractions
import KernelAbstractions: @kernel, @index

function set!(f::ProductField, value::Number)
    Nx, Ny, Nxi, Neta = size(f)
    Hx, Hy, iz = product_field_data_indices(f)
    arch = architecture(f)
    kernel = _set_product_field_scalar!(device(arch), (8, 8, 1, 1), (Nx, Ny, Nxi, Neta))
    kernel(f.flat_data, convert(eltype(f), value), Hx, Hy, iz)
    KernelAbstractions.synchronize(device(arch))
    return f
end

function set!(f::ProductField; N=nothing)
    N === nothing && return f
    return set!(f, N)
end

function set!(f::ProductField, values::AbstractArray)
    size(values) == size(f) || throw(ArgumentError("array size $(size(values)) does not match field size $(size(f))"))
    Nx, Ny, Nxi, Neta = size(f)
    Hx, Hy, iz = product_field_data_indices(f)
    arch = architecture(f)
    values_on_arch = on_architecture(arch, values)
    kernel = _set_product_field_array!(device(arch), (8, 8, 1, 1), (Nx, Ny, Nxi, Neta))
    kernel(f.flat_data, values_on_arch, Hx, Hy, iz)
    KernelAbstractions.synchronize(device(arch))
    return f
end

function set!(f::ProductField, fun)
    _, _, Nxi, Neta = size(f)
    flat_vertical = has_flat_vertical_topology(grid(f))
    for n in 1:Neta, m in 1:Nxi
        kx, ky = k_components(coordinate_grid(f), m, n)
        setter = flat_vertical ? ((x, y) -> fun(x, y, kx, ky)) :
                                 ((x, y, z) -> fun(x, y, kx, ky))
        set!(physical_field(f, m, n), setter)
    end
    return f
end

function set!(model; N=nothing)
    N === nothing || set!(model.action, N)
    N === nothing || (model.previous_tendencies_ready = false)
    return model
end

@kernel function _set_product_field_scalar!(data, value, Hx, Hy, iz)
    i, j, m, n = @index(Global, NTuple)
    @inbounds data[i + Hx, j + Hy, iz, m, n] = value
end

@kernel function _set_product_field_array!(data, values, Hx, Hy, iz)
    i, j, m, n = @index(Global, NTuple)
    @inbounds data[i + Hx, j + Hy, iz, m, n] = values[i, j, m, n]
end
