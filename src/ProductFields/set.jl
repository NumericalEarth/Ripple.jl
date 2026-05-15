function set!(f::ProductField, value::Number)
    _, _, Nxi, Neta = size(f)
    for n in 1:Neta, m in 1:Nxi
        set!(physical_field(f, m, n), value)
    end
    return f
end

function set!(f::ProductField; N=nothing)
    N === nothing && return f
    return set!(f, N)
end

function set!(f::ProductField, values::AbstractArray)
    size(values) == size(f) || throw(ArgumentError("array size $(size(values)) does not match field size $(size(f))"))
    _, _, Nxi, Neta = size(f)
    for n in 1:Neta, m in 1:Nxi
        field = physical_field(f, m, n)
        interior(field)[:, :, 1] .= view(values, :, :, m, n)
    end
    return f
end

function set!(f::ProductField, fun)
    _, _, Nxi, Neta = size(f)
    for n in 1:Neta, m in 1:Nxi
        ξ, η = spectral_coordinates(coordinate_grid(f), m, n)
        set!(physical_field(f, m, n), (x, y, z) -> fun(x, y, ξ, η))
    end
    return f
end

function set!(model; N=nothing)
    N === nothing || set!(model.action, N)
    N === nothing || (model.previous_tendencies_ready = false)
    return model
end
