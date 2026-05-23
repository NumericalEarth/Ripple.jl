function physical_transport_enabled(model)
    model.horizontal_advection !== nothing && return true
    return model.coupling isa AbstractCWCMCurrentCoupling && model.spectral_advection !== nothing
end

function maximum_absolute_transport_velocity(u)
    u isa ZeroField && return zero(eltype(u))
    u isa ConstantField && return abs(u.constant)
    if u isa Field
        data = field_storage(u)
        return maximum(abs, Array(data))
    end
    return abs(u)
end

function cfl(model)
    physical_transport_enabled(model) || return zero(eltype(model.action))

    dt = model.clock.last_Δt
    dx = xspacings(model.grid)
    dy = yspacings(model.grid)
    _, _, Nxi, Neta = size(model.action)
    max_cfl = zero(eltype(model.action))

    for n in 1:Neta, m in 1:Nxi
        U = transport_velocity_fields(model, m, n)
        u = maximum_absolute_transport_velocity(U.u)
        v = maximum_absolute_transport_velocity(U.v)
        local_cfl = u * dt / minimum(dx) + v * dt / minimum(dy)
        max_cfl = max(max_cfl, local_cfl)
    end

    return max_cfl
end
