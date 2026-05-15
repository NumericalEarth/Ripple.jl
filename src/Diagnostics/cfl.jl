function cfl(model)
    model.horizontal_advection === nothing && return zero(eltype(model.action))

    dt = model.clock.last_Δt
    dx = xspacings(model.grid)
    dy = yspacings(model.grid)
    _, _, Nxi, Neta = size(model.action)
    max_cfl = zero(eltype(model.action))

    for n in 1:Neta, m in 1:Nxi
        u, v = transport_velocity(model, m, n)
        local_cfl = abs(u) * dt / minimum(dx) + abs(v) * dt / minimum(dy)
        max_cfl = max(max_cfl, local_cfl)
    end

    return max_cfl
end
