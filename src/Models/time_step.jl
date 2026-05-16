import Oceananigans.TimeSteppers: time_step!, update_state!, tick!

function add_scaled!(N::ProductField, G::ProductField, dt)
    Nx, Ny, Nxi, Neta = size(N)
    for n in 1:Neta, m in 1:Nxi, j in 1:Ny, i in 1:Nx
        N[i, j, m, n] = max(zero(eltype(N)), N[i, j, m, n] + dt * G[i, j, m, n])
    end
    return N
end

function add_scaled_adams_bashforth2!(N::ProductField, G::ProductField, Gprevious::ProductField, dt)
    Nx, Ny, Nxi, Neta = size(N)
    for n in 1:Neta, m in 1:Nxi, j in 1:Ny, i in 1:Nx
        tendency = 1.5 * G[i, j, m, n] - 0.5 * Gprevious[i, j, m, n]
        N[i, j, m, n] = max(zero(eltype(N)), N[i, j, m, n] + dt * tendency)
    end
    return N
end

function copy_field!(dest::ProductField, src::ProductField)
    return copy_product_field!(dest, src)
end

function add_scaled_semi_implicit!(N::ProductField, G::ProductField, dt, model)
    Nx, Ny, Nxi, Neta = size(N)
    damping = similar(N)
    explicit_part = similar(N)

    for n in 1:Neta, m in 1:Nxi, j in 1:Ny, i in 1:Nx
        _, λ = source_split(model.sources, model, i, j, m, n)
        damping[i, j, m, n] = λ
        explicit_part[i, j, m, n] = G[i, j, m, n] + λ * N[i, j, m, n]
    end

    for n in 1:Neta, m in 1:Nxi, j in 1:Ny, i in 1:Nx
        λ = damping[i, j, m, n]
        N[i, j, m, n] = max(zero(eltype(N)), (N[i, j, m, n] + dt * explicit_part[i, j, m, n]) / (1 + dt * λ))
    end
    return N
end

function combine!(dest::ProductField, a, A::ProductField, b, B::ProductField)
    Nx, Ny, Nxi, Neta = size(dest)
    for n in 1:Neta, m in 1:Nxi, j in 1:Ny, i in 1:Nx
        dest[i, j, m, n] = max(zero(eltype(dest)), a * A[i, j, m, n] + b * B[i, j, m, n])
    end
    return dest
end

function combine_with_increment!(dest::ProductField, a, A::ProductField, b, dt, G::ProductField)
    Nx, Ny, Nxi, Neta = size(dest)
    for n in 1:Neta, m in 1:Nxi, j in 1:Ny, i in 1:Nx
        stage_value = max(zero(eltype(dest)), dest[i, j, m, n] + dt * G[i, j, m, n])
        dest[i, j, m, n] = max(zero(eltype(dest)), a * A[i, j, m, n] + b * stage_value)
    end
    return dest
end

is_low_storage_rk3(timestepper) = timestepper === :LowStorageRK3 || timestepper === :LSRK3

function time_step!(model::SpectralWaveModel, dt; callbacks=[])
    dt > 0 || throw(ArgumentError("time step must be positive"))
    if model.timestepper === :ForwardEuler
        compute_tendencies!(model)
        add_scaled!(model.action, model.tendencies, dt)
        model.previous_tendencies_ready = false
    elseif model.timestepper === :SemiImplicitEuler
        compute_tendencies!(model)
        add_scaled_semi_implicit!(model.action, model.tendencies, dt, model)
        model.previous_tendencies_ready = false
    elseif model.timestepper === :AB2
        compute_tendencies!(model)
        if model.previous_tendencies_ready
            add_scaled_adams_bashforth2!(model.action, model.tendencies, model.previous_tendencies, dt)
        else
            add_scaled!(model.action, model.tendencies, dt)
        end
        copy_field!(model.previous_tendencies, model.tendencies)
        model.previous_tendencies_ready = true
    elseif model.timestepper === :RK3
        model.previous_tendencies_ready = false
        N0 = copy(model.action)
        compute_tendencies!(model)
        add_scaled!(model.action, model.tendencies, dt)

        compute_tendencies!(model)
        stage = copy(model.action)
        add_scaled!(stage, model.tendencies, dt)
        combine!(model.action, 0.75, N0, 0.25, stage)

        compute_tendencies!(model)
        stage = copy(model.action)
        add_scaled!(stage, model.tendencies, dt)
        combine!(model.action, 1/3, N0, 2/3, stage)
    elseif is_low_storage_rk3(model.timestepper)
        model.previous_tendencies_ready = false
        N0 = copy(model.action)
        compute_tendencies!(model)
        add_scaled!(model.action, model.tendencies, dt)

        compute_tendencies!(model)
        combine_with_increment!(model.action, 0.75, N0, 0.25, dt, model.tendencies)

        compute_tendencies!(model)
        combine_with_increment!(model.action, 1/3, N0, 2/3, dt, model.tendencies)
    else
        throw(ArgumentError("unsupported timestepper $(model.timestepper)"))
    end
    apply_propagation_smoothing!(model, model.propagation_smoothing, dt)
    tick!(model.clock, dt)
    return model
end

update_state!(model::SpectralWaveModel; callbacks=[], kwargs...) = model
