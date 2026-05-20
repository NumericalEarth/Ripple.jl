import Oceananigans.TimeSteppers: time_step!, update_state!, tick!
import Oceananigans.TimeSteppers: RungeKutta3TimeStepper
import Oceananigans.Architectures: architecture, device
import KernelAbstractions
import KernelAbstractions: @kernel, @index

function launch_product_field_update!(kernel!, reference::ProductField, args...)
    Nx, Ny, Nxi, Neta = size(reference)
    Hx, Hy, iz = product_field_data_indices(reference)
    arch = architecture(reference)
    kernel = kernel!(device(arch), (8, 8, 1, 1), (Nx, Ny, Nxi, Neta))
    kernel(args..., Hx, Hy, iz)
    KernelAbstractions.synchronize(device(arch))
    return nothing
end

function add_scaled!(N::ProductField, G::ProductField, dt)
    launch_product_field_update!(_add_scaled_kernel!, N,
                                 flat_data(N), flat_data(G), convert(eltype(N), dt))
    return N
end

function add_scaled_adams_bashforth2!(N::ProductField, G::ProductField, Gprevious::ProductField, dt)
    launch_product_field_update!(_add_scaled_adams_bashforth2_kernel!, N,
                                 flat_data(N), flat_data(G), flat_data(Gprevious),
                                 convert(eltype(N), dt))
    return N
end

function add_scaled_runge_kutta_3!(N::ProductField, G::ProductField, Gprevious::ProductField, dt, γ, ζ)
    FT = eltype(N)
    launch_product_field_update!(_add_scaled_runge_kutta_3_kernel!, N,
                                 flat_data(N), flat_data(G), flat_data(Gprevious),
                                 convert(FT, dt), convert(FT, γ), convert(FT, ζ))
    return N
end

function add_scaled_runge_kutta_3!(N::ProductField, G::ProductField, Gprevious::ProductField, dt, γ, ::Nothing)
    FT = eltype(N)
    launch_product_field_update!(_add_scaled_runge_kutta_3_first_stage_kernel!, N,
                                 flat_data(N), flat_data(G),
                                 convert(FT, dt), convert(FT, γ))
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

    launch_product_field_update!(_add_scaled_semi_implicit_finalize_kernel!, N,
                                 flat_data(N), flat_data(explicit_part), flat_data(damping),
                                 convert(eltype(N), dt))
    return N
end

@kernel function _add_scaled_kernel!(N, G, dt, Hx, Hy, iz)
    i, j, m, n = @index(Global, NTuple)
    ix = i + Hx
    jy = j + Hy
    @inbounds begin
        N[ix, jy, iz, m, n] = max(zero(eltype(N)), N[ix, jy, iz, m, n] + dt * G[ix, jy, iz, m, n])
    end
end

@kernel function _add_scaled_adams_bashforth2_kernel!(N, G, Gprevious, dt, Hx, Hy, iz)
    i, j, m, n = @index(Global, NTuple)
    ix = i + Hx
    jy = j + Hy
    half = one(dt) / 2
    @inbounds begin
        tendency = 3 * half * G[ix, jy, iz, m, n] - half * Gprevious[ix, jy, iz, m, n]
        N[ix, jy, iz, m, n] = max(zero(eltype(N)), N[ix, jy, iz, m, n] + dt * tendency)
    end
end

@kernel function _add_scaled_runge_kutta_3_kernel!(N, G, Gprevious, dt, γ, ζ, Hx, Hy, iz)
    i, j, m, n = @index(Global, NTuple)
    ix = i + Hx
    jy = j + Hy
    @inbounds begin
        N[ix, jy, iz, m, n] = max(zero(eltype(N)),
                                  N[ix, jy, iz, m, n] +
                                  dt * (γ * G[ix, jy, iz, m, n] + ζ * Gprevious[ix, jy, iz, m, n]))
    end
end

@kernel function _add_scaled_runge_kutta_3_first_stage_kernel!(N, G, dt, γ, Hx, Hy, iz)
    i, j, m, n = @index(Global, NTuple)
    ix = i + Hx
    jy = j + Hy
    @inbounds begin
        N[ix, jy, iz, m, n] = max(zero(eltype(N)),
                                  N[ix, jy, iz, m, n] + dt * γ * G[ix, jy, iz, m, n])
    end
end

@kernel function _add_scaled_semi_implicit_finalize_kernel!(N, explicit_part, damping, dt, Hx, Hy, iz)
    i, j, m, n = @index(Global, NTuple)
    ix = i + Hx
    jy = j + Hy
    @inbounds begin
        λ = damping[ix, jy, iz, m, n]
        numerator = N[ix, jy, iz, m, n] + dt * explicit_part[ix, jy, iz, m, n]
        N[ix, jy, iz, m, n] = max(zero(eltype(N)), numerator / (one(dt) + dt * λ))
    end
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
    elseif model.timestepper isa RungeKutta3TimeStepper
        model.previous_tendencies_ready = false
        timestepper = model.timestepper

        compute_tendencies!(model)
        add_scaled_runge_kutta_3!(model.action, model.tendencies, model.previous_tendencies,
                                  dt, timestepper.γ¹, nothing)
        copy_field!(model.previous_tendencies, model.tendencies)

        compute_tendencies!(model)
        add_scaled_runge_kutta_3!(model.action, model.tendencies, model.previous_tendencies,
                                  dt, timestepper.γ², timestepper.ζ²)
        copy_field!(model.previous_tendencies, model.tendencies)

        compute_tendencies!(model)
        add_scaled_runge_kutta_3!(model.action, model.tendencies, model.previous_tendencies,
                                  dt, timestepper.γ³, timestepper.ζ³)
        copy_field!(model.previous_tendencies, model.tendencies)
    else
        throw(ArgumentError("unsupported timestepper $(model.timestepper)"))
    end
    apply_propagation_smoothing!(model, model.propagation_smoothing, dt)
    tick!(model.clock, dt)
    return model
end

update_state!(model::SpectralWaveModel; callbacks=[], kwargs...) = model
