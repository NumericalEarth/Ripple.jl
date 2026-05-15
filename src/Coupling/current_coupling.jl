abstract type AbstractCurrentCoupling end

struct NoCurrentCoupling <: AbstractCurrentCoupling end

struct PrescribedLagrangianMeanCurrent{U, V, D}
    u :: U
    v :: V
    depth :: D
end

current_data(a) = Base.invokelatest(field_storage, a)

function PrescribedLagrangianMeanCurrent(; u, v, depth)
    size(current_data(u)) == size(current_data(v)) || throw(ArgumentError("u and v must have matching size"))
    return PrescribedLagrangianMeanCurrent(u, v, depth)
end

mutable struct CWCMPrescribedCurrentCoupling{Current, QT, K, UxCache, UyCache, DUx, DUy} <: AbstractCurrentCoupling
    current :: Current
    qtransform :: QT
    kappa :: K
    Ux :: UxCache
    Uy :: UyCache
    dUxdkappa :: DUx
    dUydkappa :: DUy
    u_transport_scratch :: Any  # lazily allocated CenterField, reused across transport_velocity_fields calls
    v_transport_scratch :: Any
    Ux_x :: Any                 # spatial gradients of Doppler velocity caches, lazily allocated
    Ux_y :: Any
    Uy_x :: Any
    Uy_y :: Any
    cg_x_table :: Any           # intrinsic group velocity table per (κ, φ), lazily filled
    cg_y_table :: Any
    cos_table :: Any            # cos(φ), sin(φ) per direction index
    sin_table :: Any
    N_flat :: Any               # flat 4D scratch for the fused KA kernel
    G_flat :: Any
end

function current_cache_like(a, ::Type{FT}, dims::Tuple) where FT
    cache = similar(a, FT, dims)
    fill!(cache, zero(FT))
    return cache
end

function CWCMPrescribedCurrentCoupling(current::PrescribedLagrangianMeanCurrent,
                                       qtransform::QTransform,
                                       kappa)
    u = current_data(current.u)
    v = current_data(current.v)
    Nx, Ny, _ = size(u)
    kc = collect(float.(kappa))
    Ux = current_cache_like(u, eltype(u), (Nx, Ny, length(kc)))
    Uy = current_cache_like(v, eltype(v), (Nx, Ny, length(kc)))
    dUxdkappa = similar(Ux)
    dUydkappa = similar(Uy)
    coupling = CWCMPrescribedCurrentCoupling(current, qtransform, kc, Ux, Uy, dUxdkappa, dUydkappa,
                                              nothing, nothing,
                                              nothing, nothing, nothing, nothing,
                                              nothing, nothing, nothing, nothing,
                                              nothing, nothing)
    update_coupling!(coupling)
    return coupling
end

update_coupling!(::NoCurrentCoupling) = nothing
update_coupling!(::Nothing) = nothing

function update_coupling!(coupling::CWCMPrescribedCurrentCoupling)
    current = coupling.current
    u = current_data(current.u)
    v = current_data(current.v)
    compute_doppler_velocity!(coupling.Ux, coupling.Uy,
                              u, v, current.depth,
                              coupling.kappa, coupling.qtransform)
    compute_doppler_velocity_derivative!(coupling.dUxdkappa, coupling.dUydkappa,
                                         u, v, current.depth,
                                         coupling.kappa, coupling.qtransform)
    return coupling
end

function update_coupling!(model)
    update_coupling!(model.coupling)
    return model
end
