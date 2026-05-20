abstract type AbstractCurrentCoupling end
abstract type AbstractCWCMCurrentCoupling <: AbstractCurrentCoupling end

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

mutable struct CWCMPrescribedCurrentCoupling{Current, QT, K, UᴰxCache, UᴰyCache, DκUᴰx, DκUᴰy} <: AbstractCWCMCurrentCoupling
    current :: Current
    qtransform :: QT
    kappa :: K
    uᴰx :: UᴰxCache
    uᴰy :: UᴰyCache
    duᴰxdκ :: DκUᴰx
    duᴰydκ :: DκUᴰy
    u_transport_scratch :: Any  # lazily allocated CenterField, reused across transport_velocity_fields calls
    v_transport_scratch :: Any
    uᴰx_x :: Any                 # spatial gradients of Doppler velocity caches, lazily allocated
    uᴰx_y :: Any
    uᴰy_x :: Any
    uᴰy_y :: Any
    cg_x_table :: Any           # intrinsic group velocity table per (κ, φ), lazily filled
    cg_y_table :: Any
    cos_table :: Any            # cos(φ), sin(φ) per direction index
    sin_table :: Any
    N_flat :: Any               # flat 4D scratch for the fused KA kernel
    G_flat :: Any
end

mutable struct CWCMPseudomomentumCoupling{QT, D, K, UᴰxCache, UᴰyCache, DκUᴰx, DκUᴰy, KM, YM, O, DO} <: AbstractCWCMCurrentCoupling
    qtransform :: QT
    depth :: D
    kappa :: K
    uᴰx :: UᴰxCache
    uᴰy :: UᴰyCache
    duᴰxdκ :: DκUᴰx
    duᴰydκ :: DκUᴰy
    kx_measure :: KM
    ky_measure :: YM
    overlap :: O
    derivative_overlap :: DO
    u_transport_scratch :: Any
    v_transport_scratch :: Any
    uᴰx_x :: Any
    uᴰx_y :: Any
    uᴰy_x :: Any
    uᴰy_y :: Any
    cg_x_table :: Any
    cg_y_table :: Any
    cos_table :: Any
    sin_table :: Any
    N_flat :: Any
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
    uᴰx = current_cache_like(u, eltype(u), (Nx, Ny, length(kc)))
    uᴰy = current_cache_like(v, eltype(v), (Nx, Ny, length(kc)))
    duᴰxdκ = similar(uᴰx)
    duᴰydκ = similar(uᴰy)
    coupling = CWCMPrescribedCurrentCoupling(current, qtransform, kc, uᴰx, uᴰy, duᴰxdκ, duᴰydκ,
                                              nothing, nothing,
                                              nothing, nothing, nothing, nothing,
                                              nothing, nothing, nothing, nothing,
                                              nothing, nothing)
    update_coupling!(coupling)
    return coupling
end

function CWCMPseudomomentumCoupling(model_grid,
                                    qtransform::QTransform,
                                    spectral_grid::PolarWaveVectorGrid,
                                    depth)
    Nx, Ny = horizontal_size(model_grid)
    arch = architecture(model_grid)
    DepthFT = depth isa Number ? typeof(float(depth)) : eltype(depth)
    FT = promote_type(grid_float_type(model_grid), coordinate_float_type(spectral_grid), DepthFT)
    kappa = collect(FT, Array(spectral_grid.κ))
    Nκ, Nφ = coordinate_size(spectral_grid)

    uᴰx = device_zeros(arch, FT, (Nx, Ny, Nκ))
    uᴰy = device_zeros(arch, FT, (Nx, Ny, Nκ))
    duᴰxdκ = similar(uᴰx)
    duᴰydκ = similar(uᴰy)
    fill!(duᴰxdκ, zero(FT))
    fill!(duᴰydκ, zero(FT))
    kx_measure, ky_measure = pseudomomentum_moment_measure_tables(spectral_grid, FT, arch)
    overlap, derivative_overlap = pseudomomentum_overlap_tables(qtransform, kappa, depth, FT, arch)

    return CWCMPseudomomentumCoupling(qtransform, depth, kappa,
                                      uᴰx, uᴰy, duᴰxdκ, duᴰydκ,
                                      kx_measure, ky_measure,
                                      overlap, derivative_overlap,
                                      nothing, nothing,
                                      nothing, nothing, nothing, nothing,
                                      nothing, nothing, nothing, nothing,
                                      nothing, nothing)
end

update_coupling!(::NoCurrentCoupling) = nothing
update_coupling!(::Nothing) = nothing
update_coupling!(coupling, model) = update_coupling!(coupling)

function update_coupling!(coupling::CWCMPrescribedCurrentCoupling)
    current = coupling.current
    u = current_data(current.u)
    v = current_data(current.v)
    compute_doppler_velocity!(coupling.uᴰx, coupling.uᴰy,
                              u, v, current.depth,
                              coupling.kappa, coupling.qtransform)
    compute_doppler_velocity_derivative!(coupling.duᴰxdκ, coupling.duᴰydκ,
                                         u, v, current.depth,
                                         coupling.kappa, coupling.qtransform)
    return coupling
end

function update_coupling!(coupling::CWCMPseudomomentumCoupling, model)
    compute_pseudomomentum_doppler_velocity!(coupling, model.action)
    return coupling
end

function update_coupling!(model)
    update_coupling!(model.coupling, model)
    return model
end
