import Oceananigans.Architectures: architecture, device, on_architecture
import KernelAbstractions
import KernelAbstractions: @kernel, @index

function compute_doppler_velocity!(uᴰx, uᴰy, uL, vL, depth, kappa, qtransform::QTransform)
    Nxᵘ, Nyᵘ, Nzᵘ = size(uL)
    Nxᵛ, Nyᵛ, Nzᵛ = size(vL)
    Nzᵛ == Nzᵘ || throw(ArgumentError("uL and vL must have matching vertical size"))
    size(uᴰx) == (Nxᵘ, Nyᵘ, length(kappa)) || throw(ArgumentError("uᴰx has wrong size"))
    size(uᴰy) == (Nxᵛ, Nyᵛ, length(kappa)) || throw(ArgumentError("uᴰy has wrong size"))
    z = vertical_nodes(qtransform)
    faces = vertical_faces(qtransform)
    length(z) == Nzᵘ || throw(ArgumentError("RectilinearGrid vertical cells do not match velocity fields"))

    arch = architecture(qtransform.grid)
    faces_on_arch = on_architecture(arch, faces)
    kappa_on_arch = on_architecture(arch, kappa)
    depth_on_arch = q_depth_on_architecture(arch, depth)
    u_kernel = _compute_doppler_velocity_x_component_kernel!(device(arch), (8, 8, 1), (Nxᵘ, Nyᵘ, length(kappa)))
    v_kernel = _compute_doppler_velocity_y_component_kernel!(device(arch), (8, 8, 1), (Nxᵛ, Nyᵛ, length(kappa)))
    topology = Oceananigans.Grids.topology(qtransform.grid)
    xperiodic = topology[1] === Oceananigans.Grids.Periodic
    yperiodic = topology[2] === Oceananigans.Grids.Periodic
    Nxq, Nyq = horizontal_size(qtransform.grid)
    u_kernel(uᴰx, uL, depth_on_arch, kappa_on_arch, faces_on_arch,
             qtransform.kernel, qtransform.cache_policy, Nzᵘ, Nxq, Nyq, xperiodic, yperiodic)
    v_kernel(uᴰy, vL, depth_on_arch, kappa_on_arch, faces_on_arch,
             qtransform.kernel, qtransform.cache_policy, Nzᵘ, Nxq, Nyq, xperiodic, yperiodic)
    KernelAbstractions.synchronize(device(arch))
    return uᴰx, uᴰy
end

function compute_doppler_velocity_derivative!(duᴰxdκ, duᴰydκ, uL, vL, depth, kappa, qtransform::QTransform)
    Nxᵘ, Nyᵘ, Nzᵘ = size(uL)
    Nxᵛ, Nyᵛ, Nzᵛ = size(vL)
    Nzᵛ == Nzᵘ || throw(ArgumentError("uL and vL must have matching vertical size"))
    size(duᴰxdκ) == (Nxᵘ, Nyᵘ, length(kappa)) || throw(ArgumentError("duᴰxdκ has wrong size"))
    size(duᴰydκ) == (Nxᵛ, Nyᵛ, length(kappa)) || throw(ArgumentError("duᴰydκ has wrong size"))
    z = vertical_nodes(qtransform)
    faces = vertical_faces(qtransform)
    length(z) == Nzᵘ || throw(ArgumentError("RectilinearGrid vertical cells do not match velocity fields"))

    arch = architecture(qtransform.grid)
    faces_on_arch = on_architecture(arch, faces)
    kappa_on_arch = on_architecture(arch, kappa)
    depth_on_arch = q_depth_on_architecture(arch, depth)
    u_kernel = _compute_doppler_velocity_x_component_derivative_kernel!(device(arch), (8, 8, 1), (Nxᵘ, Nyᵘ, length(kappa)))
    v_kernel = _compute_doppler_velocity_y_component_derivative_kernel!(device(arch), (8, 8, 1), (Nxᵛ, Nyᵛ, length(kappa)))
    topology = Oceananigans.Grids.topology(qtransform.grid)
    xperiodic = topology[1] === Oceananigans.Grids.Periodic
    yperiodic = topology[2] === Oceananigans.Grids.Periodic
    Nxq, Nyq = horizontal_size(qtransform.grid)
    u_kernel(duᴰxdκ, uL, depth_on_arch, kappa_on_arch, faces_on_arch,
             qtransform.kernel, qtransform.cache_policy, Nzᵘ, Nxq, Nyq, xperiodic, yperiodic)
    v_kernel(duᴰydκ, vL, depth_on_arch, kappa_on_arch, faces_on_arch,
             qtransform.kernel, qtransform.cache_policy, Nzᵘ, Nxq, Nyq, xperiodic, yperiodic)
    KernelAbstractions.synchronize(device(arch))
    return duᴰxdκ, duᴰydκ
end

@inline _q_index(i, N) = ifelse(i < 1, 1, ifelse(i > N, N, i))

@kernel function _compute_doppler_velocity_x_component_kernel!(uᴰ, uL, depth, kappa, faces,
                                                               qkernel, qpolicy, Nz, Nxq, Nyq, xperiodic, yperiodic)
    i, j, m = @index(Global, NTuple)
    d = q_depth_at_x_face(depth, i, j, xperiodic, yperiodic)
    iq = _q_index(i, Nxq)
    jq = _q_index(j, Nyq)
    a = zero(eltype(uᴰ))

    @inbounds for k in 1:Nz
        qΔz = q_cell_weight_kernel(qpolicy, qkernel, iq, jq, k, m, kappa[m], faces[k], faces[k+1], d)
        a += uL[i, j, k] * qΔz
    end

    @inbounds uᴰ[i, j, m] = a
end

@kernel function _compute_doppler_velocity_y_component_kernel!(uᴰ, uL, depth, kappa, faces,
                                                               qkernel, qpolicy, Nz, Nxq, Nyq, xperiodic, yperiodic)
    i, j, m = @index(Global, NTuple)
    d = q_depth_at_y_face(depth, i, j, xperiodic, yperiodic)
    iq = _q_index(i, Nxq)
    jq = _q_index(j, Nyq)
    a = zero(eltype(uᴰ))

    @inbounds for k in 1:Nz
        qΔz = q_cell_weight_kernel(qpolicy, qkernel, iq, jq, k, m, kappa[m], faces[k], faces[k+1], d)
        a += uL[i, j, k] * qΔz
    end

    @inbounds uᴰ[i, j, m] = a
end

@kernel function _compute_doppler_velocity_x_component_derivative_kernel!(duᴰdκ, uL,
                                                                          depth, kappa, faces,
                                                                          qkernel, qpolicy, Nz, Nxq, Nyq, xperiodic, yperiodic)
    i, j, m = @index(Global, NTuple)
    d = q_depth_at_x_face(depth, i, j, xperiodic, yperiodic)
    iq = _q_index(i, Nxq)
    jq = _q_index(j, Nyq)
    a = zero(eltype(duᴰdκ))

    @inbounds for k in 1:Nz
        dqΔz = q_cell_weight_kappa_derivative_kernel(qpolicy, qkernel, iq, jq, k, m, kappa[m], faces[k], faces[k+1], d)
        a += uL[i, j, k] * dqΔz
    end

    @inbounds duᴰdκ[i, j, m] = a
end

@kernel function _compute_doppler_velocity_y_component_derivative_kernel!(duᴰdκ, uL,
                                                                          depth, kappa, faces,
                                                                          qkernel, qpolicy, Nz, Nxq, Nyq, xperiodic, yperiodic)
    i, j, m = @index(Global, NTuple)
    d = q_depth_at_y_face(depth, i, j, xperiodic, yperiodic)
    iq = _q_index(i, Nxq)
    jq = _q_index(j, Nyq)
    a = zero(eltype(duᴰdκ))

    @inbounds for k in 1:Nz
        dqΔz = q_cell_weight_kappa_derivative_kernel(qpolicy, qkernel, iq, jq, k, m, kappa[m], faces[k], faces[k+1], d)
        a += uL[i, j, k] * dqΔz
    end

    @inbounds duᴰdκ[i, j, m] = a
end
