import Oceananigans.Architectures: architecture, device, on_architecture
import KernelAbstractions
import KernelAbstractions: @kernel, @index

function compute_doppler_velocity!(uᴰx, uᴰy, uL, vL, depth, kappa, qtransform::QTransform)
    Nx, Ny, Nz = size(uL)
    size(vL) == size(uL) || throw(ArgumentError("uL and vL must have matching size"))
    size(uᴰx) == (Nx, Ny, length(kappa)) || throw(ArgumentError("uᴰx has wrong size"))
    size(uᴰy) == (Nx, Ny, length(kappa)) || throw(ArgumentError("uᴰy has wrong size"))
    z = vertical_nodes(qtransform)
    faces = vertical_faces(qtransform)
    length(z) == Nz || throw(ArgumentError("RectilinearGrid vertical cells do not match velocity fields"))

    arch = architecture(qtransform.grid)
    faces_on_arch = on_architecture(arch, faces)
    kappa_on_arch = on_architecture(arch, kappa)
    depth_on_arch = q_depth_on_architecture(arch, depth)
    kernel = _compute_doppler_velocity_kernel!(device(arch), (8, 8, 1), (Nx, Ny, length(kappa)))
    kernel(uᴰx, uᴰy, uL, vL, depth_on_arch, kappa_on_arch, faces_on_arch,
           qtransform.kernel, qtransform.cache_policy, Nz)
    KernelAbstractions.synchronize(device(arch))
    return uᴰx, uᴰy
end

function compute_doppler_velocity_derivative!(duᴰxdκ, duᴰydκ, uL, vL, depth, kappa, qtransform::QTransform)
    Nx, Ny, Nz = size(uL)
    size(vL) == size(uL) || throw(ArgumentError("uL and vL must have matching size"))
    size(duᴰxdκ) == (Nx, Ny, length(kappa)) || throw(ArgumentError("duᴰxdκ has wrong size"))
    size(duᴰydκ) == (Nx, Ny, length(kappa)) || throw(ArgumentError("duᴰydκ has wrong size"))
    z = vertical_nodes(qtransform)
    faces = vertical_faces(qtransform)
    length(z) == Nz || throw(ArgumentError("RectilinearGrid vertical cells do not match velocity fields"))

    arch = architecture(qtransform.grid)
    faces_on_arch = on_architecture(arch, faces)
    kappa_on_arch = on_architecture(arch, kappa)
    depth_on_arch = q_depth_on_architecture(arch, depth)
    kernel = _compute_doppler_velocity_derivative_kernel!(device(arch), (8, 8, 1), (Nx, Ny, length(kappa)))
    kernel(duᴰxdκ, duᴰydκ, uL, vL, depth_on_arch, kappa_on_arch, faces_on_arch,
           qtransform.kernel, qtransform.cache_policy, Nz)
    KernelAbstractions.synchronize(device(arch))
    return duᴰxdκ, duᴰydκ
end

@kernel function _compute_doppler_velocity_kernel!(uᴰx, uᴰy, uL, vL, depth, kappa, faces,
                                                   qkernel, qpolicy, Nz)
    i, j, m = @index(Global, NTuple)
    d = q_depth_at(depth, i, j)
    ax = zero(eltype(uᴰx))
    ay = zero(eltype(uᴰy))

    @inbounds for k in 1:Nz
        qΔz = q_cell_weight_kernel(qpolicy, qkernel, i, j, k, m, kappa[m], faces[k], faces[k+1], d)
        ax += uL[i, j, k] * qΔz
        ay += vL[i, j, k] * qΔz
    end

    @inbounds begin
        uᴰx[i, j, m] = ax
        uᴰy[i, j, m] = ay
    end
end

@kernel function _compute_doppler_velocity_derivative_kernel!(duᴰxdκ, duᴰydκ, uL, vL,
                                                              depth, kappa, faces,
                                                              qkernel, qpolicy, Nz)
    i, j, m = @index(Global, NTuple)
    d = q_depth_at(depth, i, j)
    ax = zero(eltype(duᴰxdκ))
    ay = zero(eltype(duᴰydκ))

    @inbounds for k in 1:Nz
        dqΔz = q_cell_weight_kappa_derivative_kernel(qpolicy, qkernel, i, j, k, m, kappa[m], faces[k], faces[k+1], d)
        ax += uL[i, j, k] * dqΔz
        ay += vL[i, j, k] * dqΔz
    end

    @inbounds begin
        duᴰxdκ[i, j, m] = ax
        duᴰydκ[i, j, m] = ay
    end
end
