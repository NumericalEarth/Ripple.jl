import Oceananigans.Architectures: architecture, device, on_architecture
import KernelAbstractions
import KernelAbstractions: @kernel, @index

function compute_doppler_velocity!(Ux, Uy, uL, vL, depth, kappa, qtransform::QTransform)
    Nx, Ny, Nz = size(uL)
    size(vL) == size(uL) || throw(ArgumentError("uL and vL must have matching size"))
    size(Ux) == (Nx, Ny, length(kappa)) || throw(ArgumentError("Ux has wrong size"))
    size(Uy) == (Nx, Ny, length(kappa)) || throw(ArgumentError("Uy has wrong size"))
    z = vertical_nodes(qtransform)
    faces = vertical_faces(qtransform)
    length(z) == Nz || throw(ArgumentError("RectilinearGrid vertical cells do not match velocity fields"))

    arch = architecture(qtransform.grid)
    faces_on_arch = on_architecture(arch, faces)
    kappa_on_arch = on_architecture(arch, kappa)
    depth_on_arch = q_depth_on_architecture(arch, depth)
    kernel = _compute_doppler_velocity_kernel!(device(arch), (8, 8, 1), (Nx, Ny, length(kappa)))
    kernel(Ux, Uy, uL, vL, depth_on_arch, kappa_on_arch, faces_on_arch,
           qtransform.kernel, qtransform.cache_policy, Nz)
    KernelAbstractions.synchronize(device(arch))
    return Ux, Uy
end

function compute_doppler_velocity_derivative!(dUxdkappa, dUydkappa, uL, vL, depth, kappa, qtransform::QTransform)
    Nx, Ny, Nz = size(uL)
    size(vL) == size(uL) || throw(ArgumentError("uL and vL must have matching size"))
    size(dUxdkappa) == (Nx, Ny, length(kappa)) || throw(ArgumentError("dUxdkappa has wrong size"))
    size(dUydkappa) == (Nx, Ny, length(kappa)) || throw(ArgumentError("dUydkappa has wrong size"))
    z = vertical_nodes(qtransform)
    faces = vertical_faces(qtransform)
    length(z) == Nz || throw(ArgumentError("RectilinearGrid vertical cells do not match velocity fields"))

    arch = architecture(qtransform.grid)
    faces_on_arch = on_architecture(arch, faces)
    kappa_on_arch = on_architecture(arch, kappa)
    depth_on_arch = q_depth_on_architecture(arch, depth)
    kernel = _compute_doppler_velocity_derivative_kernel!(device(arch), (8, 8, 1), (Nx, Ny, length(kappa)))
    kernel(dUxdkappa, dUydkappa, uL, vL, depth_on_arch, kappa_on_arch, faces_on_arch,
           qtransform.kernel, qtransform.cache_policy, Nz)
    KernelAbstractions.synchronize(device(arch))
    return dUxdkappa, dUydkappa
end

@kernel function _compute_doppler_velocity_kernel!(Ux, Uy, uL, vL, depth, kappa, faces,
                                                   qkernel, qpolicy, Nz)
    i, j, m = @index(Global, NTuple)
    d = q_depth_at(depth, i, j)
    ax = zero(eltype(Ux))
    ay = zero(eltype(Uy))

    @inbounds for k in 1:Nz
        qΔz = q_cell_weight_kernel(qpolicy, qkernel, i, j, k, m, kappa[m], faces[k], faces[k+1], d)
        ax += uL[i, j, k] * qΔz
        ay += vL[i, j, k] * qΔz
    end

    @inbounds begin
        Ux[i, j, m] = ax
        Uy[i, j, m] = ay
    end
end

@kernel function _compute_doppler_velocity_derivative_kernel!(dUxdkappa, dUydkappa, uL, vL,
                                                              depth, kappa, faces,
                                                              qkernel, qpolicy, Nz)
    i, j, m = @index(Global, NTuple)
    d = q_depth_at(depth, i, j)
    ax = zero(eltype(dUxdkappa))
    ay = zero(eltype(dUydkappa))

    @inbounds for k in 1:Nz
        dqΔz = q_cell_weight_kappa_derivative_kernel(qpolicy, qkernel, i, j, k, m, kappa[m], faces[k], faces[k+1], d)
        ax += uL[i, j, k] * dqΔz
        ay += vL[i, j, k] * dqΔz
    end

    @inbounds begin
        dUxdkappa[i, j, m] = ax
        dUydkappa[i, j, m] = ay
    end
end
