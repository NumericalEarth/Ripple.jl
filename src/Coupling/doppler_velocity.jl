function compute_doppler_velocity!(Ux, Uy, uL, vL, depth, kappa, qtransform::QTransform)
    Nx, Ny, Nz = size(uL)
    size(vL) == size(uL) || throw(ArgumentError("uL and vL must have matching size"))
    size(Ux) == (Nx, Ny, length(kappa)) || throw(ArgumentError("Ux has wrong size"))
    size(Uy) == (Nx, Ny, length(kappa)) || throw(ArgumentError("Uy has wrong size"))
    z = vertical_nodes(qtransform)
    faces = vertical_faces(qtransform)
    length(z) == Nz || throw(ArgumentError("RectilinearGrid vertical cells do not match velocity fields"))

    for m in eachindex(kappa), j in 1:Ny, i in 1:Nx
        d = depth isa Number ? depth : depth[i, j]
        ax = zero(eltype(Ux))
        ay = zero(eltype(Uy))
        for k in 1:Nz
            qΔz = q_cell_weight(qtransform, i, j, k, m, kappa[m], faces[k], faces[k+1], d)
            ax += uL[i, j, k] * qΔz
            ay += vL[i, j, k] * qΔz
        end
        Ux[i, j, m] = ax
        Uy[i, j, m] = ay
    end
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

    for m in eachindex(kappa), j in 1:Ny, i in 1:Nx
        d = depth isa Number ? depth : depth[i, j]
        ax = zero(eltype(dUxdkappa))
        ay = zero(eltype(dUydkappa))
        for k in 1:Nz
            dqΔz = q_cell_weight_kappa_derivative(qtransform, i, j, k, m, kappa[m], faces[k], faces[k+1], d)
            ax += uL[i, j, k] * dqΔz
            ay += vL[i, j, k] * dqΔz
        end
        dUxdkappa[i, j, m] = ax
        dUydkappa[i, j, m] = ay
    end
    return dUxdkappa, dUydkappa
end
