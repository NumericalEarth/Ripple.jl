import Oceananigans.Architectures: architecture, on_architecture

abstract type AbstractQStoragePolicy end
struct OnTheFlyQ <: AbstractQStoragePolicy end
struct CacheDopplerVelocity <: AbstractQStoragePolicy end
struct CacheDopplerVelocityAndDerivative <: AbstractQStoragePolicy end

struct PrecomputeQWeights{K, D, W} <: AbstractQStoragePolicy
    kappa :: K
    depth :: D
    weights :: W
end

struct QTransform{Q, G, Policy}
    kernel :: Q
    grid :: G
    cache_policy :: Policy
end

QTransform(kernel::QKernel, grid::AbstractGrid,
           cache_policy::AbstractQStoragePolicy=CacheDopplerVelocityAndDerivative()) =
    has_flat_vertical_topology(grid) ?
    throw(ArgumentError("QTransform requires a grid with a resolved vertical coordinate; pass 3D velocity fields, an explicit velocity `q_grid`, or finite model `depth` when the wave model grid is Flat")) :
    QTransform{typeof(kernel), typeof(grid), typeof(cache_policy)}(
        kernel, grid, cache_policy)

vertical_nodes(qtransform::QTransform) = znodes(qtransform.grid)
vertical_faces(qtransform::QTransform) = zfaces(qtransform.grid)

function PrecomputeQWeights(kernel::QKernel, grid::AbstractGrid,
                            kappa, depth::Number)
    kc = collect(float.(kappa))
    faces = zfaces(grid)
    Nz = length(faces) - 1
    weights = [q_cell_integral(kernel, kc[m], faces[k], faces[k+1], float(depth))
               for k in 1:Nz, m in eachindex(kc)]
    arch = architecture(grid)
    return PrecomputeQWeights(on_architecture(arch, kc),
                              float(depth),
                              on_architecture(arch, weights))
end

function PrecomputeQWeights(kernel::QKernel, grid::AbstractGrid,
                            kappa, depth::AbstractMatrix)
    kc = collect(float.(kappa))
    depths = collect(float.(depth))
    faces = zfaces(grid)
    Nx, Ny = size(depths)
    Nz = length(faces) - 1
    weights = zeros(eltype(depths), Nx, Ny, Nz, length(kc))

    for m in eachindex(kc), k in 1:Nz, j in 1:Ny, i in 1:Nx
        weights[i, j, k, m] = q_cell_integral(kernel, kc[m], faces[k], faces[k+1], depths[i, j])
    end

    arch = architecture(grid)
    return PrecomputeQWeights(on_architecture(arch, kc),
                              on_architecture(arch, depths),
                              on_architecture(arch, weights))
end

PrecomputeQWeights(qtransform::QTransform, kappa, depth) =
    PrecomputeQWeights(qtransform.kernel, qtransform.grid, kappa, depth)

function check_precomputed_kappa(policy::PrecomputeQWeights, m, kappa)
    1 <= m <= length(policy.kappa) ||
        throw(ArgumentError("precomputed Q weights do not contain radial index $m"))
    policy.kappa[m] == kappa ||
        throw(ArgumentError("precomputed Q weights were built for kappa=$(policy.kappa[m]), got kappa=$kappa"))
    return nothing
end

function precomputed_q_cell_weight(policy::PrecomputeQWeights{K, <:Number},
                                   i, j, k, m, kappa, depth) where K
    check_precomputed_kappa(policy, m, kappa)
    depth == policy.depth ||
        throw(ArgumentError("precomputed Q weights were built for depth=$(policy.depth), got depth=$depth"))
    return policy.weights[k, m]
end

function precomputed_q_cell_weight(policy::PrecomputeQWeights{K, D},
                                   i, j, k, m, kappa, depth) where {K, D}
    check_precomputed_kappa(policy, m, kappa)
    depth == policy.depth[i, j] ||
        throw(ArgumentError("precomputed Q weights were built for depth=$(policy.depth[i, j]), got depth=$depth at ($i, $j)"))
    return policy.weights[i, j, k, m]
end

function q_cell_weight(qtransform::QTransform, i, j, k, m, kappa, z₁, z₂, depth)
    return q_cell_integral(qtransform.kernel, kappa, z₁, z₂, depth)
end

function q_cell_weight(qtransform::QTransform{Q, VG, <:PrecomputeQWeights},
                       i, j, k, m, kappa, z₁, z₂, depth) where {Q, VG}
    return precomputed_q_cell_weight(qtransform.cache_policy, i, j, k, m, kappa, depth)
end

function q_cell_weight_kappa_derivative(qtransform::QTransform, i, j, k, m, kappa, z₁, z₂, depth)
    return q_cell_integral_kappa_derivative(qtransform.kernel, kappa, z₁, z₂, depth)
end

function q_cell_weight_kappa_derivative(qtransform::QTransform{Q, VG, <:PrecomputeQWeights},
                                        i, j, k, m, kappa, z₁, z₂, depth) where {Q, VG}
    precomputed_q_cell_weight(qtransform.cache_policy, i, j, k, m, kappa, depth)
    return q_cell_integral_kappa_derivative(qtransform.kernel, kappa, z₁, z₂, depth)
end

@inline q_cell_weight_kernel(::AbstractQStoragePolicy, kernel::QKernel,
                             i, j, k, m, kappa, z₁, z₂, depth) =
    q_cell_integral_kernel(kernel, kappa, z₁, z₂, depth)

@inline q_cell_weight_kernel(policy::PrecomputeQWeights{K, <:Number}, kernel::QKernel,
                             i, j, k, m, kappa, z₁, z₂, depth) where K =
    policy.weights[k, m]

@inline q_cell_weight_kernel(policy::PrecomputeQWeights, kernel::QKernel,
                             i, j, k, m, kappa, z₁, z₂, depth) =
    policy.weights[i, j, k, m]

@inline q_cell_weight_kappa_derivative_kernel(::AbstractQStoragePolicy, kernel::QKernel,
                                              i, j, k, m, kappa, z₁, z₂, depth) =
    q_cell_integral_kappa_derivative_kernel(kernel, kappa, z₁, z₂, depth)

@inline q_depth_at(depth::Number, i, j) = depth
@inline q_depth_at(depth, i, j) = depth[i, j]

q_depth_on_architecture(arch, depth::Number) = depth
q_depth_on_architecture(arch, depth) = on_architecture(arch, depth)
