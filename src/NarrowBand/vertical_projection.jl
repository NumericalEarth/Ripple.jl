import Oceananigans.Architectures: architecture, on_architecture
import Oceananigans.Grids: znodes, Face
import Oceananigans.Utils: launch!
import KernelAbstractions: @kernel, @index

#####
##### Depth-weighted vertical projections Ū, Ũ of the Lagrangian-mean current
#####
##### The narrow-band wave equation is driven by two vertically integrated
##### projections of the horizontal Lagrangian-mean velocity Uᴸ:
#####
#####     Ūᵢ = ∫_{-h}^{0} Φ²   Uᴸᵢ dz ,      Ũᵢ = (1/κ²) ∫_{-h}^{0} Φ_z² Uᴸᵢ dz .
#####
##### With Uᴸ stored as finite-volume cell averages, each integral is an exact
##### sum of per-cell weights against the column values. The weights are the
##### analytic cell integrals of Φ² and Φ_z²/κ² — mirroring the exact-FV pattern
##### of the Q-transform (src/Coupling/q_kernel.jl). Both weight sets share the
##### deep-water limit W = e^{2κz₂} − e^{2κz₁} (there Φ_z² = κ²Φ², so Ũ = Ū).

# Cell integral of Φ² over [z₁, z₂] (finite depth). With u = z + h,
#   ∫ Φ² dz = P₀ [u/2 + sinh(2κu)/(4κ)],   P₀ = 1 / (h/2 + sinh(2κh)/(4κ)),
# and P₀ = C²/cosh²(κh) is exactly the normalization that makes ∫_{-h}^0 Φ² = 1.
@inline function phi_squared_cell_integral(κ, h::Number, z₁, z₂)
    P₀ = inv(h / 2 + sinh(2κ * h) / 4κ)
    F(z) = (u = z + h; P₀ * (u / 2 + sinh(2κ * u) / 4κ))
    return F(z₂) - F(z₁)
end

# Cell integral of Φ_z²/κ² over [z₁, z₂] (finite depth):
#   (1/κ²) ∫ Φ_z² dz = P₀ [sinh(2κu)/(4κ) − u/2].
@inline function phi_z_squared_cell_integral(κ, h::Number, z₁, z₂)
    P₀ = inv(h / 2 + sinh(2κ * h) / 4κ)
    F(z) = (u = z + h; P₀ * (sinh(2κ * u) / 4κ - u / 2))
    return F(z₂) - F(z₁)
end

# Deep water: Φ² = 2κ e^{2κz}, Φ_z² = κ²Φ², so both weights collapse to the
# exponential cell difference and Ũ = Ū.
@inline phi_squared_cell_integral(κ, ::InfiniteDepth, z₁, z₂) = exp(2κ * z₂) - exp(2κ * z₁)
@inline phi_z_squared_cell_integral(κ, depth::InfiniteDepth, z₁, z₂) =
    phi_squared_cell_integral(κ, depth, z₁, z₂)

"""
    VerticalProjectionWeights(dispersion, grid)

Precompute the per-cell weights of the depth-weighted projections `Ū` and `Ũ`
for the vertical faces of `grid`. `Ū` uses the analytic cell integral of `Φ²`;
`Ũ` uses that of `Φ_z²/κ²`. The weights depend only on `κ`, the depth, and the
`z`-faces, so they are computed once and reused every tendency evaluation.
"""
struct VerticalProjectionWeights{W}
    Ū_weights :: W    # ∫_cell Φ² dz per vertical cell
    Ũ_weights :: W    # ∫_cell Φ_z²/κ² dz per vertical cell
end

function VerticalProjectionWeights(dispersion::NarrowBandDispersion, grid)
    κ = dispersion.κ
    depth = dispersion.depth
    zf = znodes(grid, Face())              # Nz+1 vertical faces (interior)
    Nz = length(zf) - 1
    FT = eltype(grid)
    Ū = zeros(FT, Nz)
    Ũ = zeros(FT, Nz)
    for k in 1:Nz
        Ū[k] = phi_squared_cell_integral(κ, depth, zf[k], zf[k+1])
        Ũ[k] = phi_z_squared_cell_integral(κ, depth, zf[k], zf[k+1])
    end
    arch = architecture(grid)
    return VerticalProjectionWeights(on_architecture(arch, Ū), on_architecture(arch, Ũ))
end

# Depth-weighted vertical sum of a 3-D velocity column against precomputed
# weights, writing the 2-D projection at the column's horizontal location. Used
# for each velocity component; the projected field inherits the component's
# horizontal (Face/Center) location, so u → (Face, Center) and v → (Center, Face)
# with no horizontal interpolation.
@kernel function _project_column!(projected, velocity, weights, Nz)
    i, j = @index(Global, NTuple)
    acc = zero(eltype(projected))
    @inbounds for k in 1:Nz
        acc += weights[k] * velocity[i, j, k]
    end
    @inbounds projected[i, j, 1] = acc
end

# Project one 3-D velocity component onto the 2-D `projected` field on `wave_grid`.
function project_velocity!(projected, velocity, weights, wave_grid, Nz)
    launch!(architecture(wave_grid), wave_grid, :xy, _project_column!,
            projected, velocity, weights, Nz)
    return projected
end
