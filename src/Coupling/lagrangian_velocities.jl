import Oceananigans.Grids: column_depthᶜᶜᵃ

"""
    AbstractLagrangianVelocities

Tag selecting how the Lagrangian-mean velocity `uᴸ` that drives Doppler
shifting and kinematic refraction is supplied. Three concrete choices ship
with Ripple; see `ZeroVelocities`, `PrescribedVelocities`, and
`PseudomomentumVelocities`.
"""
abstract type AbstractLagrangianVelocities end

"""
    ZeroVelocities()

`uᴸ = 0`. Waves propagate at their intrinsic deep-water group velocity with
no Doppler shift and no kinematic refraction.
"""
struct ZeroVelocities <: AbstractLagrangianVelocities end

"""
    PrescribedVelocities(; u, v)

`uᴸ` supplied as user-provided `u` and `v` fields (arrays or functions
accepted by `PrescribedLagrangianMeanCurrent`). Local water depth is read
from the physical grid.
"""
struct PrescribedVelocities{U, V} <: AbstractLagrangianVelocities
    u :: U
    v :: V
end

PrescribedVelocities(; u, v) = PrescribedVelocities(u, v)

"""
    PseudomomentumVelocities()

`uᴸ = p`, where `p(x, y, z)` is the wave pseudomomentum derived from the
current action `N`. This is a self-coupled mode — there is no ocean model;
the waves drive their own Doppler shift. The coupling refreshes once per
RK3 stage. Local water depth is read from the physical grid.
"""
struct PseudomomentumVelocities <: AbstractLagrangianVelocities end

"""
    build_coupling(velocities, grid, spectral_grid; FT)

Materialize the per-architecture coupling state that backs the chosen
`velocities` paradigm. Called once during `SpectralWaveModel` construction.
"""
# Water depth driving the Q-transform vertical projection. Uses Oceananigans'
# staggered `column_depthᶜᶜᵃ` accessor so flat-bottom grids resolve to
# `grid.Lz` and ImmersedBoundaryGrid resolves to the local column depth.
function grid_depth(grid)
    Nx, Ny = grid.Nx, grid.Ny
    d11 = column_depthᶜᶜᵃ(1, 1, grid)
    depth = fill(d11, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        depth[i, j] = column_depthᶜᶜᵃ(i, j, grid)
    end
    return all(==(d11), depth) ? d11 : depth  # scalar when uniform
end

build_coupling(::ZeroVelocities, grid, spectral_grid; FT=Float64) = nothing

function build_coupling(v::PrescribedVelocities, grid, spectral_grid; FT=Float64)
    current = PrescribedLagrangianMeanCurrent(u=v.u, v=v.v, depth=grid_depth(grid))
    qtransform = QTransform(QKernel(FT), grid)
    return CWCMPrescribedCurrentCoupling(current, qtransform, spectral_grid.κ)
end

function build_coupling(::PseudomomentumVelocities, grid, spectral_grid; FT=Float64)
    throw(ArgumentError("PseudomomentumVelocities coupling is not yet wired into SpectralWaveModel; pending implementation"))
end

# Convenience: a bare NamedTuple `(; u, v)` is interpreted as prescribed
# velocities. Reserves the longer-form `PrescribedVelocities` for callers that
# want to be explicit about the velocity paradigm.
function build_coupling(nt::NamedTuple, grid, spectral_grid; FT=Float64)
    haskey(nt, :u) && haskey(nt, :v) ||
        throw(ArgumentError("velocities NamedTuple must contain `u` and `v` (got keys $(keys(nt)))"))
    return build_coupling(PrescribedVelocities(u=nt.u, v=nt.v), grid, spectral_grid; FT)
end
