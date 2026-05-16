import Oceananigans.Grids: column_depthᶜᶜᵃ
import Oceananigans.Architectures: architecture, device
import Oceananigans.Fields: ConstantField, Field, field
import KernelAbstractions
import KernelAbstractions: @kernel, @index

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

`uᴸ = 0`. Waves propagate at their intrinsic group velocity, determined by
the model `depth`, with no Doppler shift and no kinematic refraction.
"""
struct ZeroVelocities <: AbstractLagrangianVelocities end

"""
    PrescribedVelocities(; u, v, q_grid=nothing)

`uᴸ` supplied as user-provided `u` and `v` fields (arrays or functions
accepted by `PrescribedLagrangianMeanCurrent`). When `u` and `v` are
Oceananigans `Field`s, their grid supplies the vertical Q-transform grid.
Array inputs may supply `q_grid` explicitly when the wave model grid is Flat.
"""
struct PrescribedVelocities{U, V, QG, NZ, SS, VS} <: AbstractLagrangianVelocities
    u :: U
    v :: V
    q_grid :: QG
    Nz :: NZ
    surface_spacing :: SS
    vertical_stretching :: VS
end

PrescribedVelocities(u, v) = PrescribedVelocities(u, v, nothing, nothing, nothing, 2)

function PrescribedVelocities(; u, v, q_grid=nothing, vertical_grid=nothing,
                              depth=nothing, Nz=nothing,
                              surface_spacing=nothing, vertical_stretching=2)
    depth === nothing ||
        throw(ArgumentError("`depth` is a SpectralWaveModel kwarg; pass `SpectralWaveModel(...; depth=...)` instead of putting `depth` in `velocities`"))
    q_grid !== nothing && vertical_grid !== nothing &&
        throw(ArgumentError("PrescribedVelocities cannot contain both `q_grid` and `vertical_grid`"))
    return PrescribedVelocities(u, v, q_grid === nothing ? vertical_grid : q_grid,
                                Nz, surface_spacing, vertical_stretching)
end

"""
    PseudomomentumVelocities(; q_grid=nothing)

`uᴸ = p`, where `p(x, y, z)` is the wave pseudomomentum derived from the
current action `N`. This is a self-coupled mode — there is no ocean model;
the waves drive their own Doppler shift. The coupling refreshes once per RK3
stage. If `q_grid` is omitted for a Flat wave grid, `SpectralWaveModel` must
be constructed with finite `depth` so Ripple can build a stretched vertical Q
grid from the depth and spectral wavenumber range.
"""
struct PseudomomentumVelocities{QG, NZ, SS, VS} <: AbstractLagrangianVelocities
    q_grid :: QG
    Nz :: NZ
    surface_spacing :: SS
    vertical_stretching :: VS
end

function PseudomomentumVelocities(; q_grid=nothing, vertical_grid=nothing,
                                  depth=nothing, Nz=nothing,
                                  surface_spacing=nothing, vertical_stretching=2)
    depth === nothing ||
        throw(ArgumentError("`depth` is a SpectralWaveModel kwarg; pass `SpectralWaveModel(...; depth=...)` instead of putting `depth` in `velocities`"))
    q_grid !== nothing && vertical_grid !== nothing &&
        throw(ArgumentError("PseudomomentumVelocities cannot contain both `q_grid` and `vertical_grid`"))
    return PseudomomentumVelocities(q_grid === nothing ? vertical_grid : q_grid,
                                    Nz, surface_spacing, vertical_stretching)
end

"""
    build_coupling(velocities, grid, spectral_grid, model_depth; FT)

Materialize the per-architecture coupling state that backs the chosen
`velocities` paradigm. Called once during `SpectralWaveModel` construction.
"""
# Water depth driving the Q-transform vertical projection. Uses Oceananigans'
# staggered `column_depthᶜᶜᵃ` accessor so flat-bottom grids resolve to
# `grid.Lz` and ImmersedBoundaryGrid resolves to the local column depth.
function grid_depth(grid)
    Nx, Ny = grid.Nx, grid.Ny
    d11 = column_depthᶜᶜᵃ(1, 1, grid)
    arch = architecture(grid)
    depth = device_zeros(arch, typeof(d11), (Nx, Ny))
    kernel = _grid_depth_kernel!(device(arch), (16, 16), (Nx, Ny))
    kernel(depth, grid)
    KernelAbstractions.synchronize(device(arch))
    depth_host = Array(depth)
    return all(==(d11), depth_host) ? d11 : depth_host  # scalar when uniform
end

@kernel function _grid_depth_kernel!(depth, grid)
    i, j = @index(Global, NTuple)
    @inbounds depth[i, j] = column_depthᶜᶜᵃ(i, j, grid)
end

velocity_grid(a::Field) = a.grid
velocity_grid(a) = nothing

const horizontal_depth_location = (Center, Center, Nothing)

function validate_horizontal_depth(depth::Number, model_grid; name="depth")
    depth > 0 || throw(ArgumentError("$name must be positive"))
    return depth
end

function validate_horizontal_depth(depth::AbstractArray, model_grid; name="depth")
    data = collect(float.(depth))
    size(data) == horizontal_size(model_grid) ||
        throw(ArgumentError("$name must be a scalar or have size $(horizontal_size(model_grid)) on the horizontal wave grid"))
    all(>(0), data) ||
        throw(ArgumentError("$name values must be positive"))
    return data
end

function horizontal_depth_field_data(depth::Field, model_grid; name="depth")
    compatible_horizontal_grid(depth.grid, model_grid) ||
        throw(ArgumentError("$name Field must live on the horizontal wave grid"))

    architecture(depth.grid) == architecture(model_grid) ||
        throw(ArgumentError("$name Field must live on the same architecture as the wave model grid"))

    data = field_storage(depth)
    Nx, Ny = horizontal_size(model_grid)
    depth_data = if size(data) == (Nx, Ny, 1)
        @view data[:, :, 1]
    elseif size(data) == (Nx, Ny)
        data
    else
        throw(ArgumentError("$name Field must have horizontal size $(horizontal_size(model_grid))"))
    end

    validate_horizontal_depth(depth_data, model_grid; name)
    return depth_data
end

finite_depth_values(depth::Number, model_grid; name="depth") =
    validate_horizontal_depth(depth, model_grid; name)

finite_depth_values(depth::AbstractArray, model_grid; name="depth") =
    validate_horizontal_depth(depth, model_grid; name)

finite_depth_values(depth::ConstantField, model_grid; name="depth") =
    validate_horizontal_depth(depth.constant, model_grid; name)

finite_depth_values(depth::Field, model_grid; name="depth") =
    horizontal_depth_field_data(depth, model_grid; name)

function validate_materialized_model_depth(depth::ConstantField, model_grid)
    finite_depth_values(depth, model_grid; name="model `depth`")
    return depth
end

function validate_materialized_model_depth(depth::Field, model_grid)
    finite_depth_values(depth, model_grid; name="model `depth`")
    return depth
end

function validate_materialized_model_depth(depth, model_grid)
    depth = Field(depth)
    finite_depth_values(depth, model_grid; name="model `depth`")
    return depth
end

validate_model_depth(depth::InfiniteDepth, model_grid) = depth
function validate_model_depth(depth::Union{Number, Function, ConstantField}, model_grid)
    depth = field(horizontal_depth_location, depth, model_grid)
    return validate_materialized_model_depth(depth, model_grid)
end
validate_model_depth(depth::Field, model_grid) =
    validate_materialized_model_depth(depth, model_grid)
validate_model_depth(depth::AbstractArray, model_grid) =
    throw(ArgumentError("model `depth` must be InfiniteDepth(), a positive number, a function, or an Oceananigans Field on the horizontal wave grid; raw arrays are not accepted"))
validate_model_depth(depth, model_grid) =
    throw(ArgumentError("model `depth` must be InfiniteDepth(), a positive number, a function, or an Oceananigans Field on the horizontal wave grid; raw arrays are not accepted"))

max_depth(depth::Number) = depth
max_depth(depth::AbstractArray) = maximum(Array(depth))

q_projection_depth(depth::InfiniteDepth, q_grid) = grid_depth(q_grid)
q_projection_depth(depth, q_grid) =
    finite_depth_values(depth, q_grid; name="model `depth`")

function automatic_q_grid(model_grid, spectral_grid, depth;
                          Nz=nothing, surface_spacing=nothing,
                          vertical_stretching=2, FT=grid_float_type(model_grid))
    spectral_grid isa PolarWaveVectorGrid ||
        throw(ArgumentError("automatic Q-grid selection requires a PolarWaveVectorGrid"))

    is_infinite_depth(depth) &&
        throw(ArgumentError("automatic Q-grid selection requires finite model `depth`"))
    depth_values = finite_depth_values(depth, model_grid; name="model `depth`")
    H = FT(max_depth(depth_values))
    κ = Array(spectral_grid.κ)
    κmax = maximum(κ)
    default_surface_spacing = inv(20 * FT(κmax))
    Δz_surface = surface_spacing === nothing ? default_surface_spacing : FT(surface_spacing)
    Δz_surface > 0 || throw(ArgumentError("automatic Q-grid `surface_spacing` must be positive"))

    stretch = FT(vertical_stretching)
    stretch > 1 || throw(ArgumentError("automatic Q-grid `vertical_stretching` must be greater than 1"))
    Nz = Nz === nothing ? max(16, ceil(Int, (H / Δz_surface)^(inv(stretch)))) : Int(Nz)
    Nz > 0 || throw(ArgumentError("automatic Q-grid `Nz` must be positive"))

    z = [ -H * (one(FT) - FT(k) / FT(Nz))^stretch for k in 0:Nz ]
    TX, TY, _ = OceanGrids.topology(model_grid)
    return RectilinearGrid(architecture(model_grid), FT;
                           size=(model_grid.Nx, model_grid.Ny, Nz),
                           x=xfaces(model_grid),
                           y=yfaces(model_grid),
                           z,
                           topology=(TX, TY, Bounded))
end

function compatible_horizontal_grid(a, b)
    return horizontal_size(a) == horizontal_size(b) &&
           xnodes(a) == xnodes(b) &&
           ynodes(a) == ynodes(b) &&
           xfaces(a) == xfaces(b) &&
           yfaces(a) == yfaces(b) &&
           OceanGrids.topology(a)[1:2] == OceanGrids.topology(b)[1:2]
end

function validate_q_grid(q_grid, model_grid)
    has_flat_vertical_topology(q_grid) &&
        throw(ArgumentError("velocity `q_grid` must have a resolved vertical coordinate; the wave model grid may be Flat, but the Q-transform grid cannot be Flat"))

    compatible_horizontal_grid(q_grid, model_grid) ||
        throw(ArgumentError("velocity `q_grid` must match the wave model grid horizontally"))

    architecture(q_grid) == architecture(model_grid) ||
        throw(ArgumentError("velocity `q_grid` must live on the same architecture as the wave model grid"))

    return q_grid
end

function validate_velocity_field_grid(field_grid, q_grid, name)
    field_grid === nothing && return nothing
    compatible_model_physical_grid(field_grid, q_grid) ||
        throw(ArgumentError("velocity field `$name` must live on the same grid as the inferred or supplied velocity `q_grid`"))
    return nothing
end

function prescribed_q_grid(v::PrescribedVelocities, model_grid, spectral_grid, model_depth;
                           FT=grid_float_type(model_grid))
    u_grid = velocity_grid(v.u)
    v_grid = velocity_grid(v.v)
    q_grid = if v.q_grid !== nothing
        v.q_grid
    elseif u_grid !== nothing
        u_grid
    elseif v_grid !== nothing
        v_grid
    elseif has_flat_vertical_topology(model_grid)
        is_infinite_depth(model_depth) &&
            throw(ArgumentError("Flat wave grids require finite model `depth` or `q_grid` for array-valued prescribed velocities"))
        Nz = v.Nz === nothing ? size(current_data(v.u), 3) : v.Nz
        automatic_q_grid(model_grid, spectral_grid, model_depth;
                         Nz, surface_spacing=v.surface_spacing,
                         vertical_stretching=v.vertical_stretching, FT)
    else
        model_grid
    end

    validate_q_grid(q_grid, model_grid)
    validate_velocity_field_grid(u_grid, q_grid, "u")
    validate_velocity_field_grid(v_grid, q_grid, "v")

    return q_grid
end

build_coupling(::ZeroVelocities, grid, spectral_grid, model_depth; FT=Float64) = nothing

function build_coupling(v::PrescribedVelocities, grid, spectral_grid, model_depth; FT=Float64)
    q_grid = prescribed_q_grid(v, grid, spectral_grid, model_depth; FT)
    q_depth = q_projection_depth(model_depth, q_grid)
    current = PrescribedLagrangianMeanCurrent(u=v.u, v=v.v, depth=q_depth)
    qtransform = QTransform(QKernel(FT), q_grid)
    return CWCMPrescribedCurrentCoupling(current, qtransform, spectral_grid.κ)
end

function build_coupling(v::PseudomomentumVelocities, grid, spectral_grid, model_depth; FT=Float64)
    q_grid = if v.q_grid !== nothing
        v.q_grid
    elseif has_flat_vertical_topology(grid)
        is_infinite_depth(model_depth) &&
            throw(ArgumentError("Flat wave grids require finite model `depth` or `q_grid` for PseudomomentumVelocities"))
        automatic_q_grid(grid, spectral_grid, model_depth;
                         Nz=v.Nz, surface_spacing=v.surface_spacing,
                         vertical_stretching=v.vertical_stretching, FT)
    else
        grid
    end

    validate_q_grid(q_grid, grid)
    qtransform = QTransform(QKernel(FT), q_grid)
    q_depth = q_projection_depth(model_depth, q_grid)
    return CWCMPseudomomentumCoupling(grid, qtransform, spectral_grid, q_depth)
end

# Convenience: a bare NamedTuple `(; u, v)` is interpreted as prescribed
# velocities. Reserves the longer-form `PrescribedVelocities` for callers that
# want to be explicit about the velocity paradigm.
function build_coupling(nt::NamedTuple, grid, spectral_grid, model_depth; FT=Float64)
    haskey(nt, :u) && haskey(nt, :v) ||
        throw(ArgumentError("velocities NamedTuple must contain `u` and `v` (got keys $(keys(nt)))"))
    haskey(nt, :depth) &&
        throw(ArgumentError("`depth` is a SpectralWaveModel kwarg; pass `SpectralWaveModel(...; depth=...)` instead of putting `depth` in `velocities`"))
    haskey(nt, :q_grid) && haskey(nt, :vertical_grid) &&
        throw(ArgumentError("velocities NamedTuple cannot contain both `q_grid` and `vertical_grid`"))
    q_grid = haskey(nt, :q_grid) ? nt.q_grid : (haskey(nt, :vertical_grid) ? nt.vertical_grid : nothing)
    Nz = haskey(nt, :Nz) ? nt.Nz : nothing
    surface_spacing = haskey(nt, :surface_spacing) ? nt.surface_spacing : nothing
    vertical_stretching = haskey(nt, :vertical_stretching) ? nt.vertical_stretching : 2
    return build_coupling(PrescribedVelocities(nt.u, nt.v, q_grid,
                                               Nz, surface_spacing, vertical_stretching),
                          grid, spectral_grid, model_depth; FT)
end
