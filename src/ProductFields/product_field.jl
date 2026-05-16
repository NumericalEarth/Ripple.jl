import Oceananigans
import Oceananigans.Fields: Field, interior, set!
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Architectures: architecture, device, on_architecture
import KernelAbstractions
import KernelAbstractions: @kernel, @index
import OffsetArrays: OffsetArray

struct ProductGrid{PG, CG}
    physical :: PG
    coordinate :: CG
end

# Bundle of metadata needed to construct any per-bin Oceananigans Field over
# the same physical grid + location. Shared by every spectral cell in a
# ProductField, so `physical_field(f, m, n)` builds a Field on demand that
# just wraps a view into `flat_data` plus this cached metadata.
struct ProductFieldStencil{L, I, O, BC}
    loc :: L
    indices :: I
    offsets :: O
    bcs :: BC
end

# ProductField is backed by a single contiguous 5D array `flat_data` that
# stores `(x_with_halo, y_with_halo, z_slab, κ, φ)`. There is no Matrix of
# Fields; per-bin Fields are constructed on demand by `physical_field` using
# views into `flat_data` and the cached metadata in `stencil`.
struct ProductField{LX, LY, LZ, LXi, LEta, PG, CG, S, D, T} <: AbstractArray{T, 4}
    grid :: PG
    coordinate_grid :: CG
    stencil :: S
    flat_data :: D
end

coordinate_float_type(coordinate_grid) = Float64
default_product_field_eltype(grid, coordinate_grid) =
    promote_type(grid_float_type(grid), coordinate_float_type(coordinate_grid))

canonical_location_marker(::Type{Center}) = Center
canonical_location_marker(::Type{Face}) = Face
canonical_location_marker(::Type{Nothing}) = Nothing
canonical_location_marker(::Center) = Center
canonical_location_marker(::Face) = Face
canonical_location_marker(::Nothing) = Nothing

instantiate_location_marker(::Type{Center}) = Center()
instantiate_location_marker(::Type{Face}) = Face()
instantiate_location_marker(::Type{Nothing}) = nothing

ocean_field_location_marker(::Type{Nothing}) = Center
ocean_field_location_marker(L) = L

surface_indices(grid, ::Type{Nothing}) = (:, :, grid.Nz)
surface_indices(grid, LZ) = (:, :, :)

function ocean_field_location(::Type{LX}, ::Type{LY}, ::Type{LZ}) where {LX, LY, LZ}
    OZ = ocean_field_location_marker(LZ)
    return (instantiate_location_marker(LX),
            instantiate_location_marker(LY),
            instantiate_location_marker(OZ))
end

# Build the shared metadata stencil and the contiguous 5D backing.
function product_field_storage(::Type{LX}, ::Type{LY}, ::Type{LZ},
                               grid, coordinate_grid, ::Type{FT}) where {LX, LY, LZ, FT}
    Nxi, Neta = coordinate_size(coordinate_grid)
    loc = ocean_field_location(LX, LY, LZ)
    indices = surface_indices(grid, LZ)

    probe = Field(loc, grid, FT; indices)
    probe_data = probe.data
    parent_size = size(parent(probe_data))
    ax = axes(probe_data)
    offsets = ntuple(d -> first(ax[d]) - 1, ndims(probe_data))
    bcs = probe.boundary_conditions

    flat_data = on_architecture(architecture(grid), zeros(FT, parent_size..., Nxi, Neta))

    stencil = ProductFieldStencil(loc, indices, offsets, bcs)
    return stencil, flat_data
end

function ProductField{LX, LY, LZ, LXi, LEta}(grid, coordinate_grid;
                                             boundary_conditions=nothing,
                                             eltype=default_product_field_eltype(grid, coordinate_grid),
                                             kwargs...) where {LX, LY, LZ, LXi, LEta}
    isempty(kwargs) ||
        throw(ArgumentError("unsupported ProductField keywords $(keys(kwargs)); ProductField storage is owned by Oceananigans fields"))

    boundary_conditions === nothing ||
        throw(ArgumentError("ProductField physical boundary conditions are defined by Oceananigans fields; pass boundary conditions through the Oceananigans grid/field interface"))

    lx = canonical_location_marker(LX)
    ly = canonical_location_marker(LY)
    lz = canonical_location_marker(LZ)
    lxi = canonical_location_marker(LXi)
    leta = canonical_location_marker(LEta)
    FT = eltype
    stencil, flat_data = product_field_storage(lx, ly, lz, grid, coordinate_grid, FT)

    return ProductField{lx, ly, lz, lxi, leta,
                        typeof(grid), typeof(coordinate_grid),
                        typeof(stencil), typeof(flat_data), FT}(
        grid, coordinate_grid, stencil, flat_data)
end

function ProductField(grid, coordinate_grid;
                      location=(Center, Center, Nothing),
                      coordinate_location=(Center, Center),
                      kwargs...)
    length(location) == 3 ||
        throw(ArgumentError("ProductField location must have length 3"))
    length(coordinate_location) == 2 ||
        throw(ArgumentError("ProductField coordinate_location must have length 2"))

    LX, LY, LZ = map(canonical_location_marker, location)
    LXi, LEta = map(canonical_location_marker, coordinate_location)

    return ProductField{LX, LY, LZ, LXi, LEta}(grid, coordinate_grid; kwargs...)
end

WaveActionField(grid, coordinate_grid; kwargs...) =
    ProductField(grid, coordinate_grid; location=(Center, Center, Nothing),
                 coordinate_location=(Center, Center), kwargs...)

# `parent` returns the contiguous 5D backing; `flat_data` is the public alias.
Base.parent(f::ProductField) = f.flat_data
flat_data(f::ProductField) = f.flat_data
Base.eltype(::ProductField{LX, LY, LZ, LXi, LEta, PG, CG, S, D, T}) where {LX, LY, LZ, LXi, LEta, PG, CG, S, D, T} = T
architecture(f::ProductField) = architecture(f.grid)
grid(f::ProductField) = f.grid
physical_grid(f::ProductField) = f.grid
coordinate_grid(f::ProductField) = f.coordinate_grid
product_grid(f::ProductField) = ProductGrid(f.grid, f.coordinate_grid)
location(::ProductField{LX, LY, LZ}) where {LX, LY, LZ} = (LX, LY, LZ)
coordinate_location(::ProductField{LX, LY, LZ, LXi, LEta}) where {LX, LY, LZ, LXi, LEta} = (LXi, LEta)
product_location(f::ProductField) = (location(f)..., coordinate_location(f)...)
active_product_location(f::ProductField) = (location(f)[1], location(f)[2], coordinate_location(f)...)
boundary_conditions(f::ProductField) = f.stencil.bcs

# Build a per-bin Oceananigans Field on demand. The data is a view into
# `flat_data`, the bcs are shared from the cached stencil, so this allocates
# only a thin Field wrapper.
function physical_field(f::ProductField, m, n)
    s = f.stencil
    ncolon = length(s.offsets)
    slab = view(f.flat_data, ntuple(_ -> Colon(), ncolon)..., m, n)
    backed = OffsetArray(slab, s.offsets...)
    return Field(s.loc, f.grid, eltype(f);
                 indices=s.indices, data=backed, boundary_conditions=s.bcs)
end

function Base.size(f::ProductField)
    Nx, Ny = horizontal_size(f.grid)
    Nxi, Neta = coordinate_size(f.coordinate_grid)
    return (Nx, Ny, Nxi, Neta)
end

Base.size(f::ProductField, dim::Integer) = dim <= 4 ? size(f)[dim] : 1
Base.ndims(::ProductField) = 4
Base.axes(f::ProductField) = map(Base.OneTo, size(f))
Base.axes(f::ProductField, dim::Integer) = dim <= 4 ? axes(f)[dim] : Base.OneTo(1)
Base.IndexStyle(::Type{<:ProductField}) = IndexCartesian()

function product_field_data_indices(f::ProductField)
    s = f.stencil
    Hx, Hy = -s.offsets[1], -s.offsets[2]
    iz = first(axes(f.flat_data, 3))
    return Hx, Hy, iz
end

@inline function getnode(f::ProductField, i, j, m, n)
    s = f.stencil
    Hx, Hy = s.offsets[1], s.offsets[2]
    iz = first(axes(f.flat_data, 3))
    @inbounds return f.flat_data[i - Hx, j - Hy, iz, m, n]
end

@inline function setnode!(f::ProductField, value, i, j, m, n)
    s = f.stencil
    Hx, Hy = s.offsets[1], s.offsets[2]
    iz = first(axes(f.flat_data, 3))
    @inbounds f.flat_data[i - Hx, j - Hy, iz, m, n] = value
    return value
end

Base.getindex(f::ProductField, i::Int, j::Int, m::Int, n::Int) = getnode(f, i, j, m, n)
Base.setindex!(f::ProductField, value, i::Int, j::Int, m::Int, n::Int) = setnode!(f, value, i, j, m, n)
Base.getindex(f::ProductField, I::CartesianIndex{4}) = getindex(f, Tuple(I)...)
Base.setindex!(f::ProductField, value, I::CartesianIndex{4}) = setindex!(f, value, Tuple(I)...)

function Base.similar(f::ProductField; eltype=Base.eltype(f))
    return ProductField(f.grid, f.coordinate_grid;
                        location=location(f),
                        coordinate_location=coordinate_location(f),
                        eltype=eltype)
end

Base.similar(f::ProductField, ::Type{T}) where T = similar(f; eltype=T)

function Base.similar(f::ProductField, ::Type{T}, dims::Dims{4}) where T
    dims == size(f) ||
        throw(ArgumentError("ProductField similar construction requires logical dims $(size(f)); got $dims"))
    return similar(f; eltype=T)
end

function copy_product_field!(dest::ProductField, src::ProductField)
    size(dest) == size(src) ||
        throw(DimensionMismatch("cannot copy ProductField of size $(size(src)) to size $(size(dest))"))
    copyto!(dest.flat_data, src.flat_data)
    return dest
end

function Base.copy(f::ProductField)
    g = similar(f)
    copy_product_field!(g, f)
    return g
end

function interior(f::ProductField)
    Nx, Ny, Nxi, Neta = size(f)
    values = device_zeros(architecture(f), eltype(f), (Nx, Ny, Nxi, Neta))
    Hx, Hy, iz = product_field_data_indices(f)
    arch = architecture(f)
    kernel = _copy_product_field_interior!(device(arch), (8, 8, 1, 1), (Nx, Ny, Nxi, Neta))
    kernel(values, f.flat_data, Hx, Hy, iz)
    KernelAbstractions.synchronize(device(arch))
    return Array(values)
end

@kernel function _copy_product_field_interior!(values, data, Hx, Hy, iz)
    i, j, m, n = @index(Global, NTuple)
    @inbounds values[i, j, m, n] = data[i + Hx, j + Hy, iz, m, n]
end
