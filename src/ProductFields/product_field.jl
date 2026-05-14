import Oceananigans
import Oceananigans.Fields: Field, interior, set!
import Oceananigans.BoundaryConditions: fill_halo_regions!

struct ProductGrid{PG, CG}
    physical :: PG
    coordinate :: CG
end

struct ProductField{LX, LY, LZ, LXi, LEta, PG, CG, F, T} <: AbstractArray{T, 4}
    grid :: PG
    coordinate_grid :: CG
    fields :: Matrix{F}
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

function product_field_storage(::Type{LX}, ::Type{LY}, ::Type{LZ},
                               grid, coordinate_grid, ::Type{FT}) where {LX, LY, LZ, FT}
    Nxi, Neta = coordinate_size(coordinate_grid)
    loc = ocean_field_location(LX, LY, LZ)
    indices = surface_indices(grid, LZ)
    first_field = Field(loc, grid, FT; indices)
    fields = Matrix{typeof(first_field)}(undef, Nxi, Neta)
    fields[1, 1] = first_field

    for n in 1:Neta, m in 1:Nxi
        m == 1 && n == 1 && continue
        fields[m, n] = Field(loc, grid, FT; indices)
    end

    return fields
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
    fields = product_field_storage(lx, ly, lz, grid, coordinate_grid, FT)
    field_type = Base.eltype(fields)

    return ProductField{lx, ly, lz, lxi, leta,
                        typeof(grid), typeof(coordinate_grid), field_type, FT}(
        grid, coordinate_grid, fields)
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

Base.parent(f::ProductField) = f.fields
Base.eltype(::ProductField{LX, LY, LZ, LXi, LEta, PG, CG, F, T}) where {LX, LY, LZ, LXi, LEta, PG, CG, F, T} = T
architecture(f::ProductField) = architecture(f.grid)
grid(f::ProductField) = f.grid
physical_grid(f::ProductField) = f.grid
coordinate_grid(f::ProductField) = f.coordinate_grid
product_grid(f::ProductField) = ProductGrid(f.grid, f.coordinate_grid)
location(::ProductField{LX, LY, LZ}) where {LX, LY, LZ} = (LX, LY, LZ)
coordinate_location(::ProductField{LX, LY, LZ, LXi, LEta}) where {LX, LY, LZ, LXi, LEta} = (LXi, LEta)
product_location(f::ProductField) = (location(f)..., coordinate_location(f)...)
active_product_location(f::ProductField) = (location(f)[1], location(f)[2], coordinate_location(f)...)
boundary_conditions(f::ProductField) = map(field -> field.boundary_conditions, f.fields)

physical_field(f::ProductField, m, n) = f.fields[m, n]

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

@inline function getnode(f::ProductField, i, j, m, n)
    data = interior(physical_field(f, m, n))
    @inbounds return data[i, j, 1]
end

@inline function setnode!(f::ProductField, value, i, j, m, n)
    data = interior(physical_field(f, m, n))
    @inbounds data[i, j, 1] = value
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

    _, _, Nxi, Neta = size(dest)
    for n in 1:Neta, m in 1:Nxi
        set!(physical_field(dest, m, n), physical_field(src, m, n))
    end

    return dest
end

function Base.copy(f::ProductField)
    g = similar(f)
    copy_product_field!(g, f)
    return g
end

function interior(f::ProductField)
    Nx, Ny, Nxi, Neta = size(f)
    values = Array{eltype(f)}(undef, Nx, Ny, Nxi, Neta)

    for n in 1:Neta, m in 1:Nxi
        values[:, :, m, n] .= view(interior(physical_field(f, m, n)), :, :, 1)
    end

    return values
end
