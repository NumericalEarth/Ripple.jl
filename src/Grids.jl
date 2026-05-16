import Oceananigans
import Oceananigans.Grids: RectilinearGrid, AbstractGrid
import Oceananigans.Grids: xnodes, ynodes, znodes
import Oceananigans.Architectures: architecture, on_architecture

const OceanGrids = Oceananigans.Grids

"""
    InfiniteDepth()

Marker for deep-water wave dispersion. Use as `SpectralWaveModel(...;
depth=InfiniteDepth())` when the wave model should use the deep-water
dispersion relation even if coupled velocities live on a finite-depth grid.
"""
struct InfiniteDepth end

is_infinite_depth(::InfiniteDepth) = true
is_infinite_depth(depth) = false

canonical_topology(topology::NoFlux) = topology
canonical_topology(::Type{NoFlux}) = NoFlux()
canonical_topology(::Type{T}) where T<:OceanGrids.AbstractTopology = T()
canonical_topology(topology::OceanGrids.AbstractTopology) = topology
canonical_topology(topology) =
    throw(ArgumentError("unsupported topology marker $(repr(topology)); use Periodic, Bounded, NoFlux, or their instances"))

function canonical_topology_tuple(topology, expected_length, name)
    length(topology) == expected_length ||
        throw(ArgumentError("$name topology must have length $expected_length"))
    return map(canonical_topology, topology)
end

grid_float_type(grid::AbstractGrid) = eltype(grid)
adapt_physical_grid(grid::AbstractGrid) = grid
adapt_physical_grid(grid) = grid
horizontal_size(g::AbstractGrid) = (g.Nx, g.Ny)
vertical_size(g::AbstractGrid) = g.Nz

cpu_nodes(nodes) = collect(on_architecture(Oceananigans.CPU(), nodes))
has_flat_vertical_topology(g::AbstractGrid) = OceanGrids.topology(g, 3) === Flat

xnodes(g::AbstractGrid) = cpu_nodes(OceanGrids.xnodes(g, Center()))
ynodes(g::AbstractGrid) = cpu_nodes(OceanGrids.ynodes(g, Center()))
znodes(g::AbstractGrid) = has_flat_vertical_topology(g) ? grid_float_type(g)[] : cpu_nodes(OceanGrids.znodes(g, Center()))
dimension_size(g, dim) = dim == 1 ? g.Nx : dim == 2 ? g.Ny : g.Nz
dimension_length(g, dim) = dim == 1 ? g.Lx : dim == 2 ? g.Ly : g.Lz

function closed_faces(g::AbstractGrid, faces, dim)
    T = OceanGrids.topology(g, dim)
    T === Periodic && length(faces) == dimension_size(g, dim) &&
        return vcat(faces, first(faces) + dimension_length(g, dim))
    return faces
end

xfaces(g::AbstractGrid) = closed_faces(g, cpu_nodes(OceanGrids.xnodes(g, Face())), 1)
yfaces(g::AbstractGrid) = closed_faces(g, cpu_nodes(OceanGrids.ynodes(g, Face())), 2)
zfaces(g::AbstractGrid) = has_flat_vertical_topology(g) ? grid_float_type(g)[] : closed_faces(g, cpu_nodes(OceanGrids.znodes(g, Face())), 3)

xspacings(g::AbstractGrid) = diff(xfaces(g))
yspacings(g::AbstractGrid) = diff(yfaces(g))
zspacings(g::AbstractGrid) = diff(zfaces(g))

@inline periodic_index(i, N) = mod1(i, N)
