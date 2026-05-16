import Oceananigans
import Oceananigans.Grids: RectilinearGrid, AbstractGrid
import Oceananigans.Grids: xnodes, ynodes, znodes
import Oceananigans.Architectures: architecture, on_architecture

const OceanGrids = Oceananigans.Grids

canonical_topology(::Type{T}) where T<:OceanGrids.AbstractTopology = T()
canonical_topology(topology::OceanGrids.AbstractTopology) = topology
canonical_topology(topology) =
    throw(ArgumentError("unsupported topology marker $(repr(topology)); use Periodic, Bounded, Flat, or their instances"))

function canonical_topology_tuple(topology, expected_length, name)
    length(topology) == expected_length ||
        throw(ArgumentError("$name topology must have length $expected_length"))
    return map(canonical_topology, topology)
end

# Boundary-condition markers used by spectral grids and product fields.
# NoFlux is Ripple's wave-action no-flux BC; Periodic mirrors Oceananigans'
# periodic topology when used as a face condition.
canonical_bc(::Type{NoFlux}) = NoFlux()
canonical_bc(bc::NoFlux) = bc
canonical_bc(::Type{T}) where T<:OceanGrids.AbstractTopology = T()
canonical_bc(bc::OceanGrids.AbstractTopology) = bc
canonical_bc(bc) =
    throw(ArgumentError("unsupported boundary-condition marker $(repr(bc)); use NoFlux, Periodic, Bounded, or their instances"))

function canonical_bc_tuple(bcs, expected_length, name)
    length(bcs) == expected_length ||
        throw(ArgumentError("$name boundary_conditions must have length $expected_length"))
    return map(canonical_bc, bcs)
end

grid_float_type(grid::AbstractGrid) = eltype(grid)
adapt_physical_grid(grid::AbstractGrid) = grid
adapt_physical_grid(grid) = grid
horizontal_size(g::AbstractGrid) = (g.Nx, g.Ny)
vertical_size(g::AbstractGrid) = g.Nz

cpu_nodes(nodes) = collect(on_architecture(Oceananigans.CPU(), nodes))

xnodes(g::AbstractGrid) = cpu_nodes(OceanGrids.xnodes(g, Center()))
ynodes(g::AbstractGrid) = cpu_nodes(OceanGrids.ynodes(g, Center()))
znodes(g::AbstractGrid) = cpu_nodes(OceanGrids.znodes(g, Center()))
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
zfaces(g::AbstractGrid) = closed_faces(g, cpu_nodes(OceanGrids.znodes(g, Face())), 3)

xspacings(g::AbstractGrid) = diff(xfaces(g))
yspacings(g::AbstractGrid) = diff(yfaces(g))
zspacings(g::AbstractGrid) = diff(zfaces(g))

@inline periodic_index(i, N) = mod1(i, N)
