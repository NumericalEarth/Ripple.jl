struct ProductBoundaryConditions{PBC, CBC}
    physical :: PBC
    coordinate :: CBC
end

function ProductBoundaryConditions(; physical=(x=Periodic(), y=Periodic()),
                                     coordinate=(xi=Bounded(), eta=Bounded()))
    physical = canonical_named_topology_tuple(physical, 2, "physical boundary conditions")
    coordinate = canonical_named_topology_tuple(coordinate, 2, "coordinate boundary conditions")
    return ProductBoundaryConditions(physical, coordinate)
end

function canonical_named_topology_tuple(topology::NamedTuple, expected_length, name)
    length(topology) == expected_length ||
        throw(ArgumentError("$name topology must have length $expected_length"))
    values = map(canonical_topology, Tuple(topology))
    return NamedTuple{keys(topology)}(values)
end

canonical_named_topology_tuple(topology, expected_length, name) =
    canonical_topology_tuple(topology, expected_length, name)

function default_coordinate_bcs(cgrid)
    topo = cgrid.topology
    return (xi=topo[1], eta=topo[2])
end

function default_product_bcs(grid, cgrid)
    TX, TY, _ = OceanGrids.topology(grid)
    return ProductBoundaryConditions(physical=(x=TX, y=TY),
                                     coordinate=default_coordinate_bcs(cgrid))
end

default_wave_action_bcs(grid, cgrid) = default_product_bcs(grid, cgrid)
