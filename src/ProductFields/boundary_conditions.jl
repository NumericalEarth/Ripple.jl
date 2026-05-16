struct ProductBoundaryConditions{PBC, CBC}
    physical :: PBC
    coordinate :: CBC
end

function ProductBoundaryConditions(; physical=(x=Periodic(), y=Periodic()),
                                     coordinate=(xi=NoFlux(), eta=Periodic()))
    physical   = canonical_named_bc_tuple(physical,   2, "physical boundary conditions")
    coordinate = canonical_named_bc_tuple(coordinate, 2, "coordinate boundary conditions")
    return ProductBoundaryConditions(physical, coordinate)
end

function canonical_named_bc_tuple(bcs::NamedTuple, expected_length, name)
    length(bcs) == expected_length ||
        throw(ArgumentError("$name must have length $expected_length"))
    values = map(canonical_bc, Tuple(bcs))
    return NamedTuple{keys(bcs)}(values)
end

canonical_named_bc_tuple(bcs, expected_length, name) =
    canonical_bc_tuple(bcs, expected_length, name)

function default_coordinate_bcs(cgrid)
    bcs = cgrid.boundary_conditions
    return (xi=bcs[1], eta=bcs[2])
end

function default_product_bcs(grid, cgrid)
    TX, TY, _ = OceanGrids.topology(grid)
    return ProductBoundaryConditions(physical=(x=TX, y=TY),
                                     coordinate=default_coordinate_bcs(cgrid))
end

default_wave_action_bcs(grid, cgrid) = default_product_bcs(grid, cgrid)
