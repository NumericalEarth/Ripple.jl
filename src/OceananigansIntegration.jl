import Oceananigans

const OceanFields = Oceananigans.Fields
const OceanOutputWriters = isdefined(Oceananigans, :OutputWriters) ?
                           Oceananigans.OutputWriters : nothing

if OceanOutputWriters !== nothing && isdefined(OceanOutputWriters, :fetch_output)
    @eval OceanOutputWriters.fetch_output(field::ProductField, model) =
        interior(field)
end

field_storage(field::AbstractArray) = field
field_storage(field::ProductField) = interior(field)
field_storage(field::OceanFields.Field) = OceanFields.interior(field)

grid(field::OceanFields.Field) = field.grid
