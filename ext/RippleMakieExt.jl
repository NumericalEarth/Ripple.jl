module RippleMakieExt

using Ripple
using Makie

function physical_nodes(field::Ripple.ProductField)
    grid = Ripple.grid(field)
    return collect(Ripple.xnodes(grid)), collect(Ripple.ynodes(grid))
end

function Makie.convert_arguments(::Type{<:Makie.Heatmap}, field::Ripple.ProductField)
    x, y = physical_nodes(field)
    m0_slab = Array(Ripple.interior(Ripple.m0(field)))
    return x, y, m0_slab[:, :, 1]
end

function Makie.convert_arguments(::Type{<:Makie.Heatmap}, model::Ripple.SpectralWaveModel)
    return Makie.convert_arguments(Makie.Heatmap, model.action)
end

end
