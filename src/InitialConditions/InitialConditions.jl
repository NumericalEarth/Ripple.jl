abstract type AbstractCartesianSpectrumProfile end

include("jonswap.jl")
include("gaussian_wave_packet.jl")

function set!(f::ProductField, ic::AbstractCartesianSpectrumProfile)
    _, _, Nxi, Neta = size(f)
    for n in 1:Neta, m in 1:Nxi
        kx, ky = k_components(coordinate_grid(f), m, n)
        set!(physical_field(f, m, n), (x, y, z) -> ic(x, y, kx, ky))
    end
    return f
end

