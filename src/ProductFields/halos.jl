function fill_halo_regions!(f::ProductField; kwargs...)
    _, _, Nxi, Neta = size(f)
    for n in 1:Neta, m in 1:Nxi
        fill_halo_regions!(physical_field(f, m, n); kwargs...)
    end
    return f
end
