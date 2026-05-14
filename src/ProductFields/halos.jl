function fill_halo_regions!(f::ProductField; kwargs...)
    for field in parent(f)
        fill_halo_regions!(field; kwargs...)
    end

    return f
end
