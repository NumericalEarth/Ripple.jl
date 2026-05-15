#####
##### Bulk statistics. Like the moment diagnostics, single-cell quantities
##### return 2D-slab Fields backed by a KernelFunctionOperation; volume
##### integrals materialize a per-cell area-density Field and then `sum`
##### it, so all reductions stay on the field's architecture.
#####

# `m0(N)` is ≥ 0 by construction, so we don't need `max(., 0)` paranoia.
significant_wave_height(N::ProductField) = Field(4 * sqrt(m0(N)))

#####
##### Per-cell measure builders for deep-water energy / group speed.
##### Polar / FrequencyDirection grids use the exact annular finite-volume
##### measure (`spectral_radial_power_measure(..., 1/2)` for energy and
##### `..., -1/2` for inverse-phase-speed). Generic grids fall back to the
##### midpoint approximation.
#####

@inline _deep_water_frequency_cell(cgrid, m, n, g) =
    sqrt(g * max(radial_wavenumber(cgrid, m, n),
                 zero(radial_wavenumber(cgrid, m, n)))) *
    spectral_cell_measure(cgrid, m, n)

@inline _deep_water_frequency_cell(cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid},
                                   m, n, g) =
    sqrt(g) * spectral_radial_power_measure(cgrid, m, n, 1//2)

@inline function _deep_water_group_speed_cell(cgrid, m, n, g)
    k = max(radial_wavenumber(cgrid, m, n), zero(radial_wavenumber(cgrid, m, n)))
    c = iszero(k) ? oftype(float(k), Inf) : sqrt(g / k) / 2
    return c * spectral_cell_measure(cgrid, m, n)
end

@inline _deep_water_group_speed_cell(cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid},
                                     m, n, g) =
    sqrt(g) / 2 * spectral_radial_power_measure(cgrid, m, n, -1//2)

function _deep_water_energy_measures(cgrid, g, FT)
    Nκ, Nφ = coordinate_size(cgrid)
    M = Array{FT}(undef, Nκ, Nφ)
    @inbounds for n in 1:Nφ, m in 1:Nκ
        M[m, n] = _deep_water_frequency_cell(cgrid, m, n, g)
    end
    return on_architecture(architecture(cgrid), M)
end

function _deep_water_group_speed_measures(cgrid, g, FT)
    Nκ, Nφ = coordinate_size(cgrid)
    M = Array{FT}(undef, Nκ, Nφ)
    @inbounds for n in 1:Nφ, m in 1:Nκ
        M[m, n] = _deep_water_group_speed_cell(cgrid, m, n, g)
    end
    return on_architecture(architecture(cgrid), M)
end

#####
##### Deep-water energy density and group speed.
#####

function deep_water_energy_density(N::ProductField; gravity=9.81)
    cgrid = coordinate_grid(N)
    FT = eltype(N)
    measures = _deep_water_energy_measures(cgrid, convert(FT, gravity), FT)
    kfo = _weighted_spectral_sum(N, measures)
    return _slab_field(kfo, physical_grid(N))
end

function mean_deep_water_group_speed(N::ProductField; gravity=9.81)
    cgrid = coordinate_grid(N)
    FT = eltype(N)
    g = convert(FT, gravity)
    cg_measures = _deep_water_group_speed_measures(cgrid, g, FT)
    weights = spectral_cell_measures(cgrid)
    kfo = _weighted_spectral_ratio(N, cg_measures, weights, zero(FT))
    return _slab_field(kfo, physical_grid(N))
end

#####
##### Volume integrals.
#####
##### Build a per-cell `density(i, j) * Δx * Δy` Field via
##### `SpectralAreaIntegrandKernel` and reduce with `sum`. The Field's
##### `sum` runs on the same architecture as the field data, so this
##### works on GPU without scalar indexing.
#####

function total_action(N::ProductField)
    cgrid = coordinate_grid(N)
    kfo = _spectral_area_integrand_kfo(N, spectral_cell_measures(cgrid))
    return sum(_slab_field(kfo, physical_grid(N)))
end

function total_deep_water_energy(N::ProductField; gravity=9.81)
    cgrid = coordinate_grid(N)
    FT = eltype(N)
    measures = _deep_water_energy_measures(cgrid, convert(FT, gravity), FT)
    kfo = _spectral_area_integrand_kfo(N, measures)
    return sum(_slab_field(kfo, physical_grid(N)))
end
