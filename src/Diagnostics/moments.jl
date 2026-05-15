#####
##### Spectral coordinate accessors used by Sources/ and by the
##### user-facing diagnostics below. Kept here for backward compatibility
##### with the pre-refactor layout, where `moments.jl` was the canonical
##### home for these one-liners.
#####

@inline spectral_direction(g::Union{PolarWaveVectorGrid, FrequencyDirectionGrid}, m, n) = g.φ[n]

@inline function spectral_direction(g, m, n)
    kx, ky = k_components(g, m, n)
    return atan(ky, kx)
end

@inline spectral_frequency(g::FrequencyDirectionGrid, m, n) = g.frequency[m]

spectral_frequency(g, m, n) =
    throw(ArgumentError("frequency diagnostics require FrequencyDirectionGrid; got $(typeof(g))"))

#####
##### User-facing moment diagnostics.
#####
##### Each function returns a 2D-slab `Field` over the physical grid at
##### `k = Nz`; the underlying compute is a `KernelFunctionOperation`, so
##### the same path runs on CPU and GPU. Vector-valued diagnostics return
##### a tuple of two/three Fields.
#####

#####
##### Zeroth moment m₀(x, y) = ∫∫ N(x, y, κ, φ) dκ dφ
#####

function m0(N::ProductField)
    cgrid = coordinate_grid(N)
    kfo = _weighted_spectral_sum(N, spectral_cell_measures(cgrid))
    return _slab_field(kfo, physical_grid(N))
end

#####
##### First moment (mx, my) = ∫∫ N k dκ dφ.
#####

function _first_moment_measures(N::ProductField)
    cgrid = coordinate_grid(N)
    Nκ, Nφ = coordinate_size(cgrid)
    FT = eltype(N)
    μx = Array{FT}(undef, Nκ, Nφ)
    μy = Array{FT}(undef, Nκ, Nφ)
    @inbounds for n in 1:Nφ, m in 1:Nκ
        mx_m, my_m = spectral_first_moment_measures(cgrid, m, n)
        μx[m, n] = mx_m
        μy[m, n] = my_m
    end
    arch = architecture(cgrid)
    return on_architecture(arch, μx), on_architecture(arch, μy)
end

function first_moment(N::ProductField)
    μx, μy = _first_moment_measures(N)
    grid = physical_grid(N)
    return _slab_field(_weighted_spectral_sum(N, μx), grid),
           _slab_field(_weighted_spectral_sum(N, μy), grid)
end

#####
##### Second moment (mxx, mxy, myy) = ∫∫ N kᵢkⱼ dκ dφ.
#####

function _second_moment_measures(N::ProductField)
    cgrid = coordinate_grid(N)
    Nκ, Nφ = coordinate_size(cgrid)
    FT = eltype(N)
    μxx = Array{FT}(undef, Nκ, Nφ)
    μxy = Array{FT}(undef, Nκ, Nφ)
    μyy = Array{FT}(undef, Nκ, Nφ)
    @inbounds for n in 1:Nφ, m in 1:Nκ
        xx, xy, yy = spectral_second_moment_measures(cgrid, m, n)
        μxx[m, n] = xx
        μxy[m, n] = xy
        μyy[m, n] = yy
    end
    arch = architecture(cgrid)
    return (on_architecture(arch, μxx),
            on_architecture(arch, μxy),
            on_architecture(arch, μyy))
end

function second_moment(N::ProductField)
    μxx, μxy, μyy = _second_moment_measures(N)
    grid = physical_grid(N)
    return _slab_field(_weighted_spectral_sum(N, μxx), grid),
           _slab_field(_weighted_spectral_sum(N, μxy), grid),
           _slab_field(_weighted_spectral_sum(N, μyy), grid)
end

#####
##### Mean-square wavenumber (mxx + myy) / m₀.
#####

function mean_square_wavenumber(N::ProductField)
    μxx, _, μyy = _second_moment_measures(N)
    weights = spectral_cell_measures(coordinate_grid(N))
    arch = architecture(coordinate_grid(N))
    trace = on_architecture(arch, Array(μxx) .+ Array(μyy))
    kfo = _weighted_spectral_ratio(N, trace, weights, zero(eltype(N)))
    return _slab_field(kfo, physical_grid(N))
end

# `mean_square_wavenumber` is ≥ 0 by construction; `sqrt` is a registered
# unary AbstractOp, so this stays GPU-portable and lazily composes.
root_mean_square_wavenumber(N::ProductField) = Field(sqrt(mean_square_wavenumber(N)))

#####
##### Mean direction (scalar atan and unit vector).
#####

function mean_direction(N::ProductField)
    μx, μy = _first_moment_measures(N)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    func = SpectralAngleKernel(flat, μx, μy, Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

function mean_direction_vector(N::ProductField)
    μx, μy = _first_moment_measures(N)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    grid = physical_grid(N)
    func_x = DirectionVectorComponentKernel(flat, μx, μx, μy, Hx, Hy, iz, Nκ, Nφ)
    func_y = DirectionVectorComponentKernel(flat, μy, μx, μy, Hx, Hy, iz, Nκ, Nφ)
    return _slab_field(KernelFunctionOperation{Center, Center, Center}(func_x, grid), grid),
           _slab_field(KernelFunctionOperation{Center, Center, Center}(func_y, grid), grid)
end

#####
##### Peak direction — argmax of band-integrated action.
#####

peak_direction(N::ProductField) = peak_direction(N, coordinate_grid(N))

function peak_direction(N::ProductField,
                        cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid})
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    weights = spectral_cell_measures(cgrid)
    func = PeakDirectionBandKernel(flat, weights, cgrid.φ, Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

function peak_direction(N::ProductField, cgrid)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    weights = spectral_cell_measures(cgrid)
    directions = _build_cell_direction_table(cgrid, eltype(N))
    func = PeakDirectionCellKernel(flat, weights, directions, Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

#####
##### Peak wavenumber — argmax with collapse over the orthogonal axis.
#####

peak_wavenumber(N::ProductField) = peak_wavenumber(N, coordinate_grid(N))

function peak_wavenumber(N::ProductField,
                         cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid})
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    weights = spectral_cell_measures(cgrid)
    func = PeakBandKernel(flat, weights, cgrid.κ, Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

function peak_wavenumber(N::ProductField, cgrid)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    weights = spectral_cell_measures(cgrid)
    wavenumbers = _build_cell_radial_wavenumber(cgrid, eltype(N))
    func = PeakCellKernel(flat, weights, wavenumbers, Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

#####
##### Deep-water peak phase speed √(g / κ_peak).
#####
##### Implemented by precomputing a per-band (or per-cell) phase speed
##### table and reusing the generic peak-argmax kernels with that table as
##### the reported coordinate. κ = 0 maps to typemax(FT) so empty action
##### → +Inf, matching the host loop's behaviour.
#####

deep_water_peak_phase_speed(N::ProductField; gravity=9.81) =
    deep_water_peak_phase_speed(N, coordinate_grid(N); gravity)

function deep_water_peak_phase_speed(N::ProductField,
                                     cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid};
                                     gravity=9.81)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    FT = eltype(N)
    speeds = _phase_speed_table_per_κ(cgrid, convert(FT, gravity))
    weights = spectral_cell_measures(cgrid)
    func = PeakBandKernel(flat, weights, speeds, Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

function deep_water_peak_phase_speed(N::ProductField, cgrid; gravity=9.81)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    FT = eltype(N)
    g = convert(FT, gravity)
    speeds = _phase_speed_table_per_cell(cgrid, g, FT)
    weights = spectral_cell_measures(cgrid)
    func = PeakCellKernel(flat, weights, speeds, Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

function _phase_speed_table_per_κ(cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid},
                                  g)
    FT = typeof(g)
    κ_host = Array(cgrid.κ)
    speeds = Array{FT}(undef, length(κ_host))
    @inbounds for m in eachindex(speeds)
        k = κ_host[m]
        speeds[m] = k > 0 ? sqrt(g / k) : typemax(FT)
    end
    return on_architecture(architecture(cgrid), speeds)
end

function _phase_speed_table_per_cell(cgrid, g, FT)
    Nκ, Nφ = coordinate_size(cgrid)
    speeds = Array{FT}(undef, Nκ, Nφ)
    @inbounds for n in 1:Nφ, m in 1:Nκ
        k = radial_wavenumber(cgrid, m, n)
        speeds[m, n] = k > 0 ? sqrt(g / k) : typemax(FT)
    end
    return on_architecture(architecture(cgrid), speeds)
end

#####
##### Wave age = c_peak / wind.
#####
##### Number wind folds wind into the precomputed phase-speed table.
##### Array / callable winds use dedicated kernels that argmax then divide
##### by `wind[i, j]` or `wind(x, y, t)` at the cell.
#####

wave_age(N::ProductField, wind; time=0.0, gravity=9.81) =
    _wave_age_dispatch(N, wind, coordinate_grid(N); time, gravity)

_wave_age_dispatch(N, wind::Number, cgrid; time, gravity) =
    _wave_age_number(N, wind, cgrid; time, gravity)
_wave_age_dispatch(N, wind::AbstractArray, cgrid; time, gravity) =
    _wave_age_array(N, wind, cgrid; time, gravity)
_wave_age_dispatch(N, wind, cgrid; time, gravity) =
    _wave_age_callable(N, wind, cgrid; time, gravity)

function _wave_age_number(N::ProductField, wind,
                          cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid};
                          time, gravity)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    FT = eltype(N)
    g = convert(FT, gravity)
    u = convert(FT, wind)
    κ_host = Array(cgrid.κ)
    speeds_host = Array{FT}(undef, length(κ_host))
    @inbounds for m in eachindex(speeds_host)
        k = κ_host[m]
        c = k > 0 ? sqrt(g / k) : typemax(FT)
        speeds_host[m] = u > 0 ? c / u : typemax(FT)
    end
    speeds = on_architecture(architecture(cgrid), speeds_host)
    weights = spectral_cell_measures(cgrid)
    func = PeakBandKernel(flat, weights, speeds, Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

function _wave_age_number(N::ProductField, wind, cgrid; time, gravity)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    FT = eltype(N)
    g = convert(FT, gravity)
    u = convert(FT, wind)
    ages = Array{FT}(undef, Nκ, Nφ)
    @inbounds for n in 1:Nφ, m in 1:Nκ
        k = radial_wavenumber(cgrid, m, n)
        c = k > 0 ? sqrt(g / k) : typemax(FT)
        ages[m, n] = u > 0 ? c / u : typemax(FT)
    end
    ages_dev = on_architecture(architecture(cgrid), ages)
    weights = spectral_cell_measures(cgrid)
    func = PeakCellKernel(flat, weights, ages_dev, Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

function _wave_age_array(N::ProductField, wind,
                         cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid};
                         time, gravity)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    FT = eltype(N)
    speeds = _phase_speed_table_per_κ(cgrid, convert(FT, gravity))
    weights = spectral_cell_measures(cgrid)
    wind_dev = on_architecture(architecture(physical_grid(N)), wind)
    func = WaveAgeArrayBandKernel(flat, weights, speeds, wind_dev,
                                  typemax(FT),
                                  Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

_wave_age_array(N::ProductField, wind, cgrid; time, gravity) =
    throw(ArgumentError("wave_age with an AbstractArray wind requires \
                         PolarWaveVectorGrid or FrequencyDirectionGrid; \
                         got $(typeof(cgrid))"))

function _wave_age_callable(N::ProductField, wind,
                            cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid};
                            time, gravity)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    FT = eltype(N)
    speeds = _phase_speed_table_per_κ(cgrid, convert(FT, gravity))
    weights = spectral_cell_measures(cgrid)
    func = WaveAgeCallableBandKernel(flat, weights, speeds, wind,
                                     convert(FT, time), typemax(FT),
                                     Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

_wave_age_callable(N::ProductField, wind, cgrid; time, gravity) =
    throw(ArgumentError("wave_age with a callable wind requires \
                         PolarWaveVectorGrid or FrequencyDirectionGrid; \
                         got $(typeof(cgrid))"))

#####
##### Mean / peak frequency and mean / peak period — FrequencyDirectionGrid only.
#####

_require_frequency_direction_grid(cgrid::FrequencyDirectionGrid, _) = cgrid
_require_frequency_direction_grid(cgrid, name) =
    throw(ArgumentError("$name requires FrequencyDirectionGrid; got $(typeof(cgrid))"))

function _frequency_measures(cgrid::FrequencyDirectionGrid, FT)
    Nκ, Nφ = coordinate_size(cgrid)
    M = Array{FT}(undef, Nκ, Nφ)
    @inbounds for n in 1:Nφ, m in 1:Nκ
        M[m, n] = spectral_frequency_power_measure(cgrid, m, n, 1)
    end
    return on_architecture(architecture(cgrid), M)
end

function mean_frequency(N::ProductField)
    cgrid = _require_frequency_direction_grid(coordinate_grid(N), "mean_frequency")
    FT = eltype(N)
    f_measures = _frequency_measures(cgrid, FT)
    weights = spectral_cell_measures(cgrid)
    kfo = _weighted_spectral_ratio(N, f_measures, weights, zero(FT))
    return _slab_field(kfo, physical_grid(N))
end

function mean_period(N::ProductField)
    cgrid = _require_frequency_direction_grid(coordinate_grid(N), "mean_period")
    FT = eltype(N)
    f_measures = _frequency_measures(cgrid, FT)
    weights = spectral_cell_measures(cgrid)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    func = MeanPeriodKernel(flat, f_measures, weights, Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

function peak_frequency(N::ProductField)
    cgrid = _require_frequency_direction_grid(coordinate_grid(N), "peak_frequency")
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    weights = spectral_cell_measures(cgrid)
    func = PeakBandKernel(flat, weights, cgrid.frequency, Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end

function peak_period(N::ProductField)
    cgrid = _require_frequency_direction_grid(coordinate_grid(N), "peak_period")
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    FT = eltype(N)
    f_host = Array(cgrid.frequency)
    periods_host = Array{FT}(undef, length(f_host))
    @inbounds for m in eachindex(periods_host)
        f = f_host[m]
        periods_host[m] = f > 0 ? inv(f) : typemax(FT)
    end
    periods = on_architecture(architecture(cgrid), periods_host)
    weights = spectral_cell_measures(cgrid)
    func = PeakBandKernel(flat, weights, periods, Hx, Hy, iz, Nκ, Nφ)
    kfo = KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
    return _slab_field(kfo, physical_grid(N))
end
