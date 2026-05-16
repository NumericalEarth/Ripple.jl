import Adapt
using Oceananigans: Center
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ

#####
##### Kernel-function-operation infrastructure for ProductField diagnostics.
#####
##### Each diagnostic is expressed as a callable struct that holds only
##### Adapt-traversable state (raw arrays + scalar offsets). The struct is
##### invoked as `func(i, j, k, grid)` and wrapped in a
##### `KernelFunctionOperation{Center, Center, Center}` over the physical
##### grid; computing the operation onto a `Field(kfo; indices=(:, :, Nz))`
##### dispatches through Oceananigans' `compute!` path so the same code
##### runs on CPU and GPU. The inner spectral loop is small, type-stable,
##### and allocation-free — KernelAbstractions inlines it.
#####

@inline function _product_field_anatomy(N::ProductField)
    flat = N.flat_data
    Hx = N.stencil.offsets[1]
    Hy = N.stencil.offsets[2]
    iz = first(axes(flat, 3))
    _, _, Nκ, Nφ = size(N)
    return flat, Hx, Hy, iz, Nκ, Nφ
end

# Build a Nκ×Nφ device array of per-cell direction angles atan(ky, kx).
function _build_cell_direction_table(cgrid, FT)
    Nκ, Nφ = coordinate_size(cgrid)
    M = Array{FT}(undef, Nκ, Nφ)
    @inbounds for n in 1:Nφ, m in 1:Nκ
        kx, ky = k_components(cgrid, m, n)
        M[m, n] = atan(ky, kx)
    end
    return on_architecture(architecture(cgrid), M)
end

# Build a Nκ×Nφ device array of per-cell radial wavenumber.
function _build_cell_radial_wavenumber(cgrid, FT)
    Nκ, Nφ = coordinate_size(cgrid)
    M = Array{FT}(undef, Nκ, Nφ)
    @inbounds for n in 1:Nφ, m in 1:Nκ
        M[m, n] = radial_wavenumber(cgrid, m, n)
    end
    return on_architecture(architecture(cgrid), M)
end

# Wrap a KFO in a 2D-slab Field at z = Nz and compute it.
@inline _slab_field(kfo, grid) = Field(kfo; indices=(:, :, grid.Nz))

#####
##### Single weighted-sum kernel: out(i, j) = Σₘₙ N[i, j, m, n] * w[m, n].
#####

struct WeightedSpectralSumKernel{D, W}
    flat_data :: D
    measures :: W
    Hx :: Int
    Hy :: Int
    iz :: Int
    Nκ :: Int
    Nφ :: Int
end

Adapt.adapt_structure(to, k::WeightedSpectralSumKernel) =
    WeightedSpectralSumKernel(Adapt.adapt(to, k.flat_data),
                              Adapt.adapt(to, k.measures),
                              k.Hx, k.Hy, k.iz, k.Nκ, k.Nφ)

@inline function (k::WeightedSpectralSumKernel)(i, j, _kz, grid)
    acc = zero(eltype(k.flat_data))
    @inbounds for n in 1:k.Nφ
        for m in 1:k.Nκ
            acc += k.flat_data[i - k.Hx, j - k.Hy, k.iz, m, n] * k.measures[m, n]
        end
    end
    return acc
end

function _weighted_spectral_sum(N::ProductField, measures)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    func = WeightedSpectralSumKernel(flat, measures, Hx, Hy, iz, Nκ, Nφ)
    return KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
end

#####
##### Ratio kernel: out(i, j) = Σ N wₙᵤₘ / Σ N w_dₑₙ, with safe divide.
#####

struct WeightedSpectralRatioKernel{D, NW, DW, FB}
    flat_data :: D
    numerator_measures :: NW
    denominator_measures :: DW
    fallback :: FB
    Hx :: Int
    Hy :: Int
    iz :: Int
    Nκ :: Int
    Nφ :: Int
end

Adapt.adapt_structure(to, k::WeightedSpectralRatioKernel) =
    WeightedSpectralRatioKernel(Adapt.adapt(to, k.flat_data),
                                Adapt.adapt(to, k.numerator_measures),
                                Adapt.adapt(to, k.denominator_measures),
                                k.fallback,
                                k.Hx, k.Hy, k.iz, k.Nκ, k.Nφ)

@inline function (k::WeightedSpectralRatioKernel)(i, j, _kz, grid)
    FT = eltype(k.flat_data)
    num = zero(FT)
    den = zero(FT)
    @inbounds for n in 1:k.Nφ
        for m in 1:k.Nκ
            Nmn = k.flat_data[i - k.Hx, j - k.Hy, k.iz, m, n]
            num += Nmn * k.numerator_measures[m, n]
            den += Nmn * k.denominator_measures[m, n]
        end
    end
    return ifelse(iszero(den), oftype(num, k.fallback), num / den)
end

function _weighted_spectral_ratio(N::ProductField, num_measures, den_measures, fallback)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    func = WeightedSpectralRatioKernel(flat, num_measures, den_measures,
                                       convert(eltype(N), fallback),
                                       Hx, Hy, iz, Nκ, Nφ)
    return KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
end

#####
##### Angle kernel: out(i, j) = atan(Σ N μy, Σ N μx).
#####

struct SpectralAngleKernel{D, XW, YW}
    flat_data :: D
    x_measures :: XW
    y_measures :: YW
    Hx :: Int
    Hy :: Int
    iz :: Int
    Nκ :: Int
    Nφ :: Int
end

Adapt.adapt_structure(to, k::SpectralAngleKernel) =
    SpectralAngleKernel(Adapt.adapt(to, k.flat_data),
                        Adapt.adapt(to, k.x_measures),
                        Adapt.adapt(to, k.y_measures),
                        k.Hx, k.Hy, k.iz, k.Nκ, k.Nφ)

@inline function (k::SpectralAngleKernel)(i, j, _kz, grid)
    FT = eltype(k.flat_data)
    sx = zero(FT)
    sy = zero(FT)
    @inbounds for n in 1:k.Nφ
        for m in 1:k.Nκ
            Nmn = k.flat_data[i - k.Hx, j - k.Hy, k.iz, m, n]
            sx += Nmn * k.x_measures[m, n]
            sy += Nmn * k.y_measures[m, n]
        end
    end
    return atan(sy, sx)
end

#####
##### Direction-vector component kernel: out = Σ N μᵢ / hypot(Σ N μx, Σ N μy).
#####

struct DirectionVectorComponentKernel{D, CW, XW, YW}
    flat_data :: D
    component_measures :: CW
    x_measures :: XW
    y_measures :: YW
    Hx :: Int
    Hy :: Int
    iz :: Int
    Nκ :: Int
    Nφ :: Int
end

Adapt.adapt_structure(to, k::DirectionVectorComponentKernel) =
    DirectionVectorComponentKernel(Adapt.adapt(to, k.flat_data),
                                   Adapt.adapt(to, k.component_measures),
                                   Adapt.adapt(to, k.x_measures),
                                   Adapt.adapt(to, k.y_measures),
                                   k.Hx, k.Hy, k.iz, k.Nκ, k.Nφ)

@inline function (k::DirectionVectorComponentKernel)(i, j, _kz, grid)
    FT = eltype(k.flat_data)
    sx = zero(FT)
    sy = zero(FT)
    sc = zero(FT)
    @inbounds for n in 1:k.Nφ
        for m in 1:k.Nκ
            Nmn = k.flat_data[i - k.Hx, j - k.Hy, k.iz, m, n]
            sx += Nmn * k.x_measures[m, n]
            sy += Nmn * k.y_measures[m, n]
            sc += Nmn * k.component_measures[m, n]
        end
    end
    r = hypot(sx, sy)
    return ifelse(iszero(r), zero(FT), sc / r)
end

#####
##### Peak (argmax) kernels.
#####
##### "Band" variants collapse one spectral axis first, then take the argmax
##### over the other. Used for grids with rotational symmetry (Polar / FD).
##### "Cell" variants iterate over all (m, n) cells.
#####

# Collapse m, argmax over n. Returns directions[n_max].
struct PeakDirectionBandKernel{D, W, C}
    flat_data :: D
    weights :: W
    directions :: C   # length-Nφ
    Hx :: Int
    Hy :: Int
    iz :: Int
    Nκ :: Int
    Nφ :: Int
end

Adapt.adapt_structure(to, k::PeakDirectionBandKernel) =
    PeakDirectionBandKernel(Adapt.adapt(to, k.flat_data),
                            Adapt.adapt(to, k.weights),
                            Adapt.adapt(to, k.directions),
                            k.Hx, k.Hy, k.iz, k.Nκ, k.Nφ)

@inline function (k::PeakDirectionBandKernel)(i, j, _kz, grid)
    FT = eltype(k.flat_data)
    best = typemin(FT)
    best_dir = zero(FT)
    @inbounds for n in 1:k.Nφ
        band = zero(FT)
        for m in 1:k.Nκ
            band += k.flat_data[i - k.Hx, j - k.Hy, k.iz, m, n] * k.weights[m, n]
        end
        better = band > best
        best = ifelse(better, band, best)
        best_dir = ifelse(better, k.directions[n], best_dir)
    end
    return best_dir
end

# Cell argmax, directions Nκ × Nφ.
struct PeakDirectionCellKernel{D, W, C}
    flat_data :: D
    weights :: W
    directions :: C
    Hx :: Int
    Hy :: Int
    iz :: Int
    Nκ :: Int
    Nφ :: Int
end

Adapt.adapt_structure(to, k::PeakDirectionCellKernel) =
    PeakDirectionCellKernel(Adapt.adapt(to, k.flat_data),
                            Adapt.adapt(to, k.weights),
                            Adapt.adapt(to, k.directions),
                            k.Hx, k.Hy, k.iz, k.Nκ, k.Nφ)

@inline function (k::PeakDirectionCellKernel)(i, j, _kz, grid)
    FT = eltype(k.flat_data)
    best = typemin(FT)
    best_dir = zero(FT)
    @inbounds for n in 1:k.Nφ
        for m in 1:k.Nκ
            v = k.flat_data[i - k.Hx, j - k.Hy, k.iz, m, n] * k.weights[m, n]
            better = v > best
            best = ifelse(better, v, best)
            best_dir = ifelse(better, k.directions[m, n], best_dir)
        end
    end
    return best_dir
end

# Band over κ: collapse n, argmax over m. Returns coordinate[m_max].
# Reused for peak_wavenumber, deep_water_peak_phase_speed, peak_frequency
# (just pass different `coordinate` arrays).
struct PeakBandKernel{D, W, C}
    flat_data :: D
    weights :: W
    coordinate :: C   # length-Nκ
    Hx :: Int
    Hy :: Int
    iz :: Int
    Nκ :: Int
    Nφ :: Int
end

Adapt.adapt_structure(to, k::PeakBandKernel) =
    PeakBandKernel(Adapt.adapt(to, k.flat_data),
                   Adapt.adapt(to, k.weights),
                   Adapt.adapt(to, k.coordinate),
                   k.Hx, k.Hy, k.iz, k.Nκ, k.Nφ)

@inline function (k::PeakBandKernel)(i, j, _kz, grid)
    FT = eltype(k.flat_data)
    best = typemin(FT)
    best_coord = zero(FT)
    @inbounds for m in 1:k.Nκ
        band = zero(FT)
        for n in 1:k.Nφ
            band += k.flat_data[i - k.Hx, j - k.Hy, k.iz, m, n] * k.weights[m, n]
        end
        better = band > best
        best = ifelse(better, band, best)
        best_coord = ifelse(better, k.coordinate[m], best_coord)
    end
    return best_coord
end

# Cell argmax over (m, n). Reused: pass a Nκ × Nφ coordinate matrix.
struct PeakCellKernel{D, W, C}
    flat_data :: D
    weights :: W
    coordinate :: C
    Hx :: Int
    Hy :: Int
    iz :: Int
    Nκ :: Int
    Nφ :: Int
end

Adapt.adapt_structure(to, k::PeakCellKernel) =
    PeakCellKernel(Adapt.adapt(to, k.flat_data),
                   Adapt.adapt(to, k.weights),
                   Adapt.adapt(to, k.coordinate),
                   k.Hx, k.Hy, k.iz, k.Nκ, k.Nφ)

@inline function (k::PeakCellKernel)(i, j, _kz, grid)
    FT = eltype(k.flat_data)
    best = typemin(FT)
    best_coord = zero(FT)
    @inbounds for n in 1:k.Nφ
        for m in 1:k.Nκ
            v = k.flat_data[i - k.Hx, j - k.Hy, k.iz, m, n] * k.weights[m, n]
            better = v > best
            best = ifelse(better, v, best)
            best_coord = ifelse(better, k.coordinate[m, n], best_coord)
        end
    end
    return best_coord
end

#####
##### Wave-age kernels for Array and Callable wind inputs.
#####
##### Number-wind wave age folds the wind into a precomputed `coordinate`
##### matrix and reuses `PeakBandKernel` / `PeakCellKernel`. Array and
##### callable winds vary with (i, j) so they get dedicated kernels.
#####

struct WaveAgeArrayBandKernel{D, W, C, A, FT}
    flat_data :: D
    weights :: W
    speeds :: C       # length-Nκ
    wind :: A         # 2D array indexed at (i, j)
    inf_value :: FT
    Hx :: Int
    Hy :: Int
    iz :: Int
    Nκ :: Int
    Nφ :: Int
end

Adapt.adapt_structure(to, k::WaveAgeArrayBandKernel) =
    WaveAgeArrayBandKernel(Adapt.adapt(to, k.flat_data),
                           Adapt.adapt(to, k.weights),
                           Adapt.adapt(to, k.speeds),
                           Adapt.adapt(to, k.wind),
                           k.inf_value,
                           k.Hx, k.Hy, k.iz, k.Nκ, k.Nφ)

@inline function (k::WaveAgeArrayBandKernel)(i, j, _kz, grid)
    FT = eltype(k.flat_data)
    best = typemin(FT)
    best_speed = zero(FT)
    @inbounds for m in 1:k.Nκ
        band = zero(FT)
        for n in 1:k.Nφ
            band += k.flat_data[i - k.Hx, j - k.Hy, k.iz, m, n] * k.weights[m, n]
        end
        better = band > best
        best = ifelse(better, band, best)
        best_speed = ifelse(better, k.speeds[m], best_speed)
    end
    @inbounds u = k.wind[i, j]
    return ifelse(iszero(u), k.inf_value, best_speed / u)
end

struct WaveAgeCallableBandKernel{D, W, C, F, FT}
    flat_data :: D
    weights :: W
    speeds :: C
    wind :: F
    time :: FT
    inf_value :: FT
    Hx :: Int
    Hy :: Int
    iz :: Int
    Nκ :: Int
    Nφ :: Int
end

Adapt.adapt_structure(to, k::WaveAgeCallableBandKernel) =
    WaveAgeCallableBandKernel(Adapt.adapt(to, k.flat_data),
                              Adapt.adapt(to, k.weights),
                              Adapt.adapt(to, k.speeds),
                              Adapt.adapt(to, k.wind),
                              k.time, k.inf_value,
                              k.Hx, k.Hy, k.iz, k.Nκ, k.Nφ)

@inline function (k::WaveAgeCallableBandKernel)(i, j, kz, grid)
    FT = eltype(k.flat_data)
    best = typemin(FT)
    best_speed = zero(FT)
    @inbounds for m in 1:k.Nκ
        band = zero(FT)
        for n in 1:k.Nφ
            band += k.flat_data[i - k.Hx, j - k.Hy, k.iz, m, n] * k.weights[m, n]
        end
        better = band > best
        best = ifelse(better, band, best)
        best_speed = ifelse(better, k.speeds[m], best_speed)
    end
    x = Oceananigans.Grids.xnode(i, j, kz, grid, Center(), Center(), Center())
    y = Oceananigans.Grids.ynode(i, j, kz, grid, Center(), Center(), Center())
    u = k.wind(x, y, k.time)
    return ifelse(iszero(u), k.inf_value, best_speed / u)
end

#####
##### Mean-period kernel: out(i, j) = (Σ N w) / (Σ N μ_f), safe-divided.
#####

struct MeanPeriodKernel{D, NW, DW}
    flat_data :: D
    numerator_measures :: NW
    weights :: DW
    Hx :: Int
    Hy :: Int
    iz :: Int
    Nκ :: Int
    Nφ :: Int
end

Adapt.adapt_structure(to, k::MeanPeriodKernel) =
    MeanPeriodKernel(Adapt.adapt(to, k.flat_data),
                     Adapt.adapt(to, k.numerator_measures),
                     Adapt.adapt(to, k.weights),
                     k.Hx, k.Hy, k.iz, k.Nκ, k.Nφ)

@inline function (k::MeanPeriodKernel)(i, j, _kz, grid)
    FT = eltype(k.flat_data)
    num = zero(FT)
    den = zero(FT)
    @inbounds for n in 1:k.Nφ
        for m in 1:k.Nκ
            Nmn = k.flat_data[i - k.Hx, j - k.Hy, k.iz, m, n]
            num += Nmn * k.numerator_measures[m, n]
            den += Nmn * k.weights[m, n]
        end
    end
    return ifelse(iszero(num), typemax(FT), den / num)
end

#####
##### Area-weighted integrand kernel: density(i, j) * Δx * Δy.
#####
##### Used to express ∫∫ density dx dy as `sum(Field(kfo))` so the
##### reduction stays on whatever architecture the field lives on.
#####

struct SpectralAreaIntegrandKernel{D, W}
    flat_data :: D
    measures :: W
    Hx :: Int
    Hy :: Int
    iz :: Int
    Nκ :: Int
    Nφ :: Int
end

Adapt.adapt_structure(to, k::SpectralAreaIntegrandKernel) =
    SpectralAreaIntegrandKernel(Adapt.adapt(to, k.flat_data),
                                Adapt.adapt(to, k.measures),
                                k.Hx, k.Hy, k.iz, k.Nκ, k.Nφ)

@inline function (k::SpectralAreaIntegrandKernel)(i, j, kz, grid)
    acc = zero(eltype(k.flat_data))
    @inbounds for n in 1:k.Nφ
        for m in 1:k.Nκ
            acc += k.flat_data[i - k.Hx, j - k.Hy, k.iz, m, n] * k.measures[m, n]
        end
    end
    return acc * Δxᶜᶜᶜ(i, j, kz, grid) * Δyᶜᶜᶜ(i, j, kz, grid)
end

function _spectral_area_integrand_kfo(N::ProductField, measures)
    flat, Hx, Hy, iz, Nκ, Nφ = _product_field_anatomy(N)
    func = SpectralAreaIntegrandKernel(flat, measures, Hx, Hy, iz, Nκ, Nφ)
    return KernelFunctionOperation{Center, Center, Center}(func, physical_grid(N))
end
