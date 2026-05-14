"""
    integrate_spectrum(values, grid)

Integrate cell-averaged spectral `values` over the spectral finite-volume grid.

The entries of `values` are interpreted as exact averages over spectral control
volumes. The returned integral is therefore a finite-volume sum of cell averages
times cell measures, not a point-sample rule.
"""
function integrate_spectrum(values::AbstractMatrix, grid::AbstractSpectralGrid)
    size(values) == size(spectral_cell_measures(grid)) ||
        throw(ArgumentError("values and spectral cell measures must have matching size"))

    measures = spectral_cell_measures(grid)
    total = zero(promote_type(eltype(values), eltype(measures)))

    for n in axes(values, 2), m in axes(values, 1)
        total += values[m, n] * measures[m, n]
    end

    return total
end

@inline function radial_power_annular_measure(k₁, k₂, φ₁, φ₂, power)
    power > -2 || throw(ArgumentError("radial power must be greater than -2 for annular finite-volume integration"))
    return (φ₂ - φ₁) * (k₂^(power + 2) - k₁^(power + 2)) / (power + 2)
end

@inline function spectral_radial_power_measure(g::PolarWaveVectorGrid, m, n, power)
    return radial_power_annular_measure(g.κ_faces[m], g.κ_faces[m+1],
                                        g.φ_faces[n], g.φ_faces[n+1],
                                        power)
end

@inline function spectral_radial_power_measure(g::FrequencyDirectionGrid, m, n, power)
    return radial_power_annular_measure(g.κ_faces[m], g.κ_faces[m+1],
                                        g.φ_faces[n], g.φ_faces[n+1],
                                        power)
end

@inline function cartesian_radial_first_moment_antiderivative(kx, ky)
    r = hypot(kx, ky)
    value = kx * ky * r / 3
    iszero(kx) || (value += kx^3 * asinh(ky / abs(kx)) / 6)
    iszero(ky) || (value += ky^3 * asinh(kx / abs(ky)) / 6)
    return value
end

@inline function cartesian_radial_first_moment(x₁, x₂, y₁, y₂)
    F = cartesian_radial_first_moment_antiderivative
    return F(x₂, y₂) - F(x₁, y₂) - F(x₂, y₁) + F(x₁, y₁)
end

@inline function spectral_radial_power_measure(g::CartesianWaveVectorGrid, m, n, power)
    power >= 0 || throw(ArgumentError("radial power must be nonnegative"))
    power == 0 && return spectral_cell_measure(g, m, n)

    x₁, x₂ = g.kx_faces[m], g.kx_faces[m+1]
    y₁, y₂ = g.ky_faces[n], g.ky_faces[n+1]

    power == 1 && return cartesian_radial_first_moment(x₁, x₂, y₁, y₂)

    if power == 2
        xx, _, yy = spectral_second_moment_measures(g, m, n)
        return xx + yy
    end

    throw(ArgumentError("exact Cartesian radial-power integration supports powers 0, 1, and 2"))
end

function spectral_radial_power_average(g::AbstractSpectralGrid, m, n, power)
    return spectral_radial_power_measure(g, m, n, power) / spectral_cell_measure(g, m, n)
end

function spectral_frequency_power_measure(g::AbstractSpectralGrid, m, n, power)
    throw(ArgumentError("frequency power integration requires a FrequencyDirectionGrid"))
end

@inline function spectral_frequency_power_measure(g::FrequencyDirectionGrid, m, n, power)
    power >= 0 || throw(ArgumentError("frequency power must be nonnegative"))
    f₁, f₂ = g.frequency_faces[m], g.frequency_faces[m+1]
    coeff = g.κ_faces[m+1] / f₂^2
    return (g.φ_faces[n+1] - g.φ_faces[n]) *
           2 * coeff^2 * (f₂^(power + 4) - f₁^(power + 4)) / (power + 4)
end

function spectral_frequency_power_average(g::AbstractSpectralGrid, m, n, power)
    return spectral_frequency_power_measure(g, m, n, power) / spectral_cell_measure(g, m, n)
end
