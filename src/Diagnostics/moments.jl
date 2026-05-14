function m0(N::ProductField)
    Nx, Ny, Nxi, Neta = size(N)
    out = zeros(eltype(N), Nx, Ny)
    cgrid = coordinate_grid(N)
    for j in 1:Ny, i in 1:Nx
        acc = zero(eltype(N))
        for n in 1:Neta, m in 1:Nxi
            acc += N[i, j, m, n] * spectral_weight(cgrid, m, n)
        end
        out[i, j] = acc
    end
    return out
end

function first_moment(N::ProductField)
    Nx, Ny, Nxi, Neta = size(N)
    mx = zeros(eltype(N), Nx, Ny)
    my = zeros(eltype(N), Nx, Ny)
    cgrid = coordinate_grid(N)
    for j in 1:Ny, i in 1:Nx
        ax = zero(eltype(N))
        ay = zero(eltype(N))
        for n in 1:Neta, m in 1:Nxi
            kx_measure, ky_measure = spectral_first_moment_measures(cgrid, m, n)
            ax += N[i, j, m, n] * kx_measure
            ay += N[i, j, m, n] * ky_measure
        end
        mx[i, j] = ax
        my[i, j] = ay
    end
    return mx, my
end

function second_moment(N::ProductField)
    Nx, Ny, Nxi, Neta = size(N)
    mxx = zeros(eltype(N), Nx, Ny)
    mxy = zeros(eltype(N), Nx, Ny)
    myy = zeros(eltype(N), Nx, Ny)
    cgrid = coordinate_grid(N)
    for j in 1:Ny, i in 1:Nx
        axx = zero(eltype(N))
        axy = zero(eltype(N))
        ayy = zero(eltype(N))
        for n in 1:Neta, m in 1:Nxi
            xx_measure, xy_measure, yy_measure = spectral_second_moment_measures(cgrid, m, n)
            action = N[i, j, m, n]
            axx += action * xx_measure
            axy += action * xy_measure
            ayy += action * yy_measure
        end
        mxx[i, j] = axx
        mxy[i, j] = axy
        myy[i, j] = ayy
    end
    return mxx, mxy, myy
end

function mean_square_wavenumber(N::ProductField)
    zeroth = m0(N)
    mxx, _, myy = second_moment(N)
    out = similar(zeroth)

    for I in eachindex(zeroth)
        out[I] = zeroth[I] == 0 ? zero(eltype(out)) : (mxx[I] + myy[I]) / zeroth[I]
    end

    return out
end

function root_mean_square_wavenumber(N::ProductField)
    mean_square = mean_square_wavenumber(N)
    out = similar(mean_square)

    for I in eachindex(mean_square)
        out[I] = sqrt(max(mean_square[I], zero(eltype(out))))
    end

    return out
end

function mean_direction_vector(N::ProductField)
    mx, my = first_moment(N)
    mag = hypot.(mx, my)
    ux = similar(mx)
    uy = similar(my)
    for I in eachindex(mx)
        if mag[I] == 0
            ux[I] = zero(eltype(mx))
            uy[I] = zero(eltype(my))
        else
            ux[I] = mx[I] / mag[I]
            uy[I] = my[I] / mag[I]
        end
    end
    return ux, uy
end

function mean_direction(N::ProductField)
    mx, my = first_moment(N)
    return atan.(my, mx)
end

spectral_direction(g::Union{PolarWaveVectorGrid, FrequencyDirectionGrid}, m, n) = g.φ[n]
function spectral_direction(g, m, n)
    kx, ky = k_components(g, m, n)
    return atan(ky, kx)
end

function peak_direction(N::ProductField)
    return peak_direction(N, coordinate_grid(N))
end

function peak_direction(N::ProductField, cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid})
    Nx, Ny, Nxi, Neta = size(N)
    out = zeros(eltype(N), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        best = -Inf
        best_direction = zero(eltype(N))
        for n in 1:Neta
            band = zero(eltype(N))
            for m in 1:Nxi
                band += N[i, j, m, n] * spectral_weight(cgrid, m, n)
            end
            if band > best
                best = band
                best_direction = spectral_direction(cgrid, 1, n)
            end
        end
        out[i, j] = best_direction
    end

    return out
end

function peak_direction(N::ProductField, cgrid)
    Nx, Ny, Nxi, Neta = size(N)
    out = zeros(eltype(N), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        best = -Inf
        best_direction = zero(eltype(N))
        for n in 1:Neta, m in 1:Nxi
            value = N[i, j, m, n] * spectral_weight(cgrid, m, n)
            if value > best
                best = value
                best_direction = spectral_direction(cgrid, m, n)
            end
        end
        out[i, j] = best_direction
    end

    return out
end

function peak_wavenumber(N::ProductField)
    return peak_wavenumber(N, coordinate_grid(N))
end

function peak_wavenumber(N::ProductField, cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid})
    Nx, Ny, Nxi, Neta = size(N)
    out = zeros(eltype(N), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        best = -Inf
        best_wavenumber = zero(eltype(N))
        for m in 1:Nxi
            band = zero(eltype(N))
            for n in 1:Neta
                band += N[i, j, m, n] * spectral_weight(cgrid, m, n)
            end
            if band > best
                best = band
                best_wavenumber = radial_wavenumber(cgrid, m, 1)
            end
        end
        out[i, j] = best_wavenumber
    end

    return out
end

function peak_wavenumber(N::ProductField, cgrid)
    Nx, Ny, Nxi, Neta = size(N)
    out = zeros(eltype(N), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        best = -Inf
        best_wavenumber = zero(eltype(N))
        for n in 1:Neta, m in 1:Nxi
            value = N[i, j, m, n] * spectral_weight(cgrid, m, n)
            if value > best
                best = value
                best_wavenumber = radial_wavenumber(cgrid, m, n)
            end
        end
        out[i, j] = best_wavenumber
    end

    return out
end

function deep_water_peak_phase_speed(N::ProductField; gravity=9.81)
    k = peak_wavenumber(N)
    out = similar(k)
    for I in eachindex(k)
        out[I] = k[I] == 0 ? oftype(k[I], Inf) : sqrt(gravity / k[I])
    end
    return out
end

function diagnostic_wind_speed(wind::Number, N, i, j, time)
    return wind
end

function diagnostic_wind_speed(wind::AbstractArray, N, i, j, time)
    return wind[i, j]
end

function diagnostic_wind_speed(wind, N, i, j, time)
    x = xnodes(grid(N))[i]
    y = ynodes(grid(N))[j]

    if applicable(wind_speed, wind, x, y, time)
        return wind_speed(wind, x, y, time)
    elseif applicable(wind, x, y, time)
        return wind(x, y, time)
    elseif applicable(wind, x, y)
        return wind(x, y)
    elseif applicable(wind, time)
        return wind(time)
    else
        throw(ArgumentError("wind speed diagnostics require a number, array, wind object, or callable accepting (x, y, t), (x, y), or (t)"))
    end
end

function wave_age(N::ProductField, wind; time=0.0, gravity=9.81)
    c = deep_water_peak_phase_speed(N; gravity)
    out = similar(c)
    Nx, Ny = size(c)

    for j in 1:Ny, i in 1:Nx
        u = diagnostic_wind_speed(wind, N, i, j, time)
        out[i, j] = u == 0 ? oftype(c[i, j], Inf) : c[i, j] / u
    end

    return out
end

spectral_frequency(g::FrequencyDirectionGrid, m, n) = g.frequency[m]
spectral_frequency(g, m, n) =
    throw(ArgumentError("frequency diagnostics require FrequencyDirectionGrid; got $(typeof(g))"))

function mean_frequency(N::ProductField)
    Nx, Ny, Nxi, Neta = size(N)
    cgrid = coordinate_grid(N)
    out = zeros(eltype(N), Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        numerator = zero(eltype(N))
        denominator = zero(eltype(N))
        for n in 1:Neta, m in 1:Nxi
            action = N[i, j, m, n]
            numerator += action * spectral_frequency_power_measure(cgrid, m, n, 1)
            denominator += action * spectral_weight(cgrid, m, n)
        end
        out[i, j] = denominator == 0 ? zero(eltype(N)) : numerator / denominator
    end
    return out
end

function mean_period(N::ProductField)
    frequency = mean_frequency(N)
    out = similar(frequency)
    for I in eachindex(frequency)
        out[I] = frequency[I] == 0 ? oftype(frequency[I], Inf) : inv(frequency[I])
    end
    return out
end

function peak_frequency(N::ProductField)
    Nx, Ny, Nxi, Neta = size(N)
    cgrid = coordinate_grid(N)
    out = zeros(eltype(N), Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        best = -Inf
        best_frequency = zero(eltype(N))
        for m in 1:Nxi
            band = zero(eltype(N))
            for n in 1:Neta
                band += N[i, j, m, n] * spectral_weight(cgrid, m, n)
            end
            if band > best
                best = band
                best_frequency = spectral_frequency(cgrid, m, 1)
            end
        end
        out[i, j] = best_frequency
    end
    return out
end

function peak_period(N::ProductField)
    frequency = peak_frequency(N)
    out = similar(frequency)
    for I in eachindex(frequency)
        out[I] = frequency[I] == 0 ? oftype(frequency[I], Inf) : inv(frequency[I])
    end
    return out
end
