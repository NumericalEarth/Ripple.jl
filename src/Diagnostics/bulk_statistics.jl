significant_wave_height(N::ProductField) = 4 .* sqrt.(max.(m0(N), zero(eltype(N))))

function total_action(N::ProductField)
    Nx, Ny, Nxi, Neta = size(N)
    cgrid = coordinate_grid(N)
    dx = xspacings(grid(N))
    dy = yspacings(grid(N))
    total = zero(eltype(N))
    for n in 1:Neta, m in 1:Nxi, j in 1:Ny, i in 1:Nx
        total += N[i, j, m, n] * spectral_weight(cgrid, m, n) * dx[i] * dy[j]
    end
    return total
end

deep_water_intrinsic_frequency(cgrid, m, n; gravity=9.81) =
    sqrt(gravity * max(radial_wavenumber(cgrid, m, n), zero(radial_wavenumber(cgrid, m, n))))

deep_water_intrinsic_frequency_measure(cgrid, m, n; gravity=9.81) =
    deep_water_intrinsic_frequency(cgrid, m, n; gravity) * spectral_weight(cgrid, m, n)

deep_water_intrinsic_frequency_measure(cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid},
                                       m, n; gravity=9.81) =
    sqrt(gravity) * spectral_radial_power_measure(cgrid, m, n, 1 / 2)

function deep_water_intrinsic_group_speed(cgrid, m, n; gravity=9.81)
    k = max(radial_wavenumber(cgrid, m, n), zero(radial_wavenumber(cgrid, m, n)))
    return iszero(k) ? oftype(float(k), Inf) : sqrt(gravity / k) / 2
end

deep_water_intrinsic_group_speed_measure(cgrid, m, n; gravity=9.81) =
    deep_water_intrinsic_group_speed(cgrid, m, n; gravity) * spectral_weight(cgrid, m, n)

deep_water_intrinsic_group_speed_measure(cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid},
                                         m, n; gravity=9.81) =
    sqrt(gravity) / 2 * spectral_radial_power_measure(cgrid, m, n, -1 / 2)

function deep_water_energy_density(N::ProductField; gravity=9.81)
    Nx, Ny, Nxi, Neta = size(N)
    cgrid = coordinate_grid(N)
    energy = zeros(eltype(N), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        acc = zero(eltype(N))
        for n in 1:Neta, m in 1:Nxi
            acc += N[i, j, m, n] *
                   deep_water_intrinsic_frequency_measure(cgrid, m, n; gravity)
        end
        energy[i, j] = acc
    end

    return energy
end

function total_deep_water_energy(N::ProductField; gravity=9.81)
    energy = deep_water_energy_density(N; gravity)
    dx = xspacings(grid(N))
    dy = yspacings(grid(N))
    total = zero(eltype(energy))

    Nx, Ny = size(energy)
    for j in 1:Ny, i in 1:Nx
        total += energy[i, j] * dx[i] * dy[j]
    end

    return total
end

function mean_deep_water_group_speed(N::ProductField; gravity=9.81)
    Nx, Ny, Nxi, Neta = size(N)
    cgrid = coordinate_grid(N)
    speed = zeros(eltype(N), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        numerator = zero(eltype(N))
        denominator = zero(eltype(N))
        for n in 1:Neta, m in 1:Nxi
            action = N[i, j, m, n]
            numerator += action *
                         deep_water_intrinsic_group_speed_measure(cgrid, m, n; gravity)
            denominator += action * spectral_weight(cgrid, m, n)
        end
        speed[i, j] = denominator == 0 ? zero(eltype(N)) : numerator / denominator
    end

    return speed
end
