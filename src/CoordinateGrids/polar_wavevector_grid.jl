struct PolarWaveVectorGrid{FT, A, K, Φ, KF, ΦF, W, Topo} <: AbstractSpectralGrid
    architecture :: A
    κ :: K
    φ :: Φ
    κ_faces :: KF
    φ_faces :: ΦF
    weights :: W
    topology :: Topo
end

PolarWaveVectorGrid(FT::DataType; kwargs...) = PolarWaveVectorGrid(CPU(), FT; kwargs...)

function PolarWaveVectorGrid(arch::AbstractArchitecture = CPU(),
                             FT::DataType = Float64;
                             κ = nothing,
                             φ = nothing,
                             κ_faces = nothing,
                             φ_faces = nothing,
                             topology = (NoFlux(), Periodic()))
    κ === nothing && throw(ArgumentError("PolarWaveVectorGrid requires `κ`"))
    φ === nothing && throw(ArgumentError("PolarWaveVectorGrid requires `φ`"))
    topology = canonical_topology_tuple(topology, 2, "PolarWaveVectorGrid")
    κc_host = collect(FT, κ)
    φc_host = collect(FT, φ)
    validate_coordinate_centers("φ", φc_host)
    κf_host = κ_faces === nothing ?
        lower_bounded_centers_to_faces(κc_host, zero(FT)) :
        collect(FT, κ_faces)
    if φ_faces === nothing
        dφ = length(φc_host) == 1 ? FT(2pi) : FT(2pi / length(φc_host))
        φf_host = [φc_host[1] - dφ / 2 + (i - 1) * dφ for i in 1:(length(φc_host) + 1)]
    else
        φf_host = collect(FT, φ_faces)
    end
    validate_positive_coordinate_centers("κ", κc_host)
    validate_coordinate_faces("κ", κc_host, κf_host)
    validate_coordinate_faces_lower_bound("κ", κf_host, zero(FT))
    validate_coordinate_faces("φ", φc_host, φf_host)
    radial_weights = [(κf_host[m+1]^2 - κf_host[m]^2) / 2 for m in eachindex(κc_host)]
    angular_weights = diff(φf_host)
    weights_host = [radial_weights[m] * angular_weights[n]
                    for m in eachindex(κc_host), n in eachindex(φc_host)]
    κc = on_architecture(arch, κc_host)
    φc = on_architecture(arch, φc_host)
    κf = on_architecture(arch, κf_host)
    φf = on_architecture(arch, φf_host)
    weights = on_architecture(arch, weights_host)
    return PolarWaveVectorGrid{FT, typeof(arch), typeof(κc), typeof(φc),
                               typeof(κf), typeof(φf), typeof(weights), typeof(topology)}(
        arch, κc, φc, κf, φf, weights, topology)
end

architecture(g::PolarWaveVectorGrid) = g.architecture

coordinate_centers(g::PolarWaveVectorGrid, dim) = dim == 1 ? g.κ : g.φ
coordinate_faces(g::PolarWaveVectorGrid, dim) = dim == 1 ? g.κ_faces : g.φ_faces
coordinate_float_type(::PolarWaveVectorGrid{FT}) where FT = FT

@inline function k_components(g::PolarWaveVectorGrid, m, n)
    κ = g.κ[m]
    φ = g.φ[n]
    return (κ * cos(φ), κ * sin(φ))
end

@inline spectral_coordinates(g::PolarWaveVectorGrid, m, n) = (g.κ[m], g.φ[n])

@inline radial_wavenumber(g::PolarWaveVectorGrid, m, n) = g.κ[m]
@inline metric_jacobian(g::PolarWaveVectorGrid, m, n) = g.κ[m]

@inline function radial_first_moment_measure(k₁, k₂)
    return (k₂^3 - k₁^3) / 3
end

@inline function radial_second_moment_measure(k₁, k₂)
    return (k₂^4 - k₁^4) / 4
end

@inline function angular_first_moment_measures(φ₁, φ₂)
    return sin(φ₂) - sin(φ₁),
           cos(φ₁) - cos(φ₂)
end

@inline function angular_second_moment_measures(φ₁, φ₂)
    Δφ = φ₂ - φ₁
    sin2Δ = sin(2φ₂) - sin(2φ₁)
    cos² = Δφ / 2 + sin2Δ / 4
    sin² = Δφ / 2 - sin2Δ / 4
    cossin = (sin(φ₂)^2 - sin(φ₁)^2) / 2
    return cos², cossin, sin²
end

@inline function polar_first_moment_measures(k₁, k₂, φ₁, φ₂)
    radial = radial_first_moment_measure(k₁, k₂)
    angular_x, angular_y = angular_first_moment_measures(φ₁, φ₂)
    return radial * angular_x, radial * angular_y
end

@inline function polar_second_moment_measures(k₁, k₂, φ₁, φ₂)
    radial = radial_second_moment_measure(k₁, k₂)
    angular_xx, angular_xy, angular_yy = angular_second_moment_measures(φ₁, φ₂)
    return radial * angular_xx, radial * angular_xy, radial * angular_yy
end

@inline function spectral_first_moment_measures(g::PolarWaveVectorGrid, m, n)
    return polar_first_moment_measures(g.κ_faces[m], g.κ_faces[m+1],
                                       g.φ_faces[n], g.φ_faces[n+1])
end

@inline function spectral_second_moment_measures(g::PolarWaveVectorGrid, m, n)
    return polar_second_moment_measures(g.κ_faces[m], g.κ_faces[m+1],
                                        g.φ_faces[n], g.φ_faces[n+1])
end
