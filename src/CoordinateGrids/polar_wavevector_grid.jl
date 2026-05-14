struct PolarWaveVectorGrid{FT, K, Th, KF, ThF, W, Topo} <: AbstractSpectralGrid
    kappa :: K
    theta :: Th
    kappa_faces :: KF
    theta_faces :: ThF
    weights :: W
    topology :: Topo
end

function PolarWaveVectorGrid(::Type{FT}=Float64; kappa=nothing, theta=nothing,
                             κ=nothing,
                             θ=nothing,
                             kappa_faces=nothing,
                             theta_faces=nothing,
                             κ_faces=nothing,
                             θ_faces=nothing,
                             topology=(NoFlux(), Periodic())) where FT
    kappa = required_coordinate_keyword("kappa", kappa, "κ", κ)
    theta = required_coordinate_keyword("theta", theta, "θ", θ)
    kappa_faces = optional_coordinate_keyword("kappa_faces", kappa_faces, "κ_faces", κ_faces)
    theta_faces = optional_coordinate_keyword("theta_faces", theta_faces, "θ_faces", θ_faces)
    topology = canonical_topology_tuple(topology, 2, "PolarWaveVectorGrid")
    kc = collect(FT, kappa)
    tc = collect(FT, theta)
    validate_coordinate_centers("theta", tc)
    kf = kappa_faces === nothing ?
        lower_bounded_centers_to_faces(kc, zero(FT)) :
        collect(FT, kappa_faces)
    if theta_faces === nothing
        dtheta = length(tc) == 1 ? FT(2pi) : FT(2pi / length(tc))
        tf = [tc[1] - dtheta / 2 + (i - 1) * dtheta for i in 1:(length(tc)+1)]
    else
        tf = collect(FT, theta_faces)
    end
    validate_positive_coordinate_centers("kappa", kc)
    validate_coordinate_faces("kappa", kc, kf)
    validate_coordinate_faces_lower_bound("kappa", kf, zero(FT))
    validate_coordinate_faces("theta", tc, tf)
    radial_weights = [(kf[m+1]^2 - kf[m]^2) / 2 for m in eachindex(kc)]
    angular_weights = diff(tf)
    weights = [radial_weights[m] * angular_weights[n] for m in eachindex(kc), n in eachindex(tc)]
    return PolarWaveVectorGrid{FT, typeof(kc), typeof(tc), typeof(kf), typeof(tf), typeof(weights), typeof(topology)}(
        kc, tc, kf, tf, weights, topology)
end

coordinate_centers(g::PolarWaveVectorGrid, dim) = dim == 1 ? g.kappa : g.theta
coordinate_faces(g::PolarWaveVectorGrid, dim) = dim == 1 ? g.kappa_faces : g.theta_faces
coordinate_float_type(::PolarWaveVectorGrid{FT}) where FT = FT

@inline function k_components(g::PolarWaveVectorGrid, m, n)
    kappa = g.kappa[m]
    theta = g.theta[n]
    return (kappa * cos(theta), kappa * sin(theta))
end

@inline radial_wavenumber(g::PolarWaveVectorGrid, m, n) = g.kappa[m]
@inline metric_jacobian(g::PolarWaveVectorGrid, m, n) = g.kappa[m]

@inline function radial_first_moment_measure(k₁, k₂)
    return (k₂^3 - k₁^3) / 3
end

@inline function radial_second_moment_measure(k₁, k₂)
    return (k₂^4 - k₁^4) / 4
end

@inline function angular_first_moment_measures(θ₁, θ₂)
    return sin(θ₂) - sin(θ₁),
           cos(θ₁) - cos(θ₂)
end

@inline function angular_second_moment_measures(θ₁, θ₂)
    Δθ = θ₂ - θ₁
    sin2Δ = sin(2θ₂) - sin(2θ₁)
    cos² = Δθ / 2 + sin2Δ / 4
    sin² = Δθ / 2 - sin2Δ / 4
    cossin = (sin(θ₂)^2 - sin(θ₁)^2) / 2
    return cos², cossin, sin²
end

@inline function polar_first_moment_measures(k₁, k₂, θ₁, θ₂)
    radial = radial_first_moment_measure(k₁, k₂)
    angular_x, angular_y = angular_first_moment_measures(θ₁, θ₂)
    return radial * angular_x, radial * angular_y
end

@inline function polar_second_moment_measures(k₁, k₂, θ₁, θ₂)
    radial = radial_second_moment_measure(k₁, k₂)
    angular_xx, angular_xy, angular_yy = angular_second_moment_measures(θ₁, θ₂)
    return radial * angular_xx, radial * angular_xy, radial * angular_yy
end

@inline function spectral_first_moment_measures(g::PolarWaveVectorGrid, m, n)
    return polar_first_moment_measures(g.kappa_faces[m], g.kappa_faces[m+1],
                                       g.theta_faces[n], g.theta_faces[n+1])
end

@inline function spectral_second_moment_measures(g::PolarWaveVectorGrid, m, n)
    return polar_second_moment_measures(g.kappa_faces[m], g.kappa_faces[m+1],
                                        g.theta_faces[n], g.theta_faces[n+1])
end
