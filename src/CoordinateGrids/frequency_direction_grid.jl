struct FrequencyDirectionGrid{FT, F, Th, FF, ThF, K, KF, W, Topo} <: AbstractSpectralGrid
    frequency :: F
    theta :: Th
    frequency_faces :: FF
    theta_faces :: ThF
    kappa :: K
    kappa_faces :: KF
    weights :: W
    topology :: Topo
end

function FrequencyDirectionGrid(::Type{FT}=Float64; frequency=nothing, theta=nothing,
                                f=nothing,
                                θ=nothing,
                                frequency_faces=nothing,
                                theta_faces=nothing,
                                f_faces=nothing,
                                θ_faces=nothing,
                                gravity=FT(9.81),
                                topology=(Bounded(), Periodic())) where FT
    frequency = required_coordinate_keyword("frequency", frequency, "f", f)
    theta = required_coordinate_keyword("theta", theta, "θ", θ)
    frequency_faces = optional_coordinate_keyword("frequency_faces", frequency_faces, "f_faces", f_faces)
    theta_faces = optional_coordinate_keyword("theta_faces", theta_faces, "θ_faces", θ_faces)
    topology = canonical_topology_tuple(topology, 2, "FrequencyDirectionGrid")
    fc = collect(FT, frequency)
    tc = collect(FT, theta)
    validate_coordinate_centers("theta", tc)
    ff = frequency_faces === nothing ?
        lower_bounded_centers_to_faces(fc, zero(FT)) :
        collect(FT, frequency_faces)
    if theta_faces === nothing
        dtheta = length(tc) == 1 ? FT(2pi) : FT(2pi / length(tc))
        tf = [tc[1] - dtheta / 2 + (i - 1) * dtheta for i in 1:(length(tc)+1)]
    else
        tf = collect(FT, theta_faces)
    end
    gravity > 0 || throw(ArgumentError("gravity must be positive"))
    validate_positive_coordinate_centers("frequency", fc)
    validate_coordinate_faces("frequency", fc, ff)
    validate_coordinate_faces_lower_bound("frequency", ff, zero(FT))
    validate_coordinate_faces("theta", tc, tf)

    frequency_to_kappa(f) = (2FT(pi) * f)^2 / gravity
    kc = frequency_to_kappa.(fc)
    kf = frequency_to_kappa.(ff)
    radial_weights = [(kf[m+1]^2 - kf[m]^2) / 2 for m in eachindex(kc)]
    angular_weights = diff(tf)
    weights = [radial_weights[m] * angular_weights[n] for m in eachindex(kc), n in eachindex(tc)]

    return FrequencyDirectionGrid{FT, typeof(fc), typeof(tc), typeof(ff), typeof(tf),
                                  typeof(kc), typeof(kf), typeof(weights), typeof(topology)}(
        fc, tc, ff, tf, kc, kf, weights, topology)
end

coordinate_centers(g::FrequencyDirectionGrid, dim) = dim == 1 ? g.frequency : g.theta
coordinate_faces(g::FrequencyDirectionGrid, dim) = dim == 1 ? g.frequency_faces : g.theta_faces
coordinate_float_type(::FrequencyDirectionGrid{FT}) where FT = FT

@inline function k_components(g::FrequencyDirectionGrid, m, n)
    kappa = g.kappa[m]
    theta = g.theta[n]
    return (kappa * cos(theta), kappa * sin(theta))
end

@inline radial_wavenumber(g::FrequencyDirectionGrid, m, n) = g.kappa[m]
@inline metric_jacobian(g::FrequencyDirectionGrid, m, n) = g.kappa[m]

@inline function spectral_first_moment_measures(g::FrequencyDirectionGrid, m, n)
    return polar_first_moment_measures(g.kappa_faces[m], g.kappa_faces[m+1],
                                       g.theta_faces[n], g.theta_faces[n+1])
end

@inline function spectral_second_moment_measures(g::FrequencyDirectionGrid, m, n)
    return polar_second_moment_measures(g.kappa_faces[m], g.kappa_faces[m+1],
                                        g.theta_faces[n], g.theta_faces[n+1])
end
