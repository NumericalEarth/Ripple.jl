struct FrequencyDirectionGrid{FT, A, F, Φ, FF, ΦF, K, KF, W, Topo} <: AbstractSpectralGrid
    architecture :: A
    frequency :: F
    φ :: Φ
    frequency_faces :: FF
    φ_faces :: ΦF
    κ :: K
    κ_faces :: KF
    weights :: W
    topology :: Topo
end

FrequencyDirectionGrid(FT::DataType; kwargs...) = FrequencyDirectionGrid(CPU(), FT; kwargs...)

function FrequencyDirectionGrid(arch::AbstractArchitecture = CPU(),
                                FT::DataType = Float64;
                                frequency = nothing,
                                φ = nothing,
                                frequency_faces = nothing,
                                φ_faces = nothing,
                                gravity = FT(9.81),
                                topology = (Bounded(), Periodic()))
    frequency === nothing && throw(ArgumentError("FrequencyDirectionGrid requires `frequency`"))
    φ === nothing && throw(ArgumentError("FrequencyDirectionGrid requires `φ`"))
    topology = canonical_topology_tuple(topology, 2, "FrequencyDirectionGrid")
    fc_host = collect(FT, frequency)
    φc_host = collect(FT, φ)
    validate_coordinate_centers("φ", φc_host)
    ff_host = frequency_faces === nothing ?
        lower_bounded_centers_to_faces(fc_host, zero(FT)) :
        collect(FT, frequency_faces)
    if φ_faces === nothing
        dφ = length(φc_host) == 1 ? FT(2pi) : FT(2pi / length(φc_host))
        φf_host = [φc_host[1] - dφ / 2 + (i - 1) * dφ for i in 1:(length(φc_host) + 1)]
    else
        φf_host = collect(FT, φ_faces)
    end
    gravity > 0 || throw(ArgumentError("gravity must be positive"))
    validate_positive_coordinate_centers("frequency", fc_host)
    validate_coordinate_faces("frequency", fc_host, ff_host)
    validate_coordinate_faces_lower_bound("frequency", ff_host, zero(FT))
    validate_coordinate_faces("φ", φc_host, φf_host)

    frequency_to_κ(f) = (2FT(pi) * f)^2 / gravity
    κc_host = frequency_to_κ.(fc_host)
    κf_host = frequency_to_κ.(ff_host)
    radial_weights = [(κf_host[m+1]^2 - κf_host[m]^2) / 2 for m in eachindex(κc_host)]
    angular_weights = diff(φf_host)
    weights_host = [radial_weights[m] * angular_weights[n]
                    for m in eachindex(κc_host), n in eachindex(φc_host)]

    fc = on_architecture(arch, fc_host)
    φc = on_architecture(arch, φc_host)
    ff = on_architecture(arch, ff_host)
    φf = on_architecture(arch, φf_host)
    κc = on_architecture(arch, κc_host)
    κf = on_architecture(arch, κf_host)
    weights = on_architecture(arch, weights_host)

    return FrequencyDirectionGrid{FT, typeof(arch), typeof(fc), typeof(φc),
                                  typeof(ff), typeof(φf), typeof(κc), typeof(κf),
                                  typeof(weights), typeof(topology)}(
        arch, fc, φc, ff, φf, κc, κf, weights, topology)
end

architecture(g::FrequencyDirectionGrid) = g.architecture

coordinate_centers(g::FrequencyDirectionGrid, dim) = dim == 1 ? g.frequency : g.φ
coordinate_faces(g::FrequencyDirectionGrid, dim) = dim == 1 ? g.frequency_faces : g.φ_faces
coordinate_float_type(::FrequencyDirectionGrid{FT}) where FT = FT

@inline function k_components(g::FrequencyDirectionGrid, m, n)
    κ = g.κ[m]
    φ = g.φ[n]
    return (κ * cos(φ), κ * sin(φ))
end

@inline radial_wavenumber(g::FrequencyDirectionGrid, m, n) = g.κ[m]
@inline metric_jacobian(g::FrequencyDirectionGrid, m, n) = g.κ[m]

@inline function spectral_first_moment_measures(g::FrequencyDirectionGrid, m, n)
    return polar_first_moment_measures(g.κ_faces[m], g.κ_faces[m+1],
                                       g.φ_faces[n], g.φ_faces[n+1])
end

@inline function spectral_second_moment_measures(g::FrequencyDirectionGrid, m, n)
    return polar_second_moment_measures(g.κ_faces[m], g.κ_faces[m+1],
                                        g.φ_faces[n], g.φ_faces[n+1])
end

function Base.summary(g::FrequencyDirectionGrid)
    return string(spectral_size_string(g),
                  " FrequencyDirectionGrid", spectral_type_signature(g),
                  " on ", summary(architecture(g)))
end

function Base.show(io::IO, g::FrequencyDirectionGrid)
    println(io, summary(g))
    κ_line = string("derived κ ∈ ", axis_domain_string(g.topology[1], g.κ_faces),
                    " via deep-water dispersion")
    _print_spectral_axes(io, (axis_summary(g.topology[1], g.frequency_faces, "f"),
                              axis_summary(g.topology[2], g.φ_faces, "φ"),
                              κ_line))
end
