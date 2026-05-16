abstract type AbstractCoordinateGrid end
abstract type AbstractSpectralGrid <: AbstractCoordinateGrid end

function centers_to_faces(centers::AbstractVector)
    N = length(centers)
    N == 0 && throw(ArgumentError("coordinate vectors must be non-empty"))
    FT = eltype(float.(centers))
    faces = Vector{FT}(undef, N + 1)
    if N == 1
        c = FT(centers[1])
        faces[1] = c - FT(0.5)
        faces[2] = c + FT(0.5)
    else
        for i in 2:N
            faces[i] = (centers[i-1] + centers[i]) / 2
        end
        faces[1] = centers[1] - (faces[2] - centers[1])
        faces[end] = centers[end] + (centers[end] - faces[end-1])
    end
    return faces
end

function lower_bounded_centers_to_faces(centers::AbstractVector, lower)
    faces = centers_to_faces(centers)
    faces[1] = max(faces[1], lower)
    return faces
end

function required_coordinate_keyword(name, value, alias_name, alias_value)
    if value === nothing
        alias_value === nothing &&
            throw(ArgumentError("coordinate grid construction requires `$name` or `$alias_name`"))
        return alias_value
    elseif alias_value !== nothing
        throw(ArgumentError("provide either `$name` or `$alias_name`, not both"))
    end

    return value
end

function optional_coordinate_keyword(name, value, alias_name, alias_value)
    value === nothing && return alias_value
    alias_value === nothing || throw(ArgumentError("provide either `$name` or `$alias_name`, not both"))
    return value
end

function validate_coordinate_centers(name, centers)
    length(centers) > 0 || throw(ArgumentError("$name centers must be non-empty"))
    for i in 1:(length(centers)-1)
        centers[i+1] > centers[i] ||
            throw(ArgumentError("$name centers must be strictly increasing"))
    end
    return nothing
end

function validate_positive_coordinate_centers(name, centers)
    validate_coordinate_centers(name, centers)
    for value in centers
        value > 0 ||
            throw(ArgumentError("$name centers must be positive"))
    end
    return nothing
end

function validate_coordinate_faces(name, centers, faces)
    length(faces) == length(centers) + 1 ||
        throw(ArgumentError("$name faces must have length length(centers) + 1"))
    for i in 1:(length(faces)-1)
        faces[i+1] > faces[i] ||
            throw(ArgumentError("$name faces must be strictly increasing"))
    end
    for i in eachindex(centers)
        faces[i] <= centers[i] <= faces[i+1] ||
            throw(ArgumentError("$name center $i must lie inside its cell faces"))
    end
    return nothing
end

function validate_coordinate_faces_lower_bound(name, faces, lower)
    for value in faces
        value >= lower ||
            throw(ArgumentError("$name faces must be >= $lower"))
    end
    return nothing
end

function validate_coordinate_topology(name, centers, faces)
    validate_coordinate_centers(name, centers)
    validate_coordinate_faces(name, centers, faces)
    return nothing
end

coordinate_size(g::AbstractSpectralGrid) = (length(coordinate_centers(g, 1)), length(coordinate_centers(g, 2)))
coordinate_float_type(g::AbstractSpectralGrid) = Float64
coordinate_spacings(g::AbstractSpectralGrid, dim) = diff(coordinate_faces(g, dim))
spectral_cell_measures(g::AbstractSpectralGrid) = g.weights
spectral_cell_measure(g::AbstractSpectralGrid, m, n) = g.weights[m, n]
spectral_weights(g::AbstractSpectralGrid) = spectral_cell_measures(g)
spectral_weight(g::AbstractSpectralGrid, m, n) = spectral_cell_measure(g, m, n)

function spectral_first_moment_measures(g::AbstractSpectralGrid, m, n)
    kx, ky = k_components(g, m, n)
    weight = spectral_cell_measure(g, m, n)
    return kx * weight, ky * weight
end

function spectral_second_moment_measures(g::AbstractSpectralGrid, m, n)
    kx, ky = k_components(g, m, n)
    weight = spectral_cell_measure(g, m, n)
    return kx^2 * weight, kx * ky * weight, ky^2 * weight
end

#####
##### Pretty-printing helpers shared by all spectral grids. The visual
##### style mirrors `Oceananigans.Grids.RectilinearGrid`'s `show` method:
##### a one-line summary above a tree of per-axis dimension summaries.
#####

import Printf: @sprintf

# Spectral-axis topology is a property of the grid type; see
# `spectral_topology` on each `<:AbstractSpectralGrid`. Boundary
# conditions live separately in `g.boundary_conditions`.

axis_topology_label(::Oceananigans.Grids.Periodic) = "Periodic"
axis_topology_label(::Oceananigans.Grids.Bounded)  = "Bounded "
axis_topology_label(::Oceananigans.Grids.Flat)     = "Flat    "
axis_topology_label(t)                             = string(nameof(typeof(t)))

axis_domain_close(::Oceananigans.Grids.Bounded)  = "]"
axis_domain_close(::Oceananigans.Grids.Periodic) = ")"
axis_domain_close(::Oceananigans.Grids.Flat)     = "]"
axis_domain_close(_)                             = ")"

topology_name(t) = nameof(typeof(t))

axis_bc_label(::NoFlux)                      = "NoFlux"
axis_bc_label(::Oceananigans.Grids.Periodic) = "Periodic"
axis_bc_label(::Oceananigans.Grids.Bounded)  = "Bounded"
axis_bc_label(bc)                            = string(nameof(typeof(bc)))

axis_pretty(x) = @sprintf("%.4g", x)

function axis_domain_string(topo, faces)
    return string("[", axis_pretty(first(faces)), ", ",
                       axis_pretty(last(faces)),
                       axis_domain_close(topo))
end

function axis_spacing_string(faces, name)
    Δ = diff(collect(faces))
    if length(Δ) > 0 && all(d -> isapprox(d, first(Δ); rtol=1e-10, atol=1e-14), Δ)
        return @sprintf("regularly spaced with Δ%s=%s", name, axis_pretty(first(Δ)))
    else
        return @sprintf("variably spaced with min(Δ%s)=%s, max(Δ%s)=%s",
                        name, axis_pretty(minimum(Δ)),
                        name, axis_pretty(maximum(Δ)))
    end
end

function axis_summary(topo, faces, name; bc = nothing)
    base = string(axis_topology_label(topo), " ", name, " ∈ ",
                  axis_domain_string(topo, faces), " ",
                  axis_spacing_string(faces, name))
    return bc === nothing ? base : string(base, "  [", axis_bc_label(bc), " BC]")
end

# Build the parametric type signature (FT, T1, T2) used in summary lines.
# Topology is the genuine grid topology returned by `spectral_topology(g)`;
# BC markers are reported separately.
function spectral_type_signature(g::AbstractSpectralGrid)
    FT = coordinate_float_type(g)
    topo_names = map(topology_name, spectral_topology(g))
    return string("{", FT, ", ", topo_names[1], ", ", topo_names[2], "}")
end

spectral_size_string(g::AbstractSpectralGrid) =
    string(coordinate_size(g)[1], "×", coordinate_size(g)[2])

function _print_spectral_axes(io::IO, axes::Tuple{Vararg{String}})
    n = length(axes)
    for i in 1:n
        prefix = i == n ? "└── " : "├── "
        write(io, prefix, axes[i])
        i < n && write(io, '\n')
    end
end
