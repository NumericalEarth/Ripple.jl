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
