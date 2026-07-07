#####
##### Carrier dispersion relation ω(κ, h) and its κ-derivatives
#####
##### These scalar utilities underpin the Onuki & Fujiwara (2026) narrow-band
##### amplitude model. The carrier wavenumber κ is a fixed parameter (not a
##### spectral coordinate); every model coefficient — the group velocity ω_κ,
##### the reconstitution parameter α, the vertical structure Φ(z) — is derived
##### from κ and the water depth h through the linear dispersion relation
#####
#####     ω² = g κ tanh(κ h) .
#####
##### All functions dispatch on `depth`: a positive `Number` selects the
##### finite-depth branch, `InfiniteDepth()` the closed-form deep-water limit.

"""
    carrier_frequency(κ, depth; gravity=9.81)

Carrier angular frequency `ω` from the linear dispersion relation
`ω² = g κ tanh(κ h)` at carrier wavenumber `κ` and water `depth` (`h`, a
positive `Number`, or `InfiniteDepth()` for the deep-water limit `ω = √(gκ)`).
"""
@inline carrier_frequency(κ, h::Number; gravity=9.81) = sqrt(gravity * κ * tanh(κ * h))
@inline carrier_frequency(κ, ::InfiniteDepth; gravity=9.81) = sqrt(gravity * κ)

"""
    carrier_group_velocity(κ, depth; gravity=9.81)

Carrier group velocity `ω_κ = dω/dκ` (the scalar group speed at the carrier),

    ω_κ = (ω / 2κ) (1 + 2κh / sinh(2κh)) ,

which reduces to `ω/2κ` in deep water and to the phase speed `ω/κ` in the
shallow-water limit.
"""
@inline function carrier_group_velocity(κ, h::Number; gravity=9.81)
    ω = carrier_frequency(κ, h; gravity)
    μ = κ * h
    return (ω / 2κ) * (1 + 2μ / sinh(2μ))
end

@inline carrier_group_velocity(κ, depth::InfiniteDepth; gravity=9.81) =
    carrier_frequency(κ, depth; gravity) / 2κ

"""
    carrier_frequency_curvature(κ, depth; gravity=9.81)

Curvature `ω_κκ = d²ω/dκ²` of the dispersion relation, obtained by
differentiating `2 ω ω_κ = g[tanh μ + μ sech²μ]` (with `μ = κh`) a second time:

    ω_κκ = (g h sech²μ (1 - μ tanh μ) - ω_κ²) / ω .

In deep water `ω_κκ → -ω / 4κ²`.
"""
@inline function carrier_frequency_curvature(κ, h::Number; gravity=9.81)
    ω = carrier_frequency(κ, h; gravity)
    ω_κ = carrier_group_velocity(κ, h; gravity)
    μ = κ * h
    sech²μ = 1 / cosh(μ)^2
    return (gravity * h * sech²μ * (1 - μ * tanh(μ)) - ω_κ^2) / ω
end

@inline carrier_frequency_curvature(κ, depth::InfiniteDepth; gravity=9.81) =
    -carrier_frequency(κ, depth; gravity) / (4κ^2)

"""
    reconstitution_parameter(κ, depth; gravity=9.81)

Reconstitution parameter (Thomas & Yamada 2018; Onuki & Fujiwara 2026, eq. 2.5)

    α = (κ ω_κκ - ω_κ) / (4 κ² ω_κ) ,

used to reconstitute the exact linear dispersion relation from its narrow-band
Taylor expansion about `κ`. Note that `α < 0` and is independent of `gravity`
(it is a ratio of `g`-proportional frequency derivatives). The deep-water limit
is `α → -3/(8κ²)` and the shallow-water limit is `α → -1/(4κ²)`.
"""
@inline function reconstitution_parameter(κ, depth; gravity=9.81)
    ω_κ  = carrier_group_velocity(κ, depth; gravity)
    ω_κκ = carrier_frequency_curvature(κ, depth; gravity)
    return (κ * ω_κκ - ω_κ) / (4κ^2 * ω_κ)
end

"""
    vertical_structure_constant(κ, depth; gravity=9.81)

Normalization constant `C` of the vertical structure function `Φ(z)`, with

    C² = g κ / (ω ω_κ) ,

chosen so that `∫_{-h}^{0} Φ² dz = 1`. In deep water `C = √(2κ)`.
"""
@inline function vertical_structure_constant(κ, depth; gravity=9.81)
    ω   = carrier_frequency(κ, depth; gravity)
    ω_κ = carrier_group_velocity(κ, depth; gravity)
    return sqrt(gravity * κ / (ω * ω_κ))
end

"""
    vertical_structure(z, κ, depth; gravity=9.81)

Vertical structure function of the carrier wave,

    Φ(z) = C cosh(κ(z + h)) / cosh(κ h) ,

normalized to `∫_{-h}^{0} Φ² dz = 1`. In deep water `Φ(z) = C exp(κ z)`.
"""
@inline function vertical_structure(z, κ, h::Number; gravity=9.81)
    C = vertical_structure_constant(κ, h; gravity)
    return C * cosh(κ * (z + h)) / cosh(κ * h)
end

@inline function vertical_structure(z, κ, depth::InfiniteDepth; gravity=9.81)
    C = vertical_structure_constant(κ, depth; gravity)
    return C * exp(κ * z)
end

"""
    vertical_structure_derivative(z, κ, depth; gravity=9.81)

Vertical derivative `Φ_z(z)` of [`vertical_structure`](@ref),

    Φ_z(z) = C κ sinh(κ(z + h)) / cosh(κ h) .

In deep water `Φ_z = κ Φ`, so `Φ_z² = κ² Φ²`.
"""
@inline function vertical_structure_derivative(z, κ, h::Number; gravity=9.81)
    C = vertical_structure_constant(κ, h; gravity)
    return C * κ * sinh(κ * (z + h)) / cosh(κ * h)
end

@inline vertical_structure_derivative(z, κ, depth::InfiniteDepth; gravity=9.81) =
    κ * vertical_structure(z, κ, depth; gravity)

#####
##### NarrowBandDispersion: a precomputed bundle of carrier coefficients
#####

"""
    NarrowBandDispersion{FT, D}

Precomputed carrier coefficients for the Onuki & Fujiwara (2026) narrow-band
amplitude model: the frequency `ω`, its κ-derivatives `ω_κ` and `ω_κκ`, the
reconstitution parameter `α`, and the vertical-structure constant `C`, all
derived from the carrier wavenumber `κ` and water `depth`.
"""
struct NarrowBandDispersion{FT, D}
    κ :: FT        # carrier wavenumber
    depth :: D     # water depth (a positive number or InfiniteDepth())
    gravity :: FT  # gravitational acceleration
    ω :: FT        # carrier frequency
    ω_κ :: FT      # carrier group velocity dω/dκ
    ω_κκ :: FT     # dispersion curvature d²ω/dκ²
    α :: FT        # reconstitution parameter
    C :: FT        # vertical-structure normalization
end

@inline _depth_float(::InfiniteDepth, FT) = InfiniteDepth()
@inline _depth_float(h::Number, FT) = convert(FT, h)

"""
    NarrowBandDispersion(κ, depth; gravity=9.81)

Construct a `NarrowBandDispersion` from a carrier wavenumber `κ` and water
`depth` (a positive `Number` or `InfiniteDepth()`). Validates that `κ` and
`depth` are positive and that the reconstituted operator `1 + α κ²` is strictly
positive, so the screened-Poisson inversion of `[1 + α(∇² + κ²)]` is nonsingular
(see [`screened_poisson_symbol`](@ref)).

```jldoctest
julia> using Ripple

julia> d = NarrowBandDispersion(0.1, InfiniteDepth());

julia> d.α ≈ -3 / (8 * 0.1^2)
true

julia> 1 + d.α * d.κ^2 ≈ 5//8
true
```
"""
function NarrowBandDispersion(κ, depth; gravity=9.81)
    κ > 0 || throw(ArgumentError("carrier wavenumber κ must be positive; got κ = $κ"))
    depth isa InfiniteDepth || depth > 0 ||
        throw(ArgumentError("water depth must be positive or `InfiniteDepth()`; got depth = $depth"))

    # The float type follows the physical inputs (κ and depth); `gravity` is
    # converted to match rather than widening the type (its default is Float64).
    FT = depth isa InfiniteDepth ?
        float(typeof(κ)) :
        float(promote_type(typeof(κ), typeof(depth)))

    κ = convert(FT, κ)
    gravity = convert(FT, gravity)
    depth = _depth_float(depth, FT)

    ω    = carrier_frequency(κ, depth; gravity)
    ω_κ  = carrier_group_velocity(κ, depth; gravity)
    ω_κκ = carrier_frequency_curvature(κ, depth; gravity)
    α    = (κ * ω_κκ - ω_κ) / (4κ^2 * ω_κ)
    C    = vertical_structure_constant(κ, depth; gravity)

    1 + α * κ^2 > 0 ||
        throw(ArgumentError("screened-Poisson operator [1 + α(∇² + κ²)] is singular: " *
                            "1 + ακ² = $(1 + α * κ^2) ≤ 0"))

    return NarrowBandDispersion{FT, typeof(depth)}(κ, depth, gravity, ω, ω_κ, ω_κκ, α, C)
end

"""
    screened_poisson_symbol(dispersion::NarrowBandDispersion)

The scalar symbol `m` of the screened-Poisson equation that diagnoses the
amplitude `A` from the reconstituted amplitude `G`. Rearranging the diagnostic
relation `[1 + α(∇² + κ²)] A = G` into the generalized-Poisson form
`(∇² + m) A = G / α` gives

    m = (1 + α κ²) / α  <  0 ,

which is passed to `Oceananigans.Solvers.solve!`. Because `α < 0` and
`1 + ακ² > 0`, `m` is strictly negative, so the operator is symmetric positive
definite and the `k = 0` null mode is never triggered.
"""
@inline screened_poisson_symbol(d::NarrowBandDispersion) = (1 + d.α * d.κ^2) / d.α

Base.summary(d::NarrowBandDispersion) =
    string("NarrowBandDispersion(κ=", d.κ, ", depth=", d.depth, ")")

function Base.show(io::IO, d::NarrowBandDispersion)
    print(io, summary(d), ":", '\n',
          "├── ω:    ", d.ω, '\n',
          "├── ω_κ:  ", d.ω_κ, '\n',
          "├── ω_κκ: ", d.ω_κκ, '\n',
          "├── α:    ", d.α, '\n',
          "└── C:    ", d.C)
end
