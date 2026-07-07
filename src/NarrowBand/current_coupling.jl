import Oceananigans
import Oceananigans.Architectures: architecture
import Oceananigans.Grids: topology, Flat, halo_size
import Oceananigans.Fields: XFaceField, YFaceField, CenterField, ZeroField, interior, set!
import Oceananigans.Operators: ∂xᶜᶜᶜ, ∂yᶜᶜᶜ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ
import Oceananigans.Utils: launch!
import Oceananigans.BoundaryConditions: fill_halo_regions!
import KernelAbstractions: @kernel, @index

#####
##### Prescribed-current coupling for the narrow-band model
#####
##### Given a (static) Lagrangian-mean current Uᴸ = (u, v) on a vertically
##### resolved grid, precompute everything the wave tendency needs:
#####   Ūx, Ūy, Ũx, Ũy   depth-weighted projections at velocity face locations
#####   veffx, veffy      effective transport velocity (κω_κ/ω)(Ū + Ũ) at faces
#####   source            compressibility source (κω_κ/ω)[∇·Ū + ½∇·Ũ] at centers
#####   Ūxᶜ, Ūyᶜ          Ū interpolated to centers (for the refraction operator R)
#####   Rr, Ri            scratch for R applied to the amplitude's real pair
##### These feed the Doppler flux divergence −∇·(vᵉᶠᶠ A) (Oceananigans tracer
##### advection), the pointwise compressibility source, and the refraction
##### operator R (wave_operators.jl).

struct NarrowBandPrescribedVelocities{U, V, G3, W, FX, FY, CF}
    u :: U               # 3-D Lagrangian-mean velocity (Face, Center, Center)
    v :: V               # 3-D Lagrangian-mean velocity (Center, Face, Center)
    grid3d :: G3          # the vertically resolved mean-flow grid
    weights :: W          # VerticalProjectionWeights
    Ūx :: FX ; Ūy :: FY   # depth-weighted Ū at (Face, Center)/(Center, Face)
    Ũx :: FX ; Ũy :: FY   # depth-weighted Ũ
    veffx :: FX ; veffy :: FY  # effective transport velocity at faces
    source :: CF          # compressibility source at centers
    Ūxᶜ :: CF ; Ūyᶜ :: CF  # Ū interpolated to centers (for R)
    Rr :: CF ; Ri :: CF   # refraction-operator scratch
end

# Compressibility source (κω_κ/ω)[∇·Ū + ½∇·Ũ] from face-located projections.
@kernel function _compressibility_source!(source, Ūx, Ūy, Ũx, Ũy, grid, coef)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        divŪ = ∂xᶜᶜᶜ(i, j, k, grid, Ūx) + ∂yᶜᶜᶜ(i, j, k, grid, Ūy)
        divŨ = ∂xᶜᶜᶜ(i, j, k, grid, Ũx) + ∂yᶜᶜᶜ(i, j, k, grid, Ũy)
        source[i, j, k] = coef * (divŪ + divŨ / 2)
    end
end

# Interpolate the face-located Ū onto cell centers for the refraction operator.
@kernel function _interpolate_Ū_to_centers!(Ūxᶜ, Ūyᶜ, Ūx, Ūy, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds Ūxᶜ[i, j, k] = ℑxᶜᵃᵃ(i, j, k, grid, Ūx)
    @inbounds Ūyᶜ[i, j, k] = ℑyᵃᶜᵃ(i, j, k, grid, Ūy)
end

function _materialize_velocity(fun, grid3d, ::Val{:u})
    f = XFaceField(grid3d)
    set!(f, fun)
    fill_halo_regions!(f)
    return f
end
function _materialize_velocity(fun, grid3d, ::Val{:v})
    f = YFaceField(grid3d)
    set!(f, fun)
    fill_halo_regions!(f)
    return f
end

"""
    build_narrow_band_velocities(velocities, grid3d, wave_grid, dispersion)

Materialize a [`NarrowBandPrescribedVelocities`](@ref) from a `(u, v)`
NamedTuple of vertically resolved Lagrangian-mean velocity components (Oceananigans
`Field`s or functions of `(x, y, z)`) on `grid3d`, projecting them onto the 2-D
`wave_grid` with the carrier's vertical weights.
"""
function build_narrow_band_velocities(velocities::NamedTuple, grid3d, wave_grid, dispersion)
    haskey(velocities, :u) && haskey(velocities, :v) ||
        throw(ArgumentError("narrow-band `velocities` must be a NamedTuple with fields `u` and `v`"))

    u3 = velocities.u isa Function ? _materialize_velocity(velocities.u, grid3d, Val(:u)) : velocities.u
    v3 = velocities.v isa Function ? _materialize_velocity(velocities.v, grid3d, Val(:v)) : velocities.v

    weights = VerticalProjectionWeights(dispersion, grid3d)
    Nz = size(grid3d, 3)
    arch = architecture(wave_grid)

    Ūx = XFaceField(wave_grid); Ūy = YFaceField(wave_grid)
    Ũx = XFaceField(wave_grid); Ũy = YFaceField(wave_grid)
    project_velocity!(Ūx, u3, weights.Ū_weights, wave_grid, Nz)
    project_velocity!(Ūy, v3, weights.Ū_weights, wave_grid, Nz)
    project_velocity!(Ũx, u3, weights.Ũ_weights, wave_grid, Nz)
    project_velocity!(Ũy, v3, weights.Ũ_weights, wave_grid, Nz)
    for f in (Ūx, Ūy, Ũx, Ũy)
        fill_halo_regions!(f)
    end

    FT = eltype(wave_grid)
    coef = convert(FT, dispersion.κ * dispersion.ω_κ / dispersion.ω)
    veffx = XFaceField(wave_grid); veffy = YFaceField(wave_grid)
    interior(veffx) .= coef .* (interior(Ūx) .+ interior(Ũx))
    interior(veffy) .= coef .* (interior(Ūy) .+ interior(Ũy))
    fill_halo_regions!(veffx); fill_halo_regions!(veffy)

    source = CenterField(wave_grid)
    launch!(arch, wave_grid, :xyz, _compressibility_source!,
            source, Ūx, Ūy, Ũx, Ũy, wave_grid, coef)
    fill_halo_regions!(source)

    Ūxᶜ = CenterField(wave_grid); Ūyᶜ = CenterField(wave_grid)
    launch!(arch, wave_grid, :xyz, _interpolate_Ū_to_centers!, Ūxᶜ, Ūyᶜ, Ūx, Ūy, wave_grid)
    fill_halo_regions!(Ūxᶜ); fill_halo_regions!(Ūyᶜ)

    Rr = CenterField(wave_grid); Ri = CenterField(wave_grid)

    return NarrowBandPrescribedVelocities(u3, v3, grid3d, weights,
                                          Ūx, Ūy, Ũx, Ũy, veffx, veffy, source,
                                          Ūxᶜ, Ūyᶜ, Rr, Ri)
end

# The velocity NamedTuple that Oceananigans tracer advection (div_Uc) expects.
# The vertical component is a ZeroField: the wave grid is Flat in z, so the
# z-flux divergence vanishes.
@inline effective_transport_velocities(vel::NarrowBandPrescribedVelocities) =
    (u = vel.veffx, v = vel.veffy, w = ZeroField())
