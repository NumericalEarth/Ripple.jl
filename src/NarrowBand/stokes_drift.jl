import Oceananigans.Architectures: architecture
import Oceananigans.Fields: CenterField, interior
import Oceananigans.Utils: launch!
import Oceananigans.BoundaryConditions: fill_halo_regions!
import KernelAbstractions: @kernel, @index

#####
##### Stokes drift diagnosed from the amplitude A (Onuki & Fujiwara 2026, eq. 2.8)
#####
##### The Stokes drift is separable in the vertical,
#####
#####     Uˢᵢ(x, y, z) = Φ²(z) Sᵢ(x, y) + Φ_z²(z) Tᵢ(x, y),      Uˢ_z = 0,
#####
##### with the horizontal functionals quadratic in A:
#####
#####     Sᵢ = (2/ω) Σⱼ (Aʳ,ⱼ Aⁱ,ᵢⱼ − Aⁱ,ⱼ Aʳ,ᵢⱼ) = (2/ω) Im(A†,ⱼ A,ᵢⱼ),
#####     Tᵢ = (2/ω) (Aʳ Aⁱ,ᵢ − Aⁱ Aʳ,ᵢ)         = (2/ω) Im(A† A,ᵢ).
#####
##### For a plane wave A = a e^{iκx} these reduce to S₁ = 2a²κ³/ω, T₁ = 2a²κ/ω,
##### and Uˢ₁ = (2a²κ/ω)(κ²Φ² + Φ_z²), which recovers the classical finite-depth
##### Stokes profile (deep water: a₀² ω κ e^{2κz} with surface amplitude a₀).

# Centered finite differences of a real component on a uniform grid.
@inline _fx(c, i, j, k, Δx) = @inbounds (c[i+1, j, k] - c[i-1, j, k]) / 2Δx
@inline _fy(c, i, j, k, Δy) = @inbounds (c[i, j+1, k] - c[i, j-1, k]) / 2Δy
@inline _fxx(c, i, j, k, Δx) = @inbounds (c[i+1, j, k] - 2c[i, j, k] + c[i-1, j, k]) / Δx^2
@inline _fyy(c, i, j, k, Δy) = @inbounds (c[i, j+1, k] - 2c[i, j, k] + c[i, j-1, k]) / Δy^2
@inline _fxy(c, i, j, k, Δx, Δy) =
    @inbounds (c[i+1, j+1, k] - c[i+1, j-1, k] - c[i-1, j+1, k] + c[i-1, j-1, k]) / (4Δx * Δy)

@kernel function _stokes_functionals!(S1, S2, T1, T2, Ar, Ai, coef, Δx, Δy)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        arx = _fx(Ar, i, j, k, Δx); ary = _fy(Ar, i, j, k, Δy)
        aix = _fx(Ai, i, j, k, Δx); aiy = _fy(Ai, i, j, k, Δy)
        arxx = _fxx(Ar, i, j, k, Δx); aryy = _fyy(Ar, i, j, k, Δy); arxy = _fxy(Ar, i, j, k, Δx, Δy)
        aixx = _fxx(Ai, i, j, k, Δx); aiyy = _fyy(Ai, i, j, k, Δy); aixy = _fxy(Ai, i, j, k, Δx, Δy)

        S1[i, j, k] = coef * ((arx * aixx - aix * arxx) + (ary * aixy - aiy * arxy))
        S2[i, j, k] = coef * ((arx * aixy - aix * arxy) + (ary * aiyy - aiy * aryy))
        T1[i, j, k] = coef * (Ar[i, j, k] * aix - Ai[i, j, k] * arx)
        T2[i, j, k] = coef * (Ar[i, j, k] * aiy - Ai[i, j, k] * ary)
    end
end

"""
    stokes_functionals!(S1, S2, T1, T2, model::NarrowBandWaveModel)

Compute the horizontal Stokes functionals `(S₁, S₂, T₁, T₂)` from the model's
current diagnostic amplitude, writing into the provided center fields.
"""
function stokes_functionals!(S1, S2, T1, T2, model::NarrowBandWaveModel)
    grid = model.grid
    fill_halo_regions!(model.Ar)
    fill_halo_regions!(model.Ai)
    FT = eltype(grid)
    coef = convert(FT, 2 / model.dispersion.ω)
    Δx = convert(FT, grid.Lx / grid.Nx)
    Δy = convert(FT, grid.Ly / grid.Ny)
    launch!(architecture(grid), grid, :xyz, _stokes_functionals!,
            S1, S2, T1, T2, model.Ar, model.Ai, coef, Δx, Δy)
    for f in (S1, S2, T1, T2)
        fill_halo_regions!(f)
    end
    return nothing
end

"""
    stokes_functionals(model::NarrowBandWaveModel)

Return freshly allocated center fields `(S₁, S₂, T₁, T₂)` of the Stokes
functionals, from which the 3-D Stokes drift is `Uˢᵢ = Φ² Sᵢ + Φ_z² Tᵢ`.
"""
function stokes_functionals(model::NarrowBandWaveModel)
    S1 = CenterField(model.grid); S2 = CenterField(model.grid)
    T1 = CenterField(model.grid); T2 = CenterField(model.grid)
    stokes_functionals!(S1, S2, T1, T2, model)
    return (S1=S1, S2=S2, T1=T1, T2=T2)
end

"""
    surface_elevation_amplitude(model::NarrowBandWaveModel)

Return the surface-elevation envelope `a₀ = 2 ω C |A| / g` over the model
interior, mapping the complex amplitude `A` to the physical wave height
(Onuki & Fujiwara 2026, eq. 4.8).
"""
function surface_elevation_amplitude(model::NarrowBandWaveModel)
    d = model.dispersion
    scale = 2 * d.ω * d.C / d.gravity
    return scale .* abs.(amplitude(model))
end
