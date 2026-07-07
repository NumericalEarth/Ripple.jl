import Oceananigans
import Oceananigans.Architectures: architecture, on_architecture
import Oceananigans.Grids: znodes, Center
import Oceananigans.Fields: CenterField, interior
import Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ,
                               ℑxzᶠᵃᶜ, ℑxyᶠᶜᵃ, ℑyzᵃᶠᶜ, ℑxyᶜᶠᵃ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶠ
import Oceananigans.StokesDrifts: ∂t_uˢ, ∂t_vˢ, ∂t_wˢ,
                                  x_curl_Uˢ_cross_U, y_curl_Uˢ_cross_U, z_curl_Uˢ_cross_U
import Oceananigans.Utils: launch!
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.TimeSteppers: time_step!, update_state!
import Adapt: adapt_structure, adapt
import KernelAbstractions: @kernel, @index

#####
##### Two-way coupling: NarrowBandWaveModel ⇄ Oceananigans NonhydrostaticModel
#####
##### The wave amplitude A supplies the Stokes drift that drives the
##### Craik–Leibovich vortex force in the current model; the current in turn
##### Doppler-transports and refracts A. The Stokes drift is separable,
#####     Uˢᵢ = Φ²(z) Sᵢ(x,y) + Φ_z²(z) Tᵢ(x,y),   Uˢ_z = 0,
##### so the pseudovorticity components the NonhydrostaticModel needs — the
##### vertical shear ∂z_uˢ, ∂z_vˢ and the horizontal Stokes vorticity
##### ζˢ = ∂x_vˢ − ∂y_uˢ — plus ∂t_uˢ, ∂t_vˢ are held as 3-D center fields and
##### refreshed from A each coupled step.

struct NarrowBandStokesDrift{ZU, ZV, ZE, TU, TV}
    ∂z_uˢ :: ZU   # vertical shear of uˢ (center)
    ∂z_vˢ :: ZV   # vertical shear of vˢ (center)
    ζˢ    :: ZE   # ∂x_vˢ − ∂y_uˢ, the vertical Stokes vorticity (center)
    ∂t_uˢ :: TU   # ∂ₜuˢ (center)
    ∂t_vˢ :: TV   # ∂ₜvˢ (center)
end

"""
    NarrowBandStokesDrift(grid)

Allocate the (zero-initialized) center fields that carry the narrow-band Stokes
drift's pseudovorticity and time-tendency into an Oceananigans
`NonhydrostaticModel(grid; stokes_drift=...)`. Populate them from a wave model
with [`refresh_stokes_drift!`](@ref) (done automatically by [`WaveCurrentModel`](@ref)).
"""
NarrowBandStokesDrift(grid) = NarrowBandStokesDrift(CenterField(grid), CenterField(grid),
                                                    CenterField(grid), CenterField(grid),
                                                    CenterField(grid))

adapt_structure(to, sd::NarrowBandStokesDrift) =
    NarrowBandStokesDrift(adapt(to, sd.∂z_uˢ), adapt(to, sd.∂z_vˢ), adapt(to, sd.ζˢ),
                          adapt(to, sd.∂t_uˢ), adapt(to, sd.∂t_vˢ))

# Time tendencies of the Stokes drift, interpolated to velocity faces (wˢ = 0).
@inline ∂t_uˢ(i, j, k, grid, sd::NarrowBandStokesDrift, time) = ℑxᶠᵃᵃ(i, j, k, grid, sd.∂t_uˢ)
@inline ∂t_vˢ(i, j, k, grid, sd::NarrowBandStokesDrift, time) = ℑyᵃᶠᵃ(i, j, k, grid, sd.∂t_vˢ)
@inline ∂t_wˢ(i, j, k, grid, sd::NarrowBandStokesDrift, time) = zero(grid)

# Vortex force (∇×Uˢ)×U with wˢ = 0. Center fields are interpolated to the
# staggered location of each component (mirrors Oceananigans' StokesDrift).
@inline function x_curl_Uˢ_cross_U(i, j, k, grid, sd::NarrowBandStokesDrift, U, time)
    wᶠᶜᶜ = ℑxzᶠᵃᶜ(i, j, k, grid, U.w)
    vᶠᶜᶜ = ℑxyᶠᶜᵃ(i, j, k, grid, U.v)
    ∂z_uˢ = ℑxᶠᵃᵃ(i, j, k, grid, sd.∂z_uˢ)
    ζˢ    = ℑxᶠᵃᵃ(i, j, k, grid, sd.ζˢ)
    return wᶠᶜᶜ * ∂z_uˢ - vᶠᶜᶜ * ζˢ
end

@inline function y_curl_Uˢ_cross_U(i, j, k, grid, sd::NarrowBandStokesDrift, U, time)
    wᶜᶠᶜ = ℑyzᵃᶠᶜ(i, j, k, grid, U.w)
    uᶜᶠᶜ = ℑxyᶜᶠᵃ(i, j, k, grid, U.u)
    ∂z_vˢ = ℑyᵃᶠᵃ(i, j, k, grid, sd.∂z_vˢ)
    ζˢ    = ℑyᵃᶠᵃ(i, j, k, grid, sd.ζˢ)
    return uᶜᶠᶜ * ζˢ + wᶜᶠᶜ * ∂z_vˢ
end

@inline function z_curl_Uˢ_cross_U(i, j, k, grid, sd::NarrowBandStokesDrift, U, time)
    uᶜᶜᶠ = ℑxzᶜᵃᶠ(i, j, k, grid, U.u)
    vᶜᶜᶠ = ℑyzᵃᶜᶠ(i, j, k, grid, U.v)
    ∂z_uˢ = ℑzᵃᵃᶠ(i, j, k, grid, sd.∂z_uˢ)
    ∂z_vˢ = ℑzᵃᵃᶠ(i, j, k, grid, sd.∂z_vˢ)
    return -vᶜᶜᶠ * ∂z_vˢ - uᶜᶜᶠ * ∂z_uˢ
end

#####
##### WaveCurrentModel — owns both models and the refresh scratch
#####

mutable struct WaveCurrentModel{W, O, SD, ZP, F2, F3, C}
    wave :: W                 # NarrowBandWaveModel
    ocean :: O                # Oceananigans NonhydrostaticModel with a NarrowBandStokesDrift
    stokes_drift :: SD        # the shared NarrowBandStokesDrift
    Φ² :: ZP ; Φz² :: ZP      # vertical profiles at ocean z-centers
    dΦ² :: ZP ; dΦz² :: ZP    # their z-derivatives
    S1 :: F2 ; S2 :: F2 ; T1 :: F2 ; T2 :: F2    # Stokes functionals (2-D)
    ζS :: F2 ; ζT :: F2                          # horizontal vorticities of S, T
    uˢ :: F3 ; vˢ :: F3                          # current Stokes drift (center, for ∂ₜ by FD)
    uˢ_prev :: F3 ; vˢ_prev :: F3
    clock :: C
end

# Vertical profile derivatives: (Φ²)' = 2ΦΦ_z and (Φ_z²)' = 2Φ_z(κ²Φ) = κ²(Φ²)'.
@inline _dΦ²(z, κ, depth) = 2 * vertical_structure(z, κ, depth) * vertical_structure_derivative(z, κ, depth)
@inline _dΦz²(z, κ, depth) = κ^2 * _dΦ²(z, κ, depth)

"""
    WaveCurrentModel(wave, ocean, stokes_drift)

Bundle a [`NarrowBandWaveModel`](@ref) and an Oceananigans `NonhydrostaticModel`
(built with the shared `stokes_drift::NarrowBandStokesDrift`) into a two-way
coupled model. The wave model must have been built with the ocean's velocities
(`velocities = (u = ocean.velocities.u, v = ocean.velocities.v)`) so its Doppler
transport tracks the current.
"""
function WaveCurrentModel(wave, ocean, stokes_drift::NarrowBandStokesDrift)
    grid3d = ocean.grid
    wave_grid = wave.grid
    d = wave.dispersion
    arch = architecture(grid3d)
    FT = eltype(grid3d)

    zc = znodes(grid3d, Center())
    Φ²   = on_architecture(arch, FT[vertical_structure(z, d.κ, d.depth)^2 for z in zc])
    Φz²  = on_architecture(arch, FT[vertical_structure_derivative(z, d.κ, d.depth)^2 for z in zc])
    dΦ²  = on_architecture(arch, FT[_dΦ²(z, d.κ, d.depth) for z in zc])
    dΦz² = on_architecture(arch, FT[_dΦz²(z, d.κ, d.depth) for z in zc])

    S1 = CenterField(wave_grid); S2 = CenterField(wave_grid)
    T1 = CenterField(wave_grid); T2 = CenterField(wave_grid)
    ζS = CenterField(wave_grid); ζT = CenterField(wave_grid)
    uˢ = CenterField(grid3d); vˢ = CenterField(grid3d)
    uˢ_prev = CenterField(grid3d); vˢ_prev = CenterField(grid3d)

    return WaveCurrentModel(wave, ocean, stokes_drift, Φ², Φz², dΦ², dΦz²,
                            S1, S2, T1, T2, ζS, ζT, uˢ, vˢ, uˢ_prev, vˢ_prev, ocean.clock)
end

# Horizontal Stokes vorticity ζS = ∂ₓS₂ − ∂_yS₁ (and likewise ζT), centered FD.
@kernel function _stokes_vorticity!(ζS, ζT, S1, S2, T1, T2, Δx, Δy)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ζS[i, j, k] = (S2[i+1, j, k] - S2[i-1, j, k]) / 2Δx - (S1[i, j+1, k] - S1[i, j-1, k]) / 2Δy
        ζT[i, j, k] = (T2[i+1, j, k] - T2[i-1, j, k]) / 2Δx - (T1[i, j+1, k] - T1[i, j-1, k]) / 2Δy
    end
end

# Fill the 3-D Stokes drift + pseudovorticity from the 2-D functionals and the
# vertical profiles: q(x,y,z) = Φ²(z) qS(x,y) + Φ_z²(z) qT(x,y).
@kernel function _assemble_stokes_3d!(uˢ, vˢ, ∂z_uˢ, ∂z_vˢ, ζˢ,
                                      S1, S2, T1, T2, ζS, ζT, Φ², Φz², dΦ², dΦz²)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        p = Φ²[k]; pz = Φz²[k]; dp = dΦ²[k]; dpz = dΦz²[k]
        uˢ[i, j, k]    = p * S1[i, j, 1] + pz * T1[i, j, 1]
        vˢ[i, j, k]    = p * S2[i, j, 1] + pz * T2[i, j, 1]
        ∂z_uˢ[i, j, k] = dp * S1[i, j, 1] + dpz * T1[i, j, 1]
        ∂z_vˢ[i, j, k] = dp * S2[i, j, 1] + dpz * T2[i, j, 1]
        ζˢ[i, j, k]    = p * ζS[i, j, 1] + pz * ζT[i, j, 1]
    end
end

"""
    refresh_stokes_drift!(model::WaveCurrentModel, Δt)

Recompute the Stokes functionals from the wave amplitude and assemble the 3-D
Stokes drift, its vertical shear and horizontal vorticity, and its time tendency
`∂ₜUˢ ≈ (Uˢ − Uˢ_prev)/Δt`. Pass `Δt = 0` to initialize the tendency to zero.
"""
function refresh_stokes_drift!(model::WaveCurrentModel, Δt)
    wave = model.wave
    wave_grid = wave.grid
    grid3d = model.ocean.grid
    arch = architecture(grid3d)
    sd = model.stokes_drift

    stokes_functionals!(model.S1, model.S2, model.T1, model.T2, wave)

    FT = eltype(wave_grid)
    Δx = convert(FT, wave_grid.Lx / wave_grid.Nx)
    Δy = convert(FT, wave_grid.Ly / wave_grid.Ny)
    launch!(architecture(wave_grid), wave_grid, :xyz, _stokes_vorticity!,
            model.ζS, model.ζT, model.S1, model.S2, model.T1, model.T2, Δx, Δy)
    fill_halo_regions!(model.ζS); fill_halo_regions!(model.ζT)

    # Save the previous Stokes drift for the finite-difference time tendency.
    interior(model.uˢ_prev) .= interior(model.uˢ)
    interior(model.vˢ_prev) .= interior(model.vˢ)

    launch!(arch, grid3d, :xyz, _assemble_stokes_3d!,
            model.uˢ, model.vˢ, sd.∂z_uˢ, sd.∂z_vˢ, sd.ζˢ,
            model.S1, model.S2, model.T1, model.T2, model.ζS, model.ζT,
            model.Φ², model.Φz², model.dΦ², model.dΦz²)

    if Δt > 0
        inv_Δt = convert(FT, 1 / Δt)
        interior(sd.∂t_uˢ) .= inv_Δt .* (interior(model.uˢ) .- interior(model.uˢ_prev))
        interior(sd.∂t_vˢ) .= inv_Δt .* (interior(model.vˢ) .- interior(model.vˢ_prev))
    else
        interior(sd.∂t_uˢ) .= 0
        interior(sd.∂t_vˢ) .= 0
    end

    for f in (sd.∂z_uˢ, sd.∂z_vˢ, sd.ζˢ, sd.∂t_uˢ, sd.∂t_vˢ)
        fill_halo_regions!(f)
    end
    return nothing
end

"""
    initialize_coupling!(model::WaveCurrentModel)

Synchronize the coupling after setting initial conditions: refresh the wave
model's depth-weighted current projections and the Stokes drift (with zero
initial `∂ₜUˢ`).
"""
function initialize_coupling!(model::WaveCurrentModel)
    refresh_narrow_band_velocities!(model.wave.velocities, model.wave.grid, model.wave.dispersion)
    update_state!(model.wave)
    refresh_stokes_drift!(model, 0)
    return model
end

"""
    coupled_time_step!(model::WaveCurrentModel, Δt)

Advance the coupled system by `Δt` with sequential (Lie) splitting: refresh the
wave model's Doppler transport from the current, step the wave amplitude, refresh
the Stokes drift from the new amplitude, then step the current with the updated
vortex force and `∂ₜUˢ`.
"""
function coupled_time_step!(model::WaveCurrentModel, Δt)
    refresh_narrow_band_velocities!(model.wave.velocities, model.wave.grid, model.wave.dispersion)
    time_step!(model.wave, Δt)
    refresh_stokes_drift!(model, Δt)
    time_step!(model.ocean, Δt)
    return model
end
