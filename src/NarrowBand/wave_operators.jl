import Oceananigans.Operators: ∇²ᶜᶜᶜ
import Oceananigans.Advection: div_Uc
import Oceananigans.Utils: launch!
import Oceananigans.Architectures: architecture
import Oceananigans.BoundaryConditions: fill_halo_regions!
import KernelAbstractions: @kernel, @index

#####
##### Wave-side operators for the narrow-band amplitude model
#####
##### The prognostic field is the reconstituted amplitude G ≡ [1 + α(∇ₕ² + κ²)]A;
##### the amplitude A is diagnosed each stage by inverting that operator (see
##### helmholtz_solve.jl). `reconstitute!` is the forward operator (A → G), used
##### to initialize G from a prescribed amplitude. The tendency ∂ₜG is assembled
##### term by term; in the wave-only regime (velocities = nothing) only the local
##### dispersion term survives.

# Forward reconstitution G = [1 + α(∇ₕ² + κ²)]A. Using αm = 1 + ακ² (m the
# screened-Poisson symbol), this is G = α(m A + ∇²A) — the exact inverse of the
# diagnostic solve, discretized with the same centered Laplacian the solver's
# eigenvalues represent, so `reconstitute!` and `solve_amplitude!` round-trip.
@kernel function _reconstitute!(G, A, grid, α, m)
    i, j, k = @index(Global, NTuple)
    @inbounds G[i, j, k] = α * (m * A[i, j, k] + ∇²ᶜᶜᶜ(i, j, k, grid, A))
end

"""
    reconstitute!(G, solver::NarrowBandHelmholtzSolver, A)

Apply the reconstitution operator `G = [1 + α(∇ₕ² + κ²)]A` to a real component
`A`, writing the result into `G`. The halo regions of `A` are filled first so
the centered Laplacian is well defined at the boundary.
"""
function reconstitute!(G, solver::NarrowBandHelmholtzSolver, A)
    grid = solver.grid
    fill_halo_regions!(A)
    launch!(architecture(grid), grid, :xyz, _reconstitute!, G, A, grid, solver.α, solver.m)
    return G
end

# Local dispersion tendency (wave-only regime, U = 0):
#     ∂ₜG = (i ω_κ / 2κα)(G - A) = iβ(G - A),   β = ω_κ / (2κα) < 0.
# In real components: ∂ₜGʳ = -β(Gⁱ - Aⁱ),  ∂ₜGⁱ = +β(Gʳ - Aʳ).
@kernel function _dispersion_tendency!(Gr_t, Gi_t, Gr, Gi, Ar, Ai, β)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Gr_t[i, j, k] = -β * (Gi[i, j, k] - Ai[i, j, k])
        Gi_t[i, j, k] =  β * (Gr[i, j, k] - Ar[i, j, k])
    end
end

@inline dispersion_frequency_coefficient(d::NarrowBandDispersion) = d.ω_κ / (2 * d.κ * d.α)

"""
    compute_tendencies!(model::NarrowBandWaveModel)

Diagnose the amplitude `A` from the prognostic reconstituted amplitude `G`
(one screened-Poisson solve per real component), then assemble the tendency
`∂ₜG` from the current `(G, A)` and the model's velocity coupling.
"""
function compute_tendencies!(model::NarrowBandWaveModel)
    solver = model.helmholtz_solver
    solve_amplitude!(model.Ar, solver, model.Gr)
    solve_amplitude!(model.Ai, solver, model.Gi)
    compute_wave_tendencies!(model, model.velocities)
    return nothing
end

# Wave-only regime: dispersion is the whole right-hand side.
function compute_wave_tendencies!(model::NarrowBandWaveModel, ::Nothing)
    grid = model.grid
    β = convert(eltype(grid), dispersion_frequency_coefficient(model.dispersion))
    launch!(architecture(grid), grid, :xyz, _dispersion_tendency!,
            model.Gr_tendency, model.Gi_tendency,
            model.Gr, model.Gi, model.Ar, model.Ai, β)
    return nothing
end

# Refraction/scattering operator R (Onuki & Fujiwara 2026), evaluated at cell
# centers with centered finite differences on a uniform grid:
#
#   R = 2 Ū·∇(HA) + 2 Ūᵢ,ⱼ A,ᵢⱼ + (∇·Ū) ∇²A + Ūᵢ,ᵢⱼ A,ⱼ,     HA = ∇²A + κ²A.
#
# `A` is one real component; `Ux`, `Uy` are Ū interpolated to centers. The
# stencil reaches two cells in each direction (∂ₓ(HA) needs ∇²A at x-neighbors),
# so the wave grid needs a halo of at least 2.
@inline function _refraction_at(i, j, k, A, Ux, Uy, κ², Δx, Δy)
    @inbounds begin
        Δx² = Δx^2
        Δy² = Δy^2
        # A derivatives at the center.
        Ax  = (A[i+1, j, k] - A[i-1, j, k]) / 2Δx
        Ay  = (A[i, j+1, k] - A[i, j-1, k]) / 2Δy
        Axx = (A[i+1, j, k] - 2A[i, j, k] + A[i-1, j, k]) / Δx²
        Ayy = (A[i, j+1, k] - 2A[i, j, k] + A[i, j-1, k]) / Δy²
        Axy = (A[i+1, j+1, k] - A[i+1, j-1, k] - A[i-1, j+1, k] + A[i-1, j-1, k]) / (4Δx * Δy)
        lap = Axx + Ayy

        # ∇²A at the four axial neighbors, for ∂ₓ(HA), ∂_y(HA).
        lap_xp = (A[i+2, j, k] - 2A[i+1, j, k] + A[i, j, k]) / Δx² +
                 (A[i+1, j+1, k] - 2A[i+1, j, k] + A[i+1, j-1, k]) / Δy²
        lap_xm = (A[i, j, k] - 2A[i-1, j, k] + A[i-2, j, k]) / Δx² +
                 (A[i-1, j+1, k] - 2A[i-1, j, k] + A[i-1, j-1, k]) / Δy²
        lap_yp = (A[i+1, j+1, k] - 2A[i, j+1, k] + A[i-1, j+1, k]) / Δx² +
                 (A[i, j+2, k] - 2A[i, j+1, k] + A[i, j, k]) / Δy²
        lap_ym = (A[i+1, j-1, k] - 2A[i, j-1, k] + A[i-1, j-1, k]) / Δx² +
                 (A[i, j, k] - 2A[i, j-1, k] + A[i, j-2, k]) / Δy²
        dHA_dx = ((lap_xp + κ² * A[i+1, j, k]) - (lap_xm + κ² * A[i-1, j, k])) / 2Δx
        dHA_dy = ((lap_yp + κ² * A[i, j+1, k]) - (lap_ym + κ² * A[i, j-1, k])) / 2Δy

        # Ū and its derivatives at the center.
        ux = Ux[i, j, k]; uy = Uy[i, j, k]
        ux_x = (Ux[i+1, j, k] - Ux[i-1, j, k]) / 2Δx
        ux_y = (Ux[i, j+1, k] - Ux[i, j-1, k]) / 2Δy
        uy_x = (Uy[i+1, j, k] - Uy[i-1, j, k]) / 2Δx
        uy_y = (Uy[i, j+1, k] - Uy[i, j-1, k]) / 2Δy
        divU = ux_x + uy_y

        # ∂ⱼ(∇·Ū): ∇·Ū at axial neighbors.
        divU_xp = (Ux[i+2, j, k] - Ux[i, j, k]) / 2Δx + (Uy[i+1, j+1, k] - Uy[i+1, j-1, k]) / 2Δy
        divU_xm = (Ux[i, j, k] - Ux[i-2, j, k]) / 2Δx + (Uy[i-1, j+1, k] - Uy[i-1, j-1, k]) / 2Δy
        divU_yp = (Ux[i+1, j+1, k] - Ux[i-1, j+1, k]) / 2Δx + (Uy[i, j+2, k] - Uy[i, j, k]) / 2Δy
        divU_ym = (Ux[i+1, j-1, k] - Ux[i-1, j-1, k]) / 2Δx + (Uy[i, j, k] - Uy[i, j-2, k]) / 2Δy
        divU_x = (divU_xp - divU_xm) / 2Δx
        divU_y = (divU_yp - divU_ym) / 2Δy

        R = 2 * (ux * dHA_dx + uy * dHA_dy) +
            2 * (ux_x * Axx + (ux_y + uy_x) * Axy + uy_y * Ayy) +
            divU * lap +
            (divU_x * Ax + divU_y * Ay)
    end
    return R
end

@kernel function _refraction!(Rr, Ri, Ar, Ai, Ūxᶜ, Ūyᶜ, κ², Δx, Δy)
    i, j, k = @index(Global, NTuple)
    @inbounds Rr[i, j, k] = _refraction_at(i, j, k, Ar, Ūxᶜ, Ūyᶜ, κ², Δx, Δy)
    @inbounds Ri[i, j, k] = _refraction_at(i, j, k, Ai, Ūxᶜ, Ūyᶜ, κ², Δx, Δy)
end

# Prescribed-current regime: dispersion + Doppler flux divergence −∇·(vᵉᶠᶠ A)
# (Oceananigans tracer advection) + pointwise compressibility source + R_coef·R.
@kernel function _prescribed_wave_tendency!(Gr_t, Gi_t, Gr, Gi, Ar, Ai,
                                            grid, advection, veffU, source, Rr, Ri, R_coef, β)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        s = source[i, j, k]
        transport_r = div_Uc(i, j, k, grid, advection, veffU, Ar)
        transport_i = div_Uc(i, j, k, grid, advection, veffU, Ai)
        Gr_t[i, j, k] = -β * (Gi[i, j, k] - Ai[i, j, k]) - transport_r + s * Ar[i, j, k] + R_coef * Rr[i, j, k]
        Gi_t[i, j, k] =  β * (Gr[i, j, k] - Ar[i, j, k]) - transport_i + s * Ai[i, j, k] + R_coef * Ri[i, j, k]
    end
end

function compute_wave_tendencies!(model::NarrowBandWaveModel, vel::NarrowBandPrescribedVelocities)
    grid = model.grid
    arch = architecture(grid)
    fill_halo_regions!(model.Ar)
    fill_halo_regions!(model.Ai)
    d = model.dispersion
    FT = eltype(grid)
    β = convert(FT, dispersion_frequency_coefficient(d))
    R_coef = convert(FT, d.ω_κ / (2 * d.κ * d.ω))
    κ² = convert(FT, d.κ^2)
    Δx = convert(FT, grid.Lx / grid.Nx)
    Δy = convert(FT, grid.Ly / grid.Ny)

    launch!(arch, grid, :xyz, _refraction!,
            vel.Rr, vel.Ri, model.Ar, model.Ai, vel.Ūxᶜ, vel.Ūyᶜ, κ², Δx, Δy)

    veffU = effective_transport_velocities(vel)
    launch!(arch, grid, :xyz, _prescribed_wave_tendency!,
            model.Gr_tendency, model.Gi_tendency, model.Gr, model.Gi, model.Ar, model.Ai,
            grid, model.advection, veffU, vel.source, vel.Rr, vel.Ri, R_coef, β)
    return nothing
end
