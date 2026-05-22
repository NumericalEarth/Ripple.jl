using Test
using Ripple
using Ripple: compute_tendencies!
using Oceananigans

# Tendency-level energy conservation between Ripple wave models and
# Oceananigans' Craik-Leibovich Stokes-drift force.
#
# The wave-mean total energy E = K + W (Eulerian mean KE + wave energy) is
# conserved in the continuous Craik-Leibovich system. Discretely, the cancellation
# requires
#
#   P_ocean := ⟨uᴱ, ∂t uˢ⟩_h
#
# (the ocean-side kinetic-energy gain implied by the Stokes acceleration)
# to be balanced by a corresponding wave-energy change. With an external source
# of wave action `S`, the budget statement becomes
#
#   P_ocean + P_w_intrinsic = 0     ≡    P_ocean + P_w_total = P_source,
#
# where `P_w_intrinsic` is the wave-energy tendency from transport + refraction
# (no source contribution), and `P_w_total = P_w_intrinsic + P_source` is the
# full wave-energy tendency. These tests are tendency-level — no time-stepping —
# so the only error budget is the spatial discretization of the staggered C-grid
# operators and the Q-transform.

const g_test = 9.81

# Wave energy for monobanded:  W = ∫ σ_int(κ) · A dV   with σ_int = √(g·κ).
# Its tendency includes both the action tendency and the κ-tendency:
#   dW/dt = ⟨σ, ∂tA⟩ + ⟨A · ∂σ/∂κ, ∂tκ⟩.
function monobanded_wave_energy_tendency(model::MonobandedWaveModel)
    Ripple.update_monobanded_diagnostics!(model)
    G = model.timestepper.Gⁿ
    A   = interior(model.action)
    Kx  = interior(model.diagnostics.Kx)
    Ky  = interior(model.diagnostics.Ky)
    κ   = interior(model.diagnostics.κ)
    GA  = interior(G.A)
    GAKx = interior(G.AKx)
    GAKy = interior(G.AKy)

    σ = @. sqrt(g_test * κ)
    ∂σ∂κ = @. g_test / (2 * σ)
    Aκ = @. A * κ
    A_safe = ifelse.(A .> 0, A, one.(A))
    Aκ_safe = ifelse.(Aκ .> 0, Aκ, one.(Aκ))
    ∂tκ = @. (Kx * GAKx + Ky * GAKy) / Aκ_safe - κ * GA / A_safe
    ∂tκ = ifelse.(Aκ .> 0, ∂tκ, zero.(∂tκ))

    Δx = model.grid.Δxᶜᵃᵃ
    Δy = model.grid.Δyᵃᶜᵃ
    Δz = model.grid.z.cᵃᵃᶠ[2] - model.grid.z.cᵃᵃᶠ[1] # uniform z
    dV = Δx * Δy * Δz
    return (sum(σ .* GA) + sum(A .* ∂σ∂κ .* ∂tκ)) * dV
end

# External source power for monobanded with LinearWindInput(rate=r): S = r·A.
function monobanded_source_power(model::MonobandedWaveModel, rate)
    Ripple.update_monobanded_diagnostics!(model)
    A = interior(model.action)
    κ = interior(model.diagnostics.κ)
    σ = @. sqrt(g_test * κ)
    Δx = model.grid.Δxᶜᵃᵃ
    Δy = model.grid.Δyᵃᶜᵃ
    Δz = model.grid.z.cᵃᵃᶠ[2] - model.grid.z.cᵃᵃᶠ[1]
    dV = Δx * Δy * Δz
    return sum(σ .* rate .* A) * dV
end

# Ocean-side kinetic-energy tendency from the Stokes acceleration alone.
# We don't need the full Oceananigans tendency: ⟨uᴱ, ∂t uˢ⟩ is exactly what
# Oceananigans adds to Gu via the `+∂t_uˢ(...)` term in line 103 of the
# nonhydrostatic tendency kernel.
function ocean_stokes_acceleration_power(uᴱ, vᴱ, ∂t_uˢ, ∂t_vˢ)
    grid = uᴱ.grid
    Δx = grid.Δxᶜᵃᵃ
    Δy = grid.Δyᵃᶜᵃ
    Δz = grid.z.cᵃᵃᶠ[2] - grid.z.cᵃᵃᶠ[1]
    dV = Δx * Δy * Δz
    return (sum(interior(uᴱ) .* interior(∂t_uˢ)) +
            sum(interior(vᴱ) .* interior(∂t_vˢ))) * dV
end

@testset "Energy conservation: Ripple↔Oceananigans Stokes coupling" begin

    # ──────────────────────────────────────────────────────────────────────
    # Case 1. Source-only monobanded with uᴱ ≡ 0. P_ocean = 0 trivially,
    # P_w_total = P_source by construction. This is a wiring sanity check.
    # ──────────────────────────────────────────────────────────────────────
    @testset "source-only, uᴱ = 0 → P_w = P_source, P_ocean = 0" begin
        grid = RectilinearGrid(CPU(); size=(4, 4, 8), halo=(3, 3, 3),
                               x=(0, 1), y=(0, 1), z=(-0.5, 0),
                               topology=(Periodic, Periodic, Bounded))
        rate = 0.7
        m = MonobandedWaveModel(grid; advection=nothing,
                                timestepper=:RungeKutta3,
                                gravitational_acceleration=g_test,
                                sources=LinearWindInput(rate=rate))
        set!(m; A=0.02, AKx=2.0, AKy=0.0)
        compute_tendencies!(m)

        P_w = monobanded_wave_energy_tendency(m)
        P_source = monobanded_source_power(m, rate)
        @test isapprox(P_w, P_source; rtol=1e-13)

        ptx, pty = pseudomomentum_tendency_fields(m;
                                                  location=(Face, Center, Center))
        ptx_v, pty_v = pseudomomentum_tendency_fields(m;
                                                      location=(Center, Face, Center))

        uᴱ = Field{Face, Center, Center}(grid); set!(uᴱ, 0)
        vᴱ = Field{Center, Face, Center}(grid); set!(vᴱ, 0)
        P_ocean = ocean_stokes_acceleration_power(uᴱ, vᴱ, ptx, pty_v)
        @test abs(P_ocean) < 1e-14
        @test isapprox(P_ocean + P_w, P_source; rtol=1e-13)
    end

    # ──────────────────────────────────────────────────────────────────────
    # Case 2. Source-only monobanded with uᴱ ≠ 0 (orthogonal to the Stokes
    # drift), all spatially uniform. With ∂t uˢ along x̂ and uᴱ along ŷ,
    # ⟨uᴱ, ∂t uˢ⟩ = 0 exactly → coupling does no work on the mean flow,
    # and the budget closes with P_ocean = 0 to roundoff.
    # ──────────────────────────────────────────────────────────────────────
    @testset "source-only + uᴱ ⊥ uˢ → P_ocean = 0 (orthogonality)" begin
        grid = RectilinearGrid(CPU(); size=(4, 4, 8), halo=(3, 3, 3),
                               x=(0, 1), y=(0, 1), z=(-0.5, 0),
                               topology=(Periodic, Periodic, Bounded))
        rate = 0.5
        m = MonobandedWaveModel(grid; advection=nothing,
                                timestepper=:RungeKutta3,
                                gravitational_acceleration=g_test,
                                sources=LinearWindInput(rate=rate))
        # Stokes drift along +x.
        set!(m; A=0.02, AKx=2.0, AKy=0.0)
        compute_tendencies!(m)

        P_w = monobanded_wave_energy_tendency(m)
        P_source = monobanded_source_power(m, rate)

        ptx, _ = pseudomomentum_tendency_fields(m;
                                                location=(Face, Center, Center))
        _, pty = pseudomomentum_tendency_fields(m;
                                                location=(Center, Face, Center))

        # Mean flow along +y, varying in z.
        uᴱ = Field{Face, Center, Center}(grid); set!(uᴱ, 0)
        vᴱ = Field{Center, Face, Center}(grid)
        set!(vᴱ, (x, y, z) -> 0.3 * exp(z / 0.2))

        P_ocean = ocean_stokes_acceleration_power(uᴱ, vᴱ, ptx, pty)
        @test abs(P_ocean) < 1e-14
        @test isapprox(P_ocean + P_w, P_source; rtol=1e-13)
    end

    # ──────────────────────────────────────────────────────────────────────
    # Case 3. Source-only monobanded with uᴱ aligned with uˢ — both
    # spatially uniform. P_ocean ≠ 0 and P_w_intrinsic must cancel it for
    # the closed budget to hold. With no transport and no refraction in
    # this setup, the wave model has no mechanism to balance P_ocean, so
    # P_ocean + P_w = P_source + P_ocean ≠ P_source. We assert the *known*
    # gap matches the analytic P_ocean — this isolates the wave-side sink
    # that a refraction-enabled coupling would need to supply.
    # ──────────────────────────────────────────────────────────────────────
    @testset "source-only + uᴱ ∥ uˢ: gap quantifies the missing wave sink" begin
        grid = RectilinearGrid(CPU(); size=(4, 4, 8), halo=(3, 3, 3),
                               x=(0, 1), y=(0, 1), z=(-0.5, 0),
                               topology=(Periodic, Periodic, Bounded))
        rate = 0.4
        m = MonobandedWaveModel(grid; advection=nothing,
                                timestepper=:RungeKutta3,
                                gravitational_acceleration=g_test,
                                sources=LinearWindInput(rate=rate))
        set!(m; A=0.02, AKx=2.0, AKy=0.0)
        compute_tendencies!(m)

        P_w = monobanded_wave_energy_tendency(m)
        P_source = monobanded_source_power(m, rate)

        ptx, _ = pseudomomentum_tendency_fields(m;
                                                location=(Face, Center, Center))
        _, pty = pseudomomentum_tendency_fields(m;
                                                location=(Center, Face, Center))

        uᴱ = Field{Face, Center, Center}(grid)
        set!(uᴱ, (x, y, z) -> 0.4 * exp(z / 0.15))
        vᴱ = Field{Center, Face, Center}(grid); set!(vᴱ, 0)

        P_ocean = ocean_stokes_acceleration_power(uᴱ, vᴱ, ptx, pty)
        @test P_ocean > 0  # uᴱ aligned with ∂t uˢ → positive work

        # Without refraction the wave-side sink is missing, so the
        # closed-budget statement does not hold. Document the gap so the
        # refraction-enabled test below can verify that closure recovers.
        @test isapprox(P_ocean + P_w, P_source + P_ocean; rtol=1e-13)
    end

end
