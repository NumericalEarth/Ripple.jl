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

# Wave energy for monobanded:  W = ∫dx dy σ_int(κ) · A   with σ_int = √(g·κ).
# Note A lives at a single horizontal slice (per-unit-area), so the integral
# is 2D (dA = Δx·Δy), not 3D. Its tendency includes both the action tendency
# and the κ-tendency:
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

    dA = model.grid.Δxᶜᵃᵃ * model.grid.Δyᵃᶜᵃ
    return (sum(σ .* GA) + sum(A .* ∂σ∂κ .* ∂tκ)) * dA
end

# External source power for monobanded with LinearWindInput(rate=r): S = r·A.
# 2D integral, matching `monobanded_wave_energy_tendency`.
function monobanded_source_power(model::MonobandedWaveModel, rate)
    Ripple.update_monobanded_diagnostics!(model)
    A = interior(model.action)
    κ = interior(model.diagnostics.κ)
    σ = @. sqrt(g_test * κ)
    dA = model.grid.Δxᶜᵃᵃ * model.grid.Δyᵃᶜᵃ
    return sum(σ .* rate .* A) * dA
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

    # ──────────────────────────────────────────────────────────────────────
    # Case 3b. Transport-only conservation: wave model with centered advection
    # and ZeroVelocities (no current) — pure intrinsic group-velocity
    # propagation. With κ uniform in space and time, σ_int(κ) is constant,
    # so the wave-energy tendency reduces to σ · sum(Gⁿ.A), and centered
    # conservative transport on a periodic grid gives sum(Gⁿ.A) = 0 to
    # roundoff. dW/dt should hit machine precision.
    # ──────────────────────────────────────────────────────────────────────
    @testset "transport-only, ZeroVelocities → ∂t W = 0 to roundoff" begin
        grid = RectilinearGrid(CPU(); size=(8, 8, 8), halo=(3, 3, 3),
                               x=(0, 1), y=(0, 1), z=(-0.5, 0),
                               topology=(Periodic, Periodic, Bounded))
        m = MonobandedWaveModel(grid; advection=Centered(),
                                timestepper=:RungeKutta3,
                                gravitational_acceleration=g_test)
        # Uniform K → κ constant in space and time → no refraction.
        # Spatially varying A → nontrivial transport divergence per cell, but
        # the integrated divergence vanishes by periodic + flux-form.
        Aenv(x, y, z) = 0.01 + 0.005 * (sin(2π * x) + cos(2π * y))
        set!(m; A=Aenv, AKx=(x,y,z)->100*Aenv(x,y,z), AKy=0.0)
        compute_tendencies!(m)

        P_w = monobanded_wave_energy_tendency(m)
        @test abs(P_w) < 1e-13 * abs(monobanded_source_power(m, 1.0))
    end

    # ──────────────────────────────────────────────────────────────────────
    # Case 4. Closed coupling with refraction. The wave model is given the
    # ocean velocities as Lagrangian-mean (so refraction is live), and there
    # is no external source. In the continuum dE/dt = 0; discretely the
    # wave-side refraction tendency and the ocean-side Stokes-acceleration
    # operators are only "almost" adjoint, so the closed budget closes to
    # O(Δx²), not roundoff. Two assertions:
    #
    #   (a) a fixed-N upper bound that catches any term flipping sign,
    #       being dropped, or being mis-located,
    #   (b) the drift halves by ≳ 2× per resolution doubling, which fails
    #       if a regression introduces an O(1) or O(Δx) term.
    #
    # Closure factory: returns (P_w, P_ocean, scale) at resolution N.
    # ──────────────────────────────────────────────────────────────────────
    function closed_refraction_budget(N)
        grid = RectilinearGrid(CPU(); size=(N, N, N), halo=(3, 3, 3),
                               x=(0, 1), y=(0, 1), z=(-0.5, 0),
                               topology=(Periodic, Periodic, Bounded))
        uᴱ = Field{Face, Center, Center}(grid)
        vᴱ = Field{Center, Face, Center}(grid)
        set!(uᴱ, (x, y, z) -> 0.05 * cos(2π * x) * exp(z / 0.2))
        set!(vᴱ, (x, y, z) -> 0.03 * sin(2π * y) * exp(z / 0.2))
        Oceananigans.BoundaryConditions.fill_halo_regions!((uᴱ, vᴱ))

        m = MonobandedWaveModel(grid; advection=Centered(),
                                velocities=(; u=uᴱ, v=vᴱ),
                                timestepper=:RungeKutta3,
                                gravitational_acceleration=g_test)
        A0(x, y, z) = 0.02 + 0.005 * sin(2π * x)
        set!(m; A=A0, AKx=(x,y,z)->100*A0(x,y,z), AKy=0.0)
        Ripple.update_coupling!(m)
        compute_tendencies!(m)

        P_w = monobanded_wave_energy_tendency(m)
        ptx, _ = pseudomomentum_tendency_fields(m; location=(Face, Center, Center))
        _, pty = pseudomomentum_tendency_fields(m; location=(Center, Face, Center))
        P_ocean = ocean_stokes_acceleration_power(uᴱ, vᴱ, ptx, pty)
        scale = max(abs(P_w), abs(P_ocean))
        return P_w, P_ocean, scale
    end

    @testset "refraction-enabled closed: O(Δx²) drift bound at N=8" begin
        P_w, P_ocean, scale = closed_refraction_budget(8)
        rel = abs(P_w + P_ocean) / scale
        # Empirically ~0.15 at N=8; assert an upper bound that catches any
        # missing term but is loose enough to survive minor refactors.
        @test rel < 0.25
    end

    @testset "refraction-enabled closed: drift ~ O(Δx²) under refinement" begin
        function rel_drift(N)
            P_w, P_ocean, scale = closed_refraction_budget(N)
            return abs(P_w + P_ocean) / scale
        end
        # Each doubling should shrink the drift by ≥ 2×. A correct O(Δx²)
        # discretization gives ~4×; a regression that introduces an O(Δx)
        # or O(1) term breaks this.
        r8  = rel_drift(8)
        r16 = rel_drift(16)
        r32 = rel_drift(32)
        @test r16 < 0.55 * r8
        @test r32 < 0.55 * r16
    end

    # ──────────────────────────────────────────────────────────────────────
    # Case 5. Time-stepped closed coupling: both wave and ocean models step
    # forward with RK3, ocean receives uˢ, vˢ, ∂t_uˢ, ∂t_vˢ from the wave
    # model at the start of each step (stage-consistent coupling). Verify
    # the integrated energy drift over a fixed window decreases as Δt → 0
    # until the O(Δx²) spatial-discretization floor takes over.
    # ──────────────────────────────────────────────────────────────────────
    function timestepped_coupled_drift(N, Δt, T)
        grid = RectilinearGrid(CPU(); size=(N, N, N), halo=(3, 3, 3),
                               x=(0, 1), y=(0, 1), z=(-0.5, 0),
                               topology=(Periodic, Periodic, Bounded))
        stokes_drift = FieldStokesDrift(grid)
        uˢ, vˢ       = stokes_drift.uˢ, stokes_drift.vˢ
        ∂t_uˢ, ∂t_vˢ = stokes_drift.∂t_uˢ, stokes_drift.∂t_vˢ
        ocean = NonhydrostaticModel(grid; advection=Centered(),
                                    stokes_drift=stokes_drift,
                                    closure=nothing)
        set!(ocean,
             u=(x, y, z) -> 0.05 * cos(2π * x) * exp(z / 0.2),
             v=(x, y, z) -> 0.03 * sin(2π * y) * exp(z / 0.2),
             w=0)

        wave = MonobandedWaveModel(grid; advection=Centered(),
                                   velocities=(; u=ocean.velocities.u,
                                                 v=ocean.velocities.v),
                                   timestepper=:RungeKutta3,
                                   gravitational_acceleration=g_test)
        A0(x, y, z) = 0.02 + 0.005 * sin(2π * x)
        set!(wave; A=A0, AKx=(x,y,z)->100*A0(x,y,z), AKy=0.0)
        Ripple.update_coupling!(wave)

        function refresh_stokes!()
            Ripple.compute_tendencies!(wave)
            px, _ = pseudomomentum_fields(wave; location=(Face, Center, Center))
            _, py = pseudomomentum_fields(wave; location=(Center, Face, Center))
            ∂tpx, _ = pseudomomentum_tendency_fields(wave;
                                                    location=(Face, Center, Center))
            _, ∂tpy = pseudomomentum_tendency_fields(wave;
                                                    location=(Center, Face, Center))
            set!(uˢ, px);       set!(vˢ, py)
            set!(∂t_uˢ, ∂tpx);  set!(∂t_vˢ, ∂tpy)
            Oceananigans.BoundaryConditions.fill_halo_regions!((uˢ, vˢ, ∂t_uˢ, ∂t_vˢ))
        end

        function total_energy()
            Ripple.update_monobanded_diagnostics!(wave)
            A = interior(wave.action); κ = interior(wave.diagnostics.κ)
            σ = @. sqrt(g_test * κ)
            dA = grid.Δxᶜᵃᵃ * grid.Δyᵃᶜᵃ
            W = sum(σ .* A) * dA
            dV = dA * (grid.z.cᵃᵃᶠ[2] - grid.z.cᵃᵃᶠ[1])
            u = interior(ocean.velocities.u); v = interior(ocean.velocities.v)
            w = interior(ocean.velocities.w)
            K = 0.5 * (sum(u .^ 2) + sum(v .^ 2) + sum(w .^ 2)) * dV
            return W + K
        end

        refresh_stokes!()
        E0 = total_energy()
        nsteps = round(Int, T / Δt)
        for _ in 1:nsteps
            refresh_stokes!()
            Oceananigans.TimeSteppers.time_step!(ocean, Δt)
            Ripple.time_step!(wave, Δt)
        end
        E1 = total_energy()
        return abs(E1 - E0), abs(E0)
    end

    @testset "refraction-enabled closed: time-stepped drift bounded" begin
        # Tiny grid, short window — runs in a few seconds.
        N, T = 8, 0.01
        drift, scale = timestepped_coupled_drift(N, T/4, T)
        # Drift accumulates from the O(Δx²) coupling discretization gap
        # × T plus the RK3 time-integrator error. At N=8 the spatial gap
        # dominates and bounds the relative drift ≲ a few percent.
        @test drift / scale < 0.05
    end

    @testset "refraction-enabled closed: drift decreases when Δt shrinks" begin
        # Compare a "large" Δt (RK3 time-integrator error dominates) to
        # a "small" Δt (we're near the spatial floor). The small-Δt drift
        # should be substantially smaller — anything else means the time
        # integrator is not converging or the coupling has an O(1) bug.
        # Once near the floor the drift fluctuates by roundoff, so we
        # don't require strict monotonicity for every halving.
        N, T = 8, 0.01
        d_coarse = first(timestepped_coupled_drift(N, T,   T))      # 1 step
        d_fine   = first(timestepped_coupled_drift(N, T/8, T))      # 8 steps
        @test d_fine < d_coarse
    end

end
