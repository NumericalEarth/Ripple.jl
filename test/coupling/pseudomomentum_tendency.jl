using Test
using Ripple
using Oceananigans

# Direct tests for `pseudomomentum_tendency_fields(model)`. The energy
# conservation test is a coupled, late failure mode; these tests isolate the
# analytic Q-projection so sign or location errors surface here first.

@testset "pseudomomentum_tendency_fields" begin

    @testset "Field locations" begin
        g = RectilinearGrid(CPU(); size=(4, 4, 6), halo=(3, 3, 3),
                            x=(0, 1), y=(0, 1), z=(-0.5, 0),
                            topology=(Periodic, Periodic, Bounded))
        m = MonobandedWaveModel(g; advection=Centered(), timestepper=:RungeKutta3,
                                gravitational_acceleration=9.81)
        set!(m; A=0.01, AKx=1.0, AKy=0.0)
        Ripple.compute_tendencies!(m)

        ptx, pty = pseudomomentum_tendency_fields(m)
        @test location(ptx) == (Center, Center, Center)
        @test location(pty) == (Center, Center, Center)

        ptx_fcc, pty_fcc = pseudomomentum_tendency_fields(m; location=(Face, Center, Center))
        @test location(ptx_fcc) == (Face, Center, Center)
        ptx_cfc, pty_cfc = pseudomomentum_tendency_fields(m; location=(Center, Face, Center))
        @test location(pty_cfc) == (Center, Face, Center)
    end

    @testset "Spectral ≡ Q-projection of model.tendencies" begin
        g = RectilinearGrid(CPU(); size=(6, 6, 6), halo=(3, 3, 3),
                            x=(0, 1), y=(0, 1), z=(-0.5, 0),
                            topology=(Periodic, Periodic, Bounded))
        κ0 = 50.0
        sg = PolarWaveVectorGrid(; κ=range(0.5κ0, 2.0κ0; length=4),
                                  φ=range(-π, π; length=8)[1:7])

        uL = Field{Face, Center, Center}(g); set!(uL, 0)
        vL = Field{Center, Face, Center}(g); set!(vL, 0)
        m = SpectralWaveModel(g, sg; depth=0.5,
                              velocities=(; u=uL, v=vL),
                              sources=nothing, timestepper=:RungeKutta3,
                              horizontal_advection=Centered(),
                              spectral_advection=Centered())

        init_action(x, y, kx, ky) =
            (1.0 + 0.2*sin(2π*x)) * exp(-((hypot(kx,ky) - κ0)/(0.2κ0))^2)
        set!(m.action, init_action)
        Ripple.update_coupling!(m)
        Ripple.compute_tendencies!(m)

        ptx_helper, pty_helper = pseudomomentum_tendency_fields(m)
        # Compare against the direct Q-projection of the action tendency.
        ptx_direct, pty_direct = pseudomomentum_fields(m.tendencies, m.depth, m.coupling.qtransform)

        @test interior(ptx_helper) ≈ interior(ptx_direct) atol=1e-14 rtol=1e-12
        @test interior(pty_helper) ≈ interior(pty_direct) atol=1e-14 rtol=1e-12
    end

    @testset "Monobanded analytic ≈ FD: no refraction (∂tκ = 0)" begin
        # Single wave train aligned with x; spatial variation along propagation
        # → A and AKx evolve under transport but κ stays constant in space and
        # time, so ∂tκ = 0 and the ∂κQ correction contributes nothing.
        g = RectilinearGrid(CPU(); size=(8, 8, 8), halo=(3, 3, 3),
                            x=(0, 1), y=(0, 1), z=(-0.5, 0),
                            topology=(Periodic, Periodic, Bounded))
        m = MonobandedWaveModel(g; advection=Centered(), timestepper=:RungeKutta3,
                                gravitational_acceleration=9.81)
        κ_target = 100.0
        A_field(x, y, z) = 0.01 + 0.005*sin(2π*x)
        set!(m; A=A_field, AKx=(x,y,z)->κ_target*A_field(x,y,z), AKy=0.0)
        Ripple.compute_tendencies!(m)

        ptx_analytic, pty_analytic = pseudomomentum_tendency_fields(m)
        p0x, p0y = pseudomomentum_fields(m)

        # One small forward-Euler step on the prognostic fields, then recompute
        # the pseudomomentum. (We avoid `time_step!` so the saved Gⁿ stays
        # pinned to the t=0 evaluation point and we read off an FD against the
        # same kernel state.)
        Δt = 1e-8
        G = m.timestepper.Gⁿ
        interior(m.action) .+= Δt .* interior(G.A)
        interior(m.wavenumber_moment.x) .+= Δt .* interior(G.AKx)
        interior(m.wavenumber_moment.y) .+= Δt .* interior(G.AKy)
        Ripple.update_monobanded_diagnostics!(m)
        p1x, p1y = pseudomomentum_fields(m)

        fd_ptx = (interior(p1x) .- interior(p0x)) ./ Δt
        fd_pty = (interior(p1y) .- interior(p0y)) ./ Δt
        @test fd_ptx ≈ interior(ptx_analytic) atol=1e-4 rtol=1e-4
        @test all(abs.(interior(pty_analytic)) .< 1e-12)
        @test all(abs.(fd_pty) .< 1e-6)
    end

    @testset "Monobanded analytic ≈ FD: refraction (∂tκ ≠ 0)" begin
        # AKy varies in y while A and AKx are uniform → Ky/A varies in space
        # and the transport divergence produces ∂t(AKy) ≠ 0 while ∂tA ≠ 0;
        # crucially the tendency is not parallel to (AKx, AKy), so ∂tκ ≠ 0
        # and the ∂κQ correction is essential for matching FD.
        g = RectilinearGrid(CPU(); size=(8, 8, 8), halo=(3, 3, 3),
                            x=(0, 1), y=(0, 1), z=(-0.5, 0),
                            topology=(Periodic, Periodic, Bounded))
        m = MonobandedWaveModel(g; advection=Centered(), timestepper=:RungeKutta3,
                                gravitational_acceleration=9.81)
        A0 = 0.01
        AKx0 = 1.0
        set!(m; A=A0,
                AKx=(x,y,z) -> AKx0 + 0.01*cos(2π*x),
                AKy=(x,y,z) -> 0.05*sin(2π*y))
        Ripple.compute_tendencies!(m)

        # Sanity: ∂tκ should be nonzero somewhere.
        G = m.timestepper.Gⁿ
        Kx = interior(m.diagnostics.Kx); Ky = interior(m.diagnostics.Ky)
        κ  = interior(m.diagnostics.κ);   A  = interior(m.action)
        GA = interior(G.A); GAKx = interior(G.AKx); GAKy = interior(G.AKy)
        ∂tκ = @. (Kx*GAKx + Ky*GAKy)/(A*κ) - κ*GA/A
        @test maximum(abs.(∂tκ)) > 1e-6

        ptx_analytic, pty_analytic = pseudomomentum_tendency_fields(m)
        p0x, p0y = pseudomomentum_fields(m)

        Δt = 1e-8
        interior(m.action) .+= Δt .* interior(G.A)
        interior(m.wavenumber_moment.x) .+= Δt .* interior(G.AKx)
        interior(m.wavenumber_moment.y) .+= Δt .* interior(G.AKy)
        Ripple.update_monobanded_diagnostics!(m)
        p1x, p1y = pseudomomentum_fields(m)

        fd_ptx = (interior(p1x) .- interior(p0x)) ./ Δt
        fd_pty = (interior(p1y) .- interior(p0y)) ./ Δt
        @test fd_ptx ≈ interior(ptx_analytic) atol=1e-4 rtol=1e-4
        @test fd_pty ≈ interior(pty_analytic) atol=1e-4 rtol=1e-4
    end

    @testset "Monobanded: ∂κQ contribution catches refraction" begin
        # Construct a state where Q·∂t(AK) alone (without ∂κQ·AK·∂tκ) would
        # give a noticeably different answer than the full formula. Compare
        # the helper output to a reference "frozen-κ" projection and verify
        # they differ — i.e., the ∂κQ term is actually contributing.
        g = RectilinearGrid(CPU(); size=(4, 4, 6), halo=(3, 3, 3),
                            x=(0, 1), y=(0, 1), z=(-0.5, 0),
                            topology=(Periodic, Periodic, Bounded))
        m = MonobandedWaveModel(g; advection=Centered(), timestepper=:RungeKutta3,
                                gravitational_acceleration=9.81)
        set!(m; A=0.01,
                AKx=(x,y,z) -> 1.0 + 0.05*cos(2π*x),
                AKy=(x,y,z) -> 0.2*sin(2π*y))
        Ripple.compute_tendencies!(m)
        ptx_full, _ = pseudomomentum_tendency_fields(m)

        # Reference: Q·G.AKx using the same coupling but no ∂κQ correction.
        # Reuse pseudomomentum_fields(::MonobandedWaveModel) with G.AK swapped
        # for AK by temporarily overwriting the prognostic fields.
        save_A = copy(interior(m.action))
        save_AKx = copy(interior(m.wavenumber_moment.x))
        save_AKy = copy(interior(m.wavenumber_moment.y))
        G = m.timestepper.Gⁿ
        # Swap moment fields with the tendencies (keep κ frozen via diagnostics).
        interior(m.wavenumber_moment.x) .= interior(G.AKx)
        interior(m.wavenumber_moment.y) .= interior(G.AKy)
        # Do NOT re-run diagnostics — we want κ to stay at the original value.
        ptx_frozen_κ, _ = pseudomomentum_fields(m; location=(Center, Center, Center))
        # Restore.
        interior(m.action) .= save_A
        interior(m.wavenumber_moment.x) .= save_AKx
        interior(m.wavenumber_moment.y) .= save_AKy
        Ripple.update_monobanded_diagnostics!(m)

        # The two should differ if and only if ∂κQ·AK·∂tκ contributes.
        @test maximum(abs.(interior(ptx_full) .- interior(ptx_frozen_κ))) > 1e-8
    end

end
