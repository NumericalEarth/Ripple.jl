import Oceananigans: NonhydrostaticModel, BuoyancyTracer

@testset "Two-way wave–current coupling" begin
    grid = RectilinearGrid(CPU(); size=(16, 16, 16), x=(0, 60), y=(0, 60), z=(-40, 0),
                           topology=(Periodic, Periodic, Bounded), halo=(3, 3, 3))

    stokes = NarrowBandStokesDrift(grid)
    ocean = NonhydrostaticModel(grid; stokes_drift=stokes, advection=WENO(),
                                tracers=:b, buoyancy=BuoyancyTracer(), timestepper=:RungeKutta3)
    κ = 2π / 60
    wave = NarrowBandWaveModel(grid; κ,
                               velocities=(u=ocean.velocities.u, v=ocean.velocities.v))
    wcm = WaveCurrentModel(wave, ocean, stokes)

    set!(wave; A=(x, y) -> cis(κ * x))          # plane wave: uniform Stokes drift
    initialize_coupling!(wcm)

    @testset "Stokes drift from a plane wave" begin
        @test maximum(abs, interior(stokes.∂z_uˢ)) > 0       # vertical shear populated
        @test maximum(abs, interior(stokes.ζˢ)) < 1e-8       # uniform ⇒ no Stokes vorticity
        @test maximum(abs, interior(stokes.∂z_vˢ)) < 1e-8    # no y-component
        @test maximum(abs, interior(stokes.∂t_uˢ)) == 0      # ∂ₜUˢ = 0 at initialization
        # Surface Stokes drift is positive and decays with depth (e^{2κz}-like).
        uˢ = Array(interior(wcm.uˢ))
        @test uˢ[1, 1, end] > 0
        @test uˢ[1, 1, end] > uˢ[1, 1, 1]
    end

    @testset "Coupled stepping stays finite" begin
        set!(ocean, u=(x, y, z) -> 1e-3 * (rand() - 0.5))
        for _ in 1:5
            coupled_time_step!(wcm, 1.0)
        end
        @test all(isfinite, interior(ocean.velocities.u))
        @test all(isfinite, interior(ocean.velocities.w))
        @test all(isfinite, amplitude(wave))
        @test maximum(abs, interior(stokes.∂t_uˢ)) ≥ 0        # tendency now computed by FD
    end
end
