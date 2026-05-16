import Oceananigans
import Oceananigans.Advection: EnergyConserving, EnstrophyConserving, VectorInvariant

@testset "Oceananigans tracer advection transport" begin
    Nx, Ny = 64, 64
    Lx, Ly = 64.0, 64.0
    dt = 0.1
    grid = RectilinearGrid(CPU();
                           size=(Nx, Ny, 1),
                           x=(0, Lx),
                           y=(0, Ly),
                           z=(-1, 0),
                           topology=(Periodic, Periodic, Bounded))

    spectral_grid = PolarWaveVectorGrid(;
                                        κ=[1.0],
                                        φ=[0.0],
                                        φ_faces=[-pi / 32, pi / 32])

    schemes = (
        Centered(),
        UpwindBiased(order=3),
        WENO(order=5),
        FluxFormAdvection(WENO(order=5), Centered(), WENO(order=5)),
    )

    for scheme in schemes
        model = SpectralWaveModel(grid, spectral_grid;
                          horizontal_advection=scheme,
                          timestepper=:ForwardEuler,
                          clock=Clock(time=0.0, last_Δt=dt))

        for j in 1:Ny, i in 1:Nx
            x = xnodes(grid)[i]
            model.action[i, j, 1, 1] = 1 + 0.1sin(2pi * x / Lx)
        end

        compute_tendencies!(model)
        @test all(isfinite, interior(model.tendencies))
        @test maximum(abs, interior(model.tendencies)) > 0
    end

    model = SpectralWaveModel(grid, spectral_grid;
                      horizontal_advection=WENO(order=5),
                      timestepper=:ForwardEuler,
                      clock=Clock(time=0.0, last_Δt=dt))

    for j in 1:Ny, i in 1:Nx
        x = xnodes(grid)[i]
        model.action[i, j, 1, 1] = 1 + 0.1sin(2pi * x / Lx)
    end

    compute_tendencies!(model)
    u, v = Ripple.transport_velocity(model, 1, 1)
    tendency_error = zero(eltype(model.action))

    for j in 1:Ny, i in 1:Nx
        x = xnodes(grid)[i]
        expected = -u * 0.1 * 2pi / Lx * cos(2pi * x / Lx)
        tendency_error = max(tendency_error, abs(model.tendencies[i, j, 1, 1] - expected))
    end

    @test iszero(v)
    @test tendency_error < 1e-7

    y_spectral_grid = PolarWaveVectorGrid(;
                                          κ=[1.0],
                                          φ=[pi / 2],
                                          φ_faces=[pi / 2 - pi / 32, pi / 2 + pi / 32])

    y_model = SpectralWaveModel(grid, y_spectral_grid;
                      horizontal_advection=FluxFormAdvection(Centered(), WENO(order=5), nothing),
                      timestepper=:ForwardEuler,
                      clock=Clock(time=0.0, last_Δt=dt))

    for j in 1:Ny, i in 1:Nx
        y = ynodes(grid)[j]
        y_model.action[i, j, 1, 1] = 1 + 0.1sin(2pi * y / Ly)
    end

    compute_tendencies!(y_model)
    uy, vy = Ripple.transport_velocity(y_model, 1, 1)
    y_tendency_error = zero(eltype(y_model.action))

    for j in 1:Ny, i in 1:Nx
        y = ynodes(grid)[j]
        expected = -vy * 0.1 * 2pi / Ly * cos(2pi * y / Ly)
        y_tendency_error = max(y_tendency_error, abs(y_model.tendencies[i, j, 1, 1] - expected))
    end

    @test uy ≈ 0 atol=10eps(eltype(y_model.action)) rtol=0
    @test y_tendency_error < 1e-7

    initial_total = total_action(model.action)
    time_step!(model, dt)
    @test total_action(model.action) ≈ initial_total atol=1e-10 rtol=0
    @test minimum(interior(model.action)) > 0
    @test cfl(model) ≈ abs(u) * dt / minimum(xspacings(grid))

    named = SpectralWaveModel(grid, spectral_grid; horizontal_advection=(; N=WENO(order=5)))
    @test named.horizontal_advection isa Oceananigans.Advection.WENO
    @test_throws ArgumentError SpectralWaveModel(grid, spectral_grid; horizontal_advection=(; Q=WENO()))
    @test_throws ArgumentError SpectralWaveModel(grid, spectral_grid; horizontal_advection=EnergyConserving())
    @test_throws ArgumentError SpectralWaveModel(grid, spectral_grid; horizontal_advection=EnstrophyConserving())
    @test_throws ArgumentError SpectralWaveModel(grid, spectral_grid; horizontal_advection=VectorInvariant())
    @test_throws ArgumentError SpectralWaveModel(grid, spectral_grid;
                      horizontal_advection=FluxFormAdvection(Centered(), EnergyConserving(), nothing))

    small_halo_grid = RectilinearGrid(CPU();
                                      size=(Nx, Ny, 1),
                                      halo=(1, 1, 1),
                                      x=(0, Lx),
                                      y=(0, Ly),
                                      z=(-1, 0),
                                      topology=(Periodic, Periodic, Bounded))
    @test_throws ArgumentError SpectralWaveModel(small_halo_grid, spectral_grid;
                      horizontal_advection=WENO(order=5))
end
