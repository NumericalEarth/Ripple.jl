import Oceananigans
import Oceananigans.Advection: EnergyConserving, EnstrophyConserving, VectorInvariant
import KernelAbstractions

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

    coupled_u = Oceananigans.Fields.Field{Face, Center, Center}(grid)
    coupled_v = Oceananigans.Fields.Field{Center, Face, Center}(grid)
    set!(coupled_u, (x, y, z) -> 0.2 + 0.05sin(2π * x / Lx))
    set!(coupled_v, 0)

    coupled_spectral_grid = PolarWaveVectorGrid(;
                                                κ=range(0.8, 1.2; length=3),
                                                φ=range(0, 2pi; length=9)[1:8])

    coupled_model = SpectralWaveModel(grid, coupled_spectral_grid;
                                      velocities=(; u=coupled_u, v=coupled_v),
                                      horizontal_advection=WENO(order=5),
                                      spectral_advection=WENO(order=5),
                                      timestepper=:ForwardEuler,
                                      clock=Clock(time=0.0, last_Δt=dt))

    Nκ, Nφ = coordinate_size(coupled_spectral_grid)
    for n in 1:Nφ, m in 1:Nκ, j in 1:Ny, i in 1:Nx
        x = xnodes(grid)[i]
        y = ynodes(grid)[j]
        coupled_model.action[i, j, m, n] = 1 + 0.1sin(2π * x / Lx) + 0.05cos(2π * y / Ly)
    end

    U = Ripple.transport_velocity_fields(coupled_model.coupling, coupled_model, 1, 1)
    @test Oceananigans.Fields.location(U.u) == (Face, Center, Center)
    @test Oceananigans.Fields.location(U.v) == (Center, Face, Center)

    compute_tendencies!(coupled_model)
    @test total_action(coupled_model.tendencies) ≈ 0 atol=1e-10 rtol=0

    coupled_initial_total = total_action(coupled_model.action)
    time_step!(coupled_model, dt)
    @test total_action(coupled_model.action) ≈ coupled_initial_total atol=1e-10 rtol=0

    current_only_model = SpectralWaveModel(grid, y_spectral_grid;
                                           velocities=(; u=coupled_u, v=coupled_v),
                                           horizontal_advection=WENO(order=5),
                                           spectral_advection=nothing,
                                           timestepper=:ForwardEuler,
                                           clock=Clock(time=0.0, last_Δt=dt))

    for j in 1:Ny, i in 1:Nx
        x = xnodes(grid)[i]
        current_only_model.action[i, j, 1, 1] = 1 + 0.1sin(2π * x / Lx)
    end

    compute_tendencies!(current_only_model)
    @test maximum(abs, interior(current_only_model.tendencies)) > 0
    @test cfl(current_only_model) > 0

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

@testset "CWCM polar spectral refraction conserves action" begin
    Nx, Ny, Nz = 4, 4, 1
    Hx, Hy, iz = 3, 3, 1
    Nκ, Nφ = 6, 8

    grid = RectilinearGrid(CPU();
                           size=(Nx, Ny, Nz),
                           halo=(Hx, Hy, 1),
                           x=(0, 1),
                           y=(0, 1),
                           z=(-1, 0),
                           topology=(Periodic, Periodic, Bounded))

    spectral_grid = PolarWaveVectorGrid(;
                                        κ=range(0.4, 1.4; length=Nκ),
                                        φ=range(0, 2π; length=Nφ+1)[1:Nφ])

    N_data = zeros(Float64, Nx + 2Hx, Ny + 2Hy, 1, Nκ, Nφ)
    G_data = similar(N_data)

    for n in 1:Nφ, m in 1:Nκ, j in 1:Ny, i in 1:Nx
        φ = spectral_grid.φ[n]
        N_data[i + Hx, j + Hy, iz, m, n] = 1 + 0.03m + 0.07cos(φ)
    end

    uᴰx = zeros(Float64, Nx, Ny, Nκ)
    uᴰy = zeros(Float64, Nx, Ny, Nκ)
    duᴰxdκ = zeros(Float64, Nx, Ny, Nκ)
    duᴰydκ = zeros(Float64, Nx, Ny, Nκ)
    uᴰx_x = fill(0.17, Nx, Ny, Nκ)
    uᴰx_y = zeros(Float64, Nx, Ny, Nκ)
    uᴰy_x = zeros(Float64, Nx, Ny, Nκ)
    uᴰy_y = fill(0.17, Nx, Ny, Nκ)
    cg_x_table = zeros(Float64, Nκ, Nφ)
    cg_y_table = zeros(Float64, Nκ, Nφ)
    cos_table = cos.(Array(spectral_grid.φ))
    sin_table = sin.(Array(spectral_grid.φ))

    arch_device = Oceananigans.Architectures.device(CPU())
    kernel = Ripple._wave_current_refraction_tendency!(arch_device, (1, 1, 4, 4), (Nx, Ny, Nκ, Nφ))
    kernel(G_data, N_data,
           uᴰx, uᴰy, duᴰxdκ, duᴰydκ,
           uᴰx_x, uᴰx_y, uᴰy_x, uᴰy_y,
           spectral_grid.κ, spectral_grid.κ_faces, spectral_grid.φ_faces,
           cg_x_table, cg_y_table,
           cos_table, sin_table,
           grid, Hx, Hy, iz, Nx, Ny, Nκ, Nφ, true, true)
    KernelAbstractions.synchronize(arch_device)

    weighted_tendency = sum(G_data[i + Hx, j + Hy, iz, m, n] *
                            spectral_weight(spectral_grid, m, n)
                            for n in 1:Nφ, m in 1:Nκ, j in 1:Ny, i in 1:Nx)

    @test maximum(abs, view(G_data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, iz, :, :)) > 0
    @test weighted_tendency ≈ 0 atol=1e-13 rtol=0

    angular_spectral_grid = PolarWaveVectorGrid(;
                                                κ=[1.0],
                                                κ_faces=[0.8, 1.2],
                                                φ=range(0, 2π; length=Nφ+1)[1:Nφ])
    Nκ_angular = 1
    angular_N_data = zeros(Float64, Nx + 2Hx, Ny + 2Hy, 1, Nκ_angular, Nφ)
    angular_G_data = similar(angular_N_data)

    for n in 1:Nφ, j in 1:Ny, i in 1:Nx
        φ = angular_spectral_grid.φ[n]
        angular_N_data[i + Hx, j + Hy, iz, 1, n] = 1 + 0.08cos(φ) - 0.04sin(2φ)
    end

    angular_uᴰx = zeros(Float64, Nx, Ny, Nκ_angular)
    angular_uᴰy = zeros(Float64, Nx, Ny, Nκ_angular)
    angular_duᴰxdκ = zeros(Float64, Nx, Ny, Nκ_angular)
    angular_duᴰydκ = zeros(Float64, Nx, Ny, Nκ_angular)
    angular_uᴰx_x = fill(0.12, Nx, Ny, Nκ_angular)
    angular_uᴰx_y = fill(0.04, Nx, Ny, Nκ_angular)
    angular_uᴰy_x = fill(-0.03, Nx, Ny, Nκ_angular)
    angular_uᴰy_y = fill(-0.09, Nx, Ny, Nκ_angular)
    angular_cg_x_table = zeros(Float64, Nκ_angular, Nφ)
    angular_cg_y_table = zeros(Float64, Nκ_angular, Nφ)
    angular_cos_table = cos.(Array(angular_spectral_grid.φ))
    angular_sin_table = sin.(Array(angular_spectral_grid.φ))

    angular_kernel = Ripple._wave_current_refraction_tendency!(arch_device, (1, 1, 1, 4), (Nx, Ny, Nκ_angular, Nφ))
    angular_kernel(angular_G_data, angular_N_data,
                   angular_uᴰx, angular_uᴰy, angular_duᴰxdκ, angular_duᴰydκ,
                   angular_uᴰx_x, angular_uᴰx_y, angular_uᴰy_x, angular_uᴰy_y,
                   angular_spectral_grid.κ, angular_spectral_grid.κ_faces, angular_spectral_grid.φ_faces,
                   angular_cg_x_table, angular_cg_y_table,
                   angular_cos_table, angular_sin_table,
                   grid, Hx, Hy, iz, Nx, Ny, Nκ_angular, Nφ, true, true)
    KernelAbstractions.synchronize(arch_device)

    angular_weighted_tendency = sum(angular_G_data[i + Hx, j + Hy, iz, 1, n] *
                                    spectral_weight(angular_spectral_grid, 1, n)
                                    for n in 1:Nφ, j in 1:Ny, i in 1:Nx)

    @test maximum(abs, view(angular_G_data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, iz, :, :)) > 0
    @test angular_weighted_tendency ≈ 0 atol=1e-13 rtol=0
end
