import Profile
import Oceananigans
import Oceananigans.TimeSteppers: RungeKutta3TimeStepper

@testset "MonobandedWaveModel API" begin
    grid = RectilinearGrid(CPU();
                           size=(8, 6, 2),
                           x=(0, 8),
                           y=(0, 6),
                           z=(-1, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    model = MonobandedWaveModel(grid; timestepper=:RungeKutta3)

    @test keys(prognostic_fields(model)) == (:A, :AKx, :AKy)
    @test fields(model).A === model.action
    @test fields(model).AKx === model.wavenumber_moment.x
    @test fields(model).AKy === model.wavenumber_moment.y
    @test haskey(fields(model), :κ)
    @test haskey(fields(model), :uᴰx)
    @test haskey(fields(model), :Ĉx)
    @test !haskey(fields(model), :G)
    @test !hasfield(typeof(model), :boundary_conditions)
    @test axes(model.action.data, 3) == grid.Nz:grid.Nz
    @test model.timestepper isa RungeKutta3TimeStepper
    @test model.timestepper.Gⁿ.A !== model.action
    @test model.timestepper.G⁻.A !== model.action
    @test !model.previous_tendencies_ready
    @test eltype(model) === eltype(model.action)
    @test model.coupling === nothing
    @test model.sources === nothing

    zero_velocity_model = MonobandedWaveModel(grid;
                                              velocities=ZeroVelocities(),
                                              advection=nothing)
    @test zero_velocity_model.coupling === nothing

    u = Oceananigans.Fields.Field{Face, Center, Center}(grid)
    v = Oceananigans.Fields.Field{Center, Face, Center}(grid)
    set!(u, 0.1)
    set!(v, -0.2)

    named_tuple_velocity_model = MonobandedWaveModel(grid;
                                                     velocities=(; u, v),
                                                     advection=nothing)
    @test named_tuple_velocity_model.coupling isa MonobandedPrescribedCurrentCoupling
    @test named_tuple_velocity_model.coupling.qtransform.grid === grid

    pseudomomentum_model = MonobandedWaveModel(grid;
                                               velocities=PseudomomentumVelocities(),
                                               advection=nothing)
    @test pseudomomentum_model.coupling isa MonobandedPseudomomentumCoupling
    @test Ripple.grid(pseudomomentum_model.coupling.px) === grid
    @test Ripple.grid(pseudomomentum_model.coupling.py) === grid

    @test MonobandedWaveModel(grid; sources=BottomFriction(rate=1.0), advection=nothing).sources isa BottomFriction
    @test_throws ArgumentError MonobandedWaveModel(grid; sources=ExponentialWindInput(rate=1.0))
    @test_throws ArgumentError MonobandedWaveModel(grid; sources=LinearWindInput(rate=ones(8, 6)))
    @test_throws ArgumentError MonobandedWaveModel(grid; coupling=:currents)
    @test_throws ArgumentError MonobandedWaveModel(grid; timestepper=:SemiImplicitEuler)
    @test_throws ArgumentError MonobandedWaveModel(grid; clock=:clock)
    @test_throws ArgumentError MonobandedWaveModel(grid; boundary_conditions=(; N=Oceananigans.BoundaryConditions.FieldBoundaryConditions()))
    @test MonobandedWaveModel(grid; timestepper=:RK3, advection=nothing).timestepper isa RungeKutta3TimeStepper
    @test MonobandedWaveModel(grid; timestepper=:RungeKutta3, advection=nothing).timestepper isa RungeKutta3TimeStepper
    @test MonobandedWaveModel(grid; timestepper=:QuasiAdamsBashforth2, advection=nothing).timestepper.name === :AB2

    bounded_grid = RectilinearGrid(CPU();
                                   size=(4, 3, 2),
                                   x=(0, 4),
                                   y=(0, 3),
                                   z=(-1, 0),
                                   halo=(3, 3, 3),
                                   topology=(Bounded, Bounded, Bounded))

    default_bounded_model = MonobandedWaveModel(bounded_grid; advection=nothing)
    @test default_bounded_model.action.boundary_conditions.west isa Oceananigans.BoundaryConditions.NoFluxBoundaryCondition

    # The transport and gradient kernels hard-code no-flux at non-periodic edges,
    # so a user-supplied non-default BC would be silently ignored. Reject up
    # front rather than lying about what BC is in effect.
    value_bcs = Oceananigans.BoundaryConditions.FieldBoundaryConditions(
        bounded_grid,
        (Oceananigans.Grids.Center(), Oceananigans.Grids.Center(), Oceananigans.Grids.Center()),
        (:, :, bounded_grid.Nz:bounded_grid.Nz);
        west = Oceananigans.BoundaryConditions.ValueBoundaryCondition(1))

    supplied_action_with_value_bc = Oceananigans.Fields.CenterField(bounded_grid;
        indices=(:, :, bounded_grid.Nz:bounded_grid.Nz),
        boundary_conditions=value_bcs)

    @test_throws ArgumentError MonobandedWaveModel(bounded_grid;
        action=supplied_action_with_value_bc, advection=nothing)

    @test_throws ArgumentError MonobandedWaveModel(bounded_grid;
        boundary_conditions=(; A=Oceananigans.BoundaryConditions.FieldBoundaryConditions(
            west = Oceananigans.BoundaryConditions.ValueBoundaryCondition(2))),
        advection=nothing)

    @test_throws ArgumentError MonobandedWaveModel(bounded_grid;
        boundary_conditions=(; AKx=Oceananigans.BoundaryConditions.FieldBoundaryConditions(
            east = Oceananigans.BoundaryConditions.ValueBoundaryCondition(4))),
        advection=nothing)

    # Stretched horizontal grids would require variable-spacing gradient,
    # face-interp, and WENO; the current kernels assume uniform Δx, Δy.
    @test_throws ArgumentError MonobandedWaveModel(
        RectilinearGrid(CPU(); size=(5, 4, 1),
                        x=Float64[0, 1, 3, 7, 15, 31],
                        y=(0, 4), z=(-1, 0),
                        halo=(3, 3, 3),
                        topology=(Bounded, Periodic, Bounded)))
end

@testset "MonobandedWaveModel no-current diagnostics" begin
    grid = RectilinearGrid(CPU();
                           size=(4, 3, 1),
                           x=(0, 4),
                           y=(0, 3),
                           z=(-1, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    model = MonobandedWaveModel(grid; advection=nothing)
    set!(model; A=2.0, AKx=1.0, AKy=0.0)

    diagnostics = model.diagnostics
    @test all(interior(diagnostics.Kx) .≈ 0.5)
    @test all(interior(diagnostics.Ky) .≈ 0)
    @test all(interior(diagnostics.κ) .≈ 0.5)
    @test all(interior(diagnostics.uᴰx) .≈ 0)
    @test all(interior(diagnostics.uᴰy) .≈ 0)
    @test all(interior(diagnostics.Γxx) .≈ 0)
    @test all(interior(diagnostics.ZK) .≈ 0)

    g = model.gravitational_acceleration
    expected_Cx = sqrt(g / (4 * 0.5^3)) * 0.5
    @test all(interior(diagnostics.Cx) .≈ expected_Cx)
    @test all(interior(diagnostics.Cy) .≈ 0)
    @test all(interior(diagnostics.Ω) .≈ sqrt(g * 0.5))

    curl_grid = RectilinearGrid(CPU();
                                size=(4, 5, 1),
                                x=(0, 4),
                                y=(0, 5),
                                z=(-1, 0),
                                halo=(3, 3, 3),
                                topology=(Periodic, Bounded, Bounded))

    curl_model = MonobandedWaveModel(curl_grid; advection=nothing)
    set!(curl_model; A=1.0, AKx=(x, y, z) -> y, AKy=0.0)
    ZK = interior(curl_model.diagnostics.ZK)
    @test all(ZK[:, 2:end-1, :] .≈ -1)

    current_grid = RectilinearGrid(CPU();
                                   size=(4, 3, 4),
                                   x=(0, 4),
                                   y=(0, 3),
                                   z=(-2, 0),
                                   halo=(3, 3, 3),
                                   topology=(Periodic, Periodic, Bounded))

    u = Oceananigans.Fields.CenterField(current_grid)
    v = Oceananigans.Fields.CenterField(current_grid)
    set!(u, 0.2)
    set!(v, -0.1)

    current_model = MonobandedWaveModel(current_grid;
                                        velocities=PrescribedVelocities(; u, v),
                                        advection=nothing)
    set!(current_model; A=2.0, AKx=1.0, AKy=0.0)

    current_diagnostics = current_model.diagnostics
    g = current_model.gravitational_acceleration
    intrinsic_Cx = sqrt(g / (4 * 0.5^3)) * 0.5

    @test all(interior(current_diagnostics.uᴰx) .≈ 0.2)
    @test all(interior(current_diagnostics.uᴰy) .≈ -0.1)
    @test maximum(abs, interior(current_diagnostics.Hx)) < 1e-12
    @test maximum(abs, interior(current_diagnostics.Hy)) < 1e-12
    @test maximum(abs, interior(current_diagnostics.Γxx)) < 1e-12
    @test maximum(abs, interior(current_diagnostics.Γyx)) < 1e-12
    @test maximum(abs, interior(current_diagnostics.Γxy)) < 1e-12
    @test maximum(abs, interior(current_diagnostics.Γyy)) < 1e-12
    @test all(interior(current_diagnostics.Cx) .≈ intrinsic_Cx + 0.2)
    @test all(interior(current_diagnostics.Cy) .≈ -0.1)
    @test all(interior(current_diagnostics.Ω) .≈ sqrt(g * 0.5) + 0.5 * 0.2)

    shear_u = Oceananigans.Fields.CenterField(current_grid)
    shear_v = Oceananigans.Fields.CenterField(current_grid)
    set!(shear_u, (x, y, z) -> 0.3 + 0.05z)
    set!(shear_v, 0)

    shear_model = MonobandedWaveModel(current_grid;
                                      velocities=PrescribedVelocities(; u=shear_u, v=shear_v),
                                      advection=nothing)
    set!(shear_model; A=2.0, AKx=1.0, AKy=0.0)

    κ₀ = 0.5
    Kx₀ = 0.5
    qkernel = shear_model.coupling.qtransform.kernel
    faces = vertical_faces(shear_model.coupling.qtransform)
    depth = shear_model.coupling.current.depth

    expected_uᴰx = sum(interior(shear_u)[1, 1, ℓ] * q_cell_integral(qkernel, κ₀, faces[ℓ], faces[ℓ+1], depth)
                       for ℓ in 1:(length(faces)-1))
    expected_Hx = sum(interior(shear_u)[1, 1, ℓ] * q_cell_integral_kappa_derivative(qkernel, κ₀, faces[ℓ], faces[ℓ+1], depth)
                      for ℓ in 1:(length(faces)-1))
    expected_Cx = intrinsic_Cx + expected_uᴰx + expected_Hx * Kx₀

    shear_diagnostics = shear_model.diagnostics
    @test abs(expected_Hx) > 1e-6
    @test all(interior(shear_diagnostics.uᴰx) .≈ expected_uᴰx)
    @test all(interior(shear_diagnostics.Hx) .≈ expected_Hx)
    @test all(interior(shear_diagnostics.Cx) .≈ expected_Cx)
    @test maximum(abs, interior(shear_diagnostics.Γxx)) < 1e-12
    @test maximum(abs, interior(shear_diagnostics.Γyx)) < 1e-12
end

@testset "MonobandedWaveModel gradient diagnostics" begin
    chain_rule_errors = Float64[]

    for Nx in (32, 64, 128)
        grid = RectilinearGrid(CPU();
                               size=(Nx, 4, 16),
                               x=(0, 2π),
                               y=(0, 1),
                               z=(-2, 0),
                               halo=(3, 3, 3),
                               topology=(Periodic, Periodic, Bounded))

        u = Oceananigans.Fields.Field{Face, Center, Center}(grid)
        v = Oceananigans.Fields.Field{Center, Face, Center}(grid)
        set!(u, (x, y, z) -> 1 + 0.2 * z)
        set!(v, 0)

        model = MonobandedWaveModel(grid;
                                    velocities=PrescribedVelocities(; u, v),
                                    advection=nothing)
        set!(model; A=1, AKx=(x, y, z) -> 0.6 + 0.05 * sin(x), AKy=0)

        push!(chain_rule_errors, maximum(abs, interior(model.diagnostics.Γxx)))
    end

    @test chain_rule_errors[1] < 3e-6
    @test chain_rule_errors[2] < 0.3 * chain_rule_errors[1]
    @test chain_rule_errors[3] < 0.3 * chain_rule_errors[2]

    curl_errors = Float64[]

    for N in (32, 64, 128)
        grid = RectilinearGrid(CPU();
                               size=(N, N, 1),
                               x=(0, 2π),
                               y=(0, 2π),
                               z=(-1, 0),
                               halo=(3, 3, 3),
                               topology=(Periodic, Periodic, Bounded))

        model = MonobandedWaveModel(grid; advection=nothing)
        set!(model; A=1, AKx=(x, y, z) -> cos(y), AKy=(x, y, z) -> sin(x))

        x = Ripple.xnodes(grid)
        y = Ripple.ynodes(grid)
        expected_ZK = [cos(xᵢ) + sin(yⱼ) for xᵢ in x, yⱼ in y]
        ZK = interior(model.diagnostics.ZK)[:, :, 1]
        push!(curl_errors, maximum(abs, ZK .- expected_ZK))
    end

    @test curl_errors[1] < 2e-2
    @test curl_errors[2] < 0.3 * curl_errors[1]
    @test curl_errors[3] < 0.3 * curl_errors[2]
end

@testset "MonobandedWaveModel pseudomomentum" begin
    grid = RectilinearGrid(CPU();
                           size=(5, 4, 5),
                           x=(0, 5),
                           y=(0, 4),
                           z=[-2.0, -1.35, -0.7, -0.25, -0.05, 0.0],
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    model = MonobandedWaveModel(grid; advection=nothing)
    set!(model;
         A=(x, y, z) -> 1.3 + 0.1 * cos(x),
         AKx=(x, y, z) -> 0.4 + 0.03 * sin(y),
         AKy=(x, y, z) -> -0.2 + 0.02 * cos(x + y))

    px, py = pseudomomentum_fields(model)
    @test Ripple.grid(px) === grid
    @test Ripple.grid(py) === grid
    @test location(px) == (Center, Center, Center)
    @test location(py) == (Center, Center, Center)

    AKx = Array(interior(model.wavenumber_moment.x)[:, :, 1])
    AKy = Array(interior(model.wavenumber_moment.y)[:, :, 1])
    @test vertical_integral(px) ≈ AKx atol=1e-12 rtol=0
    @test vertical_integral(py) ≈ AKy atol=1e-12 rtol=0

    i, j, k = 3, 2, 4
    qkernel = QKernel(eltype(model))
    faces = zfaces(grid)
    κ = interior(model.diagnostics.κ)[i, j, 1]
    qΔz = q_cell_integral(qkernel, κ, faces[k], faces[k+1], 2.0)
    Δz = abs(faces[k+1] - faces[k])
    @test Ripple.field_storage(px)[i, j, k] ≈ AKx[i, j] * qΔz / Δz atol=1e-12 rtol=0
    @test Ripple.field_storage(py)[i, j, k] ≈ AKy[i, j] * qΔz / Δz atol=1e-12 rtol=0

    model_grid = RectilinearGrid(CPU();
                                 size=(5, 4, 1),
                                 x=(0, 5),
                                 y=(0, 4),
                                 z=(-1, 0),
                                 halo=(3, 3, 3),
                                 topology=(Periodic, Periodic, Bounded))

    q_grid = RectilinearGrid(CPU();
                             size=(5, 4, 6),
                             x=(0, 5),
                             y=(0, 4),
                             z=[-3.0, -2.0, -1.2, -0.55, -0.2, -0.05, 0.0],
                             halo=(3, 3, 3),
                             topology=(Periodic, Periodic, Bounded))

    u = Oceananigans.Fields.Field{Face, Center, Center}(q_grid)
    v = Oceananigans.Fields.Field{Center, Face, Center}(q_grid)
    set!(u, 0)
    set!(v, 0)

    coupled = MonobandedWaveModel(model_grid;
                                  velocities=PrescribedVelocities(; u, v, q_grid),
                                  advection=nothing)
    set!(coupled;
         A=(x, y, z) -> 1.1 + 0.05 * sin(x),
         AKx=(x, y, z) -> 0.3 + 0.04 * cos(y),
         AKy=(x, y, z) -> 0.1 * sin(x + y))

    qpx, qpy = pseudomomentum_fields(coupled)
    @test Ripple.grid(qpx) === q_grid
    @test Ripple.grid(qpy) === q_grid

    qAKx = Array(interior(coupled.wavenumber_moment.x)[:, :, 1])
    qAKy = Array(interior(coupled.wavenumber_moment.y)[:, :, 1])
    @test vertical_integral(qpx) ≈ qAKx atol=1e-12 rtol=0
    @test vertical_integral(qpy) ≈ qAKy atol=1e-12 rtol=0

    i, j, k = 4, 3, 2
    qfaces = zfaces(q_grid)
    κ = interior(coupled.diagnostics.κ)[i, j, 1]
    qΔz = q_cell_integral(coupled.coupling.qtransform.kernel, κ, qfaces[k], qfaces[k+1], 3.0)
    Δz = abs(qfaces[k+1] - qfaces[k])
    @test Ripple.field_storage(qpx)[i, j, k] ≈ qAKx[i, j] * qΔz / Δz atol=1e-12 rtol=0
    @test Ripple.field_storage(qpy)[i, j, k] ≈ qAKy[i, j] * qΔz / Δz atol=1e-12 rtol=0
end

@testset "MonobandedWaveModel pseudomomentum velocities" begin
    grid = RectilinearGrid(CPU();
                           size=(4, 3, 6),
                           x=(0, 4),
                           y=(0, 3),
                           z=[-2.0, -1.3, -0.8, -0.45, -0.2, -0.06, 0.0],
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    model = MonobandedWaveModel(grid;
                                velocities=PseudomomentumVelocities(),
                                advection=nothing)

    set!(model; A=2.0, AKx=1.0, AKy=-0.4)

    coupling = model.coupling
    diagnostics = model.diagnostics
    @test coupling isa MonobandedPseudomomentumCoupling
    @test update_coupling!(model) === model

    AKx = Array(interior(model.wavenumber_moment.x)[:, :, 1])
    AKy = Array(interior(model.wavenumber_moment.y)[:, :, 1])
    @test vertical_integral(coupling.px) ≈ AKx atol=1e-12 rtol=0
    @test vertical_integral(coupling.py) ≈ AKy atol=1e-12 rtol=0

    px = Ripple.field_storage(coupling.px)
    py = Ripple.field_storage(coupling.py)
    qkernel = coupling.qtransform.kernel
    faces = vertical_faces(coupling.qtransform)
    depth = coupling.depth
    κ = interior(diagnostics.κ)[1, 1, 1]

    expected_uᴰx = sum(px[1, 1, ℓ] * q_cell_integral(qkernel, κ, faces[ℓ], faces[ℓ+1], depth)
                       for ℓ in 1:(length(faces)-1))
    expected_uᴰy = sum(py[1, 1, ℓ] * q_cell_integral(qkernel, κ, faces[ℓ], faces[ℓ+1], depth)
                       for ℓ in 1:(length(faces)-1))
    expected_Hx = sum(px[1, 1, ℓ] * q_cell_integral_kappa_derivative(qkernel, κ, faces[ℓ], faces[ℓ+1], depth)
                      for ℓ in 1:(length(faces)-1))
    expected_Hy = sum(py[1, 1, ℓ] * q_cell_integral_kappa_derivative(qkernel, κ, faces[ℓ], faces[ℓ+1], depth)
                      for ℓ in 1:(length(faces)-1))

    @test abs(expected_uᴰx) > 0
    @test abs(expected_Hx) > 0
    @test all(interior(diagnostics.uᴰx) .≈ expected_uᴰx)
    @test all(interior(diagnostics.uᴰy) .≈ expected_uᴰy)
    @test all(interior(diagnostics.Hx) .≈ expected_Hx)
    @test all(interior(diagnostics.Hy) .≈ expected_Hy)
    @test maximum(abs, interior(diagnostics.Γxx)) < 1e-12
    @test maximum(abs, interior(diagnostics.Γyx)) < 1e-12
    @test maximum(abs, interior(diagnostics.Γxy)) < 1e-12
    @test maximum(abs, interior(diagnostics.Γyy)) < 1e-12

    Kx = interior(diagnostics.Kx)[1, 1, 1]
    Ky = interior(diagnostics.Ky)[1, 1, 1]
    dotKH = Kx * expected_Hx + Ky * expected_Hy
    intrinsic_factor = sqrt(model.gravitational_acceleration / (4κ^3))
    expected_Cx = intrinsic_factor * Kx + expected_uᴰx + dotKH * Kx / κ
    expected_Cy = intrinsic_factor * Ky + expected_uᴰy + dotKH * Ky / κ
    @test all(interior(diagnostics.Cx) .≈ expected_Cx)
    @test all(interior(diagnostics.Cy) .≈ expected_Cy)

    q_grid = RectilinearGrid(CPU();
                             size=(4, 3, 4),
                             x=(0, 4),
                             y=(0, 3),
                             z=(-3, 0),
                             halo=(3, 3, 3),
                             topology=(Periodic, Periodic, Bounded))

    q_model = MonobandedWaveModel(grid;
                                  velocities=PseudomomentumVelocities(; q_grid),
                                  advection=nothing)
    set!(q_model; A=2.0, AKx=1.0, AKy=0.0)
    @test q_model.coupling.qtransform.grid === q_grid
    @test Ripple.grid(q_model.coupling.px) === q_grid
    @test vertical_integral(q_model.coupling.px) ≈ Array(interior(q_model.wavenumber_moment.x)[:, :, 1]) atol=1e-12 rtol=0
end

@testset "MonobandedWaveModel spectral consistency" begin
    function spectral_consistency_errors(σratio, Nκ, Nφ)
        Nx, Ny = 4, 8
        Lx, Ly = 4, 2π
        κ₀ = 0.5
        σ = σratio * κ₀
        shear = 0.03

        grid = RectilinearGrid(CPU();
                               size=(Nx, Ny, 1),
                               x=(0, Lx),
                               y=(0, Ly),
                               z=(-1, 0),
                               halo=(3, 3, 3),
                               topology=(Periodic, Periodic, Bounded))

        spectral_grid = PolarWaveVectorGrid(;
                                            κ=range(0.25, 0.75; length=Nκ),
                                            φ=range(-π, π; length=Nφ+1)[1:end-1])

        u = Oceananigans.Fields.Field{Face, Center, Center}(grid)
        v = Oceananigans.Fields.Field{Center, Face, Center}(grid)
        set!(u, (x, y, z) -> shear * sin(y))
        set!(v, 0)

        spectral = SpectralWaveModel(grid, spectral_grid;
                                     velocities=PrescribedVelocities(; u, v),
                                     timestepper=:ForwardEuler)

        set!(spectral.action, (x, y, kx, ky) -> exp(-((kx - κ₀)^2 + ky^2) / (2σ^2)))

        A = Array(interior(m0(spectral.action))[:, :, 1])
        Mx, My = first_moment(spectral.action)
        AKx = Array(interior(Mx)[:, :, 1])
        AKy = Array(interior(My)[:, :, 1])

        monobanded = MonobandedWaveModel(grid;
                                         velocities=PrescribedVelocities(; u, v),
                                         timestepper=:ForwardEuler)
        set!(monobanded;
             A=(x, y, z) -> A[1, 1],
             AKx=(x, y, z) -> AKx[1, 1],
             AKy=(x, y, z) -> AKy[1, 1])

        compute_tendencies!(spectral)
        compute_tendencies!(monobanded)

        spectral_A_t = Array(interior(m0(spectral.tendencies))[:, :, 1])
        spectral_Mx_t, spectral_My_t = first_moment(spectral.tendencies)
        spectral_AKx_t = Array(interior(spectral_Mx_t)[:, :, 1])
        spectral_AKy_t = Array(interior(spectral_My_t)[:, :, 1])

        monobanded_A_t = Array(interior(monobanded.timestepper.Gⁿ.A)[:, :, 1])
        monobanded_AKx_t = Array(interior(monobanded.timestepper.Gⁿ.AKx)[:, :, 1])
        monobanded_AKy_t = Array(interior(monobanded.timestepper.Gⁿ.AKy)[:, :, 1])

        tendency_scale = maximum(abs, monobanded_AKy_t)
        @test tendency_scale > 0

        return (A = maximum(abs, spectral_A_t .- monobanded_A_t) / tendency_scale,
                AKx = maximum(abs, spectral_AKx_t .- monobanded_AKx_t) / tendency_scale,
                AKy = maximum(abs, spectral_AKy_t .- monobanded_AKy_t) / tendency_scale)
    end

    configs = ((0.08, 25, 64), (0.04, 49, 128), (0.02, 97, 256))
    errors = map(config -> spectral_consistency_errors(config...), configs)
    decreases_or_roundoff(values; floor=1e-12) =
        maximum(values) < floor || values[3] < values[2] < values[1]

    @test all(error.AKy < 3e-2 for error in errors)
    @test decreases_or_roundoff(getproperty.(errors, :A))
    @test decreases_or_roundoff(getproperty.(errors, :AKx))
    @test errors[3].A < 1e-2
    @test errors[3].AKx < 1e-2

    function short_time_evolution_errors()
        Nx, Ny = 4, 16
        Lx, Ly = 4, 2π
        κ₀ = 0.5
        σ = 0.02 * κ₀
        shear = 0.05

        grid = RectilinearGrid(CPU();
                               size=(Nx, Ny, 1),
                               x=(0, Lx),
                               y=(0, Ly),
                               z=(-1, 0),
                               halo=(3, 3, 3),
                               topology=(Periodic, Periodic, Bounded))

        spectral_grid = PolarWaveVectorGrid(;
                                            κ=range(0.25, 0.75; length=97),
                                            φ=range(-π, π; length=385)[1:end-1])

        u = Oceananigans.Fields.Field{Face, Center, Center}(grid)
        v = Oceananigans.Fields.Field{Center, Face, Center}(grid)
        set!(u, (x, y, z) -> shear * sin(y))
        set!(v, 0)

        spectral = SpectralWaveModel(grid, spectral_grid;
                                     velocities=PrescribedVelocities(; u, v),
                                     timestepper=:ForwardEuler)

        set!(spectral.action, (x, y, kx, ky) -> exp(-((kx - κ₀)^2 + ky^2) / (2σ^2)))

        A₀ = Array(interior(m0(spectral.action))[:, :, 1])
        Mx₀, My₀ = first_moment(spectral.action)
        AKx₀ = Array(interior(Mx₀)[:, :, 1])
        AKy₀ = Array(interior(My₀)[:, :, 1])

        monobanded = MonobandedWaveModel(grid;
                                         velocities=PrescribedVelocities(; u, v),
                                         timestepper=:ForwardEuler)
        set!(monobanded;
             A=(x, y, z) -> A₀[1, 1],
             AKx=(x, y, z) -> AKx₀[1, 1],
             AKy=(x, y, z) -> AKy₀[1, 1])

        for _ in 1:10
            time_step!(spectral, 0.02)
            time_step!(monobanded, 0.02)
        end

        spectral_A = Array(interior(m0(spectral.action))[:, :, 1])
        spectral_AKx_field, spectral_AKy_field = first_moment(spectral.action)
        spectral_AKx = Array(interior(spectral_AKx_field)[:, :, 1])
        spectral_AKy = Array(interior(spectral_AKy_field)[:, :, 1])

        monobanded_A = Array(interior(monobanded.action)[:, :, 1])
        monobanded_AKx = Array(interior(monobanded.wavenumber_moment.x)[:, :, 1])
        monobanded_AKy = Array(interior(monobanded.wavenumber_moment.y)[:, :, 1])

        return (A = maximum(abs, spectral_A .- monobanded_A) / maximum(abs, monobanded_A),
                AKx = maximum(abs, spectral_AKx .- monobanded_AKx) / maximum(abs, monobanded_AKx),
                AKy = maximum(abs, spectral_AKy .- monobanded_AKy) / maximum(abs, monobanded_AKy))
    end

    short_time_errors = short_time_evolution_errors()
    @test short_time_errors.A < 5e-2
    @test short_time_errors.AKx < 5e-2
    @test short_time_errors.AKy < 5e-2
end

@testset "MonobandedWaveModel backend readiness" begin
    grid = RectilinearGrid(CPU();
                           size=(8, 8, 4),
                           x=(0, 8),
                           y=(0, 8),
                           z=(-1, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    model = MonobandedWaveModel(grid;
                                velocities=PseudomomentumVelocities(),
                                timestepper=:RungeKutta3)
    set!(model;
         A=(x, y, z) -> 1 + 0.1 * sin(x),
         AKx=(x, y, z) -> 0.4 * (1 + 0.1 * sin(x)),
         AKy=0)

    inferred_model = @inferred Oceananigans.TimeSteppers.update_state!(model)
    @test inferred_model === model

    inferred_tendencies = @inferred compute_tendencies!(model.timestepper.Gⁿ, model)
    @test inferred_tendencies === model.timestepper.Gⁿ

    inferred_cfl = @inferred cfl(model)
    @test inferred_cfl isa eltype(model)

    disabled = MonobandedWaveModel(grid; advection=nothing)
    @test (@inferred cfl(disabled)) === zero(eltype(disabled))

    function optional_gpu_backend_module(name::Symbol)
        Base.find_package(String(name)) === nothing && return nothing

        try
            return Base.require(@__MODULE__, name)
        catch err
            @info "Skipping optional $name GPU backend for MonobandedWaveModel tests" exception=(err, catch_backtrace())
            return nothing
        end
    end

    function backend_is_functional(mod)
        isdefined(mod, :functional) || return true
        # The backend package may have been precompiled inside this test
        # session; calling `functional()` directly hits a world-age error
        # because the loaded method is newer than the calling world.
        return Base.invokelatest(getproperty(mod, :functional))
    end

    function optional_monobanded_gpu_architecture()
        cuda = optional_gpu_backend_module(:CUDA)
        if cuda !== nothing && backend_is_functional(cuda) && isdefined(cuda, :CUDABackend)
            return GPU(getproperty(cuda, :CUDABackend)(always_inline=true))
        end

        metal = optional_gpu_backend_module(:Metal)
        if metal !== nothing && backend_is_functional(metal) && isdefined(metal, :MetalBackend)
            return GPU(getproperty(metal, :MetalBackend)())
        end

        amdgpu = optional_gpu_backend_module(:AMDGPU)
        if amdgpu !== nothing && backend_is_functional(amdgpu) && isdefined(amdgpu, :ROCBackend)
            return GPU(getproperty(amdgpu, :ROCBackend)())
        end

        return nothing
    end

    function monobanded_backend_snapshot(arch)
        parity_grid = RectilinearGrid(arch, Float32;
                                      size=(8, 8, 4),
                                      x=(0, 8),
                                      y=(0, 8),
                                      z=(-1, 0),
                                      halo=(3, 3, 3),
                                      topology=(Periodic, Periodic, Bounded))

        parity_model = MonobandedWaveModel(parity_grid;
                                           velocities=PseudomomentumVelocities(),
                                           timestepper=:RungeKutta3)
        set!(parity_model;
             A=(x, y, z) -> 1 + 0.05 * sin(2π * x / 8) * cos(2π * y / 8),
             AKx=(x, y, z) -> 0.4 * (1 + 0.05 * sin(2π * x / 8)),
             AKy=(x, y, z) -> 0.1 * (1 + 0.03 * cos(2π * y / 8)))

        compute_tendencies!(parity_model)
        before_step = monobanded_model_snapshot(parity_model)
        time_step!(parity_model, 0.01f0)
        after_step = monobanded_model_snapshot(parity_model)

        return (before_step=before_step, after_step=after_step)
    end

    function monobanded_model_snapshot(model)
        diagnostics = model.diagnostics
        return (A=Array(interior(model.action)),
                AKx=Array(interior(model.wavenumber_moment.x)),
                AKy=Array(interior(model.wavenumber_moment.y)),
                Kx=Array(interior(diagnostics.Kx)),
                Ky=Array(interior(diagnostics.Ky)),
                κ=Array(interior(diagnostics.κ)),
                uᴰx=Array(interior(diagnostics.uᴰx)),
                uᴰy=Array(interior(diagnostics.uᴰy)),
                Hx=Array(interior(diagnostics.Hx)),
                Hy=Array(interior(diagnostics.Hy)),
                Cx=Array(interior(diagnostics.Cx)),
                Cy=Array(interior(diagnostics.Cy)),
                Γxx=Array(interior(diagnostics.Γxx)),
                Γyx=Array(interior(diagnostics.Γyx)),
                Γxy=Array(interior(diagnostics.Γxy)),
                Γyy=Array(interior(diagnostics.Γyy)),
                GA=Array(interior(model.timestepper.Gⁿ.A)),
                GAKx=Array(interior(model.timestepper.Gⁿ.AKx)),
                GAKy=Array(interior(model.timestepper.Gⁿ.AKy)))
    end

    function maximum_snapshot_difference(a::NamedTuple, b::NamedTuple)
        differences = map(keys(a)) do name
            av = getfield(a, name)
            bv = getfield(b, name)
            av isa NamedTuple ? maximum_snapshot_difference(av, bv) :
                                maximum(abs, av .- bv)
        end

        return maximum(differences)
    end

    function snapshot_eltypes(snapshot::NamedTuple)
        types = DataType[]
        for name in keys(snapshot)
            value = getfield(snapshot, name)
            if value isa NamedTuple
                append!(types, snapshot_eltypes(value))
            else
                push!(types, eltype(value))
            end
        end
        return types
    end

    float32_snapshot = monobanded_backend_snapshot(CPU())
    @test all(==(Float32), snapshot_eltypes(float32_snapshot))

    gpu_arch = optional_monobanded_gpu_architecture()
    if gpu_arch !== nothing
        cpu_snapshot = monobanded_backend_snapshot(CPU())
        gpu_snapshot = monobanded_backend_snapshot(gpu_arch)
        @test maximum_snapshot_difference(cpu_snapshot, gpu_snapshot) <= 1e-5
    else
        @info "Skipping optional MonobandedWaveModel GPU parity test; CUDA, Metal, and AMDGPU are unavailable or nonfunctional."
    end

    function has_model_sized_array_allocation(model)
        compute_tendencies!(model)
        GC.gc()
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate=1 compute_tendencies!(model)
        allocations = Profile.Allocs.fetch().allocs
        Nx, Ny = horizontal_size(model.grid)
        model_field_bytes = Nx * Ny * sizeof(eltype(model))
        return any(allocations) do allocation
            allocation.type isa Type &&
                allocation.type <: Array &&
                allocation.size >= model_field_bytes
        end
    end

    for N in (8, 16)
        allocation_grid = RectilinearGrid(CPU();
                                          size=(N, N, 4),
                                          x=(0, N),
                                          y=(0, N),
                                          z=(-1, 0),
                                          halo=(3, 3, 3),
                                          topology=(Periodic, Periodic, Bounded))

        allocation_model = MonobandedWaveModel(allocation_grid;
                                               velocities=PseudomomentumVelocities(),
                                               timestepper=:RungeKutta3)
        set!(allocation_model;
             A=(x, y, z) -> 1 + 0.05 * sin(2π * x / N) * cos(2π * y / N),
             AKx=(x, y, z) -> 0.4 * (1 + 0.05 * sin(2π * x / N)),
             AKy=(x, y, z) -> 0.1 * (1 + 0.03 * cos(2π * y / N)))

        @test !has_model_sized_array_allocation(allocation_model)
    end

    source_path = normpath(joinpath(@__DIR__, "..", "..", "src", "Models", "monobanded_wave_model.jl"))
    source_text = read(source_path, String)
    forbidden_float_patterns = ("Float64", "0.0", "1.0")
    @test all(!occursin(pattern, source_text) for pattern in forbidden_float_patterns)

    host_grid_loop_patterns = (
        r"for\s+.*\bin\s+1\s*:\s*Nx\b",
        r"for\s+.*\bin\s+1\s*:\s*Ny\b",
        r"for\s+.*\bin\s+.*horizontal_size",
        r"for\s+.*\bin\s+axes\(.*,\s*[12]\s*\)"
    )

    @test all(!occursin(pattern, source_text) for pattern in host_grid_loop_patterns)
end

@testset "MonobandedWaveModel time stepping" begin
    grid = RectilinearGrid(CPU();
                           size=(16, 8, 1),
                           x=(0, 16),
                           y=(0, 8),
                           z=(-1, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    disabled = MonobandedWaveModel(grid; advection=nothing)
    set!(disabled;
         A=(x, y, z) -> 1 + 0.1 * sin(2π * x / 16),
         AKx=(x, y, z) -> 0.25 * (1 + 0.1 * sin(2π * x / 16)),
         AKy=0.0)

    A₀ = copy(interior(disabled.action))
    AKx₀ = copy(interior(disabled.wavenumber_moment.x))
    time_step!(disabled, 0.1)
    @test disabled.clock.iteration == 1
    @test cfl(disabled) == 0
    @test interior(disabled.action) ≈ A₀ atol=1e-14 rtol=0
    @test interior(disabled.wavenumber_moment.x) ≈ AKx₀ atol=1e-14 rtol=0

    transported = MonobandedWaveModel(grid; timestepper=:ForwardEuler)
    set!(transported;
         A=(x, y, z) -> 1 + 0.1 * sin(2π * x / 16),
         AKx=(x, y, z) -> 0.25 * (1 + 0.1 * sin(2π * x / 16)),
         AKy=0.0)

    initial_action = sum(interior(transported.action))
    initial_moment = sum(interior(transported.wavenumber_moment.x))
    compute_tendencies!(transported)
    @test maximum(abs, interior(transported.timestepper.Gⁿ.A)) > 0

    time_step!(transported, 0.05)
    @test transported.clock.iteration == 1
    @test sum(interior(transported.action)) ≈ initial_action atol=1e-12 rtol=0
    @test sum(interior(transported.wavenumber_moment.x)) ≈ initial_moment atol=1e-12 rtol=0
    @test minimum(interior(transported.action)) >= 0

    Δx = first(Ripple.xspacings(grid))
    Δy = first(Ripple.yspacings(grid))
    per_cell_ratio = abs.(interior(transported.diagnostics.Cx)) ./ Δx .+
                     abs.(interior(transported.diagnostics.Cy)) ./ Δy
    expected_cfl = maximum(per_cell_ratio) * transported.clock.last_Δt
    @test cfl(transported) ≈ expected_cfl

    flat_grid = RectilinearGrid(CPU();
                                size=(12, 1),
                                y=(0, 12),
                                z=(-1, 0),
                                halo=(3, 3),
                                topology=(Flat, Periodic, Bounded))

    flat_transport = MonobandedWaveModel(flat_grid; timestepper=:ForwardEuler)
    set!(flat_transport;
         A=(y, z) -> 1 + 0.1 * sin(2π * y / 12),
         AKx=(y, z) -> 0.5 * (1 + 0.1 * sin(2π * y / 12)),
         AKy=(y, z) -> 0.2 * (1 + 0.1 * sin(2π * y / 12)))

    compute_tendencies!(flat_transport)
    @test all(isfinite, interior(flat_transport.timestepper.Gⁿ.A))
    @test all(isfinite, interior(flat_transport.timestepper.Gⁿ.AKx))
    @test all(isfinite, interior(flat_transport.timestepper.Gⁿ.AKy))

    time_step!(flat_transport, 0.02)
    @test all(isfinite, interior(flat_transport.action))
    @test all(isfinite, interior(flat_transport.wavenumber_moment.x))
    @test all(isfinite, interior(flat_transport.wavenumber_moment.y))
    @test cfl(flat_transport) < Inf

    rk3 = MonobandedWaveModel(grid; timestepper=:RungeKutta3)
    set!(rk3;
         A=(x, y, z) -> 1 + 0.05 * sin(2π * x / 16) * cos(2π * y / 8),
         AKx=(x, y, z) -> 0.5 * (1 + 0.05 * sin(2π * x / 16) * cos(2π * y / 8)),
         AKy=0.0)

    rk3_Gⁿ_A = rk3.timestepper.Gⁿ.A
    rk3_G⁻_A = rk3.timestepper.G⁻.A
    initial_rk3_action = sum(interior(rk3.action))
    initial_rk3_moment = sum(interior(rk3.wavenumber_moment.x))

    for _ in 1:100
        time_step!(rk3, 0.02)
        @test minimum(interior(rk3.action)) >= 0
    end

    @test rk3.timestepper.Gⁿ.A === rk3_Gⁿ_A
    @test rk3.timestepper.G⁻.A === rk3_G⁻_A
    @test cfl(rk3) <= 0.2
    @test sum(interior(rk3.action)) ≈ initial_rk3_action atol=1e-10 rtol=0
    @test sum(interior(rk3.wavenumber_moment.x)) ≈ initial_rk3_moment atol=1e-10 rtol=0

    bounded_grid = RectilinearGrid(CPU();
                                   size=(16, 8, 1),
                                   x=(0, 16),
                                   y=(0, 8),
                                   z=(-1, 0),
                                   halo=(3, 3, 3),
                                   topology=(Bounded, Bounded, Bounded))

    bounded = MonobandedWaveModel(bounded_grid; timestepper=:ForwardEuler)
    set!(bounded;
         A=(x, y, z) -> 1 + 0.05 * cos(2π * x / 16) * cos(2π * y / 8),
         AKx=(x, y, z) -> 0.4 * (1 + 0.05 * cos(2π * x / 16) * cos(2π * y / 8)),
         AKy=0.0)

    initial_bounded_action = sum(interior(bounded.action))
    initial_bounded_moment = sum(interior(bounded.wavenumber_moment.x))
    time_step!(bounded, 0.02)
    @test sum(interior(bounded.action)) ≈ initial_bounded_action atol=1e-12 rtol=0
    @test sum(interior(bounded.wavenumber_moment.x)) ≈ initial_bounded_moment atol=1e-12 rtol=0
    @test minimum(interior(bounded.action)) >= 0
end

@testset "MonobandedWaveModel source terms" begin
    grid = RectilinearGrid(CPU();
                           size=(4, 4, 1),
                           x=(0, 4),
                           y=(0, 4),
                           z=(-1, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    growth = MonobandedWaveModel(grid;
                                 advection=nothing,
                                 sources=LinearWindInput(rate=0.2),
                                 timestepper=:ForwardEuler)
    set!(growth; A=2.0, AKx=1.0, AKy=-0.4)
    compute_tendencies!(growth)

    @test all(interior(growth.timestepper.Gⁿ.A) .≈ 0.4)
    @test all(interior(growth.timestepper.Gⁿ.AKx) .≈ 0.2)
    @test all(interior(growth.timestepper.Gⁿ.AKy) .≈ -0.08)

    time_step!(growth, 0.5)
    @test all(interior(growth.action) .≈ 2.2)
    @test all(interior(growth.wavenumber_moment.x) .≈ 1.1)
    @test all(interior(growth.wavenumber_moment.y) .≈ -0.44)
    @test all(interior(growth.diagnostics.Kx) .≈ 0.5)
    @test all(interior(growth.diagnostics.Ky) .≈ -0.2)

    balanced = MonobandedWaveModel(grid;
                                   advection=nothing,
                                   sources=SourceTermSet(LinearWindInput(rate=0.2),
                                                         BottomFriction(rate=0.2)),
                                   timestepper=:ForwardEuler)
    set!(balanced; A=2.0, AKx=1.0, AKy=-0.4)
    compute_tendencies!(balanced)
    @test maximum(abs, interior(balanced.timestepper.Gⁿ.A)) < 1e-14
    @test maximum(abs, interior(balanced.timestepper.Gⁿ.AKx)) < 1e-14
    @test maximum(abs, interior(balanced.timestepper.Gⁿ.AKy)) < 1e-14

    bottom = MonobandedWaveModel(grid;
                                 advection=nothing,
                                 sources=BottomFriction(rate=0.4,
                                                        depth=0.5,
                                                        reference_depth=1.0,
                                                        wavenumber_power=2,
                                                        reference_wavenumber=0.25),
                                 timestepper=:ForwardEuler)
    set!(bottom; A=2.0, AKx=1.0, AKy=0.0)
    compute_tendencies!(bottom)

    expected_rate = -0.4 * (1.0 / 0.5) * (0.5 / 0.25)^2
    expected_source_A = expected_rate * 2.0
    @test all(interior(bottom.timestepper.Gⁿ.A) .≈ expected_source_A)
    @test all(interior(bottom.timestepper.Gⁿ.AKx) .≈ 0.5 * expected_source_A)
    @test all(interior(bottom.timestepper.Gⁿ.AKy) .≈ 0)
end

@testset "MonobandedWaveModel WENO transport" begin
    function smooth_transport_error(N; advection)
        grid = RectilinearGrid(CPU();
                               size=(N, 4, 1),
                               x=(0, 2π),
                               y=(0, 1),
                               z=(-1, 0),
                               halo=(3, 3, 3),
                               topology=(Periodic, Periodic, Bounded))

        K = 0.5
        model = MonobandedWaveModel(grid; advection, timestepper=:ForwardEuler)
        set!(model;
             A=(x, y, z) -> 1 + 0.1 * sin(x),
             AKx=(x, y, z) -> K * (1 + 0.1 * sin(x)),
             AKy=0)

        compute_tendencies!(model)

        Cx = interior(model.diagnostics.Cx)[1, 1, 1]
        x = Ripple.xnodes(grid)
        tendency = Array(interior(model.timestepper.Gⁿ.A)[:, 1, 1])
        expected = [-Cx * 0.1 * cos(xᵢ) for xᵢ in x]

        return maximum(abs, tendency .- expected)
    end

    weno_errors = [smooth_transport_error(N; advection=WENO()) for N in (32, 64, 128)]
    fallback_errors = [smooth_transport_error(N; advection=Centered()) for N in (32, 64, 128)]

    @test weno_errors[2] < 0.3 * weno_errors[1]
    @test weno_errors[3] < 0.3 * weno_errors[2]
    @test all(weno_errors .< 0.1 .* fallback_errors)
end


# --- regression tests for the fixes in monobanded_wave_model_cleanup_plan.md ----

@testset "MonobandedWaveModel refraction without transport (T1)" begin
    grid = RectilinearGrid(CPU();
                           size=(4, 16, 4),
                           x=(0, 4),
                           y=(0, 2π),
                           z=(-1, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    u = Oceananigans.Fields.Field{Face, Center, Center}(grid)
    v = Oceananigans.Fields.Field{Center, Face, Center}(grid)
    set!(u, 0.1)
    set!(v, (x, y, z) -> 0.05 * sin(y))

    model = MonobandedWaveModel(grid;
                                velocities=PrescribedVelocities(; u, v),
                                advection=nothing)
    set!(model; A=1.0, AKx=0.0, AKy=0.5)
    compute_tendencies!(model)

    Γyy_int  = interior(model.diagnostics.Γyy)[:, :, 1]
    GAKy_int = interior(model.timestepper.Gⁿ.AKy)[:, :, 1]
    GAKx_int = interior(model.timestepper.Gⁿ.AKx)[:, :, 1]
    GA_int   = interior(model.timestepper.Gⁿ.A)[:, :, 1]

    # G_AKy = -A * Ky * Γyy  (the only nonzero refraction contribution here).
    @test maximum(abs, GAKy_int .+ 0.5 .* Γyy_int) < 1e-12
    # G_A and G_AKx are zero: advection is off, the current is x-independent,
    # and Kx = 0 so Kx·Γxy and Kx·Γxx vanish.
    @test maximum(abs, GA_int)   < 1e-14
    @test maximum(abs, GAKx_int) < 1e-14
end

@testset "MonobandedWaveModel split transport + refraction (T7)" begin
    # When transport is on, the split must reproduce the original combined
    # moment tendency. Sanity check: transport-only (with Γ zeroed) and the
    # full tendency differ on AKx, AKy but not on A.
    grid = RectilinearGrid(CPU();
                           size=(8, 16, 4),
                           x=(0, 8),
                           y=(0, 2π),
                           z=(-1, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    u = Oceananigans.Fields.Field{Face, Center, Center}(grid)
    v = Oceananigans.Fields.Field{Center, Face, Center}(grid)
    set!(u, (x, y, z) -> 0.05 * cos(y))
    set!(v, (x, y, z) -> 0.05 * sin(y))

    model = MonobandedWaveModel(grid;
                                velocities=PrescribedVelocities(; u, v),
                                advection=WENO(),
                                timestepper=:ForwardEuler)
    set!(model;
         A   = (x, y, z) -> 1 + 0.1 * sin(2π * x / 8),
         AKx = (x, y, z) -> 0.3 * (1 + 0.1 * sin(2π * x / 8)),
         AKy = (x, y, z) -> -0.1)

    compute_tendencies!(model)
    GA   = copy(interior(model.timestepper.Gⁿ.A))
    GAKx = copy(interior(model.timestepper.Gⁿ.AKx))
    GAKy = copy(interior(model.timestepper.Gⁿ.AKy))

    fill!(interior(model.diagnostics.Γxx), 0)
    fill!(interior(model.diagnostics.Γyx), 0)
    fill!(interior(model.diagnostics.Γxy), 0)
    fill!(interior(model.diagnostics.Γyy), 0)
    Ripple.compute_monobanded_transport_tendency!(model.timestepper.Gⁿ, model)

    @test interior(model.timestepper.Gⁿ.A) == GA
    @test interior(model.timestepper.Gⁿ.AKx) != GAKx
    @test interior(model.timestepper.Gⁿ.AKy) != GAKy
end

@testset "MonobandedWaveModel K diagnosis with negative A (T6)" begin
    grid = RectilinearGrid(CPU();
                           size=(4, 3, 1),
                           x=(0, 4),
                           y=(0, 3),
                           z=(-1, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    model = MonobandedWaveModel(grid;
                                advection=nothing,
                                minimum_action=1e-2)

    # Hand-poke a negative A and recompute diagnostics. With `max(A, min_A)`
    # the diagnosed K must not flip sign.
    fill!(interior(model.action), -1e-3)
    fill!(interior(model.wavenumber_moment.x), 0.5)
    fill!(interior(model.wavenumber_moment.y), 0.0)
    Ripple.update_monobanded_diagnostics!(model)

    @test all(interior(model.diagnostics.Kx) .>= 0)
    @test all(interior(model.diagnostics.Kx) .≈ 0.5 / 1e-2)
end

@testset "MonobandedWaveModel quadratic-shear H regression (T5)" begin
    grid = RectilinearGrid(CPU();
                           size=(4, 4, 16),
                           x=(0, 4),
                           y=(0, 4),
                           z=(-2, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    a, b, c = 0.3, 0.05, -0.02
    u = Oceananigans.Fields.Field{Face, Center, Center}(grid)
    v = Oceananigans.Fields.Field{Center, Face, Center}(grid)
    set!(u, (x, y, z) -> a + b * z + c * z^2)
    set!(v, 0)

    model = MonobandedWaveModel(grid;
                                velocities=PrescribedVelocities(; u, v),
                                advection=nothing)
    set!(model; A=1.0, AKx=0.5, AKy=0.0)

    κ = 0.5
    qkernel = model.coupling.qtransform.kernel
    faces = vertical_faces(model.coupling.qtransform)
    depth = model.coupling.current.depth

    expected_Hx = sum(let zc = 0.5 * (faces[ℓ] + faces[ℓ+1])
                          (a + b * zc + c * zc^2) *
                            q_cell_integral_kappa_derivative(qkernel, κ, faces[ℓ], faces[ℓ+1], depth)
                      end
                      for ℓ in 1:length(faces)-1)
    @test interior(model.diagnostics.Hx)[1, 1, 1] ≈ expected_Hx atol=1e-12
end

@testset "MonobandedWaveModel long-run robustness (T4)" begin
    # With `K = AK/max(A, minimum_action)`, the raised `cbrt(eps)` default,
    # and the cleanup step that zeros A, AKx, AKy together where A drops
    # below minimum_action, smooth packets stay finite and bounded over
    # thousands of steps. Without the cleanup the previous defaults NaN'd
    # within ~1500 steps as A drifted to zero and K = AK/A diverged.
    grid = RectilinearGrid(CPU();
                           size=(16, 8, 1),
                           x=(0, 16),
                           y=(0, 8),
                           z=(-1, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    m = MonobandedWaveModel(grid; timestepper=:RungeKutta3, advection=WENO())
    set!(m;
         A   = (x, y, z) -> 1 + 0.05 * sin(2π * x / 16) * cos(2π * y / 8),
         AKx = (x, y, z) -> 0.5 * (1 + 0.05 * sin(2π * x / 16) * cos(2π * y / 8)),
         AKy = 0)
    A_init = sum(interior(m.action))

    # Over 500 steps the smooth packet is well-resolved; neither the
    # positivity clamp nor the cleanup step fires and conservation is tight.
    for _ in 1:500
        time_step!(m, 0.02)
    end
    @test all(isfinite, interior(m.action))
    @test all(isfinite, interior(m.wavenumber_moment.x))
    @test all(isfinite, interior(m.wavenumber_moment.y))
    @test minimum(interior(m.action)) >= 0
    @test abs(sum(interior(m.action)) - A_init) / abs(A_init) < 1e-10

    # Past ~1000 steps WENO5 phase error gradually drifts K, but with the
    # cleanup step in place the simulation stays finite — the key safety
    # invariant the original code did not have.
    for _ in 1:1500
        time_step!(m, 0.02)
    end
    @test all(isfinite, interior(m.action))
    @test all(isfinite, interior(m.wavenumber_moment.x))
    @test all(isfinite, interior(m.wavenumber_moment.y))
    @test minimum(interior(m.action)) >= 0
end

@testset "MonobandedWaveModel AB2 lifecycle" begin
    # `model.previous_tendencies_ready` is Ripple's state machine for AB2's
    # first-step-Euler then subsequent-AB2 transition. `set!` must reset it
    # so callers can re-initialize state mid-simulation without contaminating
    # the multistep history.
    grid = RectilinearGrid(CPU();
                           size=(4, 4, 1),
                           x=(0, 4),
                           y=(0, 4),
                           z=(-1, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    model = MonobandedWaveModel(grid; timestepper=:AB2, advection=nothing,
                                sources=LinearWindInput(rate=0.1))
    set!(model; A=1.0, AKx=0.5, AKy=0.0)
    @test model.timestepper.name === :AB2
    @test !model.previous_tendencies_ready

    time_step!(model, 0.01)
    @test model.previous_tendencies_ready
    @test model.clock.iteration == 1

    time_step!(model, 0.01)
    @test model.previous_tendencies_ready
    @test model.clock.iteration == 2

    # set! must clear the flag so the next step restarts from Euler.
    set!(model; A=2.0)
    @test !model.previous_tendencies_ready
end

@testset "MonobandedWaveModel clamp-low-action kernel zeros A, AKx, AKy together" begin
    grid = RectilinearGrid(CPU();
                           size=(4, 3, 1),
                           x=(0, 4),
                           y=(0, 3),
                           z=(-1, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    model = MonobandedWaveModel(grid; advection=nothing, minimum_action=0.1)

    # Hand-poke a state with A below the floor but AK nonzero, then call the
    # cleanup kernel directly. All three fields must be zeroed in the
    # below-floor cells while above-floor cells stay untouched.
    fill!(interior(model.action), 0.05)
    fill!(interior(model.wavenumber_moment.x), 0.5)
    fill!(interior(model.wavenumber_moment.y), -0.3)

    state = (A   = model.action,
             AKx = model.wavenumber_moment.x,
             AKy = model.wavenumber_moment.y)
    Ripple.monobanded_clamp_low_action!(state, 0.1)

    @test all(interior(model.action) .== 0)
    @test all(interior(model.wavenumber_moment.x) .== 0)
    @test all(interior(model.wavenumber_moment.y) .== 0)

    # And the reverse: an above-floor cell must be left alone.
    fill!(interior(model.action), 0.5)
    fill!(interior(model.wavenumber_moment.x), 0.3)
    fill!(interior(model.wavenumber_moment.y), 0.0)
    Ripple.monobanded_clamp_low_action!(state, 0.1)
    @test all(interior(model.action) .== 0.5)
    @test all(interior(model.wavenumber_moment.x) .== 0.3)
end

@testset "MonobandedWaveModel rejects ValueBC embedded in wavenumber_moment" begin
    # The kwarg-BC rejection is covered in the API testset. This locks in
    # the symmetric case where a non-default BC arrives via a supplied
    # `wavenumber_moment` field, not the `boundary_conditions` kwarg.
    bounded_grid = RectilinearGrid(CPU();
                                   size=(4, 3, 2),
                                   x=(0, 4),
                                   y=(0, 3),
                                   z=(-1, 0),
                                   halo=(3, 3, 3),
                                   topology=(Bounded, Bounded, Bounded))

    value_bcs = Oceananigans.BoundaryConditions.FieldBoundaryConditions(
        bounded_grid,
        (Oceananigans.Grids.Center(), Oceananigans.Grids.Center(), Oceananigans.Grids.Center()),
        (:, :, bounded_grid.Nz:bounded_grid.Nz);
        east = Oceananigans.BoundaryConditions.ValueBoundaryCondition(7))

    AKx_field = Oceananigans.Fields.CenterField(bounded_grid;
        indices=(:, :, bounded_grid.Nz:bounded_grid.Nz),
        boundary_conditions=value_bcs)
    AKy_field = Oceananigans.Fields.CenterField(bounded_grid;
        indices=(:, :, bounded_grid.Nz:bounded_grid.Nz))

    @test_throws ArgumentError MonobandedWaveModel(bounded_grid;
        wavenumber_moment=(; x=AKx_field, y=AKy_field), advection=nothing)
end

@testset "MonobandedWaveModel + Oceananigans.Simulation + JLD2Writer" begin
    # Lock in the AbstractModel contract Oceananigans' Simulation calls
    # into (time_step!, update_state!, prognostic_fields, model.clock, etc.)
    # and that JLD2Writer can serialize the prognostic fields by name.
    # Mirrors the equivalent SpectralWaveModel test in test/integration/model_api.jl.
    grid = RectilinearGrid(CPU();
                           size=(4, 4, 1),
                           x=(0, 4),
                           y=(0, 4),
                           z=(-1, 0),
                           halo=(3, 3, 3),
                           topology=(Periodic, Periodic, Bounded))

    model = MonobandedWaveModel(grid; advection=nothing)
    set!(model; A=1.0, AKx=0.5, AKy=0.0)

    output_path = tempname() * ".jld2"
    simulation = Oceananigans.Simulation(model; Δt=0.01, stop_iteration=2, verbose=false)
    simulation.output_writers[:fields] =
        Oceananigans.JLD2Writer(model,
                                (; A=model.action,
                                   AKx=model.wavenumber_moment.x,
                                   AKy=model.wavenumber_moment.y);
                                filename=output_path,
                                schedule=Oceananigans.IterationInterval(1),
                                overwrite_existing=true)

    Oceananigans.run!(simulation)
    @test isfile(output_path)
    @test model.clock.iteration == 2
end
