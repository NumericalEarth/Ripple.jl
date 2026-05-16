using TOML
import Oceananigans

struct MockExternalPhysicalGrid end

function Ripple.adapt_physical_grid(::MockExternalPhysicalGrid)
    return RectilinearGrid(CPU();
                           size=(2, 1, 2),
                           x=[0.0, 0.5, 2.0],
                           y=[-1.0, 1.0],
                           z=[-2.0, -1.0, 0.0],
                           topology=(Bounded, Periodic, Bounded))
end

@testset "Model API" begin
    grid = RectilinearGrid(CPU(); size=(4, 3, 2), x=(0, 4), y=(0, 3), z=(-1, 0))
    cgrid = CartesianWaveVectorGrid(Float64; kx=range(0.4, 0.8; length=3), ky=range(-0.1, 0.1; length=3))
    @test_throws ArgumentError SpectralWaveModel(; advection=nothing, grid=:grid, spectral_grid=cgrid)
    @test_throws ArgumentError SpectralWaveModel(; advection=nothing, grid, spectral_grid=:spectral_grid)

    model = SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, timestepper=:RK3)
    set!(model, N=GaussianWavePacket(x0=1, y0=1, kx0=0.6, ky0=0.0))

    @test haskey(fields(model), :N)
    @test haskey(fields(model), :G)
    @test fields(model).N === model.action
    @test fields(model).G === model.tendencies
    @test haskey(prognostic_fields(model), :N)
    @test !haskey(prognostic_fields(model), :G)
    @test model.physics === nothing
    @test model.advection === nothing
    @test model.coupling === nothing
    @test eltype(model) === eltype(model.action)
    @test model.clock.iteration == 0
    @test_throws ArgumentError time_step!(model, 0.0)
    @test_throws ArgumentError time_step!(model, -0.01)
    initial = copy(interior(model.action))
    time_step!(model, 0.01)
    @test model.clock.iteration == 1
    @test model.clock.time ≈ 0.01
    @test interior(model.action) ≈ initial atol=1e-14 rtol=0

    @test !isdefined(Ripple, :HamiltonianFiniteVolume)
    @test !isdefined(Ripple, :Simulation)
    @test !isdefined(Ripple, :DiagnosticWriter)
    @test !isdefined(Ripple, :VerticalFiniteVolumeGrid)
    @test_throws ArgumentError SpectralWaveModel(; grid, spectral_grid=cgrid, advection=:weno)

    writer_model = SpectralWaveModel(advection=nothing, grid; spectral_grid=cgrid)
    set!(writer_model, N=1.0)
    output_path = tempname() * ".jld2"
    simulation = Oceananigans.Simulation(writer_model; Δt=0.01, stop_iteration=1, verbose=false)
    simulation.output_writers[:fields] =
        Oceananigans.JLD2Writer(writer_model, (; N=writer_model.action);
                                filename=output_path,
                                schedule=Oceananigans.IterationInterval(1),
                                overwrite_existing=true)
    Oceananigans.run!(simulation)
    @test isfile(output_path)
    @test writer_model.clock.iteration == 1

    supplied_action = WaveActionField(grid, cgrid)
    supplied_model = SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, action=supplied_action)
    @test supplied_model.action === supplied_action

    equivalent_grid = RectilinearGrid(CPU(); size=(4, 3, 2), x=(0, 4), y=(0, 3), z=(-1, 0))
    equivalent_action = WaveActionField(equivalent_grid, cgrid)
    @test SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, action=equivalent_action).action === equivalent_action

    mismatched_grid = RectilinearGrid(CPU(); size=(4, 3, 2), x=(0, 8), y=(0, 3), z=(-1, 0))
    mismatched_physical_action = WaveActionField(mismatched_grid, cgrid)
    @test_throws ArgumentError SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, action=mismatched_physical_action)

    mismatched_z_grid = RectilinearGrid(CPU(); size=(4, 3, 3), x=(0, 4), y=(0, 3), z=(-1, 0))
    mismatched_z_action = WaveActionField(mismatched_z_grid, cgrid)
    @test_throws ArgumentError SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, action=mismatched_z_action)

    mismatched_cgrid = CartesianWaveVectorGrid(Float64; kx=range(0.5, 0.9; length=3), ky=range(-0.1, 0.1; length=3))
    mismatched_spectral_action = WaveActionField(grid, mismatched_cgrid)
    @test_throws ArgumentError SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, action=mismatched_spectral_action)
    @test_throws ArgumentError SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, action=zeros(4, 3, 3, 3))

    for timestepper in (:ForwardEuler, :SemiImplicitEuler, :AB2, :RK3, :LowStorageRK3, :LSRK3)
        @test SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, timestepper).timestepper === timestepper
    end
    @test_throws ArgumentError SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, timestepper=:RK4)
    @test_throws ArgumentError SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, timestepper="RK3")
    @test_throws ArgumentError SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, clock=:clock)
end

@testset "Oceananigans-style absent component semantics" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = CartesianWaveVectorGrid(Float64; kx=[0.5], ky=[0.0])

    empty_sources = SpectralWaveModel(; advection=nothing, grid,
                                       spectral_grid=cgrid,
                                       physics=GenericPhysics())
    @test empty_sources.physics === nothing

    compatibility = SpectralWaveModel(; advection=nothing, grid,
                                       spectral_grid=cgrid,
                                       physics=NoPhysics(),
                                       coupling=NoCurrentCoupling())
    @test compatibility.physics === nothing
    @test compatibility.coupling === nothing
    @test source_tendency(compatibility.physics, compatibility, 1, 1, 1, 1) == 0
    @test implicit_source_rate(compatibility.physics, compatibility, 1, 1, 1, 1) == 0

    @test_throws ArgumentError SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, physics=:wind)
    @test_throws ArgumentError SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, physics=GenericPhysics((:wind,)))

    frequency_grid = FrequencyDirectionGrid(; frequency=[0.1], φ=[0.0])
    source_only = SpectralWaveModel(; grid, spectral_grid=frequency_grid, advection=nothing)
    @test source_only.advection === nothing
end

@testset "Source-only timestepper semantics" begin
    grid = RectilinearGrid(CPU(); size=(2, 1, 1), x=(0, 2), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(;
                                   frequency=[0.08, 0.16, 0.32],
                                   φ=[0.0, pi/2, pi, 3pi/2])
    source = FrequencyDissipation(rate=0.4, reference_frequency=0.16, power=2)
    model = SpectralWaveModel(; advection=nothing, grid,
                                spectral_grid=cgrid,
                                physics=source,
                                timestepper=:SemiImplicitEuler)
    set!(model, N=(x, y, kx, ky) -> 1 + 0.1x + 0.02hypot(kx, ky))
    compute_tendencies!(model)

    for n in eachindex(cgrid.φ), m in eachindex(cgrid.frequency), i in 1:2
        @test model.tendencies[i, 1, m, n] ≈ source_tendency(source, model, i, 1, m, n)
    end

    @test cfl(model) == 0
    initial = model.action[1, 1, 2, 1]
    _, damping = source_split(source, model, 1, 1, 2, 1)
    time_step!(model, 0.25)
    @test model.action[1, 1, 2, 1] ≈ initial / (1 + 0.25damping)
end

@testset "AB2 and low-storage RK3 source-only timesteppers" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = CartesianWaveVectorGrid(Float64; kx=[0.5], ky=[0.0])
    model = SpectralWaveModel(; advection=nothing, grid,
                                spectral_grid=cgrid,
                                physics=LinearWindInput(rate=0.2),
                                timestepper=:AB2)
    set!(model, N=1.0)
    time_step!(model, 0.5)
    @test model.action[1, 1, 1, 1] ≈ 1.1
    @test model.previous_tendencies_ready
    time_step!(model, 0.5)
    expected = 1.1 + 0.5 * (1.5 * 0.2 * 1.1 - 0.5 * 0.2)
    @test model.action[1, 1, 1, 1] ≈ expected

    rk3 = SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, physics=BottomFriction(rate=1.0), timestepper=:RK3)
    low_storage = SpectralWaveModel(; advection=nothing, grid, spectral_grid=cgrid, physics=BottomFriction(rate=1.0), timestepper=:LowStorageRK3)
    set!(rk3, N=1.0)
    set!(low_storage, N=1.0)
    time_step!(rk3, 0.1)
    time_step!(low_storage, 0.1)
    @test low_storage.action[1, 1, 1, 1] ≈ rk3.action[1, 1, 1, 1]
end

@testset "External physical grid adaptation" begin
    cgrid = CartesianWaveVectorGrid(Float64; kx=[0.4, 0.8], ky=[-0.1, 0.1])
    model = SpectralWaveModel(; advection=nothing, grid=MockExternalPhysicalGrid(), spectral_grid=cgrid)

    @test model.grid isa RectilinearGrid
    @test horizontal_size(model.grid) == (2, 1)
    @test vertical_size(model.grid) == 2
    @test xfaces(model.grid) == [0.0, 0.5, 2.0]
    @test yfaces(model.grid) == [-1.0, 1.0]
    @test zfaces(model.grid) == [-2.0, -1.0, 0.0]
    @test Oceananigans.Grids.topology(model.grid) == (Bounded, Periodic, Bounded)
    @test grid(model.action) === model.grid
end

@testset "Package dependency metadata" begin
    root = normpath(joinpath(@__DIR__, "..", ".."))
    project = TOML.parsefile(joinpath(root, "Project.toml"))
    @test project["deps"]["Oceananigans"] == "9e8cae18-63c1-5223-a75c-80ca9d6e9a09"
    @test project["weakdeps"]["Makie"] == "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    @test !haskey(project["weakdeps"], "Oceananigans")
    @test !haskey(project["weakdeps"], "JLD2")
    @test !haskey(project["weakdeps"], "NCDatasets")
    @test project["extensions"]["RippleMakieExt"] == "Makie"
    @test !haskey(project["extensions"], "RippleOceananigansExt")
end
