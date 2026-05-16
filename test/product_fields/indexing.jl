import Oceananigans

@testset "ProductField wraps Oceananigans fields" begin
    grid = RectilinearGrid(CPU();
                           size=(3, 2, 1),
                           x=(0, 1),
                           y=(0, 1),
                           z=(0, 1),
                           topology=(Periodic, Periodic, Bounded))
    cgrid = CartesianWaveVectorGrid(Float64; kx=1:4, ky=1:5)

    N = WaveActionField(grid, cgrid)
    @test size(N) == (3, 2, 4, 5)
    @test ndims(N) == 4
    @test N isa AbstractArray{Float64, 4}
    @test physical_field(N, 1, 1) isa Oceananigans.Fields.Field
    @test Ripple.grid(physical_field(N, 1, 1)) === grid
    @test architecture(N) isa CPU

    set!(N, (x, y, kx, ky) -> x - 2y + 3kx - 4ky)
    @test N[2, 1, 3, 4] == interior(N)[2, 1, 3, 4]
    @test Array(N)[2, 1, 3, 4] == N[2, 1, 3, 4]

    N .= 3.0
    @test all(interior(N) .== 3.0)
    N .+= 2.0
    @test all(interior(N) .== 5.0)

    copied = copy(N)
    @test copied !== N
    @test interior(copied) == interior(N)
    copied[1, 1, 1, 1] += 1
    @test copied[1, 1, 1, 1] != N[1, 1, 1, 1]

    scratch = similar(N; eltype=Float32)
    @test scratch isa ProductField
    @test eltype(scratch) === Float32
    @test size(scratch) == size(N)
    @test physical_field(scratch, 1, 1) isa Oceananigans.Fields.Field

    same_size_scratch = similar(N, Float32, size(N))
    @test same_size_scratch isa ProductField
    @test_throws ArgumentError similar(N, Float32, (1, 2, 4, 5))
end

@testset "ProductField public interface" begin
    grid = RectilinearGrid(CPU();
                           size=(2, 3, 1),
                           x=(0, 2),
                           y=(-1, 2),
                           z=(0, 1),
                           topology=(Bounded, Periodic, Bounded))
    cgrid = CartesianWaveVectorGrid(Float64;
                                    kx=[0.5, 1.0],
                                    ky=[-0.25, 0.25, 0.75],
                                    boundary_conditions=(NoFlux(), Periodic()))

    field = ProductField(grid, cgrid;
                         location=(Center, Center, Nothing),
                         coordinate_location=(Center, Face))

    @test physical_grid(field) === grid
    @test coordinate_grid(field) === cgrid
    @test product_grid(field).physical === grid
    @test product_grid(field).coordinate === cgrid
    @test location(field) == (Center, Center, Nothing)
    @test coordinate_location(field) == (Center, Face)
    @test product_location(field) == (Center, Center, Nothing, Center, Face)
    @test active_product_location(field) == (Center, Center, Center, Face)
    @test axes(field) == (Base.OneTo(2), Base.OneTo(3), Base.OneTo(2), Base.OneTo(3))
    @test axes(field, 5) == Base.OneTo(1)
    @test IndexStyle(field) == IndexCartesian()

    set!(field, (x, y, kx, ky) -> x + 2y + 3kx + 4ky)
    @test field[CartesianIndex(2, 3, 2, 3)] == field[2, 3, 2, 3]

    @test_throws ArgumentError ProductField(grid, cgrid; halo=(1, 1, 1, 1))
    @test_throws ArgumentError ProductField(grid, cgrid; boundary_conditions=default_wave_action_bcs(grid, cgrid))
end

@testset "Oceananigans RectilinearGrid accessors" begin
    grid = RectilinearGrid(CPU();
                           size=(3, 2, 1),
                           x=[0, 0.25, 1.0, 2.0],
                           y=[-1, 0, 3],
                           z=(0, 1),
                           topology=(Bounded, Periodic, Bounded))

    @test size(grid) == (3, 2, 1)
    @test horizontal_size(grid) == (3, 2)
    @test vertical_size(grid) == 1
    @test xfaces(grid) == [0.0, 0.25, 1.0, 2.0]
    @test yfaces(grid) == [-1.0, 0.0, 3.0]
    @test zfaces(grid) == [0.0, 1.0]
    @test xnodes(grid) == [0.125, 0.625, 1.5]
    @test xspacings(grid) == [0.25, 0.75, 1.0]
    @test Oceananigans.Grids.topology(grid) == (Bounded, Periodic, Bounded)
end
