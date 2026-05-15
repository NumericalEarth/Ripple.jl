@testset "CFL diagnostic source-only semantics" begin
    grid = RectilinearGrid(CPU();
                           size=(4, 3, 2),
                           x=(0, 8),
                           y=(0, 3),
                           z=(-1, 0),
                           topology=(Bounded, Bounded, Bounded))
    cgrid = FrequencyDirectionGrid(;
                                   frequency=[0.08, 0.16],
                                   φ=[0.0, pi])
    model = SpectralWaveModel(grid, cgrid;
                      horizontal_advection=nothing,
                      clock=Clock(time=0.0, last_Δt=10.0))
    @test cfl(model) == 0
    @test horizontal_size(grid) == (4, 3)
    @test vertical_size(grid) == 2
    @test length(zfaces(grid)) == 3
end
