import Oceananigans: Simulation, run!

@testset "NarrowBandWaveModel (wave-only)" begin
    make_grid(; Nx=64, Ny=64) = RectilinearGrid(CPU(); size=(Nx, Ny, 4),
        x=(0, 2π), y=(0, 2π), z=(-1000, 0),
        topology=(Periodic, Periodic, Bounded), halo=(3, 3, 3))

    @testset "Constructor validation and interface" begin
        grid = make_grid()
        @test_throws ArgumentError NarrowBandWaveModel(grid)                       # missing κ
        @test_throws ArgumentError NarrowBandWaveModel(grid; κ=4.0, timestepper=:AB2)

        model = NarrowBandWaveModel(grid; κ=4.0, depth=InfiniteDepth())
        @test propertynames(prognostic_fields(model)) == (:Gr, :Gi)
        @test propertynames(fields(model)) == (:Gr, :Gi, :Ar, :Ai)
        @test eltype(model) == Float64

        # Flat-vertical grid needs an explicit depth.
        flat = RectilinearGrid(CPU(); size=(16, 16), x=(0, 2π), y=(0, 2π),
                               topology=(Periodic, Periodic, Flat))
        @test_throws ArgumentError NarrowBandWaveModel(flat; κ=4.0)
        @test NarrowBandWaveModel(flat; κ=4.0, depth=100.0) isa NarrowBandWaveModel
    end

    @testset "set!/reconstitute round-trip" begin
        grid = make_grid()
        model = NarrowBandWaveModel(grid; κ=4.0, depth=InfiniteDepth())
        xc = xnodes(grid)
        yc = ynodes(grid)
        # A smooth multi-mode complex amplitude.
        A0(x, y) = cis(3x) + 0.4 * cis(-2x + y) - 0.3im * cis(x - 3y)
        set!(model; A=A0)
        A = amplitude(model)
        analytic = [A0(xc[i], yc[j]) for i in 1:64, j in 1:64]
        @test maximum(abs, A[:, :, 1] .- analytic) < 1e-10
    end

    @testset "Single-mode dispersion frequency matches Λ_d (RK3 amplification)" begin
        Nx = 64
        grid = make_grid(; Nx, Ny=Nx)
        κ = 4.0
        model = NarrowBandWaveModel(grid; κ, depth=InfiniteDepth())
        d = model.dispersion

        for n in (2, 6, 10)      # x-modes, some below and some above the carrier
            Δx = 2π / Nx
            λ = (2 * sin(n * π / Nx) / Δx)^2          # discrete -∇² eigenvalue
            M̂ = 1 + d.α * (κ^2 - λ)
            Λd = (d.ω_κ / 2κ) * (κ^2 - λ) / M̂         # semi-discrete dispersion
            Δt = 0.15 / d.ω
            z = im * Λd * Δt
            P = 1 + z + z^2 / 2 + z^3 / 6              # SSP-RK3 amplification factor

            set!(model; A=(x, y) -> cis(n * x))
            A0 = copy(amplitude(model))
            time_step!(model, Δt)
            A1 = amplitude(model)
            ratio = A1 ./ A0                            # uniform for a single eigenmode
            @test maximum(abs, ratio .- P) < 1e-10
        end
    end

    @testset "Discrete-carrier mode is steady" begin
        # On the grid a plane wave sees the modified wavenumber λ (not κ²). Choose
        # κ so that mode n is *exactly* the discrete carrier (λ = κ²); then HA = 0,
        # the tendency vanishes identically, and A is steady to roundoff.
        Nx = 64
        grid = make_grid(; Nx, Ny=Nx)
        n = 4
        Δx = 2π / Nx
        λ = (2 * sin(n * π / Nx) / Δx)^2
        κ = sqrt(λ)
        model = NarrowBandWaveModel(grid; κ, depth=InfiniteDepth())
        set!(model; A=(x, y) -> cis(n * x))
        A0 = copy(amplitude(model))
        Δt = 0.2 / model.dispersion.ω
        for _ in 1:100
            time_step!(model, Δt)
        end
        @test maximum(abs, amplitude(model) .- A0) < 1e-10
    end

    @testset "Wave action 𝒜 = Re⟨A, G⟩ conserved (U = 0)" begin
        grid = make_grid()
        κ = 4.0
        action(m) = real(sum(conj(amplitude(m)) .* reconstituted_amplitude(m)))

        drifts = Float64[]
        Δts = Float64[]
        for refine in (1, 2)
            model = NarrowBandWaveModel(grid; κ, depth=InfiniteDepth())
            set!(model; A=(x, y) -> cis(6x) + 0.5cis(-3x + 2y))
            𝒜0 = action(model)
            Δt = (0.2 / model.dispersion.ω) / refine
            for _ in 1:(100 * refine)
                time_step!(model, Δt)
            end
            push!(drifts, abs(action(model) - 𝒜0) / abs(𝒜0))
            push!(Δts, Δt)
        end
        @test drifts[1] < 1e-3                          # small at the working step
        # Third-order convergence: halving Δt shrinks the drift by ~8×.
        @test drifts[2] < drifts[1] / 4
    end

    @testset "Oceananigans.Simulation smoke" begin
        grid = make_grid(; Nx=32, Ny=32)
        model = NarrowBandWaveModel(grid; κ=3.0, depth=InfiniteDepth())
        set!(model; A=(x, y) -> cis(3x))
        simulation = Simulation(model; Δt=0.2 / model.dispersion.ω, stop_iteration=5)
        run!(simulation)
        @test model.clock.iteration == 5
    end
end
