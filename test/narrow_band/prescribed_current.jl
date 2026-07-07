import Oceananigans.Utils: launch!
import Oceananigans.Architectures: architecture
import Oceananigans.BoundaryConditions: fill_halo_regions!

# Evaluate the refraction kernel R for real Ū = (Ux, Uy) and real amplitude A on
# a periodic wave grid, returning the interior of R.
function refraction_field(Nx, κ, Ufun_x, Ufun_y, Afun)
    grid = RectilinearGrid(CPU(); size=(Nx, Nx), x=(0, 2π), y=(0, 2π),
                           topology=(Periodic, Periodic, Flat), halo=(3, 3))
    Ūxᶜ = Ripple.CenterField(grid); Ūyᶜ = Ripple.CenterField(grid)
    Ar = Ripple.CenterField(grid); Ai = Ripple.CenterField(grid)
    Rr = Ripple.CenterField(grid); Ri = Ripple.CenterField(grid)
    set!(Ūxᶜ, Ufun_x); set!(Ūyᶜ, Ufun_y); set!(Ar, Afun)
    for f in (Ūxᶜ, Ūyᶜ, Ar, Ai)
        fill_halo_regions!(f)
    end
    Δx = 2π / Nx
    launch!(architecture(grid), grid, :xyz, Ripple._refraction!,
            Rr, Ri, Ar, Ai, Ūxᶜ, Ūyᶜ, κ^2, Δx, Δx)
    return interior(Rr)[:, :, 1], collect(xnodes(grid)), collect(ynodes(grid))
end

@testset "Narrow-band prescribed current" begin
    @testset "Refraction R: x-only manufactured solution" begin
        # Ūx = cos(ax), Ūy = 0, A = cos(px):
        #   R = [a²p − 2p(κ²−p²)] cos(ax) sin(px) + 3 a p² sin(ax) cos(px)
        κ, a, p = 2.0, 1, 3
        analytic(x, y) = (a^2 * p - 2p * (κ^2 - p^2)) * cos(a * x) * sin(p * x) +
                         3 * a * p^2 * sin(a * x) * cos(p * x)
        errs = Float64[]
        for Nx in (64, 128, 256)
            R, xc, yc = refraction_field(Nx, κ, (x, y) -> cos(a * x), (x, y) -> 0.0, (x, y) -> cos(p * x))
            ref = [analytic(xc[i], yc[j]) for i in axes(R, 1), j in axes(R, 2)]
            push!(errs, maximum(abs, R .- ref) / maximum(abs, ref))   # relative error
        end
        @test errs[3] < 1e-2
        @test errs[1] / errs[2] > 3.5   # ~2nd-order convergence
        @test errs[2] / errs[3] > 3.5
    end

    @testset "Refraction R: fully 2-D manufactured solution" begin
        # Ūx = cos(by), Ūy = sin(cx), A = cos(px) cos(qy); K² = p² + q².
        κ, b, c, p, q = 1.5, 2, 1, 3, 2
        K² = p^2 + q^2
        analytic(x, y) = -2 * (κ^2 - K²) * (p * cos(b * y) * sin(p * x) * cos(q * y) +
                                            q * sin(c * x) * cos(p * x) * sin(q * y)) +
                         2p * q * sin(p * x) * sin(q * y) * (c * cos(c * x) - b * sin(b * y))
        errs = Float64[]
        for Nx in (64, 128, 256)
            R, xc, yc = refraction_field(Nx, κ, (x, y) -> cos(b * y), (x, y) -> sin(c * x),
                                         (x, y) -> cos(p * x) * cos(q * y))
            ref = [analytic(xc[i], yc[j]) for i in axes(R, 1), j in axes(R, 2)]
            push!(errs, maximum(abs, R .- ref) / maximum(abs, ref))   # relative error
        end
        @test errs[3] < 2e-2
        @test errs[1] / errs[2] > 3.5
        @test errs[2] / errs[3] > 3.5
    end

    @testset "Constructor validation" begin
        flat = RectilinearGrid(CPU(); size=(16, 16), x=(0, 100), y=(0, 100),
                               topology=(Periodic, Periodic, Flat))
        @test_throws ArgumentError NarrowBandWaveModel(flat; κ=0.5, depth=50.0,
                                                       velocities=(u=(x,y,z)->0.0, v=(x,y,z)->0.0))
        smallhalo = RectilinearGrid(CPU(); size=(16, 16, 4), x=(0, 100), y=(0, 100), z=(-50, 0),
                                    topology=(Periodic, Periodic, Bounded), halo=(1, 1, 1))
        @test_throws ArgumentError NarrowBandWaveModel(smallhalo; κ=0.5,
                                                       velocities=(u=(x,y,z)->0.0, v=(x,y,z)->0.0))
    end

    @testset "Prescribed vortex: stepping and action" begin
        Nx = 96
        grid = RectilinearGrid(CPU(); size=(Nx, Nx, 8), x=(0, 100), y=(0, 100), z=(-50, 0),
                               topology=(Periodic, Periodic, Bounded), halo=(3, 3, 3))
        Γ, x0, y0, r0 = 0.3, 50.0, 50.0, 12.0
        gauss(x, y) = exp(-((x - x0)^2 + (y - y0)^2) / (2r0^2))
        uvortex(x, y, z) = -Γ * (y - y0) / r0 * gauss(x, y)
        vvortex(x, y, z) =  Γ * (x - x0) / r0 * gauss(x, y)
        κ = 0.6

        model = NarrowBandWaveModel(grid; κ, velocities=(u=uvortex, v=vvortex), advection=Centered())
        @test model.velocities isa Ripple.NarrowBandPrescribedVelocities
        set!(model; A=(x, y) -> cis(κ * x))
        action(m) = real(sum(conj(amplitude(m)) .* reconstituted_amplitude(m)))
        𝒜0 = action(model)
        Δt = 0.1 / model.dispersion.ω
        for _ in 1:200
            time_step!(model, Δt)
        end
        @test all(isfinite, amplitude(model))
        # Divergence-free current + Centered transport ⇒ action nearly conserved.
        @test abs(action(model) - 𝒜0) / abs(𝒜0) < 5e-2

        # Default WENO() advection must resolve its weight-computation type for
        # the wave grid and step without error.
        weno_model = NarrowBandWaveModel(grid; κ, velocities=(u=uvortex, v=vvortex))
        set!(weno_model; A=(x, y) -> cis(κ * x))
        for _ in 1:20
            time_step!(weno_model, Δt)
        end
        @test all(isfinite, amplitude(weno_model))
    end
end
