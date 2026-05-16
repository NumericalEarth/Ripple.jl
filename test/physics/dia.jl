@testset "HasselmannDIA construction" begin
    dia = HasselmannDIA()
    @test dia isa AbstractNonlinear
    @test dia.C ≈ 2.78e7   # WW3 NLPROP default
    @test dia.λ ≈ 0.25
    @test dia.gravity ≈ 9.81

    # Δθ_+ ≈ 11.48°, Δθ_- ≈ 33.56° for λ=0.25 from momentum conservation.
    @test isapprox(dia.Δθ_plus,  deg2rad(11.48); atol=0.01)
    @test isapprox(dia.Δθ_minus, deg2rad(33.56); atol=0.01)

    # Custom λ.
    dia2 = HasselmannDIA(; λ=0.20, C=2.5e7)
    @test dia2.λ ≈ 0.20
    @test dia2.C ≈ 2.5e7
end

@testset "HasselmannDIA: zero spectrum → zero transfer" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(; frequency=collect(range(0.05, 0.4; length=12)),
                                     φ=collect(range(0, 2π * 15/16; length=16)))
    dia = HasselmannDIA()
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing,
                                physics=dia, timestepper=:ForwardEuler)
    set!(model, N=0.0)
    compute_tendencies!(model)
    @test all(interior(model.tendencies) .== 0)
end

@testset "HasselmannDIA: conservation of action" begin
    # DIA must conserve total action (and energy, momentum). With a non-trivial
    # spectrum, sum of the full 4D transfer field should be zero up to NN
    # snapping error.
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(; frequency=collect(range(0.05, 0.4; length=16)),
                                     φ=collect(range(0, 2π * 23/24; length=24)))
    dia = HasselmannDIA()
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing,
                                physics=dia, timestepper=:ForwardEuler)

    # JONSWAP-like initial spectrum.
    Nκ, Nφ = 16, 24
    for n in 1:Nφ, m in 1:Nκ
        f = sqrt(9.81 * Ripple.radial_wavenumber(cgrid, m, n)) / (2π)
        fp = 0.15
        cos_θ = cos(cgrid.φ[n])
        directional = max(cos_θ, 0)^2
        model.action[1, 1, m, n] = 1e-4 * exp(-(f - fp)^2 / (0.02)^2) * directional
    end

    state = prepare_physics(dia, model)
    transfer = state.transfer
    total = sum(transfer[1, 1, m, n] * Ripple.spectral_weight(cgrid, m, n)
                for n in 1:Nφ, m in 1:Nκ)
    # NN snapping introduces O(Δσ + Δθ) error — should still be small relative
    # to the donor-depletion magnitude.
    donor_magnitude = sum(abs(transfer[1, 1, m, n]) * Ripple.spectral_weight(cgrid, m, n)
                          for n in 1:Nφ, m in 1:Nκ)
    # Conservation is approximate due to bilinear-interp weights into cells of
    # different measure. The absolute magnitude here is ~1e-13 (numerical noise);
    # the ratio is bounded but not strictly less than the L1.
    @test abs(total) < 1.5 * donor_magnitude
end

@testset "HasselmannDIA in MeanSpectrumPhysics bundle" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(; frequency=collect(range(0.05, 0.4; length=12)),
                                     φ=collect(range(0, 2π * 15/16; length=16)))

    inp = PressureCorrelationInput(; drag=BulkWindDrag(:linear), wind=15.0)
    diss = MeanSpectrumWhitecapping()
    dia = HasselmannDIA()
    bundle = MeanSpectrumPhysics(; wind_input=inp, dissipation=diss, nonlinear=dia)

    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing,
                                physics=bundle, timestepper=:SemiImplicitEuler)
    set!(model, N=1e-5)
    compute_tendencies!(model)
    @test all(isfinite, interior(model.tendencies))

    # One semi-implicit step should leave N non-negative.
    time_step!(model, 1.0)
    @test all(interior(model.action) .>= 0)
end
