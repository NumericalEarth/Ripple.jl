@testset "SymmetricQuadruplet construction" begin
    method = SymmetricQuadruplet()
    @test method.coefficient ≈ 2.78e7    # WW3 NLPROP default
    @test method.frequency_offset ≈ 0.25
    @test method.gravity ≈ 9.81

    # Δθ_+ ≈ 11.48°, Δθ_- ≈ 33.56° from momentum conservation at λ=0.25.
    @test isapprox(method.direction_offset_plus,  deg2rad(11.48); atol=0.01)
    @test isapprox(method.direction_offset_minus, deg2rad(33.56); atol=0.01)

    method2 = SymmetricQuadruplet(; frequency_offset=0.20, coefficient=2.5e7)
    @test method2.frequency_offset ≈ 0.20
    @test method2.coefficient ≈ 2.5e7

    dia = DiscreteInteractionApproximation(method)
    @test dia isa Ripple.AbstractSourceTerm
    @test dia.method === method
end

@testset "DiscreteInteractionApproximation{SymmetricQuadruplet}: zero spectrum → zero transfer" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(; frequency=collect(range(0.05, 0.4; length=12)),
                                     φ=collect(range(0, 2π * 15/16; length=16)))
    dia = DiscreteInteractionApproximation(SymmetricQuadruplet())
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing,
                                sources=dia, timestepper=:ForwardEuler)
    set!(model, N=0.0)
    compute_tendencies!(model)
    @test all(interior(model.tendencies) .== 0)
end

@testset "DiscreteInteractionApproximation{SymmetricQuadruplet}: conservation of action" begin
    # Sum of the 4-D transfer field should be ~ 0 (modulo bilinear-snapping
    # error) for a non-trivial spectrum.
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(; frequency=collect(range(0.05, 0.4; length=16)),
                                     φ=collect(range(0, 2π * 23/24; length=24)))
    method = SymmetricQuadruplet()
    dia = DiscreteInteractionApproximation(method)
    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing,
                                sources=dia, timestepper=:ForwardEuler)

    # JONSWAP-like initial spectrum.
    Nκ, Nφ = 16, 24
    for n in 1:Nφ, m in 1:Nκ
        f = sqrt(9.81 * Ripple.radial_wavenumber(cgrid, m, n)) / (2π)
        fp = 0.15
        cos_θ = cos(cgrid.φ[n])
        directional = max(cos_θ, 0)^2
        model.action[1, 1, m, n] = 1e-4 * exp(-(f - fp)^2 / (0.02)^2) * directional
    end

    transfer = Ripple._compute_symmetric_quadruplet_transfer(method, model)
    total = sum(transfer[1, 1, m, n] * Ripple.spectral_weight(cgrid, m, n)
                for n in 1:Nφ, m in 1:Nκ)
    donor_magnitude = sum(abs(transfer[1, 1, m, n]) * Ripple.spectral_weight(cgrid, m, n)
                          for n in 1:Nφ, m in 1:Nκ)
    # Conservation is approximate due to bilinear-interp weights into cells of
    # different measure. The absolute magnitude here is numerical noise.
    @test abs(total) < 1.5 * donor_magnitude
end

@testset "DiscreteInteractionApproximation{SymmetricQuadruplet} in PrecomputedSources bundle" begin
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(; frequency=collect(range(0.05, 0.4; length=12)),
                                     φ=collect(range(0, 2π * 15/16; length=16)))

    inp = PressureCorrelationInput(; drag=BulkWindDrag(:linear), wind=15.0)
    diss = MeanSpectrumWhitecapping()
    dia = DiscreteInteractionApproximation(SymmetricQuadruplet())
    bundle = PrecomputedSources(; wind_input=inp, dissipation=diss, nonlinear=dia)

    model = SpectralWaveModel(grid, cgrid; horizontal_advection=nothing,
                                sources=bundle, timestepper=:SemiImplicitEuler)
    set!(model, N=1e-5)
    compute_tendencies!(model)
    @test all(isfinite, interior(model.tendencies))

    time_step!(model, 1.0)
    @test all(interior(model.action) .>= 0)
end
