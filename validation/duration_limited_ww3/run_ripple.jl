#####
##### Duration-limited growth (Ripple side).
#####
##### Spatially homogeneous single point, U10 = 17 m/s, deep water, ST3-equivalent
##### physics (PrecomputedSources: PressureCorrelationInput + MeanSpectrumWhitecapping).
##### Integrate to t = 24 h; record Hs(t) and the 1D spectrum F(f, t_final).

using Ripple
using Oceananigans: compute!
using DelimitedFiles

const U10 = 17.0                    # m/s
const T_FINAL = 24 * 3600.0         # 24 h to match WW3
const DT = 10.0                     # 10s step for local saturation stability
const OUTPUT_INTERVAL = 600.0       # snapshot every 10 min

const OUTDIR = joinpath(@__DIR__, "output")
mkpath(OUTDIR)

# Spectral grid: 25 logarithmic frequency bins f ∈ [0.0418, 0.4] Hz × 24 directions.
# Standard WW3 layout (matches ww3_grid.inp below).
const NFREQ = 25      # matches WW3 ww3_grid.inp
const NDIR  = 24      # matches WW3 ww3_grid.inp
const F0    = 0.04118
const FR    = 1.1
frequency_centers = [F0 * FR^(k-1) for k in 1:NFREQ]
direction_centers = collect(range(0, 2π * (NDIR - 1) / NDIR; length=NDIR))

cgrid = FrequencyDirectionGrid(; frequency=frequency_centers, φ=direction_centers)
grid  = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0.0, 1.0), y=(0.0, 1.0), z=(0.0, 1.0))

# Wind: aligned with φ = 0 (eastward). Wind speed is a scalar field constant in
# space and time.
wind_input  = PressureCorrelationInput(; drag=BulkWindDrag(:linear), wind=U10)
dissipation = LocalSaturationDissipation(; B_r=1.05e-2, σ_power=1.0)
nonlinear   = HasselmannDIA(; C=1.5e7)
# Note: Ripple uses ST3-lite (no τ_w iteration in Sin), so the wind input is
# unbounded by wave-supported stress. The system is bistable — at low DIA C
# it stays near Hs ≈ 1 m (peak doesn't downshift); at full C it transitions
# to a too-high equilibrium (Hs ≈ 9 m at U10=17 m/s, vs WW3's 5.9 m).
# Resolving this requires τ_w iterative drag — see TODO in
# `src/Physics/WindInput/pressure_correlation.jl`.
physics     = PrecomputedSources(; wind_input, dissipation, nonlinear)

model = SpectralWaveModel(; grid,
                            spectral_grid=cgrid,
                            advection=nothing,
                            physics,
                            timestepper=:SemiImplicitEuler)

# Seed with a small wind-aligned JONSWAP-like blob so the input has something to amplify.
# Without a linear-growth term (LN1) we need a non-zero starting energy.
seed_action = 1e-6
set!(model, N=seed_action)

# Time-integration loop with periodic Hs snapshots.
times = Float64[]
hs    = Float64[]
push!(times, 0.0)
hs_field = compute!(significant_wave_height(model.action)); push!(hs, hs_field[1,1,1])

function integrate!()
    next_output = OUTPUT_INTERVAL
    while model.clock.time < T_FINAL
        time_step!(model, DT)
        if model.clock.time >= next_output - DT/2
            push!(times, model.clock.time)
            hs_field = compute!(significant_wave_height(model.action))
            push!(hs, hs_field[1,1,1])
            next_output += OUTPUT_INTERVAL
        end
    end
end
integrate!()

# Save Hs(t) trace.
hs_path = joinpath(OUTDIR, "ripple_hs.tsv")
open(hs_path, "w") do io
    println(io, "t_seconds\tHs_meters")
    for (t, h) in zip(times, hs)
        println(io, t, "\t", h)
    end
end

# Save 1-D spectrum F(f) at final time: integrate N(f,θ) over θ and convert
# action to energy density (E(f,θ) = σ·N(f,θ), then ∫dθ).
N = model.action
ftrace = zeros(Float64, NFREQ)
for m in 1:NFREQ
    k_m = Ripple.radial_wavenumber(cgrid, m, 1)
    σ_m = sqrt(9.81 * k_m)
    band = 0.0
    for n in 1:NDIR
        band += N[1, 1, m, n] * Ripple.spectral_weight(cgrid, m, n)
    end
    # E(f) = σ²·N(f,θ) summed over θ-cells; conversion from radial-frequency
    # to f via dσ/df = 2π gives a factor of 2π. (Quick first-cut.)
    ftrace[m] = band * σ_m * 2π
end
spec_path = joinpath(OUTDIR, "ripple_spectrum_final.tsv")
open(spec_path, "w") do io
    println(io, "f_Hz\tE_m2_per_Hz")
    for (f, e) in zip(frequency_centers, ftrace)
        println(io, f, "\t", e)
    end
end

@info "Ripple done" hs_final=hs[end] t_final=times[end] hs_path spec_path
