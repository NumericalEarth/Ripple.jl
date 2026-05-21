# # Coupled Wind-Drift Instability
#
# A two-dimensional wind-drift-layer instability in ``(y, z)`` driven by a
# surface stress, with three wave-coupling treatments compared from a common
# spinup state:
#
# 1. **Prescribed**: the wave field is fixed; the Stokes shear is set once.
# 2. **Monobanded-coupled**: a `MonobandedWaveModel` is advanced from an
#    Oceananigans callback that copies the ocean's Lagrangian-mean current
#    into the wave model.
# 3. **Spectral-coupled**: same, with `SpectralWaveModel`.
#
# Both coupled cases drive `UniformStokesDrift` from a deep-water Stokes-shear
# formula `∂z uˢ = 2κ ε² c exp(2κz) K̂` evaluated on the wave-model state.
# Calibration of `∂z` of the CWCM pseudomomentum directly is left as a
# follow-up (units differ; the analytic formula gives the right magnitude).

using Oceananigans, Ripple
using Oceananigans.AbstractOperations: @at
using CairoMakie, Printf, Random, Statistics
import KernelAbstractions
using KernelAbstractions: @kernel, @index

CairoMakie.activate!(type = "png")

# ## Domain, wave scale, and run length

Ny, Nz = 128, 96
Ly, Lz = 0.060, 0.020
g, ν = 9.81, 1.1e-6
λ_wave = 0.030; κ0 = 2π / λ_wave
surface_stress, noise_speed = -4.8e-5, 0.001
Δt, spinup_iterations, continuation_iterations = 0.001, 500, 1500
frame_stride = 20
monobanded_wave_substeps, spectral_wave_substeps = 32, 4

λ_instability = Ly / 3
ℓ, m = 2π / λ_instability, π / Lz

spectral_Nκ, spectral_Nφ = 7, 12
spectral_σκ, spectral_σφ = 0.18κ0, 0.18
spectral_κ_range = range(0.5κ0, 2.0κ0; length = spectral_Nκ)
spectral_φ_range = range(-π, π; length = spectral_Nφ + 1)[1:spectral_Nφ]

# ## Wave-model factories
#
# `set_spectral_wave_state!` writes the equivalent of the monobanded
# ``A, K`` state into a normalized Gaussian in ``(\kappa, \varphi)``.

function wind_drift_grid()
    return RectilinearGrid(CPU(); size=(Ny, Nz), halo=(3, 3),
                           y=(0, Ly), z=(-Lz, 0),
                           topology=(Flat, Periodic, Bounded))
end

spectral_wave_grid() = PolarWaveVectorGrid(; κ=spectral_κ_range, φ=spectral_φ_range)

wrap(φ, φ₀) = atan(sin(φ - φ₀), cos(φ - φ₀))
peak_shape(κ, φ, κ₀, φ₀) = exp(-((κ - κ₀) / spectral_σκ)^2 - (wrap(φ, φ₀) / spectral_σφ)^2)

function set_spectral_wave_state!(model::SpectralWaveModel; action=1, Kx=κ0, Ky=0)
    κ₀ = max(hypot(Kx, Ky), sqrt(eps(Float64)))
    φ₀ = atan(Ky, Kx)
    sg = model.spectral_grid
    κs = Array(coordinate_centers(sg, 1))
    φs = Array(coordinate_centers(sg, 2))
    w  = Array(sg.weights)
    Z  = sum(peak_shape(κs[m], φs[n], κ₀, φ₀) * w[m, n] for n in eachindex(φs), m in eachindex(κs))
    Nx, Ny, Nκ, Nφ = size(model.action)
    N = [action * peak_shape(κs[m], φs[n], κ₀, φ₀) / Z for _ in 1:Nx, _ in 1:Ny, m in eachindex(κs), n in eachindex(φs)]
    set!(model.action, N)
    return model
end

const steepness = 0.16

function build_wave_model(::Val{:monobanded}, grid, uᴸ, vᴸ)
    m = MonobandedWaveModel(grid; velocities=(; u=uᴸ, v=vᴸ),
                            advection=Centered(), timestepper=:RungeKutta3,
                            gravitational_acceleration=g)
    set!(m; A=1, AKx=κ0, AKy=0)
    return m
end

function build_wave_model(::Val{:spectral}, grid, uᴸ, vᴸ)
    m = SpectralWaveModel(grid, spectral_wave_grid(); velocities=(; u=uᴸ, v=vᴸ),
                          depth=Lz, sources=nothing, timestepper=:RungeKutta3)
    set_spectral_wave_state!(m)
    Ripple.update_coupling!(m)
    return m
end

# Wave-bulk state used by the Stokes-shear kernel. Returns dense arrays of
# A, Kx, Ky, κ on the physical grid; the spectral version computes them
# from `m0`/`first_moment` of the spectral action field.
function wave_bulk_arrays(m::MonobandedWaveModel)
    return (Array(interior(m.action)),
            Array(interior(m.diagnostics.Kx)),
            Array(interior(m.diagnostics.Ky)),
            Array(interior(m.diagnostics.κ)))
end

function wave_bulk_arrays(m::SpectralWaveModel)
    A_field = m0(m.action); compute!(A_field)
    Mx_field, My_field = first_moment(m.action); compute!(Mx_field); compute!(My_field)
    A = Array(interior(A_field))
    Mx = Array(interior(Mx_field))
    My = Array(interior(My_field))
    denominator = max.(A, sqrt(eps(Float64)))
    Kx = Mx ./ denominator
    Ky = My ./ denominator
    κ = max.(hypot.(Kx, Ky), sqrt(eps(Float64)))
    return A, Kx, Ky, κ
end

wave_substeps(::MonobandedWaveModel) = monobanded_wave_substeps
wave_substeps(::SpectralWaveModel)   = spectral_wave_substeps

# ## Case setup
#
# Stokes drift is driven by ``\partial_z`` of the wave-model pseudomomentum,
# evaluated lazily via Oceananigans abstract operations. The cell-centered
# pseudomomentum from the wave model is interpolated to staggered velocity
# locations as needed.

noise_envelope(z) = exp(z / 0.012)
noisy(y, z) = noise_speed * noise_envelope(z) * randn()

# Stokes-shear kernel: builds `∂z uˢ = 2κ ε² c exp(2κz) K̂` from wave-model
# arrays (A, Kx, Ky, κ). Locations: `∂z uˢ` at (Face, Center, Face), `∂z vˢ`
# at (Center, Face, Center) — matching where Oceananigans naturally
# interpolates these for the vortex force.
@kernel function _stokes_shear_kernel!(∂z_uˢ, ∂z_vˢ, A, Kx, Ky, κ, zf,
                                       ε, ref_A, gravity, κ_max)
    j, k = @index(Global, NTuple)
    A⁺ = max(A[1, j, 1], zero(eltype(A)))
    Kxᵢ, Kyᵢ = Kx[1, j, 1], Ky[1, j, 1]
    κᵢ = max(κ[1, j, 1], sqrt(eps(eltype(κ))))
    K_norm = max(hypot(Kxᵢ, Kyᵢ), sqrt(eps(eltype(κ))))
    κˢ = min(κᵢ, κ_max)
    c  = sqrt(gravity / κˢ)
    surface_uˢ = ε^2 * (A⁺ / ref_A) * c
    expz_uˢ = exp(2κˢ * zf[k])
    @inbounds ∂z_uˢ[1, j, k] = 2κˢ * surface_uˢ * expz_uˢ * Kxᵢ / K_norm
    @inbounds ∂z_vˢ[1, j, k] = 2κˢ * surface_uˢ * expz_uˢ * Kyᵢ / K_norm
end

function refresh_stokes_shear!(∂z_uˢ, ∂z_vˢ, wave_model, ref_A)
    grid = wave_model.grid
    A, Kx, Ky, κ = wave_bulk_arrays(wave_model)
    kernel! = _stokes_shear_kernel!(KernelAbstractions.CPU(), (8, 8), (grid.Ny, grid.Nz + 1))
    kernel!(interior(∂z_uˢ), interior(∂z_vˢ), A, Kx, Ky, κ,
            zfaces(grid), steepness, ref_A, g, 1.5κ0)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    fill_halo_regions!((∂z_uˢ, ∂z_vˢ))
    return nothing
end

function build_case(; coupled_waves, wave_model_kind=:monobanded, seed=1234)
    grid = wind_drift_grid()

    ∂z_uˢ = Field{Face,   Center, Face  }(grid)
    ∂z_vˢ = Field{Center, Face,   Center}(grid)
    stokes_drift = UniformStokesDrift(grid; ∂z_uˢ, ∂z_vˢ)

    u_bc = FieldBoundaryConditions(top=FluxBoundaryCondition(surface_stress))
    ocean = NonhydrostaticModel(grid; advection=Centered(),
                                closure=ScalarDiffusivity(ν=ν),
                                stokes_drift, boundary_conditions=(; u=u_bc))

    Random.seed!(seed); set!(ocean; u=noisy, v=noisy, w=noisy)

    uᴸ = Field(@at (Center, Center, Center) ocean.velocities.u)
    vᴸ = Field(@at (Center, Center, Center) ocean.velocities.v)
    compute!(uᴸ); compute!(vᴸ); fill_halo_regions!((uᴸ, vᴸ))

    wave_model = build_wave_model(Val(wave_model_kind), grid, uᴸ, vᴸ)
    ref_A = mean(first(wave_bulk_arrays(wave_model)))
    refresh_stokes_shear!(∂z_uˢ, ∂z_vˢ, wave_model, ref_A)

    function update_wave_model!(sim)
        compute!(uᴸ); compute!(vᴸ); fill_halo_regions!((uᴸ, vᴸ))
        remaining = sim.model.clock.time - wave_model.clock.time
        if coupled_waves && remaining > 0
            substeps = wave_substeps(wave_model)
            for _ in 1:substeps; time_step!(wave_model, remaining / substeps); end
            refresh_stokes_shear!(∂z_uˢ, ∂z_vˢ, wave_model, ref_A)
        end
    end

    return (; grid, ocean, wave_model, update_wave_model!, coupled_waves)
end

# ## Diagnostics
#
# The dominant low-wavenumber Fourier amplitude of ``v`` in ``y`` is a clean
# growth-rate proxy that filters the broadband noise.

# Projection of v(y, z) onto the targeted instability mode cos(ℓ y), filters
# the broadband noise without needing an FFT dependency.
function dominant_v_amplitude(case)
    v_yz = view(interior(case.ocean.velocities.v), 1, :, :)
    v_y  = dropdims(mean(v_yz; dims=2); dims=2)
    ys   = ynodes(case.grid)
    c = sum(v_y[j] * cos(ℓ * ys[j]) for j in eachindex(v_y)) / length(v_y)
    s = sum(v_y[j] * sin(ℓ * ys[j]) for j in eachindex(v_y)) / length(v_y)
    return hypot(c, s)
end

empty_frames() = (; times=Float64[], v=Matrix{Float64}[], growth=Float64[])

function snapshot!(frames, case)
    push!(frames.times, case.ocean.clock.time)
    push!(frames.v, Array(view(interior(case.ocean.velocities.v), 1, :, :)))
    push!(frames.growth, dominant_v_amplitude(case))
end

function fitted_growth_rate(times, amplitudes)
    n = length(times)
    n ≥ 2 || return NaN
    log_amp = log.(max.(amplitudes, eps(Float64)))
    t̄ = mean(times); ā = mean(log_amp)
    num = sum((times[k] - t̄) * (log_amp[k] - ā) for k in 1:n)
    den = sum((times[k] - t̄)^2 for k in 1:n)
    return den > 0 ? num / den : NaN
end

# ## Run loop
#
# Three cases, each seeded identically so the spinup transient is shared and
# the divergence after `spinup_iterations` reflects the coupling treatment.

function run_case!(case, stop_iteration; capture=true)
    frames = empty_frames()
    sim = Simulation(case.ocean; Δt, stop_iteration, verbose=false)
    capture && add_callback!(sim, sim -> snapshot!(frames, case), IterationInterval(frame_stride))
    add_callback!(sim, case.update_wave_model!, IterationInterval(1))
    run!(sim)
    σ = capture ? fitted_growth_rate(frames.times, frames.growth) : NaN
    return merge(case, (; frames, growth_rate=σ))
end

spinup           = run_case!(build_case(coupled_waves=false), spinup_iterations; capture=false)
total_iterations = spinup_iterations + continuation_iterations
prescribed       = run_case!(build_case(coupled_waves=false),                                  total_iterations)
mono_coupled     = run_case!(build_case(coupled_waves=true,  wave_model_kind=:monobanded),     total_iterations)
spec_coupled     = run_case!(build_case(coupled_waves=true,  wave_model_kind=:spectral),       total_iterations)

@info "growth rates" prescribed=prescribed.growth_rate monobanded=mono_coupled.growth_rate spectral=spec_coupled.growth_rate
model = spec_coupled.wave_model  # exposed for the smoke harness

# ## Animation

let
    ys = collect(ynodes(spec_coupled.grid) .* 100)
    zs = collect(znodes(spec_coupled.grid) .* 100)
    frame_count = min(length(prescribed.frames.v), length(mono_coupled.frames.v), length(spec_coupled.frames.v))
    vmax = maximum(maximum(abs, f) for f in (prescribed.frames.v..., mono_coupled.frames.v..., spec_coupled.frames.v...))

    fig = Figure(size=(1500, 480))
    axes_ = (Axis(fig[1, k]; xlabel="y (cm)", ylabel="z (cm)", title=t)
             for (k, t) in enumerate(("prescribed", "monobanded coupled", "spectral coupled")))
    obs = ntuple(_ -> Observable(zeros(length(ys), length(zs))), 3)
    for (ax, ob) in zip(axes_, obs)
        heatmap!(ax, ys, zs, ob; colorrange=(-vmax, vmax), colormap=:balance)
    end
    title_obs = Observable("")
    Label(fig[0, :], title_obs; fontsize=16, halign=:center)

    record(fig, "coupled_wind_drift_instability.mp4", 1:frame_count; framerate=10) do n
        obs[1][] = prescribed.frames.v[n]
        obs[2][] = mono_coupled.frames.v[n]
        obs[3][] = spec_coupled.frames.v[n]
        title_obs[] = @sprintf("t = %.2f s  |  σ_fixed = %.3f, σ_mono = %.3f, σ_spec = %.3f s⁻¹",
                               prescribed.frames.times[n],
                               prescribed.growth_rate, mono_coupled.growth_rate, spec_coupled.growth_rate)
    end
end

# ![](coupled_wind_drift_instability.mp4)
