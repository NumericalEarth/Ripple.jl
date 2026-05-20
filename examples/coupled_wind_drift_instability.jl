# # Coupled Wind-Drift Instability
#
# This example adapts the two-dimensional wind-drift-layer instability from
# Wagner et al.'s transition-to-turbulence study to a much smaller, cheaper
# domain. The paper uses a ``19.2\,\mathrm{cm} \times 10\,\mathrm{cm}``
# ``(y, z)`` domain and a ``3\,\mathrm{cm}`` surface wave. Here we use an
# ``x``-flat Oceananigans model and a compact
# ``6\,\mathrm{cm} \times 2\,\mathrm{cm}`` domain that focuses on the
# shallow, young shear layer generated from rest.
#
# We first run a prescribed-wave spin-up from rest for ``2\,\mathrm{s}``.
# Then we save the ocean and wave state, branch three continuations from that
# identical state, and compare another ``4\,\mathrm{s}`` of prescribed waves
# against two-way wave coupling. Oceananigans advances the Craik-Leibovich
# ocean model with a mutable `UniformStokesDrift`; in the coupled branches, a
# callback copies the Lagrangian-mean ocean velocity into either a
# `MonobandedWaveModel` or a `SpectralWaveModel`, advances the wave model, and
# refreshes the Stokes-drift shear used by the next ocean step.

using Oceananigans, Ripple
using CairoMakie
import KernelAbstractions
using KernelAbstractions: @kernel, @index
using Printf
using Random
using Statistics

CairoMakie.activate!(type = "png")

# ## Domain and wave scale
#
# The resolution is deliberately coarse but two-dimensional: ``384 \times 256``
# cells in ``(y, z)`` with `Flat` topology in ``x``. The wave wavelength and
# steepness are close to the paper's short gravity-capillary wave scale.

truthy(value) = lowercase(string(value)) in ("true", "1", "yes")
env_integer(name, default) = parse(Int, get(ENV, name, string(default)))

quick_run = truthy(get(ENV, "RIPPLE_EXAMPLE_QUICK", "false"))
animate_run = truthy(get(ENV, "RIPPLE_EXAMPLE_ANIMATE", quick_run ? "false" : "true"))

default_Ny, default_Nz = quick_run ? (64, 48) : (384, 256)
Ny = env_integer("RIPPLE_EXAMPLE_NY", default_Ny)
Nz = env_integer("RIPPLE_EXAMPLE_NZ", default_Nz)
Ly, Lz = 0.060, 0.020

g = 9.81
surface_tension_parameter = 7.2e-5
λ_wave = 0.030
κ0 = 2π / λ_wave
steepness = 0.16
maximum_steepness = 0.20
maximum_stokes_wavenumber_factor = 1.5
minimum_wave_action_factor = 0.25
maximum_wave_action_factor = 4
minimum_wave_wavenumber_factor = 0.5
maximum_wave_wavenumber_factor = 2
monobanded_wave_substeps = env_integer("RIPPLE_EXAMPLE_MONOBANDED_WAVE_SUBSTEPS", 32)
spectral_wave_substeps = env_integer("RIPPLE_EXAMPLE_SPECTRAL_WAVE_SUBSTEPS", 4)
spectral_Nκ, spectral_Nφ = quick_run ? (7, 12) : (13, 24)
spectral_κ_range = range(minimum_wave_wavenumber_factor * κ0,
                         maximum_wave_wavenumber_factor * κ0;
                         length = spectral_Nκ)
spectral_φ_range = range(-π, π; length = spectral_Nφ + 1)[1:spectral_Nφ]
spectral_σκ = 0.18κ0
spectral_σφ = 0.18

λ_instability = Ly / 3
ℓ = 2π / λ_instability
m = π / Lz

ν = 1.1e-6
surface_stress = -4.8e-5
noise_speed = 0.001

Δt = 0.001
default_spinup_iterations = quick_run ? 10 : 2000
default_continuation_iterations = quick_run ? 10 : 4000
default_frame_stride = quick_run ? 10 : 20
spinup_iterations = env_integer("RIPPLE_EXAMPLE_SPINUP_ITERATIONS", default_spinup_iterations)
continuation_iterations = env_integer("RIPPLE_EXAMPLE_CONTINUATION_ITERATIONS", default_continuation_iterations)
frame_stride = env_integer("RIPPLE_EXAMPLE_FRAME_STRIDE", default_frame_stride)

# ## Ocean and wave setup
#
# We start from rest, apply the surface stress, and add seeded random
# perturbations. The organized instability is selected from the noise rather
# than imposed by the initial condition.

noise_envelope(z) = exp(z / 0.012)
u_initial(y, z) = noise_speed * noise_envelope(z) * randn()
noisy_rest(y, z) = noise_speed * noise_envelope(z) * randn()

function wind_drift_grid()
    return RectilinearGrid(CPU();
                           size     = (Ny, Nz),
                           halo     = (3, 3),
                           y        = (0, Ly),
                           z        = (-Lz, 0),
                           topology = (Flat, Periodic, Bounded))
end

spectral_wave_grid() =
    PolarWaveVectorGrid(; κ = spectral_κ_range, φ = spectral_φ_range)

wrapped_angular_distance(φ, φ₀) = atan(sin(φ - φ₀), cos(φ - φ₀))

function spectral_peak_shape(κ, φ, κ₀, φ₀)
    Δφ = wrapped_angular_distance(φ, φ₀)
    return exp(-((κ - κ₀) / spectral_σκ)^2 - (Δφ / spectral_σφ)^2)
end

function spectral_shape_normalization(spectral_grid, κ₀, φ₀)
    κ = Array(coordinate_centers(spectral_grid, 1))
    φ = Array(coordinate_centers(spectral_grid, 2))
    weights = Array(spectral_grid.weights)
    normalization = 0.0

    for n in eachindex(φ), m in eachindex(κ)
        normalization += spectral_peak_shape(κ[m], φ[n], κ₀, φ₀) * weights[m, n]
    end

    return normalization
end

function set_spectral_wave_state!(wave_model::SpectralWaveModel; action = 1, Kx = κ0, Ky = 0)
    κ₀ = max(hypot(Kx, Ky), sqrt(eps(Float64)))
    φ₀ = atan(Ky, Kx)
    normalization = spectral_shape_normalization(wave_model.spectral_grid, κ₀, φ₀)
    κ = Array(coordinate_centers(wave_model.spectral_grid, 1))
    φ = Array(coordinate_centers(wave_model.spectral_grid, 2))
    Nx, Ny, Nκ, Nφ = size(wave_model.action)
    N = Array{Float64}(undef, Nx, Ny, Nκ, Nφ)

    for n in 1:Nφ, m in 1:Nκ, j in 1:Ny, i in 1:Nx
        N[i, j, m, n] = action * spectral_peak_shape(κ[m], φ[n], κ₀, φ₀) / normalization
    end

    set!(wave_model; N)

    return wave_model
end

function wave_bulk_arrays(wave_model::MonobandedWaveModel)
    A = Array(interior(wave_model.action))
    Kx = Array(interior(wave_model.diagnostics.Kx))
    Ky = Array(interior(wave_model.diagnostics.Ky))
    κ = Array(interior(wave_model.diagnostics.κ))
    return A, Kx, Ky, κ
end

function wave_bulk_arrays(wave_model::SpectralWaveModel)
    A_field = m0(wave_model.action)
    Mx_field, My_field = first_moment(wave_model.action)
    compute!(A_field)
    compute!(Mx_field)
    compute!(My_field)

    A = Array(interior(A_field))
    Mx = Array(interior(Mx_field))
    My = Array(interior(My_field))
    denominator = max.(A, sqrt(eps(Float64)))
    Kx = Mx ./ denominator
    Ky = My ./ denominator
    κ = max.(hypot.(Kx, Ky), sqrt(eps(Float64)))
    return A, Kx, Ky, κ
end

prepare_wave_coupling!(::MonobandedWaveModel) = nothing
prepare_wave_coupling!(wave_model::SpectralWaveModel) = Ripple.update_coupling!(wave_model)
wave_time_substeps(::MonobandedWaveModel) = monobanded_wave_substeps
wave_time_substeps(::SpectralWaveModel) = spectral_wave_substeps

function build_wave_model(::Val{:monobanded}, grid, uᴸ, vᴸ)
    wave_model = MonobandedWaveModel(grid;
                                     velocities  = (; u = uᴸ, v = vᴸ),
                                     advection   = Centered(),
                                     timestepper = :RungeKutta3,
                                     gravitational_acceleration = g)

    set!(wave_model; A = 1, AKx = κ0, AKy = 0)
    return wave_model
end

function build_wave_model(::Val{:spectral}, grid, uᴸ, vᴸ)
    wave_model = SpectralWaveModel(grid, spectral_wave_grid();
                                   velocities  = (; u = uᴸ, v = vᴸ),
                                   sources     = nothing,
                                   timestepper = :RungeKutta3)

    set_spectral_wave_state!(wave_model)
    prepare_wave_coupling!(wave_model)
    return wave_model
end

# Both wave models provide a local action ``A`` and mean wavevector ``K``. We
# convert that bulk wave state into a vertically varying Stokes-drift shear,
#
# ```math
# \partial_z \boldsymbol{u}^s
# = 2 \kappa \, \epsilon^2 c \, e^{2 \kappa z} \, \hat{\boldsymbol{K}},
# \qquad c = \sqrt{g / \kappa}.
# ```
#
# The ``A / A_0`` factor lets wave-action changes modulate the Stokes drift
# magnitude while keeping the example close to the monochromatic formula.

@kernel function _refresh_stokes_shear_kernel!(∂z_uˢ, ∂z_vˢ, A, Kx, Ky, κ, zf,
                                               steepness, maximum_steepness, reference_action,
                                               reference_wavenumber, maximum_wavenumber_factor,
                                               gravity, capillary)
    i, j, k = @index(Global, NTuple)

    A⁺ = max(A[i, j, 1], zero(eltype(A)))
    Kxᵢ = Kx[i, j, 1]
    Kyᵢ = Ky[i, j, 1]
    κᵢ = max(κ[i, j, 1], sqrt(eps(eltype(κ))))
    K_norm = max(hypot(Kxᵢ, Kyᵢ), sqrt(eps(eltype(κ))))

    κˢ = min(κᵢ, maximum_wavenumber_factor * reference_wavenumber)
    phase_speed = sqrt(gravity / κˢ + capillary * κˢ)
    steepness² = min(steepness^2 * (A⁺ / reference_action), maximum_steepness^2)
    surface_stokes = steepness² * phase_speed
    vertical_shape = exp(2κˢ * zf[k])

    ∂z_uˢ[i, j, k] = 2κˢ * surface_stokes * vertical_shape * Kxᵢ / K_norm
    ∂z_vˢ[i, j, k] = 2κˢ * surface_stokes * vertical_shape * Kyᵢ / K_norm
end

function refresh_stokes_shear!(∂z_uˢ, ∂z_vˢ, wave_model;
                               steepness, maximum_steepness, reference_action,
                               reference_wavenumber, maximum_wavenumber_factor,
                               gravity, capillary)
    grid = wave_model.grid
    A, Kx, Ky, κ = wave_bulk_arrays(wave_model)
    kernel! = _refresh_stokes_shear_kernel!(KernelAbstractions.CPU(), (8, 8, 8), (grid.Nx, grid.Ny, grid.Nz + 1))
    kernel!(interior(∂z_uˢ), interior(∂z_vˢ),
            A, Kx, Ky, κ,
            zfaces(grid),
            steepness, maximum_steepness, reference_action,
            reference_wavenumber, maximum_wavenumber_factor,
            gravity, capillary)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    fill_halo_regions!((∂z_uˢ, ∂z_vˢ))
    return nothing
end

@kernel function _limit_wave_state_kernel!(A, AKx, AKy, reference_action, reference_wavenumber,
                                           minimum_action_factor, maximum_action_factor,
                                           minimum_wavenumber_factor, maximum_wavenumber_factor)
    i, j, k = @index(Global, NTuple)

    Aᵢ = A[i, j, k]
    minimum_action = minimum_action_factor * reference_action
    maximum_action = maximum_action_factor * reference_action
    A⁺ = min(max(Aᵢ, minimum_action), maximum_action)

    denominator = max(abs(Aᵢ), minimum_action)
    Kxᵢ = AKx[i, j, k] / denominator
    Kyᵢ = AKy[i, j, k] / denominator
    κᵢ = max(hypot(Kxᵢ, Kyᵢ), minimum_wavenumber_factor * reference_wavenumber)
    κˡ = min(κᵢ, maximum_wavenumber_factor * reference_wavenumber)
    rescale = κˡ / κᵢ

    A[i, j, k] = A⁺
    AKx[i, j, k] = A⁺ * Kxᵢ * rescale
    AKy[i, j, k] = A⁺ * Kyᵢ * rescale
end

function limit_wave_state!(wave_model::MonobandedWaveModel; reference_action, reference_wavenumber)
    grid = wave_model.grid
    kernel! = _limit_wave_state_kernel!(KernelAbstractions.CPU(), (16, 16, 1), (grid.Nx, grid.Ny, 1))
    kernel!(interior(wave_model.action),
            interior(wave_model.wavenumber_moment.x),
            interior(wave_model.wavenumber_moment.y),
            reference_action, reference_wavenumber,
            minimum_wave_action_factor, maximum_wave_action_factor,
            minimum_wave_wavenumber_factor, maximum_wave_wavenumber_factor)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    fill_halo_regions!((wave_model.action, wave_model.wavenumber_moment.x, wave_model.wavenumber_moment.y))
    Ripple.update_monobanded_diagnostics!(wave_model)
    return nothing
end

limit_wave_state!(::SpectralWaveModel; reference_action, reference_wavenumber) = nothing

function build_case(; coupled_waves, wave_model_kind = :monobanded, time_offset = 0)
    grid = wind_drift_grid()

    ∂z_uˢ = Field{Center, Center, Face}(grid)
    ∂z_vˢ = Field{Center, Center, Face}(grid)
    stokes_drift = UniformStokesDrift(grid; ∂z_uˢ, ∂z_vˢ)

    u_boundary_conditions = FieldBoundaryConditions(top = FluxBoundaryCondition(surface_stress))

    ocean_model = NonhydrostaticModel(grid;
                                      advection    = Centered(),
                                      closure      = ScalarDiffusivity(ν = ν),
                                      stokes_drift = stokes_drift,
                                      boundary_conditions = (; u = u_boundary_conditions))

    Random.seed!(1234)
    set!(ocean_model; u = u_initial, v = noisy_rest, w = noisy_rest)

    # Ripple's prescribed-current coupling expects center-located horizontal
    # velocity fields on the Q grid. Oceananigans keeps `u` and `v` on
    # staggered faces, so we construct computed center fields and refresh them
    # in the coupled callback and diagnostic callback.
    uᴸ = Field(@at (Center, Center, Center) ocean_model.velocities.u)
    vᴸ = Field(@at (Center, Center, Center) ocean_model.velocities.v)
    wᴸ = Field(@at (Center, Center, Center) ocean_model.velocities.w)
    compute!(uᴸ)
    compute!(vᴸ)
    compute!(wᴸ)

    wave_model = build_wave_model(Val(wave_model_kind), grid, uᴸ, vᴸ)
    reference_action = mean(first(wave_bulk_arrays(wave_model)))

    refresh_stokes_shear!(∂z_uˢ, ∂z_vˢ, wave_model;
                          steepness, reference_action, gravity = g,
                          reference_wavenumber = κ0,
                          maximum_steepness = maximum_steepness,
                          maximum_wavenumber_factor = maximum_stokes_wavenumber_factor,
                          capillary = surface_tension_parameter)

    function update_wave_model!(simulation)
        compute!(uᴸ)
        compute!(vᴸ)
        fill_halo_regions!((uᴸ, vᴸ))

        remaining_wave_time = simulation.model.clock.time - wave_model.clock.time

        if coupled_waves && remaining_wave_time > 0
            prepare_wave_coupling!(wave_model)
            substeps = wave_time_substeps(wave_model)

            for _ in 1:substeps
                time_step!(wave_model, remaining_wave_time / substeps)
                limit_wave_state!(wave_model; reference_action, reference_wavenumber = κ0)
            end
        end

        coupled_waves &&
                refresh_stokes_shear!(∂z_uˢ, ∂z_vˢ, wave_model;
                                      steepness, reference_action, gravity = g,
                                      reference_wavenumber = κ0,
                                      maximum_steepness = maximum_steepness,
                                      maximum_wavenumber_factor = maximum_stokes_wavenumber_factor,
                                      capillary = surface_tension_parameter)

        return nothing
    end

    return (; grid, ocean_model, wave_model, uᴸ, vᴸ, wᴸ, ∂z_uˢ, ∂z_vˢ,
            reference_action, wave_model_kind, time_offset,
            update_wave_model!, coupled_waves)
end

function save_wave_state(wave_model::MonobandedWaveModel)
    return (wave_model_kind = :monobanded,
            A = Array(interior(wave_model.action)),
            AKx = Array(interior(wave_model.wavenumber_moment.x)),
            AKy = Array(interior(wave_model.wavenumber_moment.y)))
end

function save_wave_state(wave_model::SpectralWaveModel)
    return (wave_model_kind = :spectral,
            N = Array(interior(wave_model.action)))
end

function restore_wave_state!(wave_model::MonobandedWaveModel, state)
    set!(wave_model; A = state.A, AKx = state.AKx, AKy = state.AKy)
    fill_halo_regions!((wave_model.action,
                        wave_model.wavenumber_moment.x,
                        wave_model.wavenumber_moment.y))
    return wave_model
end

function restore_wave_state!(wave_model::SpectralWaveModel, state)
    if haskey(state, :N)
        set!(wave_model; N = state.N)
    else
        A = state.A
        denominator = max.(A, sqrt(eps(Float64)))
        Kx = state.AKx ./ denominator
        Ky = state.AKy ./ denominator
        set_spectral_wave_state!(wave_model; action = mean(A), Kx = mean(Kx), Ky = mean(Ky))
    end

    prepare_wave_coupling!(wave_model)
    return wave_model
end

function save_state(case)
    compute!(case.uᴸ)
    compute!(case.vᴸ)
    compute!(case.wᴸ)
    fill_halo_regions!((case.uᴸ, case.vᴸ, case.wᴸ))

    ocean_state = (u = Array(interior(case.ocean_model.velocities.u)),
                   v = Array(interior(case.ocean_model.velocities.v)),
                   w = Array(interior(case.ocean_model.velocities.w)))

    return merge(ocean_state, save_wave_state(case.wave_model),
                 (time = case.ocean_model.clock.time,))
end

function restore_state!(case, state)
    set!(case.ocean_model.velocities.u, state.u)
    set!(case.ocean_model.velocities.v, state.v)
    set!(case.ocean_model.velocities.w, state.w)
    fill_halo_regions!(case.ocean_model.velocities)

    restore_wave_state!(case.wave_model, state)

    compute!(case.uᴸ)
    compute!(case.vᴸ)
    compute!(case.wᴸ)
    fill_halo_regions!((case.uᴸ, case.vᴸ, case.wᴸ))
    prepare_wave_coupling!(case.wave_model)

    refresh_stokes_shear!(case.∂z_uˢ, case.∂z_vˢ, case.wave_model;
                          steepness, reference_action = case.reference_action,
                          gravity = g, reference_wavenumber = κ0,
                          maximum_steepness = maximum_steepness,
                          maximum_wavenumber_factor = maximum_stokes_wavenumber_factor,
                          capillary = surface_tension_parameter)
    return case
end

# ## Running and measuring growth
#
# The comparison metric is the dominant low-wavenumber Fourier amplitude of
# ``v`` in ``y``. This filters the broadband random perturbation and tracks
# the coherent instability selected from the noise.

empty_frames() = (; times = Float64[],
                  v = Matrix{Float64}[],
                  growth = Float64[],
                  φ = Vector{Float64}[],
                  wave_action = Vector{Float64}[],
                  stokes_shear = Vector{Float64}[],
                  ocean_kinetic_energy = Float64[],
                  perturbation_kinetic_energy = Float64[],
                  wave_energy = Float64[],
                  total_energy = Float64[])

function ocean_velocity_arrays(case)
    compute!(case.uᴸ)
    compute!(case.vᴸ)
    compute!(case.wᴸ)
    fill_halo_regions!((case.uᴸ, case.vᴸ, case.wᴸ))

    u = Array(interior(case.uᴸ))[1, :, :]
    v = Array(interior(case.vᴸ))[1, :, :]
    w = Array(interior(case.wᴸ))[1, :, :]
    return u, v, w
end

function ocean_kinetic_energy(case)
    u, v, w = ocean_velocity_arrays(case)
    Δy = Ly / Ny
    Δz = Lz / Nz
    return sum(@. (u^2 + v^2 + w^2) / 2) * Δy * Δz
end

function ocean_perturbation_kinetic_energy(case)
    u, v, w = ocean_velocity_arrays(case)
    u′ = u .- mean(u; dims = 1)
    v′ = v .- mean(v; dims = 1)
    w′ = w .- mean(w; dims = 1)
    Δy = Ly / Ny
    Δz = Lz / Nz
    return sum(@. (u′^2 + v′^2 + w′^2) / 2) * Δy * Δz
end

function wave_linear_energy(case)
    A, _, _, κ = wave_bulk_arrays(case.wave_model)
    A⁺ = max.(A, 0)
    κ⁺ = max.(κ, sqrt(eps(Float64)))
    steepness² = min.(steepness^2 .* A⁺ ./ case.reference_action, maximum_steepness^2)
    energy_density = @. (g + surface_tension_parameter * κ⁺^2) * steepness² / (2κ⁺^2)
    Δy = Ly / Ny
    return sum(energy_density[1, :, 1]) * Δy
end

function capture_state!(frames, case)
    compute!(case.uᴸ)
    compute!(case.vᴸ)
    compute!(case.wᴸ)
    fill_halo_regions!((case.uᴸ, case.vᴸ, case.wᴸ))

    v = Array(interior(case.vᴸ))[1, :, :] .* 100 # cm s⁻¹
    A, Kx, Ky, _ = wave_bulk_arrays(case.wave_model)
    Kx_y = vec(mean(Kx; dims = (1, 3)))
    Ky_y = vec(mean(Ky; dims = (1, 3)))
    A_y = vec(mean(A; dims = (1, 3))) ./ case.reference_action
    K = ocean_kinetic_energy(case)
    K′ = ocean_perturbation_kinetic_energy(case)
    Ew = wave_linear_energy(case)

    push!(frames.times, case.time_offset + case.ocean_model.clock.time)
    push!(frames.v, v)
    push!(frames.growth, dominant_low_mode_amplitude(v))
    push!(frames.φ, atan.(Ky_y, Kx_y))
    push!(frames.wave_action, A_y)
    push!(frames.stokes_shear, Array(interior(case.∂z_uˢ))[1, 1, :])
    push!(frames.ocean_kinetic_energy, K)
    push!(frames.perturbation_kinetic_energy, K′)
    push!(frames.wave_energy, Ew)
    push!(frames.total_energy, K + Ew)
    return nothing
end

function meridional_mode_amplitude(v, n)
    Ny, Nz = size(v)
    amplitude² = 0.0

    for k in 1:Nz
        coefficient = 0.0 + 0.0im
        for j in 1:Ny
            coefficient += v[j, k] * cis(-2π * n * (j - 1) / Ny)
        end
        amplitude² += abs2(coefficient / Ny)
    end

    return sqrt(amplitude² / Nz)
end

dominant_low_mode_amplitude(v) =
    maximum(meridional_mode_amplitude(v, n) for n in 1:6)

function fitted_growth_rate(times, amplitudes)
    start = max(2, floor(Int, 0.45 * length(times)))
    fit_range = start:length(times)
    t = times[fit_range]
    y = log.(max.(amplitudes[fit_range], eps(Float64)))
    t′ = t .- mean(t)
    y′ = y .- mean(y)
    return sum(t′ .* y′) / sum(abs2, t′)
end

function relative_change(values)
    initial = first(values)
    final = last(values)
    return (final - initial) / max(abs(initial), eps(Float64))
end

function print_energy_summary(name, frames)
    ΔK = relative_change(frames.ocean_kinetic_energy)
    ΔK′ = relative_change(frames.perturbation_kinetic_energy)
    ΔEw = relative_change(frames.wave_energy)
    ΔEt = relative_change(frames.total_energy)

    println(@sprintf("%s_ocean_kinetic_energy_change = %.3e", name, ΔK))
    println(@sprintf("%s_perturbation_kinetic_energy_change = %.3e", name, ΔK′))
    println(@sprintf("%s_wave_energy_change          = %.3e", name, ΔEw))
    println(@sprintf("%s_total_energy_change         = %.3e", name, ΔEt))
end

function run_case!(case, stop_iteration; capture = true)
    frames = empty_frames()
    simulation = Simulation(case.ocean_model; Δt, stop_iteration, verbose = false)

    if case.coupled_waves
        simulation.callbacks[:wave_coupling] = Callback(case.update_wave_model!, IterationInterval(1))
    end

    if capture
        capture_state!(frames, case)
        simulation.callbacks[:capture] = Callback(sim -> capture_state!(frames, case),
                                                  IterationInterval(frame_stride))
    end

    run!(simulation)

    if simulation.model.clock.iteration < stop_iteration
        error("wind-drift instability run stopped early at iteration $(simulation.model.clock.iteration)")
    end

    σ = capture ? fitted_growth_rate(frames.times, frames.growth) : NaN
    return merge(case, (; frames, growth_rate = σ))
end

spinup = build_case(coupled_waves = false)
spinup = run_case!(spinup, spinup_iterations; capture = false)
spinup_state = save_state(spinup)

prescribed = build_case(coupled_waves = false, time_offset = spinup_state.time)
restore_state!(prescribed, spinup_state)
prescribed = run_case!(prescribed, continuation_iterations)

coupled = build_case(coupled_waves = true, time_offset = spinup_state.time)
restore_state!(coupled, spinup_state)
coupled = run_case!(coupled, continuation_iterations)

spectral_coupled = build_case(coupled_waves = true,
                              wave_model_kind = :spectral,
                              time_offset = spinup_state.time)
restore_state!(spectral_coupled, spinup_state)
spectral_coupled = run_case!(spectral_coupled, continuation_iterations)

prescribed_growth_rate = prescribed.growth_rate
coupled_growth_rate = coupled.growth_rate
spectral_coupled_growth_rate = spectral_coupled.growth_rate

println(@sprintf("spinup_time                  = %.3f s", spinup_state.time))
println(@sprintf("monobanded_wave_substep      = %.3e s", Δt / monobanded_wave_substeps))
println(@sprintf("spectral_wave_substep        = %.3e s", Δt / spectral_wave_substeps))
println(@sprintf("prescribed_wave_growth_rate          = %.4f s^-1", prescribed_growth_rate))
println(@sprintf("monobanded_coupled_wave_growth_rate  = %.4f s^-1", coupled_growth_rate))
println(@sprintf("spectral_coupled_wave_growth_rate    = %.4f s^-1", spectral_coupled_growth_rate))
println(@sprintf("monobanded_coupled / prescribed      = %.3f", coupled_growth_rate / prescribed_growth_rate))
println(@sprintf("spectral_coupled / prescribed        = %.3f", spectral_coupled_growth_rate / prescribed_growth_rate))
print_energy_summary("prescribed", prescribed.frames)
print_energy_summary("monobanded_coupled", coupled.frames)
print_energy_summary("spectral_coupled", spectral_coupled.frames)

# ## Animation

if animate_run
    ys = ynodes(spectral_coupled.grid) .* 100
    zs = znodes(spectral_coupled.grid) .* 100

    times = prescribed.frames.times
    frame_count = min(length(prescribed.frames.v),
                      length(coupled.frames.v),
                      length(spectral_coupled.frames.v))

    prescribed_vlim = maximum(maximum(abs, frame) for frame in prescribed.frames.v)
    coupled_vlim = maximum(maximum(abs, frame) for frame in coupled.frames.v)
    spectral_coupled_vlim = maximum(maximum(abs, frame) for frame in spectral_coupled.frames.v)
    prescribed_vlim = max(prescribed_vlim, 0.05)
    coupled_vlim = max(coupled_vlim, 0.01)
    spectral_coupled_vlim = max(spectral_coupled_vlim, 0.01)

    perturbation_energy_change(frames) =
        frames.perturbation_kinetic_energy .- first(frames.perturbation_kinetic_energy)

    wave_energy_change(frames) =
        frames.wave_energy .- first(frames.wave_energy)

    prescribed_δK′ = perturbation_energy_change(prescribed.frames)
    coupled_δK′ = perturbation_energy_change(coupled.frames)
    spectral_coupled_δK′ = perturbation_energy_change(spectral_coupled.frames)

    prescribed_δEw = wave_energy_change(prescribed.frames)
    coupled_δEw = wave_energy_change(coupled.frames)
    spectral_coupled_δEw = wave_energy_change(spectral_coupled.frames)

    energy_min = min(minimum(prescribed_δK′), 0)
    energy_max = max(maximum(prescribed_δK′), 0)
    energy_padding = 0.05 * max(energy_max - energy_min, eps(Float64))

    action_min = minimum(minimum(frame) for frame in (prescribed.frames.wave_action...,
                                                      coupled.frames.wave_action...,
                                                      spectral_coupled.frames.wave_action...))
    action_max = maximum(maximum(frame) for frame in (prescribed.frames.wave_action...,
                                                      coupled.frames.wave_action...,
                                                      spectral_coupled.frames.wave_action...))
    action_padding = 0.05 * max(action_max - action_min, eps(Float64))

    prescribed_v_obs = Observable(prescribed.frames.v[1])
    coupled_v_obs = Observable(coupled.frames.v[1])
    spectral_coupled_v_obs = Observable(spectral_coupled.frames.v[1])
    prescribed_δK′_obs = Observable(Point2f[(times[1], prescribed_δK′[1])])
    coupled_δK′_obs = Observable(Point2f[(times[1], coupled_δK′[1])])
    spectral_coupled_δK′_obs = Observable(Point2f[(times[1], spectral_coupled_δK′[1])])
    prescribed_δEw_obs = Observable(Point2f[(times[1], prescribed_δEw[1])])
    coupled_δEw_obs = Observable(Point2f[(times[1], coupled_δEw[1])])
    spectral_coupled_δEw_obs = Observable(Point2f[(times[1], spectral_coupled_δEw[1])])
    prescribed_action_obs = Observable(prescribed.frames.wave_action[1])
    coupled_action_obs = Observable(coupled.frames.wave_action[1])
    spectral_coupled_action_obs = Observable(spectral_coupled.frames.wave_action[1])
    title_obs = Observable("t = 0.00 s")

    fig = Figure(size = (1500, 820))

    ax1 = Axis(fig[1, 1];
               title = "Prescribed wave: v (cm s⁻¹)",
               xlabel = "y (cm)", ylabel = "z (cm)")

    ax2 = Axis(fig[1, 3];
               title = "Monobanded coupled: v (cm s⁻¹)",
               xlabel = "y (cm)", ylabel = "z (cm)")

    ax3 = Axis(fig[1, 5];
               title = "Spectral coupled: v (cm s⁻¹)",
               xlabel = "y (cm)", ylabel = "z (cm)")

    ax4 = Axis(fig[2, 1:3];
               title = "Perturbation kinetic energy and wave energy",
               xlabel = "time (s)", ylabel = "energy change / ρ per unit x (m⁴ s⁻²)",
               limits = ((times[1], times[end]), (energy_min - energy_padding, energy_max + energy_padding)))

    ax5 = Axis(fig[2, 4:6];
               title = "Horizontal wave-action distribution",
               xlabel = "y (cm)", ylabel = "A / A₀",
               limits = ((ys[1], ys[end]), (action_min - action_padding, action_max + action_padding)))

    hm1 = heatmap!(ax1, ys, zs, prescribed_v_obs; colormap = :vik, colorrange = (-prescribed_vlim, prescribed_vlim))
    hm2 = heatmap!(ax2, ys, zs, coupled_v_obs; colormap = :vik, colorrange = (-coupled_vlim, coupled_vlim))
    hm3 = heatmap!(ax3, ys, zs, spectral_coupled_v_obs; colormap = :vik, colorrange = (-spectral_coupled_vlim, spectral_coupled_vlim))

    lines!(ax4, prescribed_δK′_obs; linewidth = 3, color = :dodgerblue, label = "prescribed ΔK′")
    lines!(ax4, coupled_δK′_obs; linewidth = 3, color = :darkorange, label = "monobanded ΔK′")
    lines!(ax4, spectral_coupled_δK′_obs; linewidth = 3, color = :seagreen, label = "spectral ΔK′")
    lines!(ax4, prescribed_δEw_obs; linewidth = 2, color = :dodgerblue, linestyle = :dash, label = "prescribed ΔEwave")
    lines!(ax4, coupled_δEw_obs; linewidth = 2, color = :darkorange, linestyle = :dash, label = "monobanded ΔEwave")
    lines!(ax4, spectral_coupled_δEw_obs; linewidth = 2, color = :seagreen, linestyle = :dash, label = "spectral ΔEwave")
    hlines!(ax4, 0; color = (:black, 0.35), linestyle = :dash)
    axislegend(ax4; position = :lt)

    lines!(ax5, ys, prescribed_action_obs; linewidth = 3, color = :dodgerblue, label = "prescribed")
    lines!(ax5, ys, coupled_action_obs; linewidth = 3, color = :darkorange, label = "monobanded")
    lines!(ax5, ys, spectral_coupled_action_obs; linewidth = 3, color = :seagreen, label = "spectral")
    hlines!(ax5, 1; color = (:black, 0.35), linestyle = :dash)
    axislegend(ax5; position = :lt)

    Colorbar(fig[1, 2], hm1)
    Colorbar(fig[1, 4], hm2)
    Colorbar(fig[1, 6], hm3)

    Label(fig[0, :], title_obs; fontsize = 18, halign = :center)

    comparison_animation = "coupled_wind_drift_instability.mp4"

    record(fig, comparison_animation, 1:frame_count; framerate = 10) do n
        prescribed_v_obs[] = prescribed.frames.v[n]
        coupled_v_obs[] = coupled.frames.v[n]
        spectral_coupled_v_obs[] = spectral_coupled.frames.v[n]
        prescribed_δK′_obs[] = Point2f.(times[1:n], prescribed_δK′[1:n])
        coupled_δK′_obs[] = Point2f.(times[1:n], coupled_δK′[1:n])
        spectral_coupled_δK′_obs[] = Point2f.(times[1:n], spectral_coupled_δK′[1:n])
        prescribed_δEw_obs[] = Point2f.(times[1:n], prescribed_δEw[1:n])
        coupled_δEw_obs[] = Point2f.(times[1:n], coupled_δEw[1:n])
        spectral_coupled_δEw_obs[] = Point2f.(times[1:n], spectral_coupled_δEw[1:n])
        prescribed_action_obs[] = prescribed.frames.wave_action[n]
        coupled_action_obs[] = coupled.frames.wave_action[n]
        spectral_coupled_action_obs[] = spectral_coupled.frames.wave_action[n]

        title_obs[] = @sprintf("Wind-drift instability with wave coupling — t = %.2f s, σ_fixed = %.3f s⁻¹, σ_mono = %.3f s⁻¹, σ_spectral = %.3f s⁻¹",
                               times[n], prescribed_growth_rate, coupled_growth_rate, spectral_coupled_growth_rate)
    end

    println("comparison_animation = $(abspath(comparison_animation))")
end

model = spectral_coupled.wave_model # exposed for the example smoke harness
nothing #hide

# ![](coupled_wind_drift_instability.mp4)
