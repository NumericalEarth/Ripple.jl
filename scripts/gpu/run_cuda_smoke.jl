using Ripple

function usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/gpu/run_cuda_smoke.jl OUTPUT.tsv

    Runs an opt-in CUDA smoke test through Oceananigans.GPU(). CUDA.jl must be
    available on the Julia load path and `CUDA.functional()` must be true.
    """
end

function load_cuda!()
    try
        @eval using CUDA
    catch err
        error("CUDA.jl must be available to run this optional smoke test. Original error: $err")
    end

    isdefined(Base, :retry_load_extensions) && Base.retry_load_extensions()
    Base.invokelatest(CUDA.functional) ||
        error("CUDA.jl is available but `CUDA.functional()` is false; no usable CUDA device/runtime was detected")

    return CUDA
end

function cuda_smoke_model(arch)
    grid = RectilinearGrid(arch;
                           size=(3, 2, 2),
                           x=(0, 3),
                           y=(-1, 1),
                           z=(-1, 0))
    cgrid = PolarWaveVectorGrid(Float64;
                                κ=[0.45, 0.8, 1.2],
                                φ=range(0, 2pi; length=7)[1:6])
    sources = SourceTermSet((WaveAgeWindInput(rate=0.01,
                                              speed=3.0,
                                              inverse_wave_age_threshold=0.0,
                                              spreading_power=1.0,
                                              gravity=1.0),
                             BottomFriction(rate=0.002)))
    model = SpectralWaveModel(; grid,
                                spectral_grid=cgrid,
                                sources,
                                advection=nothing,
                                timestepper=:SemiImplicitEuler)
    set!(model, N=(x, y, κ, φ) -> 0.5 + 0.01x - 0.02y + 0.03κ + 0.02cos(φ))
    return model
end

function cuda_frequency_diagnostic_model(arch)
    grid = RectilinearGrid(arch; size=(2, 2, 1), x=(0, 2), y=(-1, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(Float64;
                                   frequency=[0.08, 0.12, 0.18, 0.27],
                                   φ=range(0, 2pi; length=9)[1:8])
    model = SpectralWaveModel(; grid, spectral_grid=cgrid, advection=nothing)
    set!(model, N=(x, y, f, φ) -> 0.4 + 0.02x - 0.01y +
                                      0.04 * (2pi * f)^2 / 9.81 + 0.03cos(φ))
    return model
end

function max_abs_difference(a, b)
    return maximum(abs.(Array(a) .- Array(b)))
end

function field_data_backend(field)
    data = parent(field)
    return hasproperty(data, :parent) ? getproperty(data, :parent) : data
end

function cuda_smoke_result()
    load_cuda!()
    cuda_architecture = GPU()
    cpu_model = cuda_smoke_model(CPU())
    gpu_model = cuda_smoke_model(cuda_architecture)
    cpu_frequency_model = cuda_frequency_diagnostic_model(CPU())
    gpu_frequency_model = cuda_frequency_diagnostic_model(cuda_architecture)

    for _ in 1:3
        time_step!(cpu_model, 0.02)
        time_step!(gpu_model, 0.02)
    end

    compute_tendencies!(cpu_model)
    compute_tendencies!(gpu_model)
    Base.invokelatest(CUDA.synchronize)

    qtransform = QTransform(QKernel(Float64), cpu_model.grid)
    cpu_px, cpu_py = pseudomomentum_fields(cpu_model.action, 1.0, qtransform)
    gpu_px, gpu_py = pseudomomentum_fields(gpu_model.action, 1.0, qtransform)
    Base.invokelatest(CUDA.synchronize)

    metrics = Dict(:action_error => max_abs_difference(interior(cpu_model.action), interior(gpu_model.action)),
                   :tendency_error => max_abs_difference(interior(cpu_model.tendencies), interior(gpu_model.tendencies)),
                   :total_action_error => abs(total_action(cpu_model.action) - total_action(gpu_model.action)),
                   :m0_error => max_abs_difference(m0(cpu_model.action), m0(gpu_model.action)),
                   :significant_wave_height_error => max_abs_difference(significant_wave_height(cpu_model.action),
                                                                        significant_wave_height(gpu_model.action)),
                   :rms_wavenumber_error => max_abs_difference(root_mean_square_wavenumber(cpu_model.action),
                                                               root_mean_square_wavenumber(gpu_model.action)),
                   :deep_water_energy_density_error => max_abs_difference(deep_water_energy_density(cpu_model.action),
                                                                          deep_water_energy_density(gpu_model.action)),
                   :mean_deep_water_group_speed_error => max_abs_difference(mean_deep_water_group_speed(cpu_model.action),
                                                                            mean_deep_water_group_speed(gpu_model.action)),
                   :mean_direction_error => max_abs_difference(mean_direction(cpu_model.action),
                                                               mean_direction(gpu_model.action)),
                   :peak_direction_error => max_abs_difference(peak_direction(cpu_model.action),
                                                               peak_direction(gpu_model.action)),
                   :peak_wavenumber_error => max_abs_difference(peak_wavenumber(cpu_model.action),
                                                                peak_wavenumber(gpu_model.action)),
                   :deep_water_peak_phase_speed_error => max_abs_difference(deep_water_peak_phase_speed(cpu_model.action),
                                                                            deep_water_peak_phase_speed(gpu_model.action)),
                   :frequency_mean_frequency_error => max_abs_difference(mean_frequency(cpu_frequency_model.action),
                                                                          mean_frequency(gpu_frequency_model.action)),
                   :frequency_mean_period_error => max_abs_difference(mean_period(cpu_frequency_model.action),
                                                                       mean_period(gpu_frequency_model.action)),
                   :frequency_peak_frequency_error => max_abs_difference(peak_frequency(cpu_frequency_model.action),
                                                                          peak_frequency(gpu_frequency_model.action)),
                   :frequency_peak_period_error => max_abs_difference(peak_period(cpu_frequency_model.action),
                                                                       peak_period(gpu_frequency_model.action)),
                   :frequency_deep_water_energy_density_error => max_abs_difference(deep_water_energy_density(cpu_frequency_model.action),
                                                                                    deep_water_energy_density(gpu_frequency_model.action)),
                   :frequency_mean_deep_water_group_speed_error => max_abs_difference(mean_deep_water_group_speed(cpu_frequency_model.action),
                                                                                      mean_deep_water_group_speed(gpu_frequency_model.action)),
                   :pseudomomentum_x_error => max_abs_difference(Ripple.field_storage(cpu_px), Ripple.field_storage(gpu_px)),
                   :pseudomomentum_y_error => max_abs_difference(Ripple.field_storage(cpu_py), Ripple.field_storage(gpu_py)),
                   :pseudomomentum_backend_error => field_data_backend(gpu_px) isa CUDA.CuArray && field_data_backend(gpu_py) isa CUDA.CuArray ? 0.0 : 1.0,
                   :backend_error => field_data_backend(physical_field(gpu_model.action, 1, 1)) isa CUDA.CuArray ? 0.0 : 1.0)
    tolerances = Dict(:action_error => 1e-12,
                      :tendency_error => 1e-12,
                      :total_action_error => 1e-12,
                      :m0_error => 1e-12,
                      :significant_wave_height_error => 1e-12,
                      :rms_wavenumber_error => 1e-12,
                      :deep_water_energy_density_error => 1e-12,
                      :mean_deep_water_group_speed_error => 1e-12,
                      :mean_direction_error => 1e-12,
                      :peak_direction_error => 1e-12,
                      :peak_wavenumber_error => 1e-12,
                      :deep_water_peak_phase_speed_error => 1e-12,
                      :frequency_mean_frequency_error => 1e-12,
                      :frequency_mean_period_error => 1e-12,
                      :frequency_peak_frequency_error => 1e-12,
                      :frequency_peak_period_error => 1e-12,
                      :frequency_deep_water_energy_density_error => 1e-12,
                      :frequency_mean_deep_water_group_speed_error => 1e-12,
                      :pseudomomentum_x_error => 1e-12,
                      :pseudomomentum_y_error => 1e-12,
                      :pseudomomentum_backend_error => 0.0,
                      :backend_error => 0.0)
    return ValidationResult(:cuda_smoke,
                            "CUDA-backed GPU architecture path matches the CPU path for a small source-coupled model, bulk diagnostics, frequency-direction diagnostics, and pseudomomentum fields.",
                            metrics,
                            tolerances)
end

function run_cuda_smoke_script(args=ARGS)
    length(args) == 1 || error(usage())
    output_path = only(args)
    write_validation_summary(output_path, (cuda_smoke_result(),))
    println(output_path)
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_cuda_smoke_script()
end
