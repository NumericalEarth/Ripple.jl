using Ripple

import Oceananigans
import Oceananigans.Fields: CenterField

function usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/gpu/run_metal_smoke.jl OUTPUT.tsv

    Runs an opt-in Metal smoke test through Oceananigans.GPU(MetalBackend()).
    Metal.jl must be available on the Julia load path and `Metal.functional()`
    must be true. Metal device arrays are Float32-only, so this smoke uses
    Float32 grids and spectral coordinates.
    """
end

function load_metal!()
    try
        @eval using Metal
    catch err
        error("Metal.jl must be available to run this optional smoke test. Original error: $err")
    end

    isdefined(Base, :retry_load_extensions) && Base.retry_load_extensions()

    functional = try
        Base.invokelatest(Metal.functional)
    catch err
        error("Metal.jl is available but `Metal.functional()` failed. Original error: $err")
    end

    functional ||
        error("Metal.jl is available but `Metal.functional()` is false; no usable Metal device/runtime was detected")

    return Metal
end

function sync_metal!()
    Base.invokelatest(getproperty(Metal, :synchronize))
    return nothing
end

metal_architecture() = GPU(Metal.MetalBackend())

function field_data_backend(field)
    data = parent(field)
    return hasproperty(data, :parent) ? getproperty(data, :parent) : data
end

function max_abs_difference(a, b)
    return maximum(abs.(Array(a) .- Array(b)))
end

function periodic_grid(arch, ::Type{FT}; size=(8, 7, 1)) where FT
    return RectilinearGrid(arch, FT;
                           size,
                           halo=(3, 3, 3),
                           x=(zero(FT), FT(size[1])),
                           y=(zero(FT), FT(size[2])),
                           z=(-one(FT), zero(FT)),
                           topology=(Periodic, Periodic, Bounded))
end

function polar_grid(arch, ::Type{FT}; radial_bins=3, direction_bins=6) where FT
    twoπ = FT(2) * FT(pi)
    κ = collect(FT, range(FT(0.45), FT(1.20); length=radial_bins))
    φ = collect(FT, range(zero(FT), twoπ; length=direction_bins + 1))[1:direction_bins]
    return PolarWaveVectorGrid(arch, FT; κ, φ)
end

function action_values(::Type{FT}, dims) where FT
    Nx, Ny, Nκ, Nφ = dims
    values = zeros(FT, dims)
    twoπ = 2pi

    @inbounds for n in 1:Nφ, m in 1:Nκ, j in 1:Ny, i in 1:Nx
        values[i, j, m, n] = FT(0.6 +
                                0.04 * sin(twoπ * i / Nx) +
                                0.03 * cos(twoπ * j / Ny) +
                                0.02 * m +
                                0.01 * cos(twoπ * n / Nφ))
    end

    return values
end

function metal_product_field_result(metal_arch)
    FT = Float32
    cpu_grid = periodic_grid(CPU(), FT; size=(4, 3, 1))
    gpu_grid = periodic_grid(metal_arch, FT; size=(4, 3, 1))
    cpu_cgrid = polar_grid(CPU(), FT; radial_bins=2, direction_bins=4)
    gpu_cgrid = polar_grid(metal_arch, FT; radial_bins=2, direction_bins=4)
    cpu_action = WaveActionField(cpu_grid, cpu_cgrid)
    gpu_action = WaveActionField(gpu_grid, gpu_cgrid)
    values = action_values(FT, size(cpu_action))

    set!(cpu_action, values)
    set!(gpu_action, values)
    sync_metal!()

    metrics = Dict(:product_field_set_error => max_abs_difference(interior(cpu_action), interior(gpu_action)),
                   :product_field_backend_error => field_data_backend(physical_field(gpu_action, 1, 1)) isa Metal.MtlArray ? 0.0 : 1.0)
    tolerances = Dict(:product_field_set_error => 1e-6,
                      :product_field_backend_error => 0.0)
    return ValidationResult(:metal_product_field,
                            "Metal-backed ProductField storage supports the KA scalar/array set and interior-copy kernels.",
                            metrics,
                            tolerances)
end

function intrinsic_transport_model(arch, ::Type{FT}) where FT
    grid = periodic_grid(arch, FT; size=(8, 7, 1))
    cgrid = polar_grid(arch, FT; radial_bins=3, direction_bins=6)
    model = SpectralWaveModel(; grid,
                                spectral_grid=cgrid,
                                horizontal_advection=WENO(order=5),
                                sources=nothing,
                                timestepper=:ForwardEuler)
    set!(model.action, action_values(FT, size(model.action)))
    return model
end

function metal_intrinsic_transport_result(metal_arch)
    FT = Float32
    cpu_model = intrinsic_transport_model(CPU(), FT)
    gpu_model = intrinsic_transport_model(metal_arch, FT)

    compute_tendencies!(cpu_model)
    compute_tendencies!(gpu_model)
    sync_metal!()
    tendency_error = max_abs_difference(interior(cpu_model.tendencies),
                                        interior(gpu_model.tendencies))

    time_step!(cpu_model, FT(0.02))
    time_step!(gpu_model, FT(0.02))
    sync_metal!()

    metrics = Dict(:intrinsic_tendency_error => tendency_error,
                   :intrinsic_action_error => max_abs_difference(interior(cpu_model.action),
                                                                 interior(gpu_model.action)),
                   :intrinsic_backend_error => field_data_backend(physical_field(gpu_model.action, 1, 1)) isa Metal.MtlArray ? 0.0 : 1.0)
    tolerances = Dict(:intrinsic_tendency_error => 1e-4,
                      :intrinsic_action_error => 1e-5,
                      :intrinsic_backend_error => 0.0)
    return ValidationResult(:metal_intrinsic_transport,
                            "Metal and CPU agree for the source-free fused WENO transport tendency and the KA time-step update kernel.",
                            metrics,
                            tolerances)
end

function current_fields(grid, ::Type{FT}) where FT
    Lx = FT(grid.Nx)
    Ly = FT(grid.Ny)
    twoπ = FT(2) * FT(pi)
    u = CenterField(grid)
    v = CenterField(grid)
    Oceananigans.set!(u, (x, y, z) -> FT(0.04) * sin(twoπ * x / Lx) +
                                     FT(0.01) * cos(twoπ * y / Ly))
    Oceananigans.set!(v, (x, y, z) -> FT(0.03) * cos(twoπ * x / Lx) -
                                     FT(0.02) * sin(twoπ * y / Ly))
    return u, v
end

function refraction_model(arch, ::Type{FT}) where FT
    grid = periodic_grid(arch, FT; size=(6, 5, 1))
    cgrid = polar_grid(arch, FT; radial_bins=4, direction_bins=6)
    u, v = current_fields(grid, FT)
    model = SpectralWaveModel(; grid,
                                spectral_grid=cgrid,
                                velocities=(; u, v),
                                horizontal_advection=WENO(order=5),
                                spectral_advection=WENO(order=5),
                                sources=nothing,
                                timestepper=:ForwardEuler)
    set!(model.action, action_values(FT, size(model.action)))
    return model
end

function metal_refraction_result(metal_arch)
    FT = Float32
    cpu_model = refraction_model(CPU(), FT)
    gpu_model = refraction_model(metal_arch, FT)

    compute_tendencies!(cpu_model)
    compute_tendencies!(gpu_model)
    sync_metal!()
    tendency_error = max_abs_difference(interior(cpu_model.tendencies),
                                        interior(gpu_model.tendencies))
    doppler_error = max_abs_difference(cpu_model.coupling.Ux, gpu_model.coupling.Ux) +
                    max_abs_difference(cpu_model.coupling.Uy, gpu_model.coupling.Uy)
    doppler_derivative_error = max_abs_difference(cpu_model.coupling.dUxdkappa,
                                                  gpu_model.coupling.dUxdkappa) +
                               max_abs_difference(cpu_model.coupling.dUydkappa,
                                                  gpu_model.coupling.dUydkappa)
    current_gradient_error = max_abs_difference(cpu_model.coupling.Ux_x,
                                                gpu_model.coupling.Ux_x) +
                             max_abs_difference(cpu_model.coupling.Uy_y,
                                                gpu_model.coupling.Uy_y)

    time_step!(cpu_model, FT(0.01))
    time_step!(gpu_model, FT(0.01))
    sync_metal!()

    backend_ok = field_data_backend(physical_field(gpu_model.action, 1, 1)) isa Metal.MtlArray &&
                 gpu_model.coupling.Ux isa Metal.MtlArray &&
                 gpu_model.coupling.dUxdkappa isa Metal.MtlArray
    metrics = Dict(:refraction_tendency_error => tendency_error,
                   :refraction_action_error => max_abs_difference(interior(cpu_model.action),
                                                                  interior(gpu_model.action)),
                   :doppler_velocity_error => doppler_error,
                   :doppler_derivative_error => doppler_derivative_error,
                   :current_gradient_error => current_gradient_error,
                   :refraction_backend_error => backend_ok ? 0.0 : 1.0)
    tolerances = Dict(:refraction_tendency_error => 2e-4,
                      :refraction_action_error => 2e-5,
                      :doppler_velocity_error => 2e-5,
                      :doppler_derivative_error => 2e-5,
                      :current_gradient_error => 2e-5,
                      :refraction_backend_error => 0.0)
    return ValidationResult(:metal_refraction,
                            "Metal and CPU agree for Doppler velocity setup, current gradients, fused refraction tendency, and time stepping.",
                            metrics,
                            tolerances)
end

function metal_pseudomomentum_result(metal_arch)
    FT = Float32
    cpu_model = intrinsic_transport_model(CPU(), FT)
    gpu_model = intrinsic_transport_model(metal_arch, FT)
    old_cpu_action = similar(cpu_model.action)
    old_gpu_action = similar(gpu_model.action)
    old_values = action_values(FT, size(cpu_model.action))
    new_values = old_values .* FT(1.01)
    depth = one(FT)
    dt = FT(0.05)

    set!(old_cpu_action, old_values)
    set!(old_gpu_action, old_values)
    set!(cpu_model.action, new_values)
    set!(gpu_model.action, new_values)

    cpu_qtransform = QTransform(QKernel(FT), cpu_model.grid)
    gpu_qtransform = QTransform(QKernel(FT), gpu_model.grid)
    cpu_px, cpu_py = pseudomomentum_fields(cpu_model.action, depth, cpu_qtransform)
    gpu_px, gpu_py = pseudomomentum_fields(gpu_model.action, depth, gpu_qtransform)
    cpu_ut, cpu_vt = pseudomomentum_tendency_fields(cpu_model.action, old_cpu_action,
                                                    dt, depth, cpu_qtransform)
    gpu_ut, gpu_vt = pseudomomentum_tendency_fields(gpu_model.action, old_gpu_action,
                                                    dt, depth, gpu_qtransform)
    cwcm_momentum_tendency_fields!(cpu_ut, cpu_vt, cpu_model.action, old_cpu_action,
                                   dt, depth, cpu_qtransform; coefficient=-one(FT))
    cwcm_momentum_tendency_fields!(gpu_ut, gpu_vt, gpu_model.action, old_gpu_action,
                                   dt, depth, gpu_qtransform; coefficient=-one(FT))
    sync_metal!()

    backend_ok = field_data_backend(gpu_px) isa Metal.MtlArray &&
                 field_data_backend(gpu_py) isa Metal.MtlArray &&
                 field_data_backend(gpu_ut) isa Metal.MtlArray &&
                 field_data_backend(gpu_vt) isa Metal.MtlArray
    metrics = Dict(:pseudomomentum_x_error => max_abs_difference(Ripple.field_storage(cpu_px),
                                                                 Ripple.field_storage(gpu_px)),
                   :pseudomomentum_y_error => max_abs_difference(Ripple.field_storage(cpu_py),
                                                                 Ripple.field_storage(gpu_py)),
                   :cwcm_tendency_x_error => max_abs_difference(Ripple.field_storage(cpu_ut),
                                                                Ripple.field_storage(gpu_ut)),
                   :cwcm_tendency_y_error => max_abs_difference(Ripple.field_storage(cpu_vt),
                                                                Ripple.field_storage(gpu_vt)),
                   :pseudomomentum_backend_error => backend_ok ? 0.0 : 1.0)
    tolerances = Dict(:pseudomomentum_x_error => 2e-4,
                      :pseudomomentum_y_error => 2e-4,
                      :cwcm_tendency_x_error => 2e-4,
                      :cwcm_tendency_y_error => 2e-4,
                      :pseudomomentum_backend_error => 0.0)
    return ValidationResult(:metal_pseudomomentum,
                            "Metal and CPU agree for pseudomomentum, pseudomomentum tendency, and CWCM tendency scaling kernels.",
                            metrics,
                            tolerances)
end

function metal_smoke_results()
    load_metal!()
    metal_arch = metal_architecture()
    return (metal_product_field_result(metal_arch),
            metal_intrinsic_transport_result(metal_arch),
            metal_refraction_result(metal_arch),
            metal_pseudomomentum_result(metal_arch))
end

function run_metal_smoke_script(args=ARGS)
    length(args) == 1 || error(usage())
    output_path = only(args)
    write_validation_summary(output_path, metal_smoke_results())
    println(output_path)
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_metal_smoke_script()
end
