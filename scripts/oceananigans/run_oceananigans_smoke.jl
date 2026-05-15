using Ripple

function usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/oceananigans/run_oceananigans_smoke.jl OUTPUT.tsv

    Runs an opt-in Oceananigans smoke test for Ripple's hard-dependency
    Oceananigans integration. Oceananigans.jl must be available on the Julia
    load path.
    """
end

function load_oceananigans!()
    try
        @eval using Oceananigans
        isdefined(Base, :retry_load_extensions) && Base.retry_load_extensions()
    catch err
        error("Oceananigans.jl must be available because Ripple depends on it. Original error: $err")
    end

    return Oceananigans
end

field_storage_latest(field) =
    Base.invokelatest(Ripple.field_storage, field)

function ocean_field(Ocean, grid)
    field_type = Ocean.Field{Ocean.Center, Ocean.Center, Ocean.Center}
    return Base.invokelatest(field_type, grid)
end

function oceananigans_grid(Ocean)
    architecture = Base.invokelatest(Ocean.CPU)
    return Base.invokelatest(Ocean.RectilinearGrid, architecture;
                             size=(3, 2, 2),
                             x=(0, 3),
                             y=(-1, 1),
                             z=(-1, 0),
                             topology=(Ocean.Periodic, Ocean.Bounded, Ocean.Bounded))
end

function fill_ocean_current_fields!(u, v)
    u_data = field_storage_latest(u)
    v_data = field_storage_latest(v)

    for k in axes(u_data, 3), j in axes(u_data, 2), i in axes(u_data, 1)
        u_data[i, j, k] = 0.1 + 0.01i - 0.02j + 0.03k
        v_data[i, j, k] = -0.05 + 0.02i + 0.01j - 0.015k
    end

    return u, v
end

function ocean_column_momentum_budget(u, v, px, py, grid, qtransform)
    u_data = field_storage_latest(u)
    v_data = field_storage_latest(v)
    px_data = field_storage_latest(px)
    py_data = field_storage_latest(py)
    dz = abs.(zspacings(grid))
    dx = xspacings(grid)
    dy = yspacings(grid)
    x_budget = 0.0
    y_budget = 0.0

    for k in axes(u_data, 3), j in axes(u_data, 2), i in axes(u_data, 1)
        cell_volume = dx[i] * dy[j] * dz[k]
        x_budget += (u_data[i, j, k] + px_data[i, j, k]) * cell_volume
        y_budget += (v_data[i, j, k] + py_data[i, j, k]) * cell_volume
    end

    return x_budget, y_budget
end

function max_abs_difference(a, b)
    return maximum(abs.(a .- b))
end

function oceananigans_smoke_result()
    Ocean = load_oceananigans!()
    ocean_grid = oceananigans_grid(Ocean)
    cgrid = PolarWaveVectorGrid(Float64;
                                κ=[0.45, 0.9],
                                φ=range(0, 2pi; length=9)[1:8])
    model = Base.invokelatest(SpectralWaveModel;
                              grid=ocean_grid,
                              spectral_grid=cgrid,
                              horizontal_advection=nothing,
                              timestepper=:ForwardEuler)
    initial_action = (x, y, kx, ky) -> 0.4 + 0.03x - 0.01y + 0.02kx - 0.015ky
    set!(model, N=initial_action)
    reference_model = SpectralWaveModel(model.grid, cgrid;
                      horizontal_advection=nothing,
                      timestepper=:ForwardEuler)
    set!(reference_model, N=initial_action)

    qtransform = QTransform(QKernel(Float64), model.grid)
    px = ocean_field(Ocean, ocean_grid)
    py = ocean_field(Ocean, ocean_grid)
    Base.invokelatest(compute_pseudomomentum_cell_averages!, px, py, model.action, 1.0, qtransform)
    expected_px, expected_py = compute_pseudomomentum_cell_averages(model.action, 1.0, qtransform)

    old_action = copy(model.action)
    time_step!(model, 0.01)
    time_step!(reference_model, 0.01)
    adapted_diagnostic_error = maximum((
        max_abs_difference(m0(model.action), m0(reference_model.action)),
        max_abs_difference(significant_wave_height(model.action), significant_wave_height(reference_model.action)),
        max_abs_difference(root_mean_square_wavenumber(model.action), root_mean_square_wavenumber(reference_model.action)),
        max_abs_difference(deep_water_energy_density(model.action), deep_water_energy_density(reference_model.action)),
        max_abs_difference(mean_deep_water_group_speed(model.action), mean_deep_water_group_speed(reference_model.action)),
        max_abs_difference(mean_direction(model.action), mean_direction(reference_model.action)),
        max_abs_difference(peak_direction(model.action), peak_direction(reference_model.action)),
        max_abs_difference(peak_wavenumber(model.action), peak_wavenumber(reference_model.action)),
        max_abs_difference(deep_water_peak_phase_speed(model.action), deep_water_peak_phase_speed(reference_model.action)),
    ))
    ptx = ocean_field(Ocean, ocean_grid)
    pty = ocean_field(Ocean, ocean_grid)
    Base.invokelatest(compute_pseudomomentum_tendency_cell_averages!, ptx, pty,
                      model.action, old_action, 0.01, 1.0, qtransform)
    expected_ptx, expected_pty = pseudomomentum_tendency_fields(model.action, old_action, 0.01, 1.0, qtransform)
    ut = ocean_field(Ocean, ocean_grid)
    vt = ocean_field(Ocean, ocean_grid)
    Base.invokelatest(cwcm_momentum_tendency_fields!, ut, vt,
                      model.action, old_action, 0.01, 1.0, qtransform)

    new_px, new_py = compute_pseudomomentum_cell_averages(model.action, 1.0, qtransform)
    u_old = ocean_field(Ocean, ocean_grid)
    v_old = ocean_field(Ocean, ocean_grid)
    u_new = ocean_field(Ocean, ocean_grid)
    v_new = ocean_field(Ocean, ocean_grid)
    fill_ocean_current_fields!(u_old, v_old)
    field_storage_latest(u_new) .= field_storage_latest(u_old)
    field_storage_latest(v_new) .= field_storage_latest(v_old)
    field_storage_latest(u_new) .+= 0.01 .* field_storage_latest(ut)
    field_storage_latest(v_new) .+= 0.01 .* field_storage_latest(vt)
    old_budget = ocean_column_momentum_budget(u_old, v_old, expected_px, expected_py, model.grid, qtransform)
    new_budget = ocean_column_momentum_budget(u_new, v_new, new_px, new_py, model.grid, qtransform)

    current_u = ocean_field(Ocean, ocean_grid)
    current_v = ocean_field(Ocean, ocean_grid)
    fill_ocean_current_fields!(current_u, current_v)
    native_current = PrescribedLagrangianMeanCurrent(u=current_u, v=current_v, depth=1.0)
    native_coupling = CWCMPrescribedCurrentCoupling(native_current, qtransform, cgrid.κ)
    Nx, Ny = horizontal_size(model.grid)
    expected_Ux = zeros(Float64, Nx, Ny, length(cgrid.κ))
    expected_Uy = zeros(Float64, Nx, Ny, length(cgrid.κ))
    compute_doppler_velocity!(expected_Ux, expected_Uy,
                              field_storage_latest(current_u),
                              field_storage_latest(current_v),
                              1.0, cgrid.κ, qtransform)
    native_current_model = Base.invokelatest(SpectralWaveModel;
                                             grid=ocean_grid,
                                             spectral_grid=cgrid,
                                             coupling=native_coupling,
                                             horizontal_advection=nothing,
                                             timestepper=:ForwardEuler)
    set!(native_current_model, N=(x, y, kx, ky) -> 0.3 + 0.02x + 0.01y + 0.01hypot(kx, ky))
    time_step!(native_current_model, 0.005)

    metrics = Dict(:grid_size_error => maximum(abs.(collect(size(model.grid)) .- [3, 2, 2])),
                   :grid_topology_error => Ocean.Grids.topology(model.grid) == (Ocean.Periodic, Ocean.Bounded, Ocean.Bounded) ? 0.0 : 1.0,
                   :field_storage_size_error => maximum(abs.(collect(size(field_storage_latest(px))) .- [3, 2, 2])),
                   :px_error => maximum(abs.(field_storage_latest(px) .- expected_px)),
                   :py_error => maximum(abs.(field_storage_latest(py) .- expected_py)),
                   :adapted_action_error => max_abs_difference(Ripple.interior(model.action),
                                                               Ripple.interior(reference_model.action)),
                   :adapted_total_action_error => abs(total_action(model.action) - total_action(reference_model.action)),
                   :adapted_diagnostic_error => adapted_diagnostic_error,
                   :ptx_error => maximum(abs.(field_storage_latest(ptx) .- field_storage_latest(expected_ptx))),
                   :pty_error => maximum(abs.(field_storage_latest(pty) .- field_storage_latest(expected_pty))),
                   :ut_error => maximum(abs.(field_storage_latest(ut) .+ field_storage_latest(expected_ptx))),
                   :vt_error => maximum(abs.(field_storage_latest(vt) .+ field_storage_latest(expected_pty))),
                   :native_field_update_error => max(maximum(abs.(field_storage_latest(u_new) .-
                                                                   (field_storage_latest(u_old) .- 0.01 .* field_storage_latest(expected_ptx)))),
                                                     maximum(abs.(field_storage_latest(v_new) .-
                                                                   (field_storage_latest(v_old) .- 0.01 .* field_storage_latest(expected_pty))))),
                   :native_momentum_budget_error => maximum(abs(new_budget[q] - old_budget[q]) for q in 1:2),
                   :native_prescribed_current_ux_error => maximum(abs.(native_coupling.Ux .- expected_Ux)),
                   :native_prescribed_current_uy_error => maximum(abs.(native_coupling.Uy .- expected_Uy)),
                   :native_prescribed_current_step_error => all(isfinite, Ripple.interior(native_current_model.action)) ?
                                                            max(-minimum(Ripple.interior(native_current_model.action)), 0.0) : 1.0)
    tolerances = Dict(:grid_size_error => 0.0,
                      :grid_topology_error => 0.0,
                      :field_storage_size_error => 0.0,
                      :px_error => 1e-12,
                      :py_error => 1e-12,
                      :adapted_action_error => 1e-12,
                      :adapted_total_action_error => 1e-12,
                      :adapted_diagnostic_error => 1e-12,
                      :ptx_error => 1e-12,
                      :pty_error => 1e-12,
                      :ut_error => 1e-12,
                      :vt_error => 1e-12,
                      :native_field_update_error => 1e-12,
                      :native_momentum_budget_error => 1e-12,
                      :native_prescribed_current_ux_error => 1e-12,
                      :native_prescribed_current_uy_error => 1e-12,
                      :native_prescribed_current_step_error => 0.0)
    return ValidationResult(:oceananigans_smoke,
                            "Oceananigans grid adaptation, native prescribed-current input, native field output, adapted-grid diagnostics, and native-field CWCM column momentum budgeting match Ripple arrays.",
                            metrics,
                            tolerances)
end

function run_oceananigans_smoke_script(args=ARGS)
    length(args) == 1 || error(usage())
    output_path = only(args)
    write_validation_summary(output_path, (oceananigans_smoke_result(),))
    println(output_path)
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_oceananigans_smoke_script()
end
