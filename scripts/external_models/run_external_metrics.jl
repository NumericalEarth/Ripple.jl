using Ripple

function usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/external_models/run_external_metrics.jl OUTPUT.tsv ENV_VAR

    ENV_VAR may contain a command that writes scalar metrics to stdout.
    Alternatively, set the matching *_FILE variable to a scalar metrics file.
    Or set the matching *_BULK_FILE variable to a columnar bulk-output table.
    Or set the matching *_EXECUTABLE variable to run a generated input deck in
    *_WORKDIR, reducing either the emitted scalar metrics file or bulk table.

    Supported metric formats:
      case.metric=value
      case.metric<TAB>value<TAB>tolerance
      case.metric<TAB>value<TAB>tolerance<TAB>description

    Supported bulk columns include x, y, dx, dy, area, Hs/Hm0/SWH, m0,
    energy_density, mean_direction, peak_direction, mean_period, peak_period,
    mean_frequency, peak_frequency, peak_wavenumber, and peak_phase_speed.
    """
end

function run_external_metrics_script(args=ARGS)
    length(args) == 2 || error(usage())

    output_path, env_var = args
    file_env_var = replace(env_var, "_COMMAND" => "_FILE")
    bulk_env_var = replace(env_var, "_METRICS_COMMAND" => "_BULK_FILE")

    if haskey(ENV, env_var)
        run_external_metrics_command(ENV[env_var], output_path;
                                     description="metrics emitted by command in `$env_var`")
    elseif haskey(ENV, file_env_var)
        metrics = parse_external_metrics(read(ENV[file_env_var], String);
                                         description="metrics read from `$file_env_var`")
        write_external_metrics_summary(output_path, metrics)
    elseif haskey(ENV, bulk_env_var)
        write_external_bulk_metrics_summary(output_path, ENV[bulk_env_var];
                                            case=Symbol(lowercase(replace(bulk_env_var, "_BULK_FILE" => ""))),
                                            description="bulk metrics reduced from `$bulk_env_var`")
	    else
	        model = Symbol(lowercase(replace(env_var, "_METRICS_COMMAND" => "")))
	        executable_env_var = external_model_executable_env_var(model)
	        workdir_env_var = external_model_workdir_env_var(model)
	        profile_env_var = replace(executable_env_var, "_EXECUTABLE" => "_LAUNCH_PROFILE")
	        if haskey(ENV, executable_env_var)
	            workdir = get(ENV, workdir_env_var, mktempdir())
	            profile = Symbol(get(ENV, profile_env_var, "default"))
	            plan = external_model_launch_plan(workdir, model;
	                                              profile,
	                                              executable=ENV[executable_env_var])
	            run_external_model_launch_plan!(plan, output_path)
	        else
	            error("Set `$env_var`, `$file_env_var`, `$bulk_env_var`, or `$executable_env_var` (optionally `$profile_env_var`). " * usage())
	        end
	    end

    println(output_path)
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_external_metrics_script()
end
