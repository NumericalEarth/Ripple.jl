function usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/validation/check_optional_runtime_gates.jl OUTPUT.tsv

    Checks whether optional runtime gates have the packages, executables, or
    environment variables needed to run their full smoke tests.
    """
end

struct OptionalGateStatus
    gate :: Symbol
    status :: Symbol
    evidence :: String
    command :: String
end

function package_path(name)
    path = Base.find_package(String(name))
    return path === nothing ? nothing : path
end

function package_evidence(name)
    path = package_path(name)
    path === nothing && return (:missing, string(name, " package missing"))
    return (:available, string(name, " package at ", path))
end

function compact_optional_gate_error(err; limit=240)
    text = join(split(sprint(showerror, err)), " ")
    length(text) <= limit && return text
    return string(first(text, limit), "...")
end

function optional_package_load_hint(name, err)
    text = sprint(showerror, err)

    if name === :Oceananigans && occursin("GPUArraysCore", text)
        return "; known Oceananigans extension-trigger manifest issue: recreate the optional runtime environment or provision a manifest where GPUArraysCore and KernelAbstractions are recorded for Oceananigans"
    end

    return ""
end

function package_load_evidence(name)
    path = package_path(name)
    path === nothing && return (:missing, string(name, " package missing"))

    try
        Core.eval(@__MODULE__, Meta.parse("using $(String(name))"))
        isdefined(Base, :retry_load_extensions) && Base.retry_load_extensions()
    catch err
        return (:missing,
                string(name,
                       " package at ",
                       path,
                       " failed to load: ",
                       compact_optional_gate_error(err),
                       optional_package_load_hint(name, err)))
    end

    return (:available, string(name, " package loaded from ", path))
end

function cuda_gate_evidence()
    status, evidence = package_load_evidence(:CUDA)
    status === :available || return status, evidence
    cuda = getfield(@__MODULE__, :CUDA)

    functional = try
        Base.invokelatest(getproperty(cuda, :functional))
    catch err
        return (:missing,
                string(evidence,
                       "; CUDA.functional() failed: ",
                       compact_optional_gate_error(err)))
    end

    functional ||
        return (:missing,
                string(evidence,
                       "; CUDA.functional() is false, so no usable CUDA device/runtime was detected"))

    return (:available, string(evidence, "; CUDA.functional() is true"))
end

function executable_evidence(names)
    for name in names
        path = Sys.which(String(name))
        path === nothing || return (:available, string(name, " executable at ", path))
    end
    return (:missing, string("missing executables: ", join(String.(names), ", ")))
end

external_gate_env_variables(prefix) =
    (string(prefix, "_METRICS_COMMAND"),
     string(prefix, "_METRICS_FILE"),
     string(prefix, "_BULK_FILE"),
     string(prefix, "_EXECUTABLE"))

function external_gate_env_configuration_evidence(prefix)
    command_var, metrics_file_var, bulk_file_var, executable_var = external_gate_env_variables(prefix)

    if haskey(ENV, command_var)
        value = ENV[command_var]
        isempty(strip(value)) && return (:missing, string(command_var, " is set but empty"), true)
        return (:available, string("set environment variable: ", command_var), true)
    end

    for file_var in (metrics_file_var, bulk_file_var)
        if haskey(ENV, file_var)
            value = ENV[file_var]
            isempty(strip(value)) && return (:missing, string(file_var, " is set but empty"), true)
            isfile(value) && return (:available, string(file_var, " file at ", value), true)
            return (:missing, string(file_var, " points to a missing file: ", value), true)
        end
    end

    if haskey(ENV, executable_var)
        value = ENV[executable_var]
        isempty(strip(value)) && return (:missing, string(executable_var, " is set but empty"), true)
        path = Sys.which(value)
        path === nothing &&
            return (:missing, string(executable_var, " is not executable or not on PATH: ", value), true)
        return (:available, string(executable_var, " executable at ", path), true)
    end

    return (:missing, string("no configured environment variables: ",
                            join(external_gate_env_variables(prefix), ", ")), false)
end

function external_gate_readiness(prefix, executable_names)
    env_status, env_evidence_text, configured = external_gate_env_configuration_evidence(prefix)
    configured && return (env_status, env_evidence_text)

    executable_status, executable_evidence_text = executable_evidence(executable_names)
    status = any_available_status((env_status, executable_status))
    return (status, string(env_evidence_text, "; ", executable_evidence_text))
end

any_available_status(statuses) =
    any(status == :available for status in statuses) ? :available : :missing

function optional_gate_statuses()
    ocean_status, ocean_evidence = package_load_evidence(:Oceananigans)
    cuda_status, cuda_evidence = cuda_gate_evidence()
    swan_status, swan_evidence = external_gate_readiness("SWAN", (:swanrun, :swan))
    wam_status, wam_evidence = external_gate_readiness("WAM", (:wam,))
    ww3_status, ww3_evidence = external_gate_readiness("WW3", (:ww3_shel, :ww3_multi, :ww3_grid, :ww3_prep))
    ecwam_status, ecwam_evidence = external_gate_readiness("ECWAM", (:ecwam,))
    picles_status, picles_evidence = external_gate_readiness("PICLES", (:picles,))

    return (
        OptionalGateStatus(:oceananigans,
                           ocean_status,
                           ocean_evidence,
                           "julia --startup-file=no --project=. scripts/oceananigans/run_oceananigans_smoke.jl OUTPUT.tsv"),
        OptionalGateStatus(:cuda,
                           cuda_status,
                           cuda_evidence,
                           "julia --startup-file=no --project=. scripts/gpu/run_cuda_smoke.jl OUTPUT.tsv"),
        OptionalGateStatus(:swan,
                           swan_status,
                           swan_evidence,
                           "julia --startup-file=no --project=. scripts/external_models/run_swan_fetch_limited.jl OUTPUT.tsv"),
        OptionalGateStatus(:wam,
                           wam_status,
                           wam_evidence,
                           "julia --startup-file=no --project=. scripts/external_models/run_wam_fetch_limited.jl OUTPUT.tsv"),
        OptionalGateStatus(:ww3,
                           ww3_status,
                           ww3_evidence,
                           "julia --startup-file=no --project=. scripts/external_models/run_ww3_fetch_limited.jl OUTPUT.tsv"),
        OptionalGateStatus(:ecwam,
                           ecwam_status,
                           ecwam_evidence,
                           "julia --startup-file=no --project=. scripts/external_models/run_ecwam_fetch_limited.jl OUTPUT.tsv"),
        OptionalGateStatus(:picles,
                           picles_status,
                           picles_evidence,
                           "julia --startup-file=no --project=. scripts/external_models/run_picles_vortex_wind.jl OUTPUT.tsv"),
    )
end

function write_optional_gate_statuses(path, statuses=optional_gate_statuses())
    open(path, "w") do io
        println(io, "gate\tstatus\tevidence\tcommand")
        for status in statuses
            println(io, join((status.gate, status.status, status.evidence, status.command), '\t'))
        end
    end
    return path
end

function run_optional_gate_status_script(args=ARGS)
    length(args) == 1 || error(usage())
    output_path = only(args)
    write_optional_gate_statuses(output_path)
    println(output_path)
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_optional_gate_status_script()
end
