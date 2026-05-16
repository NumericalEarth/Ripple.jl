isdefined(@__MODULE__, :OptionalGateStatus) ||
    Base.include(@__MODULE__, joinpath(@__DIR__, "check_optional_runtime_gates.jl"))

function available_gates_usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/validation/run_available_optional_gates.jl [--dry-run] [--require-all] OUTPUT_DIR

    Checks optional runtime gate readiness, writes `optional_runtime_gates.tsv`,
    and runs every gate reported as available. Missing gates are recorded as
    skipped. With `--require-all`, any skipped gate is treated as a failure
    after the summary is written. Flags may appear before or after positional
    arguments.
    """
end

repo_root() = normpath(joinpath(@__DIR__, "..", ".."))

const OPTIONAL_GATE_SCRIPTS = Dict(
    :oceananigans => ("scripts/oceananigans/run_oceananigans_smoke.jl", ()),
    :cuda => ("scripts/gpu/run_cuda_smoke.jl", ()),
    :metal => ("scripts/gpu/run_metal_smoke.jl", ()),
    :swan => ("scripts/external_models/run_swan_fetch_limited.jl", ()),
    :wam => ("scripts/external_models/run_wam_fetch_limited.jl", ()),
    :ww3 => ("scripts/external_models/run_ww3_fetch_limited.jl", ()),
    :ecwam => ("scripts/external_models/run_ecwam_fetch_limited.jl", ()),
    :picles => ("scripts/external_models/run_picles_vortex_wind.jl", ()),
)

const EXTERNAL_GATE_EXECUTABLES = Dict(
    :swan => ("SWAN", ("swanrun", "swan")),
    :wam => ("WAM", ("wam",)),
    :ww3 => ("WW3", ("ww3_shel", "ww3_multi", "ww3_grid", "ww3_prep")),
    :ecwam => ("ECWAM", ("ecwam",)),
    :picles => ("PICLES", ("picles",)),
)

function julia_gate_command(script, output_path, extra_args=())
    root = repo_root()
    return Cmd([Base.julia_cmd().exec...,
                "--startup-file=no",
                "--project=$root",
                joinpath(root, script),
                output_path,
                String.(extra_args)...])
end

function gate_command(status::OptionalGateStatus, output_path)
    script, extra_args = OPTIONAL_GATE_SCRIPTS[status.gate]
    return julia_gate_command(script, output_path, extra_args)
end

function external_gate_env_overrides(gate)
    haskey(EXTERNAL_GATE_EXECUTABLES, gate) || return Pair{String, String}[]
    prefix, executable_names = EXTERNAL_GATE_EXECUTABLES[gate]
    configured = (string(prefix, "_METRICS_COMMAND"),
                  string(prefix, "_METRICS_FILE"),
                  string(prefix, "_BULK_FILE"),
                  string(prefix, "_EXECUTABLE"))
    any(name -> haskey(ENV, name) && !isempty(ENV[name]), configured) &&
        return Pair{String, String}[]

    for executable_name in executable_names
        path = Sys.which(executable_name)
        path === nothing || return [string(prefix, "_EXECUTABLE") => path]
    end

    return Pair{String, String}[]
end

function validation_summary_passed(path)
    isfile(path) || throw(ArgumentError("optional gate did not write validation summary `$path`"))
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("optional gate validation summary is empty: $path"))
    split(first(lines), '\t') == ["case", "passed", "metric", "value", "tolerance", "description"] ||
        throw(ArgumentError("optional gate validation summary has an unexpected header: $(first(lines))"))

    row_count = 0
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        parts = split(line, '\t')
        length(parts) == 6 || throw(ArgumentError("optional gate validation summary row must have 6 columns: $line"))
        row_count += 1
        lowercase(parts[2]) == "true" ||
            throw(ArgumentError("optional gate validation metric failed: $line"))
    end

    row_count > 0 || throw(ArgumentError("optional gate validation summary has no metric rows: $path"))
    return true
end

function run_gate!(status::OptionalGateStatus, output_path; dry_run=false)
    status.status === :available || return :skipped
    dry_run && return :would_run

    command = gate_command(status, output_path)
    overrides = external_gate_env_overrides(status.gate)
    cd(repo_root()) do
        withenv(overrides...) do
            run(command)
        end
    end
    validation_summary_passed(output_path)
    return :passed
end

function write_optional_gate_run_summary(path, rows)
    open(path, "w") do io
        println(io, "gate\tresult\toutput\tevidence")
        for row in rows
            println(io, join((row.gate, row.result, row.output, row.evidence), '\t'))
        end
    end
    return path
end

function run_available_optional_gates(output_dir;
                                      dry_run=false,
                                      require_all=false,
                                      statuses=optional_gate_statuses())
    mkpath(output_dir)
    readiness_path = joinpath(output_dir, "optional_runtime_gates.tsv")
    summary_path = joinpath(output_dir, "optional_gate_run_summary.tsv")
    write_optional_gate_statuses(readiness_path, statuses)

    rows = NamedTuple[]
    failed = Symbol[]
    failure_evidence = String[]
    for status in statuses
        output_path = joinpath(output_dir, string(status.gate, ".tsv"))
        result = try
            run_gate!(status, output_path; dry_run)
        catch err
            evidence = string(status.evidence, "; error: ", err)
            push!(rows, (gate=status.gate,
                         result=:failed,
                         output=output_path,
                         evidence))
            push!(failed, status.gate)
            push!(failure_evidence, evidence)
            write_optional_gate_run_summary(summary_path, rows)
            continue
        end

        push!(rows, (gate=status.gate,
                     result,
                     output=result in (:passed, :would_run) ? output_path : "",
                     evidence=status.evidence))
    end

    write_optional_gate_run_summary(summary_path, rows)
    isempty(failed) ||
        throw(ArgumentError("optional gates failed: $(join(failed, ", ")); see `$summary_path`; $(join(failure_evidence, " | "))"))

    if require_all
        skipped = [row.gate for row in rows if row.result === :skipped]
        isempty(skipped) ||
            throw(ArgumentError("optional gates unavailable: $(join(skipped, ", ")); see `$summary_path`"))
    end

    return (readiness=readiness_path, summary=summary_path)
end

function parse_available_gates_args(args)
    dry_run = false
    require_all = false
    arguments = String[]

    for argument in args
        if argument == "--dry-run"
            dry_run = true
        elseif argument == "--require-all"
            require_all = true
        elseif startswith(argument, "--")
            error("unknown option `$argument`\n" * available_gates_usage())
        else
            push!(arguments, String(argument))
        end
    end

    length(arguments) == 1 || error(available_gates_usage())
    output_dir = arguments[1]
    return output_dir, dry_run, require_all
end

function run_available_optional_gates_script(args=ARGS)
    output_dir, dry_run, require_all = parse_available_gates_args(args)
    paths = run_available_optional_gates(output_dir; dry_run, require_all)
    println(paths.summary)
    return paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_available_optional_gates_script()
end
