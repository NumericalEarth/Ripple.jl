function goal_completion_usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/validation/write_goal_completion_checklist.jl OUTPUT.tsv [--default-suite-summary SUMMARY.tsv] [--require-complete]

    Writes a compact checklist for the current Oceananigans-aligned Ripple
    design.
    """
end

struct GoalChecklistItem
    item :: Symbol
    status :: Symbol
    evidence :: String
    action :: String
end

struct GoalChecklistOptions
    output_path :: String
    default_suite_summary :: Union{Nothing, String}
    require_complete :: Bool
end

repo_root() = normpath(joinpath(@__DIR__, "..", ".."))

function file_exists(root, path)
    return isfile(joinpath(root, path))
end

function file_contains(root, path, patterns)
    full_path = joinpath(root, path)
    isfile(full_path) || return false
    text = read(full_path, String)
    return all(pattern -> occursin(pattern, text), patterns)
end

function status_item(item, ok, evidence, action)
    return GoalChecklistItem(item, ok ? :available : :missing, evidence, ok ? "" : action)
end

function default_suite_summary_item(path)
    if path === nothing
        return nothing
    elseif !isfile(path)
        return GoalChecklistItem(:default_suite, :missing,
                                 "default-suite summary missing: $path",
                                 "run test/runtests.jl with RIPPLE_TEST_SUMMARY set")
    end

    try
        lines = filter(!isempty, readlines(path))
        header = split(lines[1], '\t')
        row = split(lines[2], '\t')
        expected = ["name", "passed", "failed", "errored", "broken", "total", "duration"]
        header == expected || error("unexpected header")
        passed = parse(Int, row[2])
        failed = parse(Int, row[3])
        errored = parse(Int, row[4])
        broken = parse(Int, row[5])
        total = parse(Int, row[6])
        ok = failed == 0 && errored == 0 && broken == 0 && passed == total
        return status_item(:default_suite,
                           ok,
                           "default suite passed with $passed checks",
                           "rerun the default suite and fix failing checks")
    catch err
        return GoalChecklistItem(:default_suite, :missing,
                                 "default-suite summary is invalid: $(sprint(showerror, err))",
                                 "rerun the default suite with RIPPLE_TEST_SUMMARY set")
    end
end

function goal_completion_items(root=repo_root(); default_suite_summary=nothing)
    items = GoalChecklistItem[
        status_item(:source_only_model,
                    file_contains(root, "src/Models/spectral_wave_model.jl", ("horizontal_advection=WENO()",)) &&
                    file_contains(root, "src/Models/tendencies.jl", ("source_tendency",)),
                    "SpectralWaveModel exposes horizontal_advection/spectral_advection kwargs and tendencies wire source_tendency.",
                    "restore source-only model semantics"),
        status_item(:removed_private_advection,
                    !file_exists(root, "src/Operators/Operators.jl") &&
                    !file_exists(root, "src/Operators/fluxes.jl") &&
                    !file_exists(root, "src/Operators/hamiltonian_velocities.jl") &&
                    file_contains(root, "test/integration/model_api.jl", ("!isdefined(Ripple, :HamiltonianFiniteVolume)",)),
                    "Private Hamiltonian transport operators have been removed.",
                    "delete remaining private advection code"),
        status_item(:removed_private_simulation_output,
                    !file_exists(root, "src/Models/simulation.jl") &&
                    file_contains(root, "test/integration/model_api.jl", ("!isdefined(Ripple, :Simulation)",)),
                    "Ripple-owned Simulation and writer types have been removed.",
                    "delete remaining private simulation/output code"),
        status_item(:rectilinear_vertical_grid,
                    file_contains(root, "src/Grids.jl", ("zfaces", "vertical_size")) &&
                    file_contains(root, "src/Coupling/q_transform.jl", ("QTransform(kernel::QKernel", "grid::RectilinearGrid")),
                    "The physical RectilinearGrid carries vertical faces used by QTransform.",
                    "source QTransform vertical geometry from RectilinearGrid"),
        status_item(:oceananigans_hard_dependency,
                    file_contains(root, "Project.toml", ("[deps]", "Oceananigans")) &&
                    !file_contains(root, "Project.toml", ("RippleOceananigansExt",)),
                    "Oceananigans is a hard dependency and no longer a Ripple extension trigger.",
                    "move Oceananigans out of weakdeps/extensions"),
        status_item(:examples_literate_and_curated,
                    file_contains(root, "docs/generate.jl", ("product_field_basics.jl", "exact_finite_volume_source_rates.jl")) &&
                    file_contains(root, "test/examples_smoke/run_examples.jl", ("plot_paths", "animation_paths")),
                    "The curated examples are literate, documented, and visualized.",
                    "restore literate example/docs smoke coverage"),
        status_item(:optional_gates_current,
                    file_contains(root, "scripts/validation/check_optional_runtime_gates.jl", ("oceananigans", "cuda", "swan")) &&
                    !isfile(joinpath(root, "scripts", "output", "run_optional_dataset_backend_smoke.jl")),
                    "Optional gates target Oceananigans, CUDA, and external models, not Ripple private output backends.",
                    "update optional gate scripts"),
    ]

    summary_item = default_suite_summary_item(default_suite_summary)
    summary_item === nothing || push!(items, summary_item)

    return items
end

function write_goal_completion_checklist(path, items)
    open(path, "w") do io
        println(io, "item\tstatus\tevidence\taction")
        for item in items
            println(io, item.item, '\t', item.status, '\t', item.evidence, '\t', item.action)
        end
    end
    return path
end

function parse_goal_completion_args(args)
    isempty(args) && error(goal_completion_usage())
    output_path = first(args)
    default_suite_summary = nothing
    require_complete = false

    index = 2
    while index <= length(args)
        argument = args[index]
        if argument == "--default-suite-summary"
            index == length(args) && error("missing value for --default-suite-summary\n" * goal_completion_usage())
            default_suite_summary = args[index + 1]
            index += 2
        elseif argument == "--require-complete"
            require_complete = true
            index += 1
        else
            error("unknown option `$argument`\n" * goal_completion_usage())
        end
    end

    return GoalChecklistOptions(output_path, default_suite_summary, require_complete)
end

function run_goal_completion_checklist_script(args=ARGS)
    options = parse_goal_completion_args(args)
    items = goal_completion_items(; default_suite_summary=options.default_suite_summary)
    write_goal_completion_checklist(options.output_path, items)
    println(options.output_path)

    if options.require_complete
        incomplete = [item for item in items if item.status !== :available]
        isempty(incomplete) ||
            error("goal-completion checklist has incomplete items: $(join(String.(getproperty.(incomplete, :item)), ", "))")
    end

    return options.output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_goal_completion_checklist_script()
end
