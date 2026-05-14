using Ripple

function usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/validation/run_validation_suite.jl OUTPUT.tsv [CASE...]
      julia --startup-file=no --project=. scripts/validation/run_validation_suite.jl --list

    Runs Ripple's built-in validation suite and writes the validation-summary TSV
    format used by external-comparison and smoke-test scripts. When one or more
    CASE names are supplied, only those validation cases are run.
    """
end

function validation_case_table()
    return Dict(case.name => case for case in default_validation_cases())
end

function validation_case_name(name)
    text = strip(String(name))
    startswith(text, ":") && (text = text[nextind(text, firstindex(text)):end])
    isempty(text) && throw(ArgumentError("validation case name must not be empty"))
    return Symbol(text)
end

function selected_validation_cases(names)
    all_cases = collect(default_validation_cases())
    isempty(names) && return all_cases

    cases = validation_case_table()
    selected = ValidationCase[]
    for raw_name in names
        name = validation_case_name(raw_name)
        haskey(cases, name) || throw(ArgumentError("unknown validation case `$name`; available cases: $(join(sort!(String.(keys(cases))), ", "))"))
        push!(selected, cases[name])
    end

    return selected
end

function print_validation_cases()
    for case in sort!(collect(default_validation_cases()); by=case -> String(case.name))
        println(case.name, '\t', case.description)
    end
end

function print_validation_result_summary(results)
    for result in results
        status = validation_passed(result) ? "passed" : "failed"
        println(result.name, '\t', status, '\t', length(result.metrics), " metrics")
    end
end

function run_validation_suite_script(args=ARGS)
    length(args) == 1 && args[1] == "--list" && (print_validation_cases(); return nothing)
    length(args) >= 1 || error(usage())

    output_path = first(args)
    cases = selected_validation_cases(args[2:end])
    results = run_validation(cases)
    write_validation_summary(output_path, results)
    print_validation_result_summary(results)
    println(output_path)

    failed = filter(result -> !validation_passed(result), results)
    isempty(failed) ||
        error("validation failed for cases: $(join(String.(getproperty.(failed, :name)), ", "))")

    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_validation_suite_script()
end
