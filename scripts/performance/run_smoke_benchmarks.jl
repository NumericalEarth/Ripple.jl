using Ripple

function usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/performance/run_smoke_benchmarks.jl OUTPUT.tsv

    Writes stdlib-only smoke benchmark timings and allocations for representative
    Ripple.jl kernels. The output is intended for trend tracking, not strict
    pass/fail physics validation.
    """
end

function run_smoke_benchmarks_script(args=ARGS)
    length(args) == 1 || error(usage())

    output_path = only(args)
    metrics = run_performance_smoke()
    write_performance_summary(output_path, metrics)
    println(output_path)
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_smoke_benchmarks_script()
end
