using Ripple

function usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/postprocess/external_bulk_to_metrics.jl BULK_OUTPUT.txt OUTPUT.tsv [case]

    Reduces a columnar external-model bulk-output table to scalar metrics in
    Ripple.jl's validation summary format. The optional case argument defaults
    to `external_bulk`.

    Supported columns include x, y, dx, dy, area, Hs/Hm0/SWH, m0,
    energy_density, mean_direction, peak_direction, mean_period, peak_period,
    mean_frequency, peak_frequency, peak_wavenumber, and peak_phase_speed.
    Columns may be comma-separated or whitespace-separated.
    """
end

function external_bulk_to_metrics_script(args=ARGS)
    (2 <= length(args) <= 3) || error(usage())

    bulk_table_path, output_path = args[1], args[2]
    case = length(args) == 3 ? Symbol(args[3]) : :external_bulk
    write_external_bulk_metrics_summary(output_path, bulk_table_path; case)
    println(output_path)
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    external_bulk_to_metrics_script()
end
