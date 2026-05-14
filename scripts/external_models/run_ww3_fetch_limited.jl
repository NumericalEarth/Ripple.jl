include("run_external_metrics.jl")
length(ARGS) == 1 || error("Usage: julia --startup-file=no --project=. scripts/external_models/run_ww3_fetch_limited.jl OUTPUT.tsv\nSet WW3_METRICS_COMMAND to a command that emits scalar metrics.")
run_external_metrics_script((ARGS[1], "WW3_METRICS_COMMAND"))
