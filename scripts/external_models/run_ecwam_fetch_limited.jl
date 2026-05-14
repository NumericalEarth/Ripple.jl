include("run_external_metrics.jl")
length(ARGS) == 1 || error("Usage: julia --startup-file=no --project=. scripts/external_models/run_ecwam_fetch_limited.jl OUTPUT.tsv\nSet ECWAM_METRICS_COMMAND to a command that emits scalar metrics.")
run_external_metrics_script((ARGS[1], "ECWAM_METRICS_COMMAND"))
