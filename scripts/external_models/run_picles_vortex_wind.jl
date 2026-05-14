include("run_external_metrics.jl")
length(ARGS) == 1 || error("Usage: julia --startup-file=no --project=. scripts/external_models/run_picles_vortex_wind.jl OUTPUT.tsv\nSet PICLES_METRICS_COMMAND to a command that emits scalar metrics.")
run_external_metrics_script((ARGS[1], "PICLES_METRICS_COMMAND"))
