using Ripple

function usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/external_models/write_external_input_deck.jl OUTPUT_DIR MODEL [CASE]

    MODEL may be swan, wam, ww3, ecwam, or picles.
    CASE defaults to fetch_limited for SWAN/WAM/WW3/ecWAM and stationary_vortex for PiCLES.
    """
end

function write_external_input_deck_script(args=ARGS)
    (2 <= length(args) <= 3) || error(usage())

    output_dir = args[1]
    model = Symbol(lowercase(args[2]))
    case = length(args) == 3 ? Symbol(lowercase(args[3])) : nothing
    manifest_path = write_external_model_input_deck(output_dir, model; case)
    println(manifest_path)
    return manifest_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_external_input_deck_script()
end
