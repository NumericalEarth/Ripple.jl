# Each entry maps an example basename (without `.jl`) to its displayed
# title in the Documenter table of contents. Order is the order in which
# pages appear in the sidebar.
const EXAMPLE_TUTORIALS = (
    ("quick_start.jl",                       "Quick Start"),
    ("source_only_fetch_limited_growth.jl",  "Source-Only Fetch-Limited Growth"),
    ("bounded_wave_packet_dispersion.jl",    "Bounded Wave Packet Dispersion"),
    ("vortex_refraction.jl",                 "Wave Refraction Through A Barotropic Vortex"),
)

example_slug(filename) = first(splitext(filename)) * ".md"

function generated_example_pages()
    pages = Pair{String, String}["Overview" => "examples.md"]
    for (filename, title) in EXAMPLE_TUTORIALS
        push!(pages, title => joinpath("generated", "examples", example_slug(filename)))
    end
    return pages
end

# Postamble appended to every literate example so the docs page records
# the Julia version and the top-level packages it ran against. Mirrors
# Oceananigans / Breeze.
const EXAMPLE_POSTAMBLE = """

# ---

# ### Julia version and environment information
#
# This example was executed with the following version of Julia:

using InteractiveUtils: versioninfo
versioninfo()

# These were the top-level packages installed in the environment:

import Pkg
Pkg.status()
"""

function build_literate_examples!(docs_root)
    # `Literate` must be available in the scope that invokes this function
    # (typically `docs/make.jl` adds `using Literate` before `include`ing
    # this file). Keeping the import out here lets the top-level project
    # read `EXAMPLE_TUTORIALS` without depending on Literate.
    examples_src_dir = joinpath(docs_root, "..", "examples")
    output_dir       = joinpath(docs_root, "src", "generated", "examples")
    rm(output_dir; force = true, recursive = true)
    mkpath(output_dir)

    for (filename, _title) in EXAMPLE_TUTORIALS
        script_path = joinpath(examples_src_dir, filename)
        @info "Literate: building $(filename)"
        Literate.markdown(script_path, output_dir;
                          flavor     = Literate.DocumenterFlavor(),
                          preprocess = content -> content * EXAMPLE_POSTAMBLE,
                          execute    = true)
    end

    return output_dir
end

function generate_documentation_sources!(docs_root = @__DIR__)
    generated_dir = joinpath(docs_root, "src", "generated")
    mkpath(generated_dir)
    build_literate_examples!(docs_root)
    return generated_dir
end
