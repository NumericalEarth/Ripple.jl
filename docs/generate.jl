# Each entry maps an example basename (without `.jl`) to its displayed
# title in the Documenter table of contents. Order is the order in which
# pages appear in the sidebar.
const EXAMPLE_TUTORIALS = (
    ("quick_start.jl",                       "Quick Start"),
    ("source_only_fetch_limited_growth.jl",  "Source-Only Fetch-Limited Growth"),
    ("bounded_wave_packet_dispersion.jl",    "Bounded Wave Packet Dispersion"),
    ("spectral_refraction_by_shear.jl",      "Spectral Refraction by a Sheared Current"),
    ("vortex_refraction.jl",                 "Wave Refraction Through A Barotropic Vortex"),
    ("monobanded_linear_shear_refraction.jl", "Monobanded Linear-Shear Refraction"),
    ("coupled_wind_drift_instability.jl",    "Coupled Wind-Drift Instability"),
    ("translating_hurricane_swell.jl",       "Swell Generation by a Translating Idealized Hurricane"),
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

# Per-example environment overrides used only by the docs build. Each map
# describes the env vars to set before `Literate.markdown(...; execute=true)`
# runs the example, scoped to the `try`/`finally` so the smoke harness and
# user shell are untouched.
#
# These knobs trade fidelity for build time: full-resolution runs of the
# wind-drift instability sit at ~30 min of CPU on a laptop; the docs target
# is "fast enough to iterate on, big enough to show the physics".
const EXAMPLE_DOCS_ENV_OVERRIDES = Dict{String, Dict{String, String}}(
    # Wind-drift instability: ~30 min at prod → under a minute at docs scale.
    # `Ny=96, Nz=64` is 2× larger than `quick`, 16× smaller than prod, and
    # large enough that the unstable mode is visible in the animation;
    # 100 spinup + 250 continuation iterations × 3 cases × 32 wave substeps
    # is enough to see the growth rate stabilize without burning a CI hour.
    # We deliberately don't set `RIPPLE_EXAMPLE_QUICK=true`: that would
    # override these per-knob choices with the example's built-in quick
    # defaults (10+10 iterations, too few to see growth).
    "coupled_wind_drift_instability.jl" => Dict(
        "RIPPLE_EXAMPLE_NY"                      => "96",
        "RIPPLE_EXAMPLE_NZ"                      => "64",
        "RIPPLE_EXAMPLE_SPINUP_ITERATIONS"       => "100",
        "RIPPLE_EXAMPLE_CONTINUATION_ITERATIONS" => "250",
        "RIPPLE_EXAMPLE_FRAME_STRIDE"            => "10",
    ),
    # Monobanded linear-shear refraction: quick mode trims the grid and the
    # frame count without changing the physics being shown.
    "monobanded_linear_shear_refraction.jl" => Dict(
        "RIPPLE_EXAMPLE_QUICK" => "true",
    ),
)

function with_example_env_overrides(f, filename)
    overrides = get(EXAMPLE_DOCS_ENV_OVERRIDES, filename, nothing)
    overrides === nothing && return f()

    prior = Dict{String, Union{String, Nothing}}()
    for (k, v) in overrides
        prior[k] = get(ENV, k, nothing)
        ENV[k] = v
    end
    try
        f()
    finally
        for (k, v) in prior
            v === nothing ? delete!(ENV, k) : (ENV[k] = v)
        end
    end
end

function build_literate_examples!(docs_root)
    # `Literate` must be available in the scope that invokes this function
    # (typically `docs/make.jl` adds `using Literate` before `include`ing
    # this file). Keeping the import out here lets the top-level project
    # read `EXAMPLE_TUTORIALS` without depending on Literate.
    examples_src_dir = joinpath(docs_root, "..", "examples")
    output_dir       = joinpath(docs_root, "src", "generated", "examples")
    rm(output_dir; force = true, recursive = true)
    mkpath(output_dir)

    # Force animation generation for the docs build so static `![](...mp4)`
    # references in the example markdown resolve to a file. Smoke tests
    # invoke the examples directly without Documenter and can leave this
    # off via RIPPLE_EXAMPLE_ANIMATE=false.
    prior_animate = get(ENV, "RIPPLE_EXAMPLE_ANIMATE", nothing)
    ENV["RIPPLE_EXAMPLE_ANIMATE"] = "true"

    try
        for (filename, _title) in EXAMPLE_TUTORIALS
            script_path = joinpath(examples_src_dir, filename)
            @info "Literate: building $(filename)"
            with_example_env_overrides(filename) do
                Literate.markdown(script_path, output_dir;
                                  flavor     = Literate.DocumenterFlavor(),
                                  preprocess = content -> content * EXAMPLE_POSTAMBLE,
                                  execute    = true)
            end
        end
    finally
        prior_animate === nothing ? delete!(ENV, "RIPPLE_EXAMPLE_ANIMATE") :
                                    (ENV["RIPPLE_EXAMPLE_ANIMATE"] = prior_animate)
    end

    return output_dir
end

function generate_documentation_sources!(docs_root = @__DIR__; examples = true)
    generated_dir = joinpath(docs_root, "src", "generated")
    mkpath(generated_dir)
    examples && build_literate_examples!(docs_root)
    return generated_dir
end
