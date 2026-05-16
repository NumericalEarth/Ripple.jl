using Documenter
using Literate
using CairoMakie
using Ripple

CairoMakie.activate!(type = "png")

const DOCS_ROOT = @__DIR__
const REPO_ROOT = normpath(joinpath(DOCS_ROOT, ".."))
const RIPPLE_REMOTE = Documenter.Remotes.GitHub("NumericalEarth", "Ripple.jl")
include(joinpath(DOCS_ROOT, "generate.jl"))
generate_documentation_sources!(DOCS_ROOT)

makedocs(;
    modules = [Ripple],
    sitename = "Ripple.jl",
    authors = "Ripple.jl contributors",
    remotes = Dict(REPO_ROOT => (RIPPLE_REMOTE, "main")),
    format = Documenter.HTML(;
        canonical = "https://NumericalEarth.github.io/RippleDocumentation/stable/",
        edit_link = "main",
        prettyurls = get(ENV, "CI", "false") == "true",
        # Literate-generated example pages inline base64 figures (and
        # produce multi-megabyte HTML for the vortex animation), so bump
        # the size threshold well above Documenter's default 200 KiB cap.
        size_threshold_warn  = 2_000_000,
        size_threshold       = 5_000_000,
    ),
    pages = [
        "Home" => "index.md",
        "Model API" => "model_api.md",
        "Finite-Volume Integration" => "finite_volume_integration.md",
        "API Reference" => "api_reference.md",
        "Examples" => generated_example_pages(),
    ],
    checkdocs = :none,
)

deploydocs(;
    repo = "github.com/NumericalEarth/Ripple.jl.git",
    deploy_repo = "github.com/NumericalEarth/RippleDocumentation.git",
    devbranch = "main",
    push_preview = true,
)
