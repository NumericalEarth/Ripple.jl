using Documenter
using DocumenterCitations
using Literate
using CairoMakie
using Ripple

CairoMakie.activate!(type = "png")

const DOCS_ROOT = @__DIR__
const REPO_ROOT = normpath(joinpath(DOCS_ROOT, ".."))
const RIPPLE_REMOTE = Documenter.Remotes.GitHub("NumericalEarth", "Ripple.jl")
include(joinpath(DOCS_ROOT, "generate.jl"))

const BUILD_LITERATE_EXAMPLES =
    lowercase(get(ENV, "RIPPLE_DOCS_BUILD_EXAMPLES", "true")) in ("1", "true", "yes")

generate_documentation_sources!(DOCS_ROOT; examples = BUILD_LITERATE_EXAMPLES)

const DOCS_REMOTES = BUILD_LITERATE_EXAMPLES ? Dict(REPO_ROOT => (RIPPLE_REMOTE, "main")) : nothing
const EDIT_LINK = BUILD_LITERATE_EXAMPLES ? "main" : nothing
const REPO_LINK = BUILD_LITERATE_EXAMPLES ? "https://github.com/NumericalEarth/Ripple.jl" : nothing

pages = Pair{String, Any}[
    "Home" => "index.md",
    "Notation" => "notation.md",
    "Theory and Numerics" => "theory.md",
    "Model API" => "model_api.md",
    "Monobanded Model" => "monobanded_model.md",
    "Finite-Volume Integration" => "finite_volume_integration.md",
    "API Reference" => "api_reference.md",
]

BUILD_LITERATE_EXAMPLES && push!(pages, "Examples" => generated_example_pages())

push!(pages, "References" => "references.md")

bib = CitationBibliography(joinpath(DOCS_ROOT, "src", "refs.bib"), style=:authoryear)

makedocs(;
    modules = [Ripple],
    sitename = "Ripple.jl",
    authors = "Ripple.jl contributors",
    remotes = DOCS_REMOTES,
    format = Documenter.HTML(;
        canonical = "https://NumericalEarth.github.io/RippleDocumentation/stable/",
        edit_link = EDIT_LINK,
        repolink = REPO_LINK,
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = String["assets/citations.css"],
        # Literate-generated example pages inline base64 figures (and
        # produce multi-megabyte HTML for the vortex animation), so bump
        # the size threshold well above Documenter's default 200 KiB cap.
        size_threshold_warn  = 2_000_000,
        size_threshold       = 5_000_000,
    ),
    pages = pages,
    pagesonly = !BUILD_LITERATE_EXAMPLES,
    checkdocs = :none,
    plugins = [bib],
)

deploydocs(;
    repo = "github.com/NumericalEarth/Ripple.jl.git",
    deploy_repo = "github.com/NumericalEarth/RippleDocumentation.git",
    devbranch = "main",
    push_preview = true,
)
