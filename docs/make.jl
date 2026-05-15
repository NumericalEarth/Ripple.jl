using Documenter
using Ripple

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
    ),
    pages = [
        "Home" => "index.md",
        "Model API" => "model_api.md",
        "Finite-Volume Integration" => "finite_volume_integration.md",
        "API Reference" => "api_reference.md",
        "Examples" => generated_example_pages(),
        "Validation" => "validation.md",
        "Publication" => "publication.md",
        "Implementation Status" => "generated/implementation_status.md",
        "Goal Completion Audit" => "generated/goal_completion_audit.md",
        "External Comparison Harness" => "generated/external_comparison_harness.md",
    ],
    checkdocs = :none,
)

deploydocs(;
    repo = "github.com/NumericalEarth/RippleDocumentation.git",
    devbranch = "main",
    push_preview = true,
)
