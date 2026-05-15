using TOML

function publication_readiness_usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/validation/check_publication_readiness.jl OUTPUT.tsv

    Writes a publication-readiness checklist for creating and publishing
    NumericalEarth/Ripple.jl. This script does not create repositories or make
    policy choices such as license selection.
    """
end

struct PublicationReadinessStatus
    check :: Symbol
    status :: Symbol
    evidence :: String
    action :: String
end

publication_repo_root() = normpath(joinpath(@__DIR__, "..", ".."))

function all_paths_exist(root, paths)
    missing = [path for path in paths if !isfile(joinpath(root, path))]
    isempty(missing) && return (:available, "all expected files are present")
    return (:missing, string("missing files: ", join(missing, ", ")))
end

function missing_content_patterns(root, content_patterns)
    missing = String[]

    for pair in content_patterns
        path = first(pair)
        patterns = last(pair)
        full_path = joinpath(root, path)
        if !isfile(full_path)
            push!(missing, string(path, ":", join(patterns, ",")))
            continue
        end

        text = read(full_path, String)
        for pattern in patterns
            occursin(pattern, text) || push!(missing, string(path, ":", pattern))
        end
    end

    return missing
end

function publication_content_status(root)
    content_patterns = (
        "README.md" => ("actions/workflows/ci.yml",
                        "actions/workflows/documentation.yml",
                        "NumericalEarth.github.io/Ripple.jl/stable",
                        "CONTRIBUTING.md",
                        "scripts/validation/test_publication_bundle.jl",
                        "scripts/validation/patch_oceananigans_manifest_triggers.jl",
                        "scripts/validation/publish_to_numericalearth.jl"),
        "publication_decisions.toml" => ("license",
                                         "registry_policy"),
        "CONTRIBUTING.md" => ("finite-volume",
                              "horizontal_advection=nothing",
                              "run_available_optional_gates.jl",
                              "RIPPLE_TEST_SUMMARY"),
        ".github/PULL_REQUEST_TEMPLATE.md" => ("Validation",
                                               "finite-volume",
                                               "`nothing` semantics"),
        ".github/ISSUE_TEMPLATE/bug_report.yml" => ("Bug Report",
                                                    "Reproducer",
                                                    "Environment"),
        ".github/ISSUE_TEMPLATE/validation_gap.yml" => ("Validation Gap",
                                                        "Reference Behavior",
                                                        "Gate Type"),
        ".github/ISSUE_TEMPLATE/feature_request.yml" => ("Feature Request",
                                                         "Design Constraints",
                                                         "finite-volume",
                                                         "`nothing` semantics"),
        "docs/make.jl" => ("generate_documentation_sources!",
                           "makedocs",
                           "deploydocs",
                           "github.com/NumericalEarth/Ripple.jl.git"),
        "docs/src/publication.md" => ("gh repo create NumericalEarth/Ripple.jl",
                                      "gh repo view NumericalEarth/Ripple.jl --json nameWithOwner",
                                      "DOCUMENTER_KEY",
                                      "scripts/validation/create_publication_bundle.jl",
                                      "scripts/validation/test_publication_bundle.jl",
                                      "scripts/validation/patch_oceananigans_manifest_triggers.jl",
                                      "scripts/validation/publish_to_numericalearth.jl",
                                      "publication_decisions.toml",
                                      "clean git worktree",
                                      "push readiness",
                                      "--skip-readiness",
                                      "private/internal handoff",
                                      "RIPPLE_PUBLICATION_GIT_DIR",
                                      "GPUArraysCore",
                                      "KernelAbstractions",
                                      "julia --startup-file=no --project=. scripts/validation/run_available_optional_gates.jl --require-all optional_gate_outputs"),
        "docs/src/validation.md" => ("julia --startup-file=no --project=. test/runtests.jl",
                                     "julia --startup-file=no --project=. scripts/validation/check_optional_runtime_gates.jl",
                                     "julia --startup-file=no --project=. scripts/validation/check_publication_readiness.jl",
                                     "julia --startup-file=no --project=. scripts/validation/write_goal_completion_checklist.jl",
                                     "julia --startup-file=no --project=. scripts/validation/run_available_optional_gates.jl",
                                     "patch_oceananigans_manifest_triggers.jl",
                                     "GPUArraysCore",
                                     "KernelAbstractions",
                                     "RIPPLE_PUBLICATION_GIT_DIR"),
        "docs/src/model_api.md" => ("Semantics Contract",
                                    "horizontal_advection=nothing",
                                    "sources=nothing",
                                    "coupling=nothing",
                                    "RectilinearGrid",
                                    "QTransform"),
        "docs/src/examples.md" => ("test/examples_smoke/run_examples.jl",
                                   "generated/examples/product_field_basics.md",
                                   "generated/examples/frequency_direction_source_package.md",
                                   "generated/examples/exact_finite_volume_source_rates.md"),
        "test/examples_smoke/run_examples.jl" => ("Example semantic manifest",
                                                  "RIPPLE_EXAMPLE_MODE",
                                                  "animation_paths"),
        "docs/external_comparison_harness.md" => ("julia --startup-file=no --project=. scripts/validation/check_optional_runtime_gates.jl",
                                                  "julia --startup-file=no --project=. scripts/validation/run_available_optional_gates.jl",
                                                  "julia --startup-file=no --project=. scripts/postprocess/external_bulk_to_metrics.jl"),
        "scripts/validation/publish_to_numericalearth.jl" => ("require_publication_push_readiness!",
                                                              "require_github_publication_auth!",
                                                              "ensure_clean_git_worktree",
                                                              "validate_github_publication_config",
                                                              "private or internal visibility",
                                                              "publication_decisions.toml",
                                                              "--skip-readiness"),
        ".github/workflows/ci.yml" => ("test/runtests.jl",
                                       "package-test",
                                       "RIPPLE_TEST_SUMMARY",
                                       "default_suite.tsv",
                                       "check_optional_runtime_gates.jl",
                                       "run_available_optional_gates.jl",
                                       "create_publication_bundle.jl",
                                       "test_publication_bundle.jl",
                                       "ripple-publication.bundle",
                                       "publication_bundle_default_suite.tsv",
                                       "RIPPLE_PUBLICATION_BUNDLE",
                                       "RIPPLE_PUBLICATION_BUNDLE_TEST_SUMMARY",
                                       "check_publication_readiness.jl",
                                       "write_goal_completion_checklist.jl",
                                       "--default-suite-summary",
                                       "publication_readiness.tsv",
                                       "goal_completion_checklist.tsv",
                                       "upload-artifact"),
        ".github/workflows/documentation.yml" => ("docs/make.jl",
                                                  "--startup-file=no",
                                                  "Pkg.develop",
                                                  "Pkg.instantiate",
                                                  "DOCUMENTER_KEY"),
        ".github/workflows/optional-gates.yml" => ("workflow_dispatch",
                                                   "require_all:",
                                                   "require_complete_audit:",
                                                   "GPUArraysCore",
                                                   "KernelAbstractions",
                                                   "patch_oceananigans_manifest_triggers.jl",
                                                   "default_suite.tsv",
                                                   "create_publication_bundle.jl",
                                                   "test_publication_bundle.jl",
                                                   "publication_bundle_default_suite.tsv",
                                                   "check_publication_readiness.jl",
                                                   "write_goal_completion_checklist.jl",
                                                   "--require-complete",
                                                   "run_available_optional_gates.jl",
                                                   "--require-all",
                                                   "upload-artifact"),
    )
    missing = missing_content_patterns(root, content_patterns)

    isempty(missing) &&
        return PublicationReadinessStatus(:docs_github_actions_wiring,
                                          :available,
                                          "README, Documenter, publication runbook, and GitHub Actions wiring contain expected NumericalEarth/Ripple.jl commands",
                                          "run documentation and CI workflows after pushing the repository")

    return PublicationReadinessStatus(:docs_github_actions_wiring,
                                      :missing,
                                      string("missing content patterns: ", join(missing, ", ")),
                                      "restore README, docs, or workflow wiring before publication")
end

function publication_bundle_default_suite_pass_count(summary_path)
    isfile(summary_path) ||
        throw(ArgumentError("publication bundle default-suite summary missing: $(summary_path)"))

    lines = filter(!isempty, readlines(summary_path))
    length(lines) >= 2 ||
        throw(ArgumentError("publication bundle default-suite summary is empty: $(summary_path)"))

    header = split(lines[1], '\t')
    values = split(lines[2], '\t')
    expected = ["name", "passed", "failed", "errored", "broken", "total", "duration"]
    header == expected ||
        throw(ArgumentError("unexpected publication bundle default-suite summary header: $(join(header, ","))"))
    length(values) == length(expected) ||
        throw(ArgumentError("unexpected publication bundle default-suite summary row: $(lines[2])"))

    passed = parse(Int, values[2])
    failed = parse(Int, values[3])
    errored = parse(Int, values[4])
    broken = parse(Int, values[5])
    total = parse(Int, values[6])

    (failed == 0 && errored == 0 && broken == 0 && passed == total) ||
        throw(ArgumentError("publication bundle default suite did not pass cleanly: $(lines[2])"))

    return passed
end

function compact_command_output(text; limit=280)
    compacted = join(split(chomp(text)), " ")
    length(compacted) <= limit && return compacted
    return string(first(compacted, limit), "...")
end

function command_status_and_output(command)
    mktemp() do output_path, output_io
        close(output_io)

        succeeded = try
            success(pipeline(command; stdout=output_path, stderr=output_path))
        catch err
            open(output_path, "a") do io
                println(io, sprint(showerror, err))
            end
            false
        end

        return succeeded, read(output_path, String)
    end
end

function documentation_runtime_status(root; julia_cmd=Base.julia_cmd())
    docs_project = joinpath(root, "docs")
    isfile(joinpath(docs_project, "Project.toml")) ||
        return PublicationReadinessStatus(:documentation_runtime,
                                          :missing,
                                          "docs/Project.toml missing",
                                          "restore the docs project before building documentation")

    load_documenter = `$(julia_cmd) --startup-file=no --project=$(docs_project) -e "using Documenter"`
    documenter_available, output = command_status_and_output(load_documenter)

    documenter_available &&
        return PublicationReadinessStatus(:documentation_runtime,
                                          :available,
                                          "Documenter loads in the docs project",
                                          "run `julia --startup-file=no --project=docs docs/make.jl`")

    return PublicationReadinessStatus(:documentation_runtime,
                                      :missing,
                                      string("Documenter does not load in the docs project: ",
                                             compact_command_output(output)),
                                      "run `julia --startup-file=no --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'`")
end

function license_status(root)
    decision_table = publication_decision_table(root)

    decision_table.error === nothing ||
        return PublicationReadinessStatus(:license,
                                          :missing,
                                          string("publication_decisions.toml could not be parsed: ",
                                                 compact_command_output(sprint(showerror, decision_table.error))),
                                          "fix publication_decisions.toml, choose a project license, and add a LICENSE file before public registration")

    license = publication_decision_value(decision_table, "license")
    candidates = ("LICENSE", "LICENSE.md", "LICENSE.txt")
    existing = [candidate for candidate in candidates if isfile(joinpath(root, candidate))]

    if license == "undecided"
        file_evidence = isempty(existing) ? "no LICENSE file found" : string("license file: ", first(existing))
        return PublicationReadinessStatus(:license,
                                          :manual,
                                          string(file_evidence, "; publication_decisions.toml license is undecided"),
                                          "choose a project license, record it in publication_decisions.toml, and add or review the LICENSE file before public registration")
    end

    if isempty(existing)
        return PublicationReadinessStatus(:license,
                                          :manual,
                                          string("no LICENSE file found; publication_decisions.toml license=", license),
                                          "add a LICENSE file matching publication_decisions.toml before public registration")
    end

    return PublicationReadinessStatus(:license,
                                      :available,
                                      string("license file: ", first(existing), "; publication_decisions.toml license=", license),
                                      "review license text before tagging a release")
end

function publication_decisions_path(root)
    return joinpath(root, "publication_decisions.toml")
end

function publication_decision_table(root)
    path = publication_decisions_path(root)
    isfile(path) ||
        return (path=path,
                decisions=Dict{String, Any}(),
                error=nothing)

    try
        return (path=path,
                decisions=TOML.parsefile(path),
                error=nothing)
    catch err
        return (path=path,
                decisions=Dict{String, Any}(),
                error=err)
    end
end

function publication_decision_value(decision_table, key)
    raw_value = get(decision_table.decisions, key, "undecided")
    return raw_value isa AbstractString ? strip(raw_value) : string(raw_value)
end

function registry_policy_status(root)
    decision_table = publication_decision_table(root)

    isfile(decision_table.path) ||
        return PublicationReadinessStatus(:registry_policy,
                                          :manual,
                                          "publication_decisions.toml is missing",
                                          "add publication_decisions.toml and choose General registry or organization-local distribution")

    decision_table.error === nothing ||
        return PublicationReadinessStatus(:registry_policy,
                                          :missing,
                                          string("publication_decisions.toml could not be parsed: ",
                                                 compact_command_output(sprint(showerror, decision_table.error))),
                                          "fix publication_decisions.toml before publication")

    policy = publication_decision_value(decision_table, "registry_policy")

    if policy == "general"
        return PublicationReadinessStatus(:registry_policy,
                                          :available,
                                          "publication_decisions.toml registry_policy=general",
                                          "after repository creation, follow Julia General registry registration checks")
    elseif policy == "organization-local"
        return PublicationReadinessStatus(:registry_policy,
                                          :available,
                                          "publication_decisions.toml registry_policy=organization-local",
                                          "publish the repository without General registry registration")
    elseif policy == "undecided"
        return PublicationReadinessStatus(:registry_policy,
                                          :manual,
                                          "publication_decisions.toml registry_policy is undecided",
                                          "decide General registry versus organization-local distribution")
    end

    return PublicationReadinessStatus(:registry_policy,
                                      :missing,
                                      string("publication_decisions.toml has invalid registry_policy=", policy),
                                      "set registry_policy to general, organization-local, or undecided")
end

function git_metadata_status(root; git_dir=get(ENV, "RIPPLE_PUBLICATION_GIT_DIR", ""))
    isdir(joinpath(root, ".git")) &&
        return PublicationReadinessStatus(:git_metadata,
                                          :available,
                                          ".git directory present",
                                          "commit and push to NumericalEarth/Ripple.jl")

    if !isempty(git_dir)
        normalized_git_dir = normpath(git_dir)

        isdir(normalized_git_dir) ||
            return PublicationReadinessStatus(:git_metadata,
                                              :missing,
                                              string("RIPPLE_PUBLICATION_GIT_DIR is not a directory: ",
                                                     normalized_git_dir),
                                              "create the separate publication git directory or unset RIPPLE_PUBLICATION_GIT_DIR")

        git = Sys.which("git")
        git === nothing &&
            return PublicationReadinessStatus(:git_metadata,
                                              :missing,
                                              "git executable missing; cannot verify RIPPLE_PUBLICATION_GIT_DIR",
                                              "install git or initialize the repository in an environment with git available")

        head_ok, head_output = command_status_and_output(
            `$git --git-dir=$(normalized_git_dir) --work-tree=$(root) rev-parse --verify HEAD`)
        head_ok ||
            return PublicationReadinessStatus(:git_metadata,
                                              :missing,
                                              string("could not verify separate publication git HEAD: ",
                                                     compact_command_output(head_output)),
                                              "commit the separate publication git directory before pushing")

        worktree_clean, worktree_output = command_status_and_output(
            `$git --git-dir=$(normalized_git_dir) --work-tree=$(root) diff --quiet --exit-code`)
        worktree_clean ||
            return PublicationReadinessStatus(:git_metadata,
                                              :missing,
                                              string("separate publication git dir has uncommitted tracked changes: ",
                                                     compact_command_output(worktree_output)),
                                              "commit or discard tracked worktree changes before publication")

        index_clean, index_output = command_status_and_output(
            `$git --git-dir=$(normalized_git_dir) --work-tree=$(root) diff --cached --quiet --exit-code`)
        index_clean ||
            return PublicationReadinessStatus(:git_metadata,
                                              :missing,
                                              string("separate publication git dir has staged changes: ",
                                                     compact_command_output(index_output)),
                                              "commit or unstage changes before publication")

        return PublicationReadinessStatus(:git_metadata,
                                          :available,
                                          string("separate publication git dir: ",
                                                 normalized_git_dir,
                                                 "; HEAD ",
                                                 strip(head_output)),
                                          "push this committed tree to NumericalEarth/Ripple.jl after repository access is available")
    end

    return PublicationReadinessStatus(:git_metadata,
                                      :missing,
                                      ".git directory missing",
                                      "initialize a local git repository or set RIPPLE_PUBLICATION_GIT_DIR to a committed separate git directory")
end

function verified_git_bundle(git, bundle_path)
    return mktempdir() do dir
        repo = joinpath(dir, "bundle-verify.git")
        initialized = try
            success(pipeline(`$git init --bare $repo`; stdout=devnull, stderr=devnull))
        catch
            false
        end
        initialized || return false

        try
            withenv("GIT_DIR" => repo) do
                success(pipeline(`$git bundle verify $bundle_path`; stdout=devnull, stderr=devnull))
            end
        catch
            false
        end
    end
end

function git_bundle_heads(git, bundle_path)
    return try
        strip(read(`$git bundle list-heads $bundle_path`, String))
    catch
        ""
    end
end

const LOCAL_ONLY_BUNDLE_BASENAMES = (".DS_Store",)
const LOCAL_ONLY_BUNDLE_PATHS = ("Manifest.toml",
                                 "default_suite.tsv",
                                 "publication_bundle_default_suite.tsv",
                                 "optional_runtime_gates.tsv",
                                 "publication_readiness.tsv",
                                 "goal_completion_checklist.tsv")
const LOCAL_ONLY_BUNDLE_PREFIXES = ("docs/build/",
                                    "docs/src/generated/",
                                    "optional_gate_outputs/")
const LOCAL_ONLY_BUNDLE_SUFFIXES = (".jl.cov", ".mem")

function is_julia_coverage_artifact(path)
    return endswith(path, ".jl.cov") ||
           (occursin(".jl.", path) && endswith(path, ".cov"))
end

function publication_bundle_files(git, bundle_path)
    return mktempdir() do dir
        worktree = joinpath(dir, "bundle-worktree")
        cloned = try
            success(pipeline(`$git clone $bundle_path $worktree`; stdout=devnull, stderr=devnull))
        catch
            false
        end
        cloned || return nothing

        text = try
            read(`$git -C $worktree ls-files`, String)
        catch
            return nothing
        end

        return filter(!isempty, split(text, '\n'))
    end
end

function local_only_publication_bundle_files(files)
    return [path for path in files
            if basename(path) in LOCAL_ONLY_BUNDLE_BASENAMES ||
               path in LOCAL_ONLY_BUNDLE_PATHS ||
               any(prefix -> startswith(path, prefix), LOCAL_ONLY_BUNDLE_PREFIXES) ||
               any(suffix -> endswith(path, suffix), LOCAL_ONLY_BUNDLE_SUFFIXES) ||
               is_julia_coverage_artifact(path)]
end

function publication_bundle_status(; bundle_path=get(ENV, "RIPPLE_PUBLICATION_BUNDLE", ""))
    isempty(bundle_path) &&
        return PublicationReadinessStatus(:publication_bundle,
                                          :manual,
                                          "RIPPLE_PUBLICATION_BUNDLE is not set",
                                          "set RIPPLE_PUBLICATION_BUNDLE to a verified git bundle if .git metadata is unavailable")

    isfile(bundle_path) ||
        return PublicationReadinessStatus(:publication_bundle,
                                          :missing,
                                          string("bundle file missing: ", bundle_path),
                                          "create a git bundle from the current worktree")

    git = Sys.which("git")
    git === nothing &&
        return PublicationReadinessStatus(:publication_bundle,
                                          :missing,
                                          "git executable missing; cannot verify bundle",
                                          "install git and run `git bundle verify`")

    verified = verified_git_bundle(git, bundle_path)

    verified &&
        return PublicationReadinessStatus(:publication_bundle,
                                          :available,
                                          string("verified git bundle: ", bundle_path, "; ", git_bundle_heads(git, bundle_path)),
                                          "push the bundle's main branch after NumericalEarth repository access is available")

    return PublicationReadinessStatus(:publication_bundle,
                                      :missing,
                                      string("git bundle verification failed: ", bundle_path),
                                      "recreate the bundle from the current worktree")
end

function publication_bundle_cleanliness_status(; bundle_path=get(ENV, "RIPPLE_PUBLICATION_BUNDLE", ""))
    isempty(bundle_path) &&
        return PublicationReadinessStatus(:publication_bundle_cleanliness,
                                          :manual,
                                          "RIPPLE_PUBLICATION_BUNDLE is not set",
                                          "set RIPPLE_PUBLICATION_BUNDLE and check that generated/local artifacts are absent")

    isfile(bundle_path) ||
        return PublicationReadinessStatus(:publication_bundle_cleanliness,
                                          :missing,
                                          string("bundle file missing: ", bundle_path),
                                          "recreate the publication bundle")

    git = Sys.which("git")
    git === nothing &&
        return PublicationReadinessStatus(:publication_bundle_cleanliness,
                                          :missing,
                                          "git executable missing; cannot inspect bundle contents",
                                          "install git and inspect the publication bundle")

    files = publication_bundle_files(git, bundle_path)
    files === nothing &&
        return PublicationReadinessStatus(:publication_bundle_cleanliness,
                                          :missing,
                                          string("could not inspect bundle contents: ", bundle_path),
                                          "recreate the publication bundle and run `git clone BUNDLE WORKTREE`")

    local_only = local_only_publication_bundle_files(files)
    isempty(local_only) &&
        return PublicationReadinessStatus(:publication_bundle_cleanliness,
                                          :available,
                                          "publication bundle contains no generated or local-only artifacts",
                                          "keep .gitignore and publication-bundle checks in sync")

    return PublicationReadinessStatus(:publication_bundle_cleanliness,
                                      :missing,
                                      string("publication bundle contains local-only artifacts: ", join(local_only, ", ")),
                                      "remove generated/local artifacts from the bundle before pushing")
end

function publication_bundle_default_suite_status(; summary_path=get(ENV, "RIPPLE_PUBLICATION_BUNDLE_TEST_SUMMARY", ""))
    isempty(summary_path) &&
        return PublicationReadinessStatus(:publication_bundle_default_suite,
                                          :manual,
                                          "RIPPLE_PUBLICATION_BUNDLE_TEST_SUMMARY is not set",
                                          "run `julia --startup-file=no --project=. scripts/validation/test_publication_bundle.jl BUNDLE OUTPUT.tsv`")

    try
        passed = publication_bundle_default_suite_pass_count(summary_path)
        return PublicationReadinessStatus(:publication_bundle_default_suite,
                                          :available,
                                          string("publication bundle clone default suite passed with ", passed, " tests"),
                                          "rerun the bundle clone test after changing publication contents")
    catch err
        return PublicationReadinessStatus(:publication_bundle_default_suite,
                                          :missing,
                                          string("publication bundle default-suite summary invalid: ",
                                                 compact_command_output(sprint(showerror, err))),
                                          "rerun the publication bundle default suite from a clean clone")
    end
end

function github_cli_status()
    gh = Sys.which("gh")
    gh === nothing &&
        return PublicationReadinessStatus(:github_cli,
                                          :missing,
                                          "gh executable missing",
                                          "install GitHub CLI or create NumericalEarth/Ripple.jl through the GitHub web UI")

    auth_ok = try
        success(pipeline(`$gh auth status`; stdout=devnull, stderr=devnull))
    catch err
        false
    end

    auth_ok &&
        return PublicationReadinessStatus(:github_cli,
                                          :available,
                                          string("authenticated GitHub CLI at ", gh),
                                          "create or push NumericalEarth/Ripple.jl")

    return PublicationReadinessStatus(:github_cli,
                                      :missing,
                                      string("GitHub CLI exists but auth status failed: ", gh),
                                      "run `gh auth login` with access to the NumericalEarth organization")
end

function numericalearth_repo_status(gh_status=github_cli_status();
                                    repo="NumericalEarth/Ripple.jl",
                                    repo_visible=nothing)
    gh_status.status === :available ||
        return PublicationReadinessStatus(:numericalearth_repo,
                                          :manual,
                                          string("not checked because GitHub CLI is not authenticated: ",
                                                 gh_status.evidence),
                                          "authenticate GitHub CLI with NumericalEarth access, then create or view NumericalEarth/Ripple.jl")

    visible = repo_visible === nothing ? begin
        gh = Sys.which("gh")
        if gh === nothing
            false
        else
            try
                success(pipeline(`$gh repo view $repo --json nameWithOwner`;
                                 stdout=devnull,
                                 stderr=devnull))
            catch err
                false
            end
        end
    end : Bool(repo_visible)

    visible &&
        return PublicationReadinessStatus(:numericalearth_repo,
                                          :available,
                                          string(repo, " is visible to the authenticated GitHub CLI"),
                                          "push the verified publication bundle or committed main branch")

    return PublicationReadinessStatus(:numericalearth_repo,
                                      :missing,
                                      string("authenticated GitHub CLI could not view ", repo),
                                      string("create ", repo, " or grant the authenticated account access"))
end

function publication_readiness_statuses(; root=publication_repo_root())
    expected_files = (
        "Project.toml",
        "publication_decisions.toml",
        "README.md",
        "CONTRIBUTING.md",
        "src/Ripple.jl",
        "test/runtests.jl",
        "docs/Project.toml",
        "docs/Manifest.toml",
        "docs/generate.jl",
        "docs/make.jl",
        "docs/src/index.md",
        "docs/src/model_api.md",
        "docs/src/finite_volume_integration.md",
        "docs/src/api_reference.md",
        "docs/src/examples.md",
        "docs/src/validation.md",
        "docs/src/publication.md",
        "docs/external_comparison_harness.md",
        "scripts/example_visuals.jl",
        "test/examples_smoke/run_examples.jl",
        "examples/product_field_basics.jl",
        "examples/source_only_fetch_limited_growth.jl",
        "examples/bounded_wave_packet_dispersion.jl",
        "examples/hasselmann_inertial_oscillation.jl",
        "examples/cwcm_q_transform_sheared_current.jl",
        "examples/frequency_direction_source_package.jl",
        "examples/exact_finite_volume_source_rates.jl",
        "scripts/validation/create_publication_bundle.jl",
        "scripts/validation/patch_oceananigans_manifest_triggers.jl",
        "scripts/validation/test_publication_bundle.jl",
        "scripts/validation/publish_to_numericalearth.jl",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/validation_gap.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/documentation.yml",
        ".github/workflows/optional-gates.yml",
    )

    file_status, file_evidence = all_paths_exist(root, expected_files)
    gh_status = github_cli_status()

    return (
        PublicationReadinessStatus(:expected_files,
                                   file_status,
                                   file_evidence,
                                   "restore missing publication files before pushing"),
        publication_content_status(root),
        documentation_runtime_status(root),
        license_status(root),
        git_metadata_status(root),
        publication_bundle_status(),
        publication_bundle_cleanliness_status(),
        publication_bundle_default_suite_status(),
        gh_status,
        numericalearth_repo_status(gh_status),
        registry_policy_status(root),
        PublicationReadinessStatus(:external_completion_gates,
                                   :manual,
                                   "requires provisioned optional runtime environment",
                                   "run the Optional Runtime Gates workflow with require_all enabled"),
    )
end

function write_publication_readiness_statuses(path, statuses=publication_readiness_statuses())
    open(path, "w") do io
        println(io, "check\tstatus\tevidence\taction")
        for status in statuses
            println(io, join((status.check, status.status, status.evidence, status.action), '\t'))
        end
    end

    return path
end

function run_publication_readiness_script(args=ARGS)
    length(args) == 1 || error(publication_readiness_usage())
    output_path = only(args)
    write_publication_readiness_statuses(output_path)
    println(output_path)
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_publication_readiness_script()
end
