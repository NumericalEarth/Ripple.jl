isdefined(@__MODULE__, :PublicationReadinessStatus) ||
    Base.include(@__MODULE__, joinpath(@__DIR__, "check_publication_readiness.jl"))

function github_publication_usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/validation/publish_to_numericalearth.jl [--dry-run] [--skip-readiness] [--visibility public|private|internal] [--repo OWNER/NAME] [--bundle BUNDLE_PATH] [--workdir WORKDIR]

    Creates or updates the GitHub repository for Ripple.jl. By default this
    publishes the current git worktree to NumericalEarth/Ripple.jl as a public
    repository. Use --bundle when the current checkout has no .git metadata but
    a verified publication bundle is available.

    Non-dry-run execution checks the local LICENSE file and
    publication_decisions.toml owner decisions before creating or pushing the
    repository. Use --skip-readiness only for an intentional pre-policy
    private/internal handoff.

    The script requires authenticated `gh` and `git` for non-dry-run execution.
    """
end

Base.@kwdef struct GitHubPublicationConfig
    repo :: String = "NumericalEarth/Ripple.jl"
    visibility :: Symbol = :public
    bundle :: Union{Nothing, String} = nothing
    workdir :: Union{Nothing, String} = nothing
    dry_run :: Bool = false
    skip_readiness :: Bool = false
end

github_publication_repo_root() = normpath(joinpath(@__DIR__, "..", ".."))

function validate_github_repo_name(repo)
    parts = split(repo, '/')
    length(parts) == 2 || throw(ArgumentError("repo must have OWNER/NAME form, got `$repo`"))
    all(part -> !isempty(strip(part)), parts) ||
        throw(ArgumentError("repo must have non-empty OWNER and NAME, got `$repo`"))
    return repo
end

function validate_publication_visibility(visibility)
    visibility in (:public, :private, :internal) ||
        throw(ArgumentError("visibility must be public, private, or internal, got `$visibility`"))
    return visibility
end

function validate_github_publication_config(config::GitHubPublicationConfig)
    validate_github_repo_name(config.repo)
    validate_publication_visibility(config.visibility)
    if config.skip_readiness && config.visibility === :public
        throw(ArgumentError("--skip-readiness is only allowed with private or internal visibility"))
    end
    return config
end

github_repo_ssh_url(repo) = string("git@github.com:", validate_github_repo_name(repo), ".git")

function require_publication_executable(name)
    path = Sys.which(name)
    path === nothing && error("`$name` executable missing")
    return path
end

function require_github_publication_auth!(gh)
    run(Cmd([gh, "auth", "status"]))
    return gh
end

function publication_command_success(command)
    return try
        success(pipeline(command; stdout=devnull, stderr=devnull))
    catch
        false
    end
end

function command_line(command::Cmd)
    return join(command.exec, ' ')
end

function parse_github_publication_args(args)
    config = GitHubPublicationConfig()
    i = firstindex(args)

    while i <= lastindex(args)
        argument = String(args[i])

        if argument == "--dry-run"
            config = GitHubPublicationConfig(; config.repo,
                                             config.visibility,
                                             config.bundle,
                                             config.workdir,
                                             dry_run=true,
                                             config.skip_readiness)
        elseif argument == "--skip-readiness"
            config = GitHubPublicationConfig(; config.repo,
                                             config.visibility,
                                             config.bundle,
                                             config.workdir,
                                             config.dry_run,
                                             skip_readiness=true)
        elseif startswith(argument, "--visibility=")
            value = Symbol(split(argument, "="; limit=2)[2])
            config = GitHubPublicationConfig(; config.repo,
                                             visibility=validate_publication_visibility(value),
                                             config.bundle,
                                             config.workdir,
                                             config.dry_run,
                                             config.skip_readiness)
        elseif argument == "--visibility"
            i == lastindex(args) && error(github_publication_usage())
            i += 1
            value = Symbol(args[i])
            config = GitHubPublicationConfig(; config.repo,
                                             visibility=validate_publication_visibility(value),
                                             config.bundle,
                                             config.workdir,
                                             config.dry_run,
                                             config.skip_readiness)
        elseif startswith(argument, "--repo=")
            value = validate_github_repo_name(split(argument, "="; limit=2)[2])
            config = GitHubPublicationConfig(; repo=value,
                                             config.visibility,
                                             config.bundle,
                                             config.workdir,
                                             config.dry_run,
                                             config.skip_readiness)
        elseif argument == "--repo"
            i == lastindex(args) && error(github_publication_usage())
            i += 1
            value = validate_github_repo_name(args[i])
            config = GitHubPublicationConfig(; repo=value,
                                             config.visibility,
                                             config.bundle,
                                             config.workdir,
                                             config.dry_run,
                                             config.skip_readiness)
        elseif startswith(argument, "--bundle=")
            value = split(argument, "="; limit=2)[2]
            config = GitHubPublicationConfig(; config.repo,
                                             config.visibility,
                                             bundle=value,
                                             config.workdir,
                                             config.dry_run,
                                             config.skip_readiness)
        elseif argument == "--bundle"
            i == lastindex(args) && error(github_publication_usage())
            i += 1
            config = GitHubPublicationConfig(; config.repo,
                                             config.visibility,
                                             bundle=String(args[i]),
                                             config.workdir,
                                             config.dry_run,
                                             config.skip_readiness)
        elseif startswith(argument, "--workdir=")
            value = split(argument, "="; limit=2)[2]
            config = GitHubPublicationConfig(; config.repo,
                                             config.visibility,
                                             config.bundle,
                                             workdir=value,
                                             config.dry_run,
                                             config.skip_readiness)
        elseif argument == "--workdir"
            i == lastindex(args) && error(github_publication_usage())
            i += 1
            config = GitHubPublicationConfig(; config.repo,
                                             config.visibility,
                                             config.bundle,
                                             workdir=String(args[i]),
                                             config.dry_run,
                                             config.skip_readiness)
        else
            error("unknown argument `$argument`\n" * github_publication_usage())
        end

        i += 1
    end

    return validate_github_publication_config(config)
end

function github_publication_plan(config::GitHubPublicationConfig)
    validate_github_publication_config(config)
    repo = validate_github_repo_name(config.repo)
    visibility_flag = string("--", config.visibility)
    workdir = config.workdir === nothing ? "." : config.workdir
    bundle_workdir = config.workdir === nothing ? "Ripple.jl-publication-worktree" : config.workdir
    push_workdir = config.bundle === nothing ? workdir : bundle_workdir
    remote = github_repo_ssh_url(repo)

    lines = String[]
    auth_line = "gh auth status"

    config.skip_readiness && push!(lines, auth_line)

    if config.bundle !== nothing
        push!(lines, string("git clone ", config.bundle, " ", bundle_workdir))
    end

    config.skip_readiness ||
        push!(lines, string("check publication_decisions.toml, LICENSE, and registry_policy readiness in ", push_workdir))

    config.skip_readiness || push!(lines, auth_line)
    push!(lines, string("gh repo view ", repo, " --json nameWithOwner || gh repo create ", repo, " ", visibility_flag))
    push!(lines, string("git -C ", push_workdir, " remote remove origin 2>/dev/null || true"))
    push!(lines, string("git -C ", push_workdir, " remote add origin ", remote))
    push!(lines, string("git -C ", push_workdir, " push -u origin main"))

    return lines
end

function clone_publication_bundle!(git, bundle, workdir)
    isfile(bundle) || throw(ArgumentError("publication bundle missing: $bundle"))
    ispath(workdir) && throw(ArgumentError("workdir already exists: $workdir"))
    mkpath(dirname(abspath(workdir)))
    run(Cmd([git, "clone", bundle, workdir]))
    return workdir
end

function ensure_git_worktree(git, workdir)
    publication_command_success(Cmd([git, "-C", workdir, "rev-parse", "--is-inside-work-tree"])) ||
        throw(ArgumentError("`$workdir` is not a git worktree; use --bundle with a verified publication bundle"))
    ensure_clean_git_worktree(git, workdir)
    return workdir
end

function ensure_clean_git_worktree(git, workdir)
    status_ok, output = command_status_and_output(Cmd([git, "-C", workdir, "status", "--porcelain"]))
    status_ok ||
        throw(ArgumentError("could not inspect git worktree status for `$workdir`: $(compact_command_output(output))"))

    isempty(strip(output)) ||
        throw(ArgumentError("git worktree has uncommitted or untracked changes: $(compact_command_output(output))"))

    return workdir
end

function ensure_origin_remote!(git, workdir, remote)
    has_origin = publication_command_success(Cmd([git, "-C", workdir, "remote", "get-url", "origin"]))
    if has_origin
        run(Cmd([git, "-C", workdir, "remote", "set-url", "origin", remote]))
    else
        run(Cmd([git, "-C", workdir, "remote", "add", "origin", remote]))
    end
    return remote
end

function remove_origin_remote!(git, workdir)
    has_origin = publication_command_success(Cmd([git, "-C", workdir, "remote", "get-url", "origin"]))
    has_origin && run(Cmd([git, "-C", workdir, "remote", "remove", "origin"]))
    return !publication_command_success(Cmd([git, "-C", workdir, "remote", "get-url", "origin"]))
end

function with_publication_workdir(f, git, config::GitHubPublicationConfig)
    if config.bundle === nothing
        workdir = normpath(config.workdir === nothing ? github_publication_repo_root() : config.workdir)
        ensure_git_worktree(git, workdir)
        return f(workdir)
    end

    bundle = abspath(config.bundle)
    if config.workdir === nothing
        return mktempdir() do dir
            workdir = joinpath(dir, "Ripple.jl")
            clone_publication_bundle!(git, bundle, workdir)
            f(workdir)
        end
    else
        workdir = normpath(config.workdir)
        clone_publication_bundle!(git, bundle, workdir)
        return f(workdir)
    end
end

function publication_push_readiness_statuses(workdir)
    return (license_status(workdir),
            registry_policy_status(workdir))
end

function require_publication_push_readiness!(workdir)
    statuses = publication_push_readiness_statuses(workdir)
    incomplete = [status for status in statuses if status.status !== :available]
    isempty(incomplete) && return statuses

    labels = [string(status.check, "=", status.status, " (", status.evidence, ")")
              for status in incomplete]
    throw(ArgumentError("publication push readiness has incomplete items: $(join(labels, ", "))"))
end

function publish_to_github(config::GitHubPublicationConfig)
    validate_github_publication_config(config)

    if config.dry_run
        for line in github_publication_plan(config)
            println(line)
        end
        return (repo=config.repo, dry_run=true)
    end

    gh = require_publication_executable("gh")
    git = require_publication_executable("git")
    repo = validate_github_repo_name(config.repo)
    visibility = validate_publication_visibility(config.visibility)
    remote = github_repo_ssh_url(repo)

    config.skip_readiness && require_github_publication_auth!(gh)

    return with_publication_workdir(git, config) do workdir
        config.skip_readiness || require_publication_push_readiness!(workdir)

        config.skip_readiness || require_github_publication_auth!(gh)

        repository_exists = publication_command_success(Cmd([gh, "repo", "view", repo, "--json", "nameWithOwner"]))

        if !repository_exists
            run(Cmd([gh, "repo", "create", repo, string("--", visibility)]))
        end

        remove_origin_remote!(git, workdir)
        ensure_origin_remote!(git, workdir, remote)
        run(Cmd([git, "-C", workdir, "push", "-u", "origin", "main"]))

        return (repo=repo, workdir=workdir, created=!repository_exists, pushed=true)
    end
end

function run_github_publication_script(args=ARGS)
    config = parse_github_publication_args(args)
    result = publish_to_github(config)
    config.dry_run || println(string(result.repo, "\t", result.pushed ? "pushed" : "not-pushed"))
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_github_publication_script()
end
