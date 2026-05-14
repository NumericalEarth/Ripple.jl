function publication_bundle_usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/validation/create_publication_bundle.jl BUNDLE_PATH [GIT_DIR]

    Creates a git bundle for publishing Ripple.jl without requiring a `.git`
    directory in the worktree. The script uses a separate bare git directory,
    commits the current worktree on `main`, writes BUNDLE_PATH, verifies the
    bundle, and prints the bundle path.
    """
end

publication_bundle_repo_root() = normpath(joinpath(@__DIR__, "..", ".."))

function publication_git()
    git = Sys.which("git")
    git === nothing && error("git executable missing")
    return git
end

function git_cmd(git, args...)
    return Cmd(String[git, string.(args)...])
end

function bare_git_cmd(git, git_dir, args...)
    return git_cmd(git, string("--git-dir=", git_dir), args...)
end

function worktree_git_cmd(git, git_dir, root, args...)
    return git_cmd(git, string("--git-dir=", git_dir), string("--work-tree=", root), args...)
end

function git_success(cmd)
    return try
        success(pipeline(cmd; stdout=devnull, stderr=devnull))
    catch
        false
    end
end

function ensure_publication_git_dir!(git, git_dir)
    if !isdir(git_dir)
        mkpath(dirname(git_dir))
        run(git_cmd(git, "init", "--bare", git_dir))
    end

    run(bare_git_cmd(git, git_dir, "symbolic-ref", "HEAD", "refs/heads/main"))
    return git_dir
end

function has_publication_head(git, git_dir, root)
    return git_success(worktree_git_cmd(git, git_dir, root, "rev-parse", "--verify", "HEAD"))
end

function has_staged_publication_changes(git, git_dir, root)
    return !git_success(worktree_git_cmd(git, git_dir, root, "diff", "--cached", "--quiet", "--exit-code"))
end

function publication_commit_environment(f)
    return withenv("GIT_AUTHOR_NAME" => get(ENV, "GIT_AUTHOR_NAME", "Ripple.jl publication script"),
                   "GIT_AUTHOR_EMAIL" => get(ENV, "GIT_AUTHOR_EMAIL", "ripple@example.invalid"),
                   "GIT_COMMITTER_NAME" => get(ENV, "GIT_COMMITTER_NAME", "Ripple.jl publication script"),
                   "GIT_COMMITTER_EMAIL" => get(ENV, "GIT_COMMITTER_EMAIL", "ripple@example.invalid")) do
        f()
    end
end

function commit_publication_tree!(git, git_dir, root;
                                  initial_message="Initial Ripple.jl implementation",
                                  update_message="Update Ripple.jl publication bundle")
    run(worktree_git_cmd(git, git_dir, root, "add", "--all"))

    has_head = has_publication_head(git, git_dir, root)
    staged_changes = has_staged_publication_changes(git, git_dir, root)

    if !has_head
        publication_commit_environment() do
            run(worktree_git_cmd(git, git_dir, root, "commit", "-m", initial_message))
        end
    elseif staged_changes
        publication_commit_environment() do
            run(worktree_git_cmd(git, git_dir, root, "commit", "-m", update_message))
        end
    end

    return strip(read(worktree_git_cmd(git, git_dir, root, "rev-parse", "HEAD"), String))
end

function create_publication_bundle(bundle_path;
                                   root=publication_bundle_repo_root(),
                                   git_dir=string(bundle_path, ".gitdir"))
    git = publication_git()
    root = normpath(root)
    bundle_path = abspath(bundle_path)
    git_dir = abspath(git_dir)

    isdir(root) || error("root directory missing: $(root)")
    mkpath(dirname(bundle_path))
    ensure_publication_git_dir!(git, git_dir)
    commit = commit_publication_tree!(git, git_dir, root)
    run(bare_git_cmd(git, git_dir, "bundle", "create", bundle_path, "main"))
    run(bare_git_cmd(git, git_dir, "bundle", "verify", bundle_path))
    heads = strip(read(git_cmd(git, "bundle", "list-heads", bundle_path), String))
    return (bundle=bundle_path, git_dir=git_dir, commit=commit, heads=heads)
end

function parse_publication_bundle_args(args)
    length(args) == 1 && return (args[1], string(args[1], ".gitdir"))
    length(args) == 2 && return (args[1], args[2])
    error(publication_bundle_usage())
end

function run_publication_bundle_script(args=ARGS)
    bundle_path, git_dir = parse_publication_bundle_args(args)
    result = create_publication_bundle(bundle_path; git_dir)
    println(result.bundle)
    println(result.heads)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_publication_bundle_script()
end
