function publication_bundle_test_usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/validation/test_publication_bundle.jl BUNDLE_PATH OUTPUT.tsv [WORKDIR]

    Clones BUNDLE_PATH into a clean worktree and runs Ripple's default suite
    from the cloned publication contents. OUTPUT.tsv receives the
    RIPPLE_TEST_SUMMARY written by test/runtests.jl in the clone. If WORKDIR is
    omitted, a temporary clone directory is used.
    """
end

function publication_bundle_test_git()
    git = Sys.which("git")
    git === nothing && error("git executable missing")
    return git
end

function publication_bundle_test_command(julia_cmd, worktree, output_path)
    return `$(julia_cmd) --startup-file=no --project=$(worktree) $(joinpath(worktree, "test", "runtests.jl"))`
end

function clone_publication_bundle!(git, bundle_path, worktree)
    mkpath(dirname(worktree))
    run(`$git clone $bundle_path $worktree`)
    return worktree
end

function run_publication_bundle_default_suite(bundle_path,
                                              output_path;
                                              workdir=nothing,
                                              julia_cmd=Base.julia_cmd())
    git = publication_bundle_test_git()
    bundle_path = abspath(bundle_path)
    output_path = abspath(output_path)
    isfile(bundle_path) || error("bundle file missing: $(bundle_path)")
    mkpath(dirname(output_path))

    run_in_worktree(worktree) = begin
        clone_publication_bundle!(git, bundle_path, worktree)
        command = publication_bundle_test_command(julia_cmd, worktree, output_path)
        withenv("RIPPLE_TEST_SUMMARY" => output_path) do
            run(command)
        end
        return (bundle=bundle_path, output=output_path, worktree=worktree, command=command)
    end

    if workdir === nothing
        return mktempdir() do dir
            run_in_worktree(joinpath(dir, "Ripple.jl"))
        end
    else
        return run_in_worktree(abspath(workdir))
    end
end

function parse_publication_bundle_test_args(args)
    length(args) == 2 && return (args[1], args[2], nothing)
    length(args) == 3 && return (args[1], args[2], args[3])
    error(publication_bundle_test_usage())
end

function run_publication_bundle_test_script(args=ARGS)
    bundle_path, output_path, workdir = parse_publication_bundle_test_args(args)
    result = run_publication_bundle_default_suite(bundle_path, output_path; workdir)
    println(result.output)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_publication_bundle_test_script()
end
