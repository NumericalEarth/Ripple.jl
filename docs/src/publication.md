# Publication

The repository is configured for publication as `NumericalEarth/Ripple.jl`.
README badges, Documenter deployment, and GitHub Actions workflows already use
that repository path.

To write a local readiness checklist:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. scripts/validation/patch_oceananigans_manifest_triggers.jl .
julia --startup-file=no --project=. scripts/validation/check_publication_readiness.jl publication_readiness.tsv
julia --startup-file=no --project=. scripts/validation/write_goal_completion_checklist.jl goal_completion_checklist.tsv
RIPPLE_TEST_SUMMARY=default_suite.tsv julia --startup-file=no --project=. test/runtests.jl
julia --startup-file=no --project=. scripts/validation/test_publication_bundle.jl /private/tmp/Ripple.jl-initial.bundle publication_bundle_default_suite.tsv
julia --startup-file=no --project=. scripts/validation/write_goal_completion_checklist.jl goal_completion_checklist.tsv --default-suite-summary default_suite.tsv
julia --startup-file=no --project=. scripts/validation/write_goal_completion_checklist.jl goal_completion_checklist.tsv --default-suite-summary default_suite.tsv --require-complete
julia --startup-file=no --project=. scripts/validation/publish_to_numericalearth.jl --dry-run --bundle /private/tmp/Ripple.jl-initial.bundle --workdir /private/tmp/Ripple.jl-publish
```

Use `--require-complete` only for the final provisioned completion audit: it
writes the TSV and then exits with an error if any checklist row is not
`available`.

## Prerequisites

- GitHub access to create repositories in the `NumericalEarth` organization.
- A valid local GitHub CLI session:

```bash
gh auth status
```

- A local git repository. If this checkout does not have `.git`, initialize it
  after local filesystem permissions allow git metadata creation:

```bash
git init -b main
git add .
git commit -m "Initial Ripple.jl implementation"
```

If a sandbox blocks creation of a `.git` directory in the worktree, stage the
same files with a separate git directory:

```bash
julia --startup-file=no --project=. scripts/validation/create_publication_bundle.jl /private/tmp/Ripple.jl-initial.bundle /private/tmp/Ripple.jl.gitdir
julia --startup-file=no --project=. scripts/validation/test_publication_bundle.jl /private/tmp/Ripple.jl-initial.bundle publication_bundle_default_suite.tsv
RIPPLE_PUBLICATION_GIT_DIR=/private/tmp/Ripple.jl.gitdir RIPPLE_PUBLICATION_BUNDLE=/private/tmp/Ripple.jl-initial.bundle RIPPLE_PUBLICATION_BUNDLE_TEST_SUMMARY=publication_bundle_default_suite.tsv julia --startup-file=no --project=. scripts/validation/check_publication_readiness.jl publication_readiness.tsv
```

To publish from that bundle after GitHub access is available:

```bash
julia --startup-file=no --project=. scripts/validation/publish_to_numericalearth.jl --visibility public --bundle /private/tmp/Ripple.jl-initial.bundle --workdir /private/tmp/Ripple.jl-publish
```

When the GitHub CLI is authenticated, the publication-readiness checker also
attempts `gh repo view NumericalEarth/Ripple.jl --json nameWithOwner` so the
audit can distinguish an existing accessible repository from a repository that
still needs to be created or shared with the authenticated account.

## Create And Push

Choose visibility explicitly:

```bash
gh repo create NumericalEarth/Ripple.jl --public --source=. --remote=origin --push
```

For a private or internal repository, replace `--public` with `--private` or
`--internal`.

The checked-in publication script wraps the same publication intent and also
handles the case where the repository already exists. It creates the repository
if needed, requires a clean git worktree when publishing without a bundle,
resets `origin` to `git@github.com:NumericalEarth/Ripple.jl.git`, and pushes
`main`. For non-dry-run execution, it first checks local publication
push readiness: `publication_decisions.toml` must record a decided license,
the corresponding `LICENSE` file must exist, and `registry_policy` must be
`general` or `organization-local`. Use `--skip-readiness` only for an
intentional pre-policy private/internal handoff; the script rejects
`--skip-readiness` with public visibility. For that handoff path, it checks
GitHub CLI authentication before cloning a bundle into the publish worktree:

```bash
julia --startup-file=no --project=. scripts/validation/publish_to_numericalearth.jl --visibility public
```

If the repository already exists:

```bash
git remote add origin git@github.com:NumericalEarth/Ripple.jl.git
git push -u origin main
```

## Documentation Deployment

The documentation workflow builds with Documenter.jl and deploys to the
`gh-pages` branch. The workflow uses `GITHUB_TOKEN` with `contents: write`; a
`DOCUMENTER_KEY` secret can also be supplied if the organization requires an
SSH deploy key.

After the first successful documentation workflow, verify GitHub Pages is
serving from the `gh-pages` branch if the site is not available at:

```text
https://NumericalEarth.github.io/Ripple.jl/stable/
```

## Registration Checklist

Before registering or advertising a tagged release, make these repository-level
choices explicitly:

- Review `CONTRIBUTING.md`, the pull request template, and the issue templates
  after repository labels and organization policies are known.
- Choose and add a `LICENSE` file. This is a project/legal decision and is not
  inferred by the package code.
- Record the license and registry decisions in `publication_decisions.toml`.
  Leave values as `undecided` until the project owner has made the choice.
- Decide whether to register in Julia's General registry or keep the package
  organization-local, then set `registry_policy` to `general` or
  `organization-local`.
- If registering, verify package registration requirements in a networked Julia
  environment. The CI workflow runs the direct project suite; `Pkg.test()` is
  deferred while Oceananigans 0.107 needs a Julia 1.10 manifest-trigger patch
  before package load.
- Decide whether to add release automation such as TagBot and CompatHelper
  after repository secrets and organization policies are known.

## Required Follow-Up Gates

The default suite does not require optional runtimes. Full cross-runtime
completion still requires the optional gate runner in an environment with
Oceananigans, CUDA, and external model executables installed:

```bash
export RIPPLE_OPTIONAL_RUNTIME_ENV=/tmp/ripple-optional-runtime
julia --startup-file=no --project="$RIPPLE_OPTIONAL_RUNTIME_ENV" -e 'using Pkg; Pkg.add(["Oceananigans", "GPUArraysCore", "KernelAbstractions", "CUDA"])'
julia --startup-file=no --project=. scripts/validation/patch_oceananigans_manifest_triggers.jl "$RIPPLE_OPTIONAL_RUNTIME_ENV"
export JULIA_LOAD_PATH="@:$RIPPLE_OPTIONAL_RUNTIME_ENV:@stdlib"
julia --startup-file=no --project=. scripts/validation/run_available_optional_gates.jl --require-all optional_gate_outputs
```

If Oceananigans readiness reports `KeyError("GPUArraysCore")`, the optional
environment needs to be recreated or provisioned with a manifest where
`GPUArraysCore` and `KernelAbstractions` are recorded for Oceananigans before
the Ripple smoke can run. The manifest patch helper above only edits the
throwaway optional runtime environment used by the runtime gates.

The same completion gate is available as the manual
`Optional Runtime Gates` GitHub Actions workflow. Trigger it with a runner label
for a provisioned runner and leave `require_all` enabled when the goal is full
cross-runtime validation. That workflow also uploads `default_suite.tsv`,
`publication_bundle_default_suite.tsv`, `publication_readiness.tsv`, and
`goal_completion_checklist.tsv` from the provisioned run. Enable
`require_complete_audit` only for the final completion run after license,
documentation runtime, repository access, and optional gates are all
provisioned; it adds `--require-complete` to the goal-completion checklist
writer.
