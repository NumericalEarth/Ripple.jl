using TOML

function patch_oceananigans_manifest_triggers_usage()
    return """
    Usage:
      julia --startup-file=no --project=. scripts/validation/patch_oceananigans_manifest_triggers.jl OPTIONAL_ENV_OR_MANIFEST

    Patches a throwaway optional-runtime Manifest.toml so Oceananigans 0.107
    extension trigger packages that are listed in its extensions are also
    recorded in the Oceananigans weakdeps table. This works around Julia 1.10
    manifest-trigger metadata generated from affected registry/package
    metadata. Do not run this on Ripple's root project; use it only for the
    optional runtime environment used by runtime gates.
    """
end

const OCEANANIGANS_TRIGGER_WEAKDEPS = Dict(
    "GPUArraysCore" => "46192b85-c4d5-4398-a991-12ede77f4527",
    "KernelAbstractions" => "63c18a36-062a-441e-b654-da1e3ab1ce7c",
)

function optional_manifest_path(path)
    candidate = normpath(path)
    isdir(candidate) && return joinpath(candidate, "Manifest.toml")
    return candidate
end

function oceananigans_manifest_entry(manifest)
    deps = get(manifest, "deps", nothing)
    deps isa AbstractDict ||
        throw(ArgumentError("Manifest.toml has no [deps] table"))

    entries = get(deps, "Oceananigans", nothing)
    entries isa AbstractVector ||
        throw(ArgumentError("Manifest.toml has no [[deps.Oceananigans]] entry"))
    length(entries) == 1 ||
        throw(ArgumentError("expected one [[deps.Oceananigans]] entry, found $(length(entries))"))

    entry = only(entries)
    entry isa AbstractDict ||
        throw(ArgumentError("[[deps.Oceananigans]] entry is not a table"))

    return entry
end

function oceananigans_extensions_require_trigger(entry, trigger_name)
    extensions = get(entry, "extensions", Dict{String, Any}())
    extensions isa AbstractDict || return false

    for value in values(extensions)
        if value isa AbstractVector && trigger_name in String.(value)
            return true
        elseif value == trigger_name
            return true
        end
    end

    return false
end

function patch_oceananigans_manifest_triggers!(manifest)
    entry = oceananigans_manifest_entry(manifest)
    weakdeps = get!(entry, "weakdeps", Dict{String, Any}())
    weakdeps isa AbstractDict ||
        throw(ArgumentError("[deps.Oceananigans.weakdeps] is not a table"))

    added = String[]
    for (name, uuid) in sort(collect(OCEANANIGANS_TRIGGER_WEAKDEPS); by=first)
        if oceananigans_extensions_require_trigger(entry, name) && !haskey(weakdeps, name)
            weakdeps[name] = uuid
            push!(added, name)
        end
    end

    return added
end

function patch_oceananigans_manifest_triggers(path)
    manifest_path = optional_manifest_path(path)
    isfile(manifest_path) ||
        throw(ArgumentError("Manifest.toml not found: $(manifest_path)"))

    manifest = TOML.parsefile(manifest_path)
    added = patch_oceananigans_manifest_triggers!(manifest)

    if !isempty(added)
        open(manifest_path, "w") do io
            TOML.print(io, manifest; sorted=true)
        end
    end

    return added
end

function run_patch_oceananigans_manifest_triggers_script(args=ARGS)
    length(args) == 1 || error(patch_oceananigans_manifest_triggers_usage())
    added = patch_oceananigans_manifest_triggers(only(args))
    isempty(added) ?
        println("Oceananigans manifest trigger weakdeps already recorded") :
        println("Patched Oceananigans manifest trigger weakdeps: ", join(added, ", "))
    return added
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_patch_oceananigans_manifest_triggers_script()
end
