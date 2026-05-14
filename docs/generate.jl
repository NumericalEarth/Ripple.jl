const GENERATED_DOC_FILENAMES = (
    "implementation_status.md",
    "goal_completion_audit.md",
    "external_comparison_harness.md",
)

const EXAMPLE_TUTORIALS = (
    ("product_field_basics.jl", "Product Field Basics"),
    ("source_only_fetch_limited_growth.jl", "Source-Only Fetch-Limited Growth"),
    ("bounded_wave_packet_dispersion.jl", "Bounded Wave Packet Dispersion"),
    ("hasselmann_inertial_oscillation.jl", "Hasselmann Column Growth"),
    ("cwcm_q_transform_sheared_current.jl", "CWCM Q-Transform Shear"),
    ("frequency_direction_source_package.jl", "Frequency-Direction Sources"),
    ("exact_finite_volume_source_rates.jl", "Exact Finite-Volume Source Rates"),
)

function example_slug(filename)
    return first(splitext(filename)) * ".md"
end

function generated_example_pages()
    pages = Pair{String, String}["Overview" => "examples.md"]
    for (filename, title) in EXAMPLE_TUTORIALS
        push!(pages, title => joinpath("generated", "examples", example_slug(filename)))
    end
    return pages
end

function convert_literate_example(source_path, output_path; title)
    lines = readlines(source_path)
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        code_open = false

        for line in lines
            if startswith(line, "# ")
                if code_open
                    println(io, "```")
                    println(io)
                    code_open = false
                end
                println(io, line[3:end])
            elseif line == "#"
                if code_open
                    println(io, "```")
                    println(io)
                    code_open = false
                end
                println(io)
            else
                if !code_open
                    println(io)
                    println(io, "```julia")
                    code_open = true
                end
                println(io, line)
            end
        end

        code_open && println(io, "```")

        println(io)
        println(io, "## Running The Example")
        println(io)
        println(io, "From the repository root:")
        println(io)
        println(io, "```bash")
        println(io, "julia --startup-file=no --project=. examples/$(basename(source_path))")
        println(io, "```")
        println(io)
        println(io, "Set `RIPPLE_EXAMPLE_MODE=small` for the fast smoke-test version.")
        println(io)
        println(io, "This literate page is generated from `examples/$(basename(source_path))`.")
    end

    return output_path
end

function generate_documentation_sources!(docs_root=@__DIR__)
    generated_dir = joinpath(docs_root, "src", "generated")
    mkpath(generated_dir)

    for filename in GENERATED_DOC_FILENAMES
        cp(joinpath(docs_root, filename),
           joinpath(generated_dir, filename);
           force=true)
    end

    examples_dir = joinpath(generated_dir, "examples")
    source_examples_dir = joinpath(docs_root, "..", "examples")
    rm(examples_dir; force=true, recursive=true)
    mkpath(examples_dir)

    for (filename, title) in EXAMPLE_TUTORIALS
        convert_literate_example(joinpath(source_examples_dir, filename),
                                 joinpath(examples_dir, example_slug(filename));
                                 title)
    end

    return generated_dir
end
