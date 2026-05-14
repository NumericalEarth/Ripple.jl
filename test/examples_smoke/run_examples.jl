using Test
using Oceananigans
using Ripple

function test_finite_product_field(field)
    values = interior(field)
    @test !isempty(values)
    @test all(isfinite, values)
    @test all(values .>= -1e-12)
    @test isfinite(total_action(field))
    @test total_action(field) >= -1e-12
end

function test_visual_artifacts(paths)
    @test paths isa AbstractVector || paths isa Tuple
    @test !isempty(paths)
    for path in paths
        @test path isa AbstractString
        extension = splitext(path)[2]
        @test extension in (".mp4", ".png")
        @test isfile(path)
        @test filesize(path) > 0
    end
end

function test_example_module_artifacts(example_module)
    if isdefined(example_module, :N)
        test_finite_product_field(getfield(example_module, :N))
    end

    if isdefined(example_module, :model)
        model = getfield(example_module, :model)
        @test model isa SpectralWaveModel
        @test model.advection === nothing || model.advection isa Oceananigans.Advection.AbstractAdvectionScheme
        @test model.clock.time >= 0
        @test model.clock.iteration >= 0
        test_finite_product_field(model.action)
    end

    if isdefined(example_module, :result)
        @test validation_passed(getfield(example_module, :result))
    end

    @test isdefined(example_module, :plot_paths)
    @test isdefined(example_module, :animation_paths)
    test_visual_artifacts(getfield(example_module, :plot_paths))
    test_visual_artifacts(getfield(example_module, :animation_paths))
end

function test_example_contains(example_dir, file, required_patterns)
    text = read(joinpath(example_dir, file), String)
    for pattern in required_patterns
        @test occursin(pattern, text)
    end
end

@testset "Example smoke tests" begin
    example_dir = joinpath(@__DIR__, "..", "..", "examples")
    docs_root = joinpath(@__DIR__, "..", "..", "docs")
    example_files = [
        "product_field_basics.jl",
        "source_only_fetch_limited_growth.jl",
        "bounded_wave_packet_dispersion.jl",
        "hasselmann_inertial_oscillation.jl",
        "cwcm_q_transform_sheared_current.jl",
        "frequency_direction_source_package.jl",
        "exact_finite_volume_source_rates.jl",
    ]

    discovered_examples = sort([basename(path) for path in readdir(example_dir; join=true)
                                if endswith(path, ".jl")])
    @test sort(example_files) == discovered_examples

    @testset "Literate docs manifest" begin
        include(joinpath(docs_root, "generate.jl"))
        @test sort(collect(first.(EXAMPLE_TUTORIALS))) == sort(example_files)
        generated_dir = generate_documentation_sources!(docs_root)

        examples_md = read(joinpath(docs_root, "src", "examples.md"), String)
        for file in example_files
            source_text = read(joinpath(example_dir, file), String)
            @test startswith(source_text, "# # ")

            page = joinpath(generated_dir, "examples", first(splitext(file)) * ".md")
            @test isfile(page)
            @test occursin("examples/$file", read(page, String))
            @test occursin("generated/examples/$(first(splitext(file))).md", examples_md)
        end
    end

    @testset "Example semantic manifest" begin
        semantic_examples = (
            "product_field_basics.jl" => ("# # Product Field Basics", "WaveActionField", "set!", "plot_paths", "animation_paths"),
            "source_only_fetch_limited_growth.jl" => ("# # Source-Only Fetch-Limited Growth", "fetch_limited_source_balance", "run_validation", "advection=nothing"),
            "bounded_wave_packet_dispersion.jl" => ("# # Bounded Wave Packet Dispersion", "topology=(Bounded, Periodic, Bounded)", "advection=WENO(order=5)", "packet_hovmoller"),
            "hasselmann_inertial_oscillation.jl" => ("# # Hasselmann Column Growth", "hasselmann_column", "run_validation", "advection=nothing"),
            "cwcm_q_transform_sheared_current.jl" => ("# # CWCM Q-Transform", "QTransform", "CWCMPrescribedCurrentCoupling", "advection=nothing"),
            "frequency_direction_source_package.jl" => ("# # Frequency-Direction Source Package", "FrequencyDirectionGrid", "SourceTermSet", "advection=nothing", "SemiImplicitEuler"),
            "exact_finite_volume_source_rates.jl" => ("# # Exact Finite-Volume Source Rates", "spectral_frequency_power_average", "spectral_radial_power_average", "center_frequency_factor"),
        )

        for (file, required_patterns) in semantic_examples
            @testset "$file" begin
                @test file in example_files
                test_example_contains(example_dir, file, required_patterns)
            end
        end
    end

    previous_example_mode = get(ENV, "RIPPLE_EXAMPLE_MODE", nothing)
    ENV["RIPPLE_EXAMPLE_MODE"] = "small"
    try
        for file in example_files
            @testset "$file" begin
                path = joinpath(example_dir, file)
                example_module = Module(Symbol(:RippleExampleSmoke_, replace(file, r"[^A-Za-z0-9_]" => "_")))
                success = try
                    redirect_stdout(devnull) do
                        Base.include(example_module, path)
                    end
                    true
                catch err
                    @error "Example failed" file exception=(err, catch_backtrace())
                    false
                end
                @test success
                success && test_example_module_artifacts(example_module)
            end
        end
    finally
        if previous_example_mode === nothing
            delete!(ENV, "RIPPLE_EXAMPLE_MODE")
        else
            ENV["RIPPLE_EXAMPLE_MODE"] = previous_example_mode
        end
    end
end
