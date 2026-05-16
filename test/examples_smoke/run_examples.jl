using Test
using Oceananigans
using Ripple

# Quick checks that each example runs end-to-end and leaves the model in a
# sane state. The full execution path (with Literate, plotting, and
# animations) lives in the docs build (`julia --project=docs docs/make.jl`).

function test_finite_product_field(field)
    values = interior(field)
    @test !isempty(values)
    @test all(isfinite, values)
    @test all(values .>= -1e-12)
    @test isfinite(total_action(field))
    @test total_action(field) >= -1e-12
end

function test_example_module_artifacts(example_module)
    if isdefined(example_module, :model)
        model = getfield(example_module, :model)
        @test model isa SpectralWaveModel
        @test model.clock.time >= 0
        @test model.clock.iteration >= 0
        test_finite_product_field(model.action)
    end
end

function test_example_contains(example_dir, file, required_patterns)
    text = read(joinpath(example_dir, file), String)
    for pattern in required_patterns
        @test occursin(pattern, text)
    end
end

@testset "Example smoke tests" begin
    example_dir = joinpath(@__DIR__, "..", "..", "examples")
    docs_root   = joinpath(@__DIR__, "..", "..", "docs")
    example_files = [
        "quick_start.jl",
        "source_only_fetch_limited_growth.jl",
        "bounded_wave_packet_dispersion.jl",
        "spectral_refraction_by_shear.jl",
        "vortex_refraction.jl",
        "translating_hurricane_swell.jl",
    ]

    discovered_examples = sort([basename(path) for path in readdir(example_dir; join=true)
                                if endswith(path, ".jl")])
    @test sort(example_files) == discovered_examples

    @testset "Literate docs manifest" begin
        include(joinpath(docs_root, "generate.jl"))
        @test sort(collect(first.(EXAMPLE_TUTORIALS))) == sort(example_files)

        examples_md = read(joinpath(docs_root, "src", "examples.md"), String)
        for file in example_files
            source_text = read(joinpath(example_dir, file), String)
            @test startswith(source_text, "# # ")
            @test occursin("generated/examples/$(first(splitext(file))).md", examples_md)
        end
    end

    @testset "Example semantic manifest" begin
        semantic_examples = (
            "quick_start.jl"                       => ("# # Quick Start",
                                                       "PolarWaveVectorGrid",
                                                       "velocities",
                                                       "Simulation",
                                                       ":RK3"),
            "source_only_fetch_limited_growth.jl"  => ("# # Source-Only Fetch-Limited Growth",
                                                       "horizontal_advection = nothing",
                                                       "ExponentialWindInput",
                                                       "WhitecappingDissipation"),
            "bounded_wave_packet_dispersion.jl"    => ("# # Bounded Wave Packet Dispersion",
                                                       "topology = (Bounded, Periodic, Bounded)",
                                                       "horizontal_advection = WENO(order = 5)"),
            "spectral_refraction_by_shear.jl"      => ("# # Spectral Refraction by a Sheared Current",
                                                       "PolarWaveVectorGrid",
                                                       "mean_direction",
                                                       "−T A ω cos(ω y)"),
            "vortex_refraction.jl"                 => ("# # Wave Refraction Through A Barotropic Vortex",
                                                       "velocities",
                                                       "Simulation",
                                                       ":RK3"),
            "translating_hurricane_swell.jl"       => ("# # Swell Generation by a Translating Idealized Hurricane",
                                                       "HollandHurricaneWind",
                                                       "MeanSpectrumPhysics",
                                                       "PressureCorrelationInput"),
        )

        for (file, required_patterns) in semantic_examples
            @testset "$file" begin
                @test file in example_files
                test_example_contains(example_dir, file, required_patterns)
            end
        end
    end

    # Run each example in a tempdir so `record(..., "x.mp4", ...)` output
    # does not pollute the repo. The example bodies use relative paths so
    # the tempdir becomes the working dir.
    for file in example_files
        @testset "$file" begin
            path = joinpath(example_dir, file)
            example_module = Module(Symbol(:RippleExampleSmoke_,
                                           replace(file, r"[^A-Za-z0-9_]" => "_")))
            success = mktempdir() do tmp
                cd(tmp) do
                    try
                        redirect_stdout(devnull) do
                            Base.include(example_module, path)
                        end
                        true
                    catch err
                        @error "Example failed" file exception = (err, catch_backtrace())
                        false
                    end
                end
            end
            @test success
            success && test_example_module_artifacts(example_module)
        end
    end
end
