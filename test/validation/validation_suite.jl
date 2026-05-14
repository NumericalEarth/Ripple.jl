script_project_flag() = "--project=$(dirname(Base.active_project()))"

@testset "Validation suite" begin
    cases = collect(default_validation_cases())
    case_names = getproperty.(cases, :name)
    expected_names = [
        :constant_action,
        :second_moment_tensor,
        :q_normalization,
        :q_precomputed_weights,
        :relaxation_source_solution,
        :pure_damping_decay,
        :fetch_limited_source_balance,
        :hasselmann_column,
        :finite_volume_source_rates,
    ]

    @test length(cases) == length(expected_names)
    @test case_names == expected_names
    @test all(case -> case isa ValidationCase, cases)

    results = run_validation(cases)
    @test length(results) == length(cases)
    @test all(result -> result isa ValidationResult, results)
    @test validation_passed(results)

    for result in results
        @test result.name in expected_names
        @test !isempty(result.metrics)
        @test keys(result.metrics) == keys(result.tolerances)
        @test all(key -> abs(result.metrics[key]) <= result.tolerances[key], keys(result.metrics))
        @test !isempty(result.description)
    end

    mktempdir() do dir
        path = joinpath(dir, "validation_summary.tsv")
        write_validation_summary(path, results)
        lines = readlines(path)
        @test first(lines) == "case\tpassed\tmetric\tvalue\ttolerance\tdescription"
        @test length(lines) == 1 + sum(length(result.metrics) for result in results)
        @test all(length(split(line, '\t')) == 6 for line in lines)

        parsed = read_validation_summary(path)
        @test parsed[(:constant_action, :max_tendency)] == only(filter(r -> r.name == :constant_action, results)).metrics[:max_tendency]
        @test haskey(parsed, (:q_normalization, :normalization_error))
        @test haskey(parsed, (:finite_volume_source_rates, :finite_volume_rate_error))

        comparison = compare_validation_summaries(path, path)
        @test all(result -> result isa ExternalComparisonResult, comparison)
        @test validation_passed(comparison)
        @test all(result -> result.absolute_error == 0, comparison)

        perturbed = joinpath(dir, "validation_summary_perturbed.tsv")
        write_validation_summary(perturbed, results)
        text = read(perturbed, String)
        text = replace(text, "0.0" => "1.0"; count=1)
        write(perturbed, text)

        comparison = compare_validation_summaries(path, perturbed; atol=0.0, rtol=0.0)
        @test !validation_passed(comparison)

        script = joinpath(@__DIR__, "..", "..", "scripts", "validation", "run_validation_suite.jl")
        script_output = joinpath(dir, "script_validation_summary.tsv")
        repo = joinpath(@__DIR__, "..", "..")
        project = script_project_flag()
        cd(repo) do
            run(`$(Base.julia_cmd()) --startup-file=no $project $script $script_output constant_action q_normalization`)
            list_output = read(`$(Base.julia_cmd()) --startup-file=no $project $script --list`, String)
            @test occursin("constant_action", list_output)
            @test occursin("q_normalization", list_output)
        end
        script_summary = read_validation_summary(script_output)
        @test haskey(script_summary, (:constant_action, :max_tendency))
        @test haskey(script_summary, (:q_normalization, :normalization_error))
    end
end

@testset "Performance smoke format" begin
    metrics = collect(run_performance_smoke(; Nx=3, Ny=2, Nk=3, Nθ=4))
    @test length(metrics) == 2
    @test all(metric -> metric isa PerformanceMetric, metrics)
    @test Set(getproperty.(metrics, :case)) == Set([:product_field, :sources])
    @test Set(getproperty.(metrics, :operation)) == Set([:set_and_m0, :semi_implicit_step])
    @test all(metric -> metric.seconds >= 0, metrics)
    @test all(metric -> metric.bytes >= 0, metrics)
    @test all(metric -> !isempty(metric.description), metrics)

    mktempdir() do dir
        path = joinpath(dir, "performance.tsv")
        write_performance_summary(path, metrics)
        parsed = read_performance_summary(path)
        @test length(parsed) == length(metrics)
        @test getproperty.(parsed, :case) == collect(getproperty.(metrics, :case))
        @test getproperty.(parsed, :operation) == collect(getproperty.(metrics, :operation))

        script = joinpath(@__DIR__, "..", "..", "scripts", "performance", "run_smoke_benchmarks.jl")
        script_output = joinpath(dir, "script_performance.tsv")
        repo = joinpath(@__DIR__, "..", "..")
        project = script_project_flag()
        cd(repo) do
            run(`$(Base.julia_cmd()) --startup-file=no $project $script $script_output`)
        end

        script_metrics = read_performance_summary(script_output)
        @test length(script_metrics) == length(metrics)
        @test all(metric -> metric.seconds >= 0 && metric.bytes >= 0, script_metrics)
    end

    mktempdir() do dir
        path = joinpath(dir, "bad_performance.tsv")
        write(path, "bad\theader\n")
        @test_throws ArgumentError read_performance_summary(path)
    end
end

@testset "External metric runner format" begin
    text = """
    # comment
    fetch_limited.hm0_error=0.01
    source_column.energy_error\t0.02\t0.1\tsource-column reference metric
    """

    metrics = parse_external_metrics(text; default_tolerance=0.05)
    @test length(metrics) == 2
    @test metrics[1].case == :fetch_limited
    @test metrics[1].metric == :hm0_error
    @test metrics[1].value == 0.01
    @test metrics[1].tolerance == 0.05
    @test metrics[2].case == :source_column
    @test metrics[2].metric == :energy_error
    @test metrics[2].value == 0.02
    @test metrics[2].tolerance == 0.1

    mktempdir() do dir
        path = joinpath(dir, "external.tsv")
        write_external_metrics_summary(path, metrics)
        parsed = read_validation_summary(path)
        @test parsed[(:fetch_limited, :hm0_error)] == 0.01
        @test parsed[(:source_column, :energy_error)] == 0.02

        command_path = joinpath(dir, "emit_metrics.jl")
        write(command_path, "println(\"fetch_limited.hm0_error=0.0\")")
        output_path = joinpath(dir, "command.tsv")
        run_external_metrics_command(`$(Base.julia_cmd()) --startup-file=no $command_path`, output_path;
                                     default_tolerance=0.0)

        command_metrics = read_validation_summary(output_path)
        @test command_metrics[(:fetch_limited, :hm0_error)] == 0.0

        emitter = joinpath(dir, "emit_metrics.sh")
        write(emitter, "#!/bin/sh\necho fetch_limited.hm0_error=0.0\n")
        chmod(emitter, 0o755)
        script = joinpath(@__DIR__, "..", "..", "scripts", "external_models", "run_swan_fetch_limited.jl")
        script_output = joinpath(dir, "swan.tsv")
        repo = joinpath(@__DIR__, "..", "..")
        project = script_project_flag()
        cd(repo) do
            withenv("SWAN_METRICS_COMMAND" => emitter) do
                run(`$(Base.julia_cmd()) --startup-file=no $project $script $script_output`)
            end
        end
        @test read_validation_summary(script_output)[(:fetch_limited, :hm0_error)] == 0.0

        metrics_file = joinpath(dir, "swan_metrics.txt")
        write(metrics_file, "fetch_limited.mean_period_error\t0.03\t0.1\tfile metric\n")
        file_script_output = joinpath(dir, "swan_file.tsv")
        cd(repo) do
            withenv("SWAN_METRICS_COMMAND" => nothing,
                    "SWAN_METRICS_FILE" => metrics_file) do
                run(`$(Base.julia_cmd()) --startup-file=no $project $script $file_script_output`)
            end
        end
        @test read_validation_summary(file_script_output)[(:fetch_limited, :mean_period_error)] == 0.03

        bulk_text = """
        m0 cell_area Hs
        0.0625 2.0 1.0
        0.25 1.0 2.0
        """
        bulk_table = parse_external_bulk_table(bulk_text)
        bulk_metrics = external_bulk_table_metrics(bulk_table; case=:swan)
        @test bulk_table.row_count == 2
        @test bulk_table.header == [:m0, :cell_area, :Hs]
        @test only(filter(metric -> metric.metric == :mean_Hs, bulk_metrics)).value ≈ 4 / 3
        @test only(filter(metric -> metric.metric == :total_m0, bulk_metrics)).value ≈ 0.375

        bulk_file = joinpath(dir, "swan_bulk.txt")
        write(bulk_file, bulk_text)
        bulk_summary_path = joinpath(dir, "swan_bulk_metrics.tsv")
        write_external_bulk_metrics_summary(bulk_summary_path, bulk_file; case=:swan)
        parsed_bulk_summary = read_validation_summary(bulk_summary_path)
        @test parsed_bulk_summary[(:swan, :mean_Hs)] ≈ 4 / 3
        @test parsed_bulk_summary[(:swan, :total_m0)] ≈ 0.375

        bulk_script = joinpath(@__DIR__, "..", "..", "scripts", "postprocess", "external_bulk_to_metrics.jl")
        scripted_bulk_summary = joinpath(dir, "scripted_external_bulk.tsv")
        cd(repo) do
            run(`$(Base.julia_cmd()) --startup-file=no $project $bulk_script $bulk_file $scripted_bulk_summary swan`)
        end
        @test read_validation_summary(scripted_bulk_summary)[(:swan, :mean_Hs)] ≈ 4 / 3

        bulk_env_output = joinpath(dir, "swan_bulk_env.tsv")
        cd(repo) do
            withenv("SWAN_METRICS_COMMAND" => nothing,
                    "SWAN_METRICS_FILE" => nothing,
                    "SWAN_BULK_FILE" => bulk_file) do
                run(`$(Base.julia_cmd()) --startup-file=no $project $script $bulk_env_output`)
            end
        end
        @test read_validation_summary(bulk_env_output)[(:swan, :total_m0)] ≈ 0.375

        external_deck = external_model_input_deck(:swan; case=:fetch_limited)
        @test external_deck isa ExternalModelInputDeck
        @test external_deck.model == :swan
        @test external_deck.case == :fetch_limited
        @test only(keys(external_deck.files)) == "README.txt"

        input_dir = joinpath(dir, "swan_input")
        manifest = write_external_model_input_deck(input_dir, external_deck)
        @test isfile(manifest)
        @test isfile(joinpath(input_dir, "README.txt"))
        @test first(readlines(manifest)) == "model\tcase\tfile\tbytes"

        fake_model = joinpath(dir, "fake_model.sh")
        write(fake_model, "#!/bin/sh\necho fetch_limited.launch_error=0.0\n")
        chmod(fake_model, 0o755)
        profile = external_model_launch_profile(:swan; executable=fake_model)
        @test profile isa ExternalModelLaunchProfile
        @test profile.model == :swan
        @test profile.case == :fetch_limited
        @test profile.executable == fake_model

        plan = external_model_launch_plan(joinpath(dir, "launch"), :swan; executable=fake_model)
        @test plan isa ExternalModelLaunchPlan
        @test plan.model == :swan
        @test plan.case == :fetch_limited
        @test basename(only(plan.input_files)) == "README.txt"
        launch_output = joinpath(dir, "launch.tsv")
        run_external_model_launch_plan!(plan, launch_output)
        @test read_validation_summary(launch_output)[(:fetch_limited, :launch_error)] == 0.0
    end

    @test_throws ArgumentError parse_external_metrics("bad_line")
    @test_throws ArgumentError parse_external_bulk_table("x Hs\n")
    @test_throws ArgumentError parse_external_bulk_table("x Hs\n0 1 2\n")
    @test_throws ArgumentError external_bulk_table_metrics(parse_external_bulk_table("x y\n0 1\n"))
    @test_throws ArgumentError external_model_input_deck(:unknown)
end
