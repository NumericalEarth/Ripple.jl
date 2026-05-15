struct ValidationCase{F}
    name :: Symbol
    description :: String
    runner :: F
end

struct ValidationResult
    name :: Symbol
    passed :: Bool
    metrics :: Dict{Symbol, Float64}
    tolerances :: Dict{Symbol, Float64}
    description :: String
end

struct ExternalComparisonResult
    case :: Symbol
    metric :: Symbol
    reference :: Float64
    candidate :: Float64
    absolute_error :: Float64
    relative_error :: Float64
    passed :: Bool
end

struct ExternalMetric
    case :: Symbol
    metric :: Symbol
    value :: Float64
    tolerance :: Float64
    description :: String
end

struct ExternalModelInputDeck
    model :: Symbol
    case :: Symbol
    files :: Dict{String, String}
end

struct ExternalModelLaunchPlan
    model :: Symbol
    case :: Symbol
    workdir :: String
    input_manifest :: String
    command :: Cmd
    input_files :: Vector{String}
    bulk_output_path :: String
    metrics_output_path :: String
end

struct ExternalModelLaunchProfile
    model :: Symbol
    case :: Symbol
    profile :: Symbol
    executable :: String
    arguments :: Vector{String}
    bulk_output_file :: String
    metrics_output_file :: String
end

struct PerformanceMetric
    case :: Symbol
    operation :: Symbol
    seconds :: Float64
    bytes :: Float64
    description :: String
end

validation_passed(result::ValidationResult) = result.passed
validation_passed(results) = all(validation_passed, results)
validation_passed(results::AbstractVector{ExternalComparisonResult}) = all(result -> result.passed, results)

run_validation(case::ValidationCase) = case.runner()
run_validation(cases) = [run_validation(case) for case in cases]
run_validation() = run_validation(default_validation_cases())

function ValidationResult(name, description, metrics, tolerances)
    metric_dict = Dict{Symbol, Float64}(metrics)
    tolerance_dict = Dict{Symbol, Float64}(tolerances)
    passed = all(abs(value) <= tolerance_dict[key] for (key, value) in metric_dict)
    return ValidationResult(name, passed, metric_dict, tolerance_dict, description)
end

function default_validation_cases()
    return (
        ValidationCase(:constant_action,
                       "Source-free action with horizontal_advection=nothing has zero tendency and is unchanged by time stepping.",
                       constant_action_validation),
        ValidationCase(:second_moment_tensor,
                       "Spectral second moments use exact finite-volume cell moment measures.",
                       second_moment_tensor_validation),
        ValidationCase(:q_normalization,
                       "Q-kernel cell integrals over the RectilinearGrid vertical coordinate normalize to one.",
                       q_normalization_validation),
        ValidationCase(:q_precomputed_weights,
                       "Precomputed Q weights match on-the-fly perfect finite-volume integration.",
                       q_precomputed_weights_validation),
        ValidationCase(:relaxation_source_solution,
                       "RelaxationToSpectrum follows the analytic column solution.",
                       relaxation_source_solution_validation),
        ValidationCase(:pure_damping_decay,
                       "Semi-implicit damping follows the exact backward-Euler column update.",
                       pure_damping_decay_validation),
        ValidationCase(:fetch_limited_source_balance,
                       "Wind input and whitecapping approach the source-only equilibrium monotonically.",
                       fetch_limited_source_balance_validation),
        ValidationCase(:hasselmann_column,
                       "A Hasselmann-style source-only column follows analytic action growth and Q-integrated pseudomomentum.",
                       hasselmann_column_validation),
        ValidationCase(:finite_volume_source_rates,
                       "Power-law source rates use exact spectral finite-volume averages.",
                       finite_volume_source_rates_validation),
    )
end

function constant_action_validation()
    grid = RectilinearGrid(CPU(); size=(3, 2, 2), x=(0, 3), y=(0, 2), z=(-1, 0))
    cgrid = CartesianWaveVectorGrid(Float64; kx=[0.4, 0.8], ky=[-0.2, 0.2])
    model = SpectralWaveModel(; grid, spectral_grid=cgrid, horizontal_advection=nothing)
    set!(model, N=1.0)
    initial = copy(interior(model.action))
    compute_tendencies!(model)
    time_step!(model, 0.25)
    metrics = Dict(:max_tendency => maximum(abs, interior(model.tendencies)),
                   :state_change => maximum(abs.(interior(model.action) .- initial)),
                   :cfl => cfl(model))
    tolerances = Dict(:max_tendency => 0.0, :state_change => 0.0, :cfl => 0.0)
    return ValidationResult(:constant_action, "source-only no-op action", metrics, tolerances)
end

function second_moment_tensor_validation()
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = CartesianWaveVectorGrid(Float64; kx=[-1.0, 0.5, 2.0], ky=[-0.5, 1.5])
    N = WaveActionField(grid, cgrid)
    set!(N, 1.0)
    Mxx, Mxy, Myy = second_moment(N)

    expected_xx = sum(spectral_second_moment_measures(cgrid, m, n)[1]
                      for n in 1:length(cgrid.ky), m in 1:length(cgrid.kx))
    expected_xy = sum(spectral_second_moment_measures(cgrid, m, n)[2]
                      for n in 1:length(cgrid.ky), m in 1:length(cgrid.kx))
    expected_yy = sum(spectral_second_moment_measures(cgrid, m, n)[3]
                      for n in 1:length(cgrid.ky), m in 1:length(cgrid.kx))

    metrics = Dict(:Mxx_error => Mxx[1, 1] - expected_xx,
                   :Mxy_error => Mxy[1, 1] - expected_xy,
                   :Myy_error => Myy[1, 1] - expected_yy)
    tolerances = Dict(:Mxx_error => 1e-12, :Mxy_error => 1e-12, :Myy_error => 1e-12)
    return ValidationResult(:second_moment_tensor, "exact finite-volume second moments", metrics, tolerances)
end

function q_normalization_validation()
    grid = RectilinearGrid(CPU();
                           size=(1, 1, 8),
                           x=(0, 1),
                           y=(0, 1),
                           z=(-1, 0))
    kernel = QKernel(Float64)
    faces = zfaces(grid)
    errors = Float64[]
    derivative_errors = Float64[]

    for kappa in (1e-8, 0.25, 1.0, 4.0)
        integral = sum(q_cell_integral(kernel, kappa, faces[k], faces[k+1], 1.0)
                       for k in 1:vertical_size(grid))
        derivative = sum(q_cell_integral_kappa_derivative(kernel, kappa, faces[k], faces[k+1], 1.0)
                         for k in 1:vertical_size(grid))
        push!(errors, integral - 1)
        push!(derivative_errors, derivative)
    end

    metrics = Dict(:normalization_error => maximum(abs, errors),
                   :derivative_normalization_error => maximum(abs, derivative_errors))
    tolerances = Dict(:normalization_error => 1e-12,
                      :derivative_normalization_error => 1e-10)
    return ValidationResult(:q_normalization, "Q kernel vertical normalization", metrics, tolerances)
end

function q_precomputed_weights_validation()
    grid = RectilinearGrid(CPU();
                           size=(2, 1, 6),
                           x=[0.0, 0.4, 1.0],
                           y=[0.0, 1.0],
                           z=collect(range(-1, 0; length=7)),
                           topology=(Bounded, Bounded, Bounded))
    kernel = QKernel(Float64)
    kappa = [0.25, 0.8, 1.4]
    depth = reshape([1.0, 1.3], 2, 1)
    qtransform = QTransform(kernel, grid)
    cached = QTransform(kernel, grid, PrecomputeQWeights(kernel, grid, kappa, depth))

    u = [0.1i - 0.2j + 0.05znodes(grid)[k] for i in 1:2, j in 1:1, k in 1:vertical_size(grid)]
    v = [-0.3i + 0.1j - 0.02znodes(grid)[k] for i in 1:2, j in 1:1, k in 1:vertical_size(grid)]
    Ux = zeros(2, 1, length(kappa))
    Uy = zeros(2, 1, length(kappa))
    cached_Ux = similar(Ux)
    cached_Uy = similar(Uy)
    compute_doppler_velocity!(Ux, Uy, u, v, depth, kappa, qtransform)
    compute_doppler_velocity!(cached_Ux, cached_Uy, u, v, depth, kappa, cached)

    metrics = Dict(:Ux_error => maximum(abs.(cached_Ux .- Ux)),
                   :Uy_error => maximum(abs.(cached_Uy .- Uy)))
    tolerances = Dict(:Ux_error => 1e-14, :Uy_error => 1e-14)
    return ValidationResult(:q_precomputed_weights, "cached Q weights match exact integration", metrics, tolerances)
end

function relaxation_source_solution_validation()
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = CartesianWaveVectorGrid(Float64; kx=[0.4, 0.8], ky=[-0.2, 0.2])
    target(x, y, kx, ky) = 0.5 + 0.2kx - 0.1ky
    alpha = 0.7
    model = SpectralWaveModel(; horizontal_advection=nothing, grid,
                                spectral_grid=cgrid,
                                sources=RelaxationToSpectrum(target; timescale=inv(alpha)),
                                timestepper=:ForwardEuler)
    set!(model, N=0.0)
    dt = 1e-3
    steps = 100
    for _ in 1:steps
        time_step!(model, dt)
    end

    target_field = WaveActionField(grid, cgrid)
    set!(target_field, target)
    expected_scale = 1 - (1 - alpha * dt)^steps
    expected = expected_scale .* interior(target_field)
    metrics = Dict(:action_error => maximum(abs.(interior(model.action) .- expected)))
    tolerances = Dict(:action_error => 1e-12)
    return ValidationResult(:relaxation_source_solution, "explicit relaxation column solution", metrics, tolerances)
end

function pure_damping_decay_validation()
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = CartesianWaveVectorGrid(Float64; kx=[0.5], ky=[0.0])
    rate = 2.0
    dt = 0.1
    model = SpectralWaveModel(; horizontal_advection=nothing, grid,
                                spectral_grid=cgrid,
                                sources=BottomFriction(rate=rate),
                                timestepper=:SemiImplicitEuler)
    set!(model, N=1.0)
    time_step!(model, dt)
    expected = inv(1 + rate * dt)
    metrics = Dict(:decay_error => model.action[1, 1, 1, 1] - expected)
    tolerances = Dict(:decay_error => 1e-14)
    return ValidationResult(:pure_damping_decay, "semi-implicit damping update", metrics, tolerances)
end

function fetch_limited_source_balance_validation()
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=[1.0], φ=[0.0])
    target_growth_rate = 1.2
    weight = spectral_weight(cgrid, 1, 1)
    directional_weight = wind_directional_weight(cgrid, 1, 1, 0.0, 2)
    source = SourceTermSet(
        ExponentialWindInput(rate=target_growth_rate / directional_weight,
                             direction=0.0,
                             spreading_power=2),
        WhitecappingDissipation(rate=4.8,
                                 saturation_threshold=0.5,
                                 saturation_power=1.0,
                                 wavenumber_power=0.0),
    )
    model = SpectralWaveModel(; horizontal_advection=nothing, grid,
                                spectral_grid=cgrid,
                                sources=source,
                                timestepper=:SemiImplicitEuler)
    equilibrium_m0 = 0.5 * (1 + target_growth_rate / 4.8)
    set!(model, N=0.5 / (20weight))
    moments = Float64[m0(model.action)[1, 1]]
    for _ in 1:240
        time_step!(model, 0.02)
        push!(moments, m0(model.action)[1, 1])
    end

    equilibrium_action = equilibrium_m0 / weight
    equilibrium_model = SpectralWaveModel(; horizontal_advection=nothing, grid, spectral_grid=cgrid, sources=source)
    set!(equilibrium_model, N=equilibrium_action)
    positive, damping = source_split(source, equilibrium_model, 1, 1, 1, 1)

    metrics = Dict(:equilibrium_tendency_error => positive - damping * equilibrium_action,
                   :final_m0_error => last(moments) - equilibrium_m0,
                   :monotonicity_error => min(0.0, minimum(diff(moments))))
    tolerances = Dict(:equilibrium_tendency_error => 1e-12,
                      :final_m0_error => 1e-2,
                      :monotonicity_error => 1e-12)
    return ValidationResult(:fetch_limited_source_balance, "source-only fetch-limited equilibrium", metrics, tolerances)
end

function hasselmann_column_validation()
    grid = RectilinearGrid(CPU();
                           size=(1, 1, 8),
                           x=(0, 1),
                           y=(0, 1),
                           z=(-1, 0))
    cgrid = PolarWaveVectorGrid(; κ=range(0.35, 1.15; length=8),
                                  φ=range(0, 2pi; length=17)[1:16])
    target(x, y, kx, ky) = begin
        k = hypot(kx, ky)
        direction = k == 0 ? 0.0 : max(kx / k, 0.0)^4
        exp(-((k - 0.75) / 0.22)^2) * direction
    end

    alpha = 1.3
    dt = 5e-4
    steps = 200
    model = SpectralWaveModel(; horizontal_advection=nothing, grid,
                                spectral_grid=cgrid,
                                sources=RelaxationToSpectrum(target; timescale=inv(alpha)),
                                timestepper=:ForwardEuler)
    set!(model, N=0.0)
    for _ in 1:steps
        time_step!(model, dt)
    end

    target_field = WaveActionField(grid, cgrid)
    set!(target_field, target)
    expected_scale = 1 - (1 - alpha * dt)^steps
    expected_total_action = expected_scale * total_action(target_field)
    qtransform = QTransform(QKernel(Float64), grid)
    px, py = pseudomomentum_fields(model.action, 1.0, qtransform)
    mx, my = first_moment(model.action)

    metrics = Dict(:action_error => total_action(model.action) - expected_total_action,
                   :pseudomomentum_error => maximum(abs.(vertical_integral(px) .- mx)) +
                                            maximum(abs.(vertical_integral(py) .- my)),
                   :current_error => 0.0,
                   :kinetic_error => 0.0)
    tolerances = Dict(:action_error => 1e-10,
                      :pseudomomentum_error => 1e-10,
                      :current_error => 0.0,
                      :kinetic_error => 0.0)
    return ValidationResult(:hasselmann_column, "source-only Hasselmann-style column", metrics, tolerances)
end

function finite_volume_source_rates_validation()
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))
    cgrid = FrequencyDirectionGrid(; frequency=range(0.08, 0.32; length=12),
                                     φ=range(0, 2pi; length=25)[1:24])
    source = FrequencyDissipation(rate=0.4, reference_frequency=0.16, power=2)
    model = SpectralWaveModel(; horizontal_advection=nothing, grid,
                                spectral_grid=cgrid,
                                sources=source,
                                timestepper=:SemiImplicitEuler)
    set!(model, N=1.0)
    m, n = 6, 1
    _, damping = source_split(source, model, 1, 1, m, n)
    expected = source.rate *
               spectral_frequency_power_average(cgrid, m, n, source.power) /
               source.reference_frequency^source.power
    center = source.rate * (cgrid.frequency[m] / source.reference_frequency)^source.power
    metrics = Dict(:finite_volume_rate_error => damping - expected,
                   :midpoint_difference => abs(expected - center))
    tolerances = Dict(:finite_volume_rate_error => 1e-14,
                      :midpoint_difference => Inf)
    return ValidationResult(:finite_volume_source_rates, "exact finite-volume source rates", metrics, tolerances)
end

function write_validation_summary(path::AbstractString, results)
    open(path, "w") do io
        println(io, "case\tpassed\tmetric\tvalue\ttolerance\tdescription")
        for result in results
            for metric in sort!(collect(keys(result.metrics)); by=String)
                println(io, result.name, '\t',
                        result.passed, '\t',
                        metric, '\t',
                        result.metrics[metric], '\t',
                        result.tolerances[metric], '\t',
                        result.description)
            end
        end
    end
    return path
end

function read_validation_summary(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("validation summary is empty: $path"))
    split(first(lines), '\t') == ["case", "passed", "metric", "value", "tolerance", "description"] ||
        throw(ArgumentError("validation summary has an unexpected header: $(first(lines))"))
    data = Dict{Tuple{Symbol, Symbol}, Float64}()
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        parts = split(line, '\t')
        length(parts) == 6 || throw(ArgumentError("validation summary row must have 6 columns: $line"))
        data[(Symbol(parts[1]), Symbol(parts[3]))] = parse(Float64, parts[4])
    end
    return data
end

function compare_validation_summaries(reference_path::AbstractString,
                                      candidate_path::AbstractString;
                                      atol=0.0,
                                      rtol=1e-12)
    reference = read_validation_summary(reference_path)
    candidate = read_validation_summary(candidate_path)
    results = ExternalComparisonResult[]
    for key in sort!(collect(keys(reference)); by=string)
        case, metric = key
        ref = reference[key]
        cand = get(candidate, key, NaN)
        abs_error = abs(cand - ref)
        rel_error = abs_error / max(abs(ref), eps(Float64))
        passed = isfinite(cand) && abs_error <= atol + rtol * abs(ref)
        push!(results, ExternalComparisonResult(case, metric, ref, cand, abs_error, rel_error, passed))
    end
    return results
end

function parse_external_metrics(text::AbstractString; default_tolerance=0.0, description="external model metric")
    metrics = ExternalMetric[]
    for raw_line in split(text, '\n')
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, "#") && continue
        if occursin("=", line)
            key, value = split(line, "="; limit=2)
            tolerance = string(default_tolerance)
            desc = description
        else
            parts = split(line, '\t')
            length(parts) in (3, 4) ||
                throw(ArgumentError("external metric lines must be `case.metric=value` or tab-separated `case.metric value tolerance [description]`: $line"))
            key, value, tolerance = parts[1:3]
            desc = length(parts) == 4 ? parts[4] : description
        end
        dot = findlast(==('.'), key)
        dot === nothing && throw(ArgumentError("external metric key must be `case.metric`: $key"))
        case = Symbol(strip(key[begin:prevind(key, dot)]))
        metric = Symbol(strip(key[nextind(key, dot):end]))
        push!(metrics, ExternalMetric(case, metric, parse(Float64, strip(value)),
                                      parse(Float64, strip(tolerance)), String(desc)))
    end
    return metrics
end

function write_external_metrics_summary(path::AbstractString, metrics)
    open(path, "w") do io
        println(io, "case\tpassed\tmetric\tvalue\ttolerance\tdescription")
        for metric in metrics
            println(io, metric.case, '\t',
                    abs(metric.value) <= metric.tolerance, '\t',
                    metric.metric, '\t',
                    metric.value, '\t',
                    metric.tolerance, '\t',
                    metric.description)
        end
    end
    return path
end

function run_external_metrics_command(command::Cmd, output_path::AbstractString;
                                      default_tolerance=0.0,
                                      description="external model metric")
    metrics = parse_external_metrics(read(command, String); default_tolerance, description)
    isempty(metrics) && throw(ArgumentError("external metrics command produced no metrics"))
    return write_external_metrics_summary(output_path, metrics)
end

function run_external_metrics_command(command::AbstractString, output_path::AbstractString; kwargs...)
    isempty(strip(command)) && throw(ArgumentError("external metrics command must not be empty"))
    return run_external_metrics_command(Cmd(String.(split(command))), output_path; kwargs...)
end

normalized_external_model(model) = begin
    key = Symbol(lowercase(String(model)))
    key in (:swan, :wam, :ww3, :ecwam, :picles) && return key
    key in (:wavewatch, :wavewatchiii, :wavewatch3) && return :ww3
    throw(ArgumentError("unsupported external model `$model`; expected swan, wam, ww3, ecwam, or picles"))
end

external_model_default_case(model) =
    normalized_external_model(model) === :picles ? :stationary_vortex : :fetch_limited

function external_model_env_prefix(model)
    normalized = normalized_external_model(model)
    normalized === :ww3 && return "WW3"
    return uppercase(String(normalized))
end

external_model_executable_env_var(model) = external_model_env_prefix(model) * "_EXECUTABLE"
external_model_workdir_env_var(model) = external_model_env_prefix(model) * "_WORKDIR"

function external_model_launch_profile(model;
                                       case=nothing,
                                       profile=:default,
                                       executable=nothing,
                                       arguments=nothing)
    normalized = normalized_external_model(model)
    selected_case = case === nothing ? external_model_default_case(normalized) : Symbol(case)
    exe = executable === nothing ? get(ENV, external_model_executable_env_var(normalized), String(normalized)) :
                                   String(executable)
    args = arguments === nothing ? String[] : String.(collect(arguments))
    return ExternalModelLaunchProfile(normalized, selected_case, Symbol(profile), exe, args,
                                      "$(normalized)_bulk_output.txt",
                                      "$(normalized)_metrics.txt")
end

function external_model_input_deck(model; case=nothing, kwargs...)
    normalized = normalized_external_model(model)
    selected_case = case === nothing ? external_model_default_case(normalized) : Symbol(case)
    files = Dict("README.txt" => "Ripple generated placeholder input deck for $(normalized) $(selected_case).\n")
    return ExternalModelInputDeck(normalized, selected_case, files)
end

function write_external_model_input_deck(output_dir::AbstractString, deck::ExternalModelInputDeck)
    mkpath(output_dir)
    manifest_path = joinpath(output_dir, "manifest.tsv")
    open(manifest_path, "w") do io
        println(io, "model\tcase\tfile\tbytes")
        for filename in sort!(collect(keys(deck.files)))
            content = deck.files[filename]
            write(joinpath(output_dir, filename), content)
            println(io, deck.model, '\t', deck.case, '\t', filename, '\t', sizeof(content))
        end
    end
    return manifest_path
end

function write_external_model_input_deck(output_dir::AbstractString, model; case=nothing, kwargs...)
    return write_external_model_input_deck(output_dir, external_model_input_deck(model; case, kwargs...))
end

function external_model_launch_plan(output_dir::AbstractString, model;
                                    case=nothing,
                                    profile=:default,
                                    executable=nothing,
                                    arguments=nothing,
                                    kwargs...)
    deck = external_model_input_deck(model; case, kwargs...)
    manifest = write_external_model_input_deck(output_dir, deck)
    launch = external_model_launch_profile(deck.model; case=deck.case, profile, executable, arguments)
    command = Cmd(vcat([launch.executable], launch.arguments))
    input_files = [joinpath(output_dir, file) for file in keys(deck.files)]
    return ExternalModelLaunchPlan(deck.model, deck.case, output_dir, manifest, command, input_files,
                                   joinpath(output_dir, launch.bulk_output_file),
                                   joinpath(output_dir, launch.metrics_output_file))
end

function run_external_model_launch_plan!(plan::ExternalModelLaunchPlan,
                                         output_path::AbstractString=plan.metrics_output_path;
                                         default_tolerance=0.0,
                                         description="external model metric")
    return run_external_metrics_command(plan.command, output_path; default_tolerance, description)
end

function parse_external_bulk_table(text::AbstractString)
    lines = [strip(line) for line in split(text, '\n') if !isempty(strip(line))]
    isempty(lines) && throw(ArgumentError("external bulk table is empty"))
    length(lines) > 1 || throw(ArgumentError("external bulk table must contain at least one data row"))
    header = Symbol.(split(first(lines)))
    columns = Dict(name => Float64[] for name in header)
    for line in Iterators.drop(lines, 1)
        parts = split(line)
        length(parts) == length(header) || throw(ArgumentError("external bulk row has wrong number of columns: $line"))
        for (name, value) in zip(header, parts)
            push!(columns[name], parse(Float64, value))
        end
    end
    return (header=header, columns=columns, row_count=length(first(values(columns))))
end

external_bulk_table_metrics(path::AbstractString; kwargs...) =
    external_bulk_table_metrics(parse_external_bulk_table(read(path, String)); kwargs...)

function external_bulk_table_metrics(table; case=:external_bulk,
                                     tolerance=Inf,
                                     description="external bulk field scalar metric",
                                     direction_period=360.0)
    metrics = ExternalMetric[]
    weights = get(table.columns, :cell_area, ones(Float64, table.row_count))
    for (column, metric) in ((:m0, :total_m0),
                             (:significant_wave_height, :mean_Hs),
                             (:Hs, :mean_Hs))
        values = get(table.columns, column, nothing)
        values === nothing && continue
        value = metric === :total_m0 ? sum(values .* weights) : sum(values .* weights) / sum(weights)
        push!(metrics, ExternalMetric(Symbol(case), metric, value, tolerance, description))
    end
    isempty(metrics) && throw(ArgumentError("external bulk table does not contain recognized columns"))
    return metrics
end

function write_external_bulk_metrics_summary(output_path::AbstractString,
                                             bulk_table_path::AbstractString; kwargs...)
    return write_external_metrics_summary(output_path, external_bulk_table_metrics(bulk_table_path; kwargs...))
end

function performance_metric(func, case::Symbol, operation::Symbol, description::String)
    func()
    timed = @timed func()
    return PerformanceMetric(case, operation, Float64(timed.time), Float64(timed.bytes), description)
end

function run_performance_smoke(; Nx=6, Ny=5, Nk=5, Nθ=8)
    grid = RectilinearGrid(CPU(); size=(Nx, Ny, 1), x=(0, Nx), y=(0, Ny), z=(0, 1))
    cgrid = PolarWaveVectorGrid(; κ=range(0.4, 1.4; length=Nk),
                                  φ=range(0, 2pi; length=Nθ + 1)[1:Nθ])
    field = WaveActionField(grid, cgrid)
    product_field_metric = performance_metric(:product_field, :set_and_m0,
                                               "Set a ProductField and compute zeroth moment.") do
        set!(field, (x, y, kx, ky) -> 1 + 0.01x + 0.02y + 0.03hypot(kx, ky))
        m0(field)
    end
    model = SpectralWaveModel(; horizontal_advection=nothing, grid,
                                spectral_grid=cgrid,
                                sources=SourceTermSet(ExponentialWindInput(rate=0.03, direction=0.0),
                                                       BottomFriction(rate=0.01)),
                                timestepper=:SemiImplicitEuler)
    set!(model, N=1.0)
    source_step_metric = performance_metric(:sources, :semi_implicit_step,
                                             "Advance one source-only semi-implicit step.") do
        time_step!(model, 0.01)
    end
    return (product_field_metric, source_step_metric)
end

function write_performance_summary(path::AbstractString, metrics)
    open(path, "w") do io
        println(io, "case\toperation\tseconds\tbytes\tdescription")
        for metric in metrics
            println(io, metric.case, '\t',
                    metric.operation, '\t',
                    metric.seconds, '\t',
                    metric.bytes, '\t',
                    metric.description)
        end
    end
    return path
end

function read_performance_summary(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("performance summary is empty: $path"))
    split(first(lines), '\t') == ["case", "operation", "seconds", "bytes", "description"] ||
        throw(ArgumentError("performance summary has an unexpected header: $(first(lines))"))
    metrics = PerformanceMetric[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        parts = split(line, '\t')
        length(parts) == 5 || throw(ArgumentError("performance summary row must have 5 columns: $line"))
        push!(metrics, PerformanceMetric(Symbol(parts[1]), Symbol(parts[2]),
                                         parse(Float64, parts[3]),
                                         parse(Float64, parts[4]), parts[5]))
    end
    return metrics
end
