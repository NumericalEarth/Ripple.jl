using CairoMakie

function example_mode()
    return get(ENV, "RIPPLE_EXAMPLE_MODE", "") == "small" ? :small : :normal
end

function example_output_directory(example_name)
    root = get(ENV, "RIPPLE_EXAMPLE_OUTPUT_DIR",
               joinpath(tempdir(), "ripple_example_outputs"))
    directory = joinpath(root, string(example_name))
    mkpath(directory)
    return directory
end

visual_matrix(matrix) = Matrix(matrix)
visual_matrix(field::ProductField) = field_m0_matrix(field)
visual_matrix(model::SpectralWaveModel) = field_m0_matrix(model.action)
makie_heatmap_argument(matrix) = matrix isa Union{ProductField, SpectralWaveModel} ? matrix : Matrix(matrix)

function write_heatmap(path, matrix; title="Heatmap", xlabel="", ylabel="", width=720, height=420)
    fig = Figure(size=(width, height))
    ax = Axis(fig[1, 1]; title, xlabel, ylabel)
    heatmap_plot = heatmap!(ax, makie_heatmap_argument(matrix))
    Colorbar(fig[1, 2], heatmap_plot)
    save(path, fig)
    return path
end

function write_heatmap_animation(path, frames; title="Animation", xlabel="", ylabel="",
                                 width=720, height=420, fps=4)
    frame_matrices = [visual_matrix(frame) for frame in frames]
    isempty(frame_matrices) && error("write_heatmap_animation requires at least one frame")

    fig = Figure(size=(width, height))
    ax = Axis(fig[1, 1]; title, xlabel, ylabel)
    frame = Observable(first(frame_matrices))
    heatmap_plot = heatmap!(ax, frame)
    Colorbar(fig[1, 2], heatmap_plot)

    record(fig, path, eachindex(frame_matrices); framerate=fps) do index
        frame[] = frame_matrices[index]
    end

    return path
end

function write_line(path, xs, series; title="Line plot", xlabel="", ylabel="",
                    names=nothing, width=720, height=420)
    fig = Figure(size=(width, height))
    ax = Axis(fig[1, 1]; title, xlabel, ylabel)
    xvalues = collect(xs)

    for (index, values) in enumerate(series)
        label = names === nothing ? nothing : string(names[index])
        lines!(ax, xvalues, collect(values); label)
    end

    names === nothing || axislegend(ax)
    save(path, fig)
    return path
end

function write_line_animation(path, xs, yframes; title="Animated line plot", xlabel="", ylabel="",
                              width=720, height=420, fps=4)
    frames = [collect(frame) for frame in yframes]
    isempty(frames) && error("write_line_animation requires at least one frame")

    fig = Figure(size=(width, height))
    ax = Axis(fig[1, 1]; title, xlabel, ylabel)
    xvalues = collect(xs)
    y = Observable(first(frames))
    lines!(ax, xvalues, y)

    record(fig, path, eachindex(frames); framerate=fps) do index
        y[] = frames[index]
    end

    return path
end

function field_m0_matrix(field)
    return permutedims(dropdims(Array(interior(m0(field))); dims=3))
end

# Materialize a 2D-slab `Field` (size Nx × Ny × 1) into a plain Matrix on the
# host. Useful for routing diagnostic Fields through CairoMakie utilities.
diagnostic_field_matrix(f) = Array(interior(f))[:, :, 1]

function column_spectrum_matrix(field; i=1, j=1)
    _, _, Nxi, Neta = size(field)
    return [field[i, j, m, n] for n in 1:Neta, m in 1:Nxi]
end

function x_kappa_phase_space_matrix(field; j=1, n=1)
    Nx, _, Nxi, _ = size(field)
    return [field[i, j, m, n] for m in 1:Nxi, i in 1:Nx]
end
