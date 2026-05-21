# # Monobanded Linear-Shear Refraction
#
# This example validates `MonobandedWaveModel` against a quasi-analytic
# solution. A compact monobanded wave-action packet propagates through a
# barotropic linear shear current
#
# ```math
# u^D_x(y) = U_0 + S (y - y_c), \qquad u^D_y = 0 .
# ```
#
# For a depth-independent current, the Q projection leaves the current
# unchanged and ``H = \partial u^D / \partial \kappa = 0``. The fixed-κ current
# gradient is spatially constant:
#
# ```math
# \Gamma_{xy} = \partial_y u^D_x = S .
# ```
#
# Starting from uniform wavevector ``K(0) = (K_{x0}, K_{y0})``, the monobanded
# moment equation predicts
#
# ```math
# K_x(t) = K_{x0}, \qquad
# K_y(t) = K_{y0} - S K_{x0} t .
# ```
#
# Since ``K(t)`` stays spatially uniform, the packet position follows an
# analytic ray map. We use that map to overlay reference contours in the movie
# and to compute quantitative errors.

using Oceananigans, Ripple
using CairoMakie
using Printf

CairoMakie.activate!(type = "png")

output_dir = get(ENV, "RIPPLE_EXAMPLE_OUTPUT_DIR", pwd())
mkpath(output_dir)

# ## Grid and current
#
# The monobanded transport discretization currently assumes uniform horizontal
# spacing. We use a periodic x direction so the packet can translate
# freely, and a bounded y direction. The packet remains well away from the y
# boundaries over the run, so the reference solution is the unbounded linear
# shear solution. Bump `Nx`, `Ny`, the frame count, and lower `dt` for a
# higher-fidelity reproduction.

Nx = 48
Ny = 36
Nz = 4

Lx = 72.0
Ly = 48.0
Lz = 1.0

grid = RectilinearGrid(CPU();
                       size     = (Nx, Ny, Nz),
                       halo     = (3, 3, 3),
                       x        = (0, Lx),
                       y        = (0, Ly),
                       z        = (-Lz, 0),
                       topology = (Periodic, Bounded, Bounded))

x_nodes = collect(xnodes(grid))
y_nodes = collect(ynodes(grid))

U0 = 0.10
S = 0.045
yc = Ly / 2

u = Oceananigans.Fields.CenterField(grid)
v = Oceananigans.Fields.CenterField(grid)
set!(u, (x, y, z) -> U0 + S * (y - yc))
set!(v, 0)

# ## Model and initial state

Kx0 = 0.50
Ky0 = 0.18

x0_packet = 18.0
y0_packet = 17.0
σx = 4.2
σy = 3.2

periodic_displacement(x, x0, L) = mod(x - x0 + L / 2, L) - L / 2

function initial_action(x, y)
    dx = periodic_displacement(x, x0_packet, Lx)
    dy = y - y0_packet
    return exp(-0.5 * ((dx / σx)^2 + (dy / σy)^2))
end

model = MonobandedWaveModel(grid;
                            velocities       = PrescribedVelocities(; u, v),
                            advection        = WENO(order = 5),
                            timestepper = :RungeKutta3,
                            minimum_action   = 1e-8,
                            minimum_wavenumber = 0.05)

set!(model;
     A   = (x, y, z) -> initial_action(x, y),
     AKx = (x, y, z) -> Kx0 * initial_action(x, y),
     AKy = (x, y, z) -> Ky0 * initial_action(x, y))

nothing #hide

# ## Quasi-analytic reference
#
# With ``\kappa(t) = |K(t)|`` and
# ``\alpha(t) = \sqrt{g / [4\kappa(t)^3]}``, the ray velocity is
#
# ```math
# \dot y = \alpha(t) K_y(t),
# ```
#
# ```math
# \dot x = \alpha(t) K_x + U_0 + S (y - y_c).
# ```
#
# Define ``\eta(t) = \int_0^t \dot y(s) ds``,
# ``\xi(t) = \int_0^t [\alpha(s) K_x + U_0] ds``, and
# ``I_\eta(t) = \int_0^t \eta(s) ds``. The inverse ray map is
#
# ```math
# y_0 = y - \eta(t),
# ```
#
# ```math
# x_0 = x - \xi(t) - S t (y_0 - y_c) - S I_\eta(t).
# ```

dt = 0.04
stop_time = 3.2
step_count = round(Int, stop_time / dt)
frame_count = 12
frame_stride = max(1, step_count ÷ frame_count)

times_full = collect((0:step_count) .* dt)
g = model.gravitational_acceleration

Ky_reference(t) = Ky0 - S * Kx0 * t
κ_reference(t) = hypot(Kx0, Ky_reference(t))
α_reference(t) = sqrt(g / (4 * κ_reference(t)^3))
Cx_base_reference(t) = α_reference(t) * Kx0 + U0
Cy_reference(t) = α_reference(t) * Ky_reference(t)

function cumulative_trapezoid(values, Δt)
    out = zeros(eltype(values), length(values))
    for n in 2:length(values)
        out[n] = out[n-1] + (values[n-1] + values[n]) * Δt / 2
    end
    return out
end

cx_base_full = Cx_base_reference.(times_full)
cy_full = Cy_reference.(times_full)
η_full = cumulative_trapezoid(cy_full, dt)
ξ_full = cumulative_trapezoid(cx_base_full, dt)
Iη_full = cumulative_trapezoid(η_full, dt)

function reference_action(step)
    t = times_full[step + 1]
    η = η_full[step + 1]
    ξ = ξ_full[step + 1]
    Iη = Iη_full[step + 1]

    return [begin
                y_initial = y - η
                x_initial = x - ξ - S * t * (y_initial - yc) - S * Iη
                initial_action(x_initial, y_initial)
            end for x in x_nodes, y in y_nodes]
end

reference_AKx(Aref, step) = Kx0 .* Aref
reference_AKy(Aref, step) = Ky_reference(times_full[step + 1]) .* Aref

# ## Diagnostics helpers

slab(field) = Array(interior(field))[:, :, 1]

function weighted_mean(field_data, weights)
    total_weight = sum(weights)
    return sum(field_data .* weights) / total_weight
end

function centroid(A)
    total = sum(A)
    xbar = sum(A .* reshape(x_nodes, :, 1)) / total
    ybar = sum(A .* reshape(y_nodes, 1, :)) / total
    return xbar, ybar
end

relative_l2(a, b) = sqrt(sum(abs2, a .- b) / max(sum(abs2, b), eps(eltype(b))))

function snapshot(step)
    A = slab(model.action)
    AKx = slab(model.wavenumber_moment.x)
    AKy = slab(model.wavenumber_moment.y)
    Kx = sum(AKx) / sum(A)
    Ky = sum(AKy) / sum(A)
    xbar, ybar = centroid(A)

    Aref = reference_action(step)
    AKxref = reference_AKx(Aref, step)
    AKyref = reference_AKy(Aref, step)
    xref, yref = centroid(Aref)

    return (time = model.clock.time,
            A = A,
            Aref = Aref,
            A_error = A .- Aref,
            Kx = Kx,
            Ky = Ky,
            Kx_ref = Kx0,
            Ky_ref = Ky_reference(model.clock.time),
            x = xbar,
            y = ybar,
            x_ref = xref,
            y_ref = yref,
            mass = sum(A),
            A_l2 = relative_l2(A, Aref),
            AKx_l2 = relative_l2(AKx, AKxref),
            AKy_l2 = relative_l2(AKy, AKyref))
end

frames = [snapshot(0)]

for step in 1:step_count
    time_step!(model, dt)
    if step == step_count || step % frame_stride == 0
        push!(frames, snapshot(step))
    end
end

# ## Quantitative checks

initial_mass = first(frames).mass
mass_error = maximum(abs(frame.mass / initial_mass - 1) for frame in frames)
Kx_error = maximum(abs(frame.Kx - frame.Kx_ref) for frame in frames)
Ky_error = maximum(abs(frame.Ky - frame.Ky_ref) for frame in frames)
centroid_error = maximum(hypot(frame.x - frame.x_ref, frame.y - frame.y_ref) for frame in frames)
final_A_l2 = last(frames).A_l2

Γxy = slab(model.diagnostics.Γxy)
weighted_Γxy = weighted_mean(Γxy, last(frames).A)
Γxy_error = abs(weighted_Γxy - S)

@printf("Monobanded linear-shear refraction validation\n")
@printf("  frames: %d, dt: %.4f s, stop_time: %.2f s\n", length(frames), dt, model.clock.time)
@printf("  max relative action-mass error: %.3e\n", mass_error)
@printf("  max |weighted Kx - Kx_ref|: %.3e\n", Kx_error)
@printf("  max |weighted Ky - Ky_ref|: %.3e\n", Ky_error)
@printf("  max centroid error: %.3e m\n", centroid_error)
@printf("  final relative L2(A - Aref): %.3e\n", final_A_l2)
@printf("  |weighted Γxy - S| at final time: %.3e s^-1\n", Γxy_error)

# ## Movie
#
# The left panel shows the numerical action with dashed analytic contours. The
# middle panel shows the signed error. The right panel tracks the action-weighted
# wavevector against the analytic prediction.

A_limits = (0, maximum(maximum(frame.A) for frame in frames))
error_limit = maximum(maximum(abs, frame.A_error) for frame in frames)
error_limits = (-error_limit, error_limit)

A_obs = Observable(first(frames).A)
Aref_obs = Observable(first(frames).Aref)
err_obs = Observable(first(frames).A_error)
title_obs = Observable(@sprintf("t = %.2f s", first(frames).time))
model_centroid_obs = Observable(Point2f(first(frames).x, first(frames).y))
ref_centroid_obs = Observable(Point2f(first(frames).x_ref, first(frames).y_ref))
Kx_series_obs = Observable([first(frames).Kx])
Ky_series_obs = Observable([first(frames).Ky])
Kx_ref_series_obs = Observable([first(frames).Kx_ref])
Ky_ref_series_obs = Observable([first(frames).Ky_ref])

fig = Figure(size = (1320, 460))
ax1 = Axis(fig[1, 1]; title = "A with reference contours",
                       xlabel = "x", ylabel = "y", aspect = DataAspect())
ax2 = Axis(fig[1, 3]; title = "A - Aref",
                       xlabel = "x", ylabel = "y", aspect = DataAspect())
ax3 = Axis(fig[1, 5]; title = "Action-weighted K(t)",
                       xlabel = "Kx", ylabel = "Ky", aspect = DataAspect())

hm1 = heatmap!(ax1, x_nodes, y_nodes, A_obs; colormap = :viridis, colorrange = A_limits)
contour!(ax1, x_nodes, y_nodes, Aref_obs; levels = [0.2, 0.5, 0.8],
         color = :white, linestyle = :dash, linewidth = 2)
scatter!(ax1, model_centroid_obs; marker = :circle, color = :black, markersize = 10)
scatter!(ax1, ref_centroid_obs; marker = :xcross, color = :white, markersize = 14)

hm2 = heatmap!(ax2, x_nodes, y_nodes, err_obs; colormap = :balance, colorrange = error_limits)

lines!(ax3, Kx_ref_series_obs, Ky_ref_series_obs; color = :black, linestyle = :dash,
       label = "reference")
lines!(ax3, Kx_series_obs, Ky_series_obs; color = :dodgerblue3, linewidth = 3,
       label = "model")
axislegend(ax3; position = :lb)
xlims!(ax3, Kx0 - 0.04, Kx0 + 0.04)
ky_min = minimum(min(frame.Ky, frame.Ky_ref) for frame in frames)
ky_max = maximum(max(frame.Ky, frame.Ky_ref) for frame in frames)
ylims!(ax3, ky_min - 0.03, ky_max + 0.03)

Colorbar(fig[1, 2], hm1)
Colorbar(fig[1, 4], hm2)
Label(fig[0, :], title_obs; fontsize = 18, halign = :center)

movie_path = joinpath(output_dir, "monobanded_linear_shear_refraction.mp4")
record(fig, movie_path, eachindex(frames); framerate = 12) do idx
    frame = frames[idx]
    A_obs[] = frame.A
    Aref_obs[] = frame.Aref
    err_obs[] = frame.A_error
    title_obs[] = @sprintf("Monobanded refraction through linear shear: t = %.2f s", frame.time)
    model_centroid_obs[] = Point2f(frame.x, frame.y)
    ref_centroid_obs[] = Point2f(frame.x_ref, frame.y_ref)
    Kx_series_obs[] = [frames[n].Kx for n in 1:idx]
    Ky_series_obs[] = [frames[n].Ky for n in 1:idx]
    Kx_ref_series_obs[] = [frames[n].Kx_ref for n in 1:idx]
    Ky_ref_series_obs[] = [frames[n].Ky_ref for n in 1:idx]
end

# ## Final comparison figure

final = last(frames)
fig_final = Figure(size = (1260, 380))
fax1 = Axis(fig_final[1, 1]; title = "Numerical A", xlabel = "x", ylabel = "y",
                                aspect = DataAspect())
fax2 = Axis(fig_final[1, 3]; title = "Reference A", xlabel = "x", ylabel = "y",
                                aspect = DataAspect())
fax3 = Axis(fig_final[1, 5]; title = "A - Aref", xlabel = "x", ylabel = "y",
                                aspect = DataAspect())
fhm1 = heatmap!(fax1, x_nodes, y_nodes, final.A; colormap = :viridis, colorrange = A_limits)
fhm2 = heatmap!(fax2, x_nodes, y_nodes, final.Aref; colormap = :viridis, colorrange = A_limits)
fhm3 = heatmap!(fax3, x_nodes, y_nodes, final.A_error; colormap = :balance, colorrange = error_limits)
Colorbar(fig_final[1, 2], fhm1)
Colorbar(fig_final[1, 4], fhm2)
Colorbar(fig_final[1, 6], fhm3)
Label(fig_final[0, :],
      @sprintf("Final comparison: relative L2(A - Aref) = %.3e", final.A_l2);
      fontsize = 18)

final_path = joinpath(output_dir, "monobanded_linear_shear_refraction_final.png")
save(final_path, fig_final)

# ## Diagnostic figure

times = [frame.time for frame in frames]
mass = [frame.mass / initial_mass - 1 for frame in frames]
Kx_model = [frame.Kx for frame in frames]
Ky_model = [frame.Ky for frame in frames]
Kx_ref = [frame.Kx_ref for frame in frames]
Ky_ref = [frame.Ky_ref for frame in frames]
x_model = [frame.x for frame in frames]
y_model = [frame.y for frame in frames]
x_ref = [frame.x_ref for frame in frames]
y_ref = [frame.y_ref for frame in frames]
A_l2 = [frame.A_l2 for frame in frames]
AKx_l2 = [frame.AKx_l2 for frame in frames]
AKy_l2 = [frame.AKy_l2 for frame in frames]

fig_diag = Figure(size = (1180, 820))
dax1 = Axis(fig_diag[1, 1]; title = "Action conservation",
                                xlabel = "t (s)", ylabel = "relative mass error")
dax2 = Axis(fig_diag[1, 2]; title = "Wavevector refraction",
                                xlabel = "t (s)", ylabel = "K")
dax3 = Axis(fig_diag[2, 1]; title = "Packet centroid",
                                xlabel = "x", ylabel = "y", aspect = DataAspect())
dax4 = Axis(fig_diag[2, 2]; title = "Relative L2 errors",
                                xlabel = "t (s)", ylabel = "relative L2")

lines!(dax1, times, mass; color = :black)
lines!(dax2, times, Kx_ref; color = :black, linestyle = :dash, label = "Kx ref")
lines!(dax2, times, Kx_model; color = :dodgerblue3, label = "Kx model")
lines!(dax2, times, Ky_ref; color = :gray35, linestyle = :dash, label = "Ky ref")
lines!(dax2, times, Ky_model; color = :orange2, label = "Ky model")
axislegend(dax2; position = :lb)
lines!(dax3, x_ref, y_ref; color = :black, linestyle = :dash, label = "reference")
lines!(dax3, x_model, y_model; color = :dodgerblue3, linewidth = 3, label = "model")
axislegend(dax3; position = :lt)
lines!(dax4, times, A_l2; color = :black, label = "A")
lines!(dax4, times, AKx_l2; color = :dodgerblue3, label = "AKx")
lines!(dax4, times, AKy_l2; color = :orange2, label = "AKy")
axislegend(dax4; position = :lt)

diagnostics_path = joinpath(output_dir, "monobanded_linear_shear_refraction_diagnostics.png")
save(diagnostics_path, fig_diag)

@printf("  wrote %s\n", movie_path)
@printf("  wrote %s\n", final_path)
@printf("  wrote %s\n", diagnostics_path)

nothing #hide

# ![](monobanded_linear_shear_refraction.mp4)
