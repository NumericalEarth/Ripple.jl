# # CWCM Q-Transform With A Sheared Current
#
# Coupled wave-current models project a depth-dependent current through the
# vertical coordinate of the same physical `RectilinearGrid` used by the wave
# field. This source-only example isolates that projection from the
# Oceananigans WENO transport path.

using Oceananigans, Ripple, CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "..", "scripts", "example_visuals.jl"))

output_dir = example_output_directory("cwcm_q_transform_sheared_current")
plot_paths = String[]
animation_paths = String[]

grid = RectilinearGrid(CPU();
                       size=(32, 16, 32),
                       x=(0, 8),
                       y=(0, 4),
                       z=(-1, 0),
                       topology=(Periodic, Periodic, Bounded))
spectral_grid = PolarWaveVectorGrid(Float64;
                                    κ=range(0.3, 1.2; length=8),
                                    φ=range(0, 2pi; length=17)[1:16])
qtransform = QTransform(QKernel(Float64), grid)

Nx, Ny = horizontal_size(grid)
Nz = vertical_size(grid)
z = znodes(grid)
u = zeros(Nx, Ny, Nz)
v = zeros(Nx, Ny, Nz)

for k in 1:Nz, j in 1:Ny, i in 1:Nx
    shear = z[k] + 1
    u[i, j, k] = 0.1 + 0.4shear + 0.03sin(2pi * xnodes(grid)[i] / last(xfaces(grid)))
    v[i, j, k] = 0.05shear * cos(2pi * ynodes(grid)[j] / last(yfaces(grid)))
end

current = PrescribedLagrangianMeanCurrent(u=u, v=v, depth=1.0)
coupling = CWCMPrescribedCurrentCoupling(current, qtransform, spectral_grid.κ)

model = SpectralWaveModel(; grid,
                            spectral_grid,
                            coupling,
                            advection=nothing,
                            physics=RelaxationToSpectrum((x, y, kx, ky) ->
                                exp(-((x - 4)^2 + (y - 2)^2) / 2) *
                                exp(-((hypot(kx, ky) - 0.7)^2) / 0.08);
                                timescale=0.5),
                            timestepper=:ForwardEuler)
set!(model, N=0.0)

frames = [field_m0_matrix(model.action)]
step_count = example_mode() == :small ? 6 : 36
for n in 1:step_count
    time_step!(model, 0.01)
    update_coupling!(model)
    if n == step_count || n % max(1, step_count ÷ 6) == 0
        push!(frames, field_m0_matrix(model.action))
    end
end

push!(plot_paths,
      write_line(joinpath(output_dir, "vertical_current_profile.png"),
                 z, (vec(u[1, 1, :]), vec(v[1, 1, :]));
                 title="Input Lagrangian-mean current",
                 xlabel="z", ylabel="velocity",
                 names=("u", "v")))

push!(animation_paths,
      write_heatmap_animation(joinpath(output_dir, "source_spinup_under_shear_projection.mp4"),
                              frames;
                              title="Source spinup with Q-projected current cache",
                              xlabel="x cell", ylabel="y cell"))

@assert all(isfinite, interior(model.action))
@assert all(isfinite, coupling.Ux)
@assert all(isfinite, coupling.Uy)
@assert all(isfile, plot_paths)
@assert all(isfile, animation_paths)

@show coupling.Ux[1, 1, 1]
@show coupling.Uy[1, 1, 1]
@show total_action(model.action)
@show plot_paths
@show animation_paths
