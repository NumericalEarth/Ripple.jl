# Two-way coupled Langmuir turbulence: an Oceananigans NonhydrostaticModel whose
# Stokes drift is supplied by an evolving NarrowBandWaveModel amplitude field.
# Setup follows McWilliams et al. (1997) / Wagner et al. (2021), as in the
# Oceananigans langmuir_turbulence.jl example, but the wave field is prognostic.

using Ripple, Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf
using Statistics: std
CairoMakie.activate!(type = "png")

arch = GPU()
Nx = Ny = 96
Nz = 48
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(120, 120, 48),
                       halo=(3, 3, 3), topology=(Periodic, Periodic, Bounded))

# --- surface wave / carrier parameters (monochromatic, wavelength 60 m) ---
wavelength = 60.0
κ = 2π / wavelength

# --- forcing (Wagner 2021) ---
τx = -3.72e-5           # surface momentum flux (m² s⁻²)
Jᵇ = 2.307e-8           # surface buoyancy flux (m² s⁻³)
N² = 1.936e-5           # bottom / initial stratification (s⁻²)
f  = 1e-4

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵇ),
                                bottom = GradientBoundaryCondition(N²))

# --- coupled model ---
stokes = NarrowBandStokesDrift(grid)
ocean = NonhydrostaticModel(grid;
                            coriolis = FPlane(; f),
                            advection = WENO(),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            stokes_drift = stokes,
                            boundary_conditions = (u=u_bcs, b=b_bcs),
                            timestepper = :RungeKutta3)

wave = NarrowBandWaveModel(grid; κ,
                           velocities = (u = ocean.velocities.u, v = ocean.velocities.v),
                           advection = WENO())

wcm = WaveCurrentModel(wave, ocean, stokes)
d = wave.dispersion
@info "carrier" κ ω=d.ω group_velocity=d.ω_κ

# --- initial conditions ---
# Wave amplitude giving a 0.8 m surface-elevation envelope (⇒ Uˢ(0) ≈ 0.068 m/s).
a = 0.8 * d.gravity / (2 * d.ω * d.C)
set!(wave; A = (x, y) -> a * cis(κ * x))

Ξ(z) = randn() * exp(z / 4)
mixed_layer_depth = 33.0
stratification(z) = z < -mixed_layer_depth ? N² * z : N² * (-mixed_layer_depth)
bᵢ(x, y, z) = stratification(z) + 1e-1 * Ξ(z) * N² * grid.Lz
u★ = sqrt(abs(τx))
uᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
set!(ocean, u=uᵢ, w=uᵢ, b=bᵢ)

initialize_coupling!(wcm)
@info "surface Stokes drift" Uˢ_surface=maximum(Array(interior(wcm.uˢ))[:, :, end])

# --- coupled time stepping ---
Δt = 1.0
stop_time = 90minutes
Nsteps = round(Int, stop_time / Δt)
sample_every = round(Int, 5minutes / Δt)

times = Float64[]
rms_w = Float64[]           # Langmuir intensity
Astruct = Float64[]         # std(|A|)/mean(|A|): wave-field structuring
w_xy_frames = Array{Float64,2}[]
w_xz_frames = Array{Float64,2}[]
A_frames = Array{Float64,2}[]

wint = interior(ocean.velocities.w)
k_surf = Nz                       # near-surface index for w slices (w at faces)
j_mid = Ny ÷ 2

function record_snapshot!()
    push!(times, ocean.clock.time / 60)   # minutes
    w = Array(interior(ocean.velocities.w))
    push!(rms_w, sqrt(sum(w .^ 2) / length(w)))
    Amag = abs.(amplitude(wave))[:, :, 1]
    push!(Astruct, std(Amag) / (sum(Amag) / length(Amag)))
    push!(w_xy_frames, w[:, :, max(k_surf - 6, 1)])   # a few m below surface
    push!(w_xz_frames, w[:, j_mid, :])
    push!(A_frames, Amag)
    return nothing
end

record_snapshot!()
t0 = time()
for n in 1:Nsteps
    coupled_time_step!(wcm, Δt)
    if n % sample_every == 0 || n == Nsteps
        record_snapshot!()
        @info @sprintf("step %d/%d  t=%.0f min  rms(w)=%.2e  A-struct=%.3f  max|u|=%.3f",
                       n, Nsteps, ocean.clock.time/60, rms_w[end], Astruct[end],
                       maximum(abs, Array(interior(ocean.velocities.u))))
        flush(stdout)
    end
end
@info "done stepping" wallclock_s=round(time()-t0, digits=1)

# --- figures ---
xc = Array(xnodes(grid, Center()))
yc = Array(ynodes(grid, Center()))
zf = Array(znodes(grid, Face()))
zc = Array(znodes(grid, Center()))

fig = Figure(size = (1200, 900))

axw_xy = Axis(fig[1, 1]; title="w (m/s) at z ≈ $(round(zf[max(k_surf-6,1)];digits=1)) m  [final]",
              xlabel="x (m)", ylabel="y (m)", aspect=1)
wxy = last(w_xy_frames); wlim = maximum(abs, wxy) + eps()
hm1 = heatmap!(axw_xy, xc, yc, wxy; colormap=:balance, colorrange=(-wlim, wlim))
Colorbar(fig[1, 2], hm1)

axw_xz = Axis(fig[1, 3]; title="w (m/s) at y=$(round(yc[j_mid];digits=0)) m  [final]",
              xlabel="x (m)", ylabel="z (m)")
wxz = last(w_xz_frames); wlim2 = maximum(abs, wxz) + eps()
hm2 = heatmap!(axw_xz, xc, zf, wxz; colormap=:balance, colorrange=(-wlim2, wlim2))
Colorbar(fig[1, 4], hm2)

axA = Axis(fig[2, 1]; title="wave-height envelope |A| [final]",
           xlabel="x (m)", ylabel="y (m)", aspect=1)
hmA = heatmap!(axA, xc, yc, last(A_frames); colormap=:viridis)
Colorbar(fig[2, 2], hmA)

axts = Axis(fig[2, 3]; title="Langmuir spin-up & wave-field structuring",
            xlabel="time (min)", ylabel="rms(w) (m/s)")
lines!(axts, times, rms_w; color=:dodgerblue, label="rms(w)")
axislegend(axts; position=:lt)
axts2 = Axis(fig[2, 3]; yaxisposition=:right, ylabel="std|A| / mean|A|")
hidespines!(axts2); hidexdecorations!(axts2)
lines!(axts2, times, Astruct; color=:darkorange, label="A structure")
axislegend(axts2; position=:rb)

save("langmuir_two_way.png", fig)
@info "saved figure" path="langmuir_two_way.png"

# animation of the vertical-velocity streaks
figa = Figure(size=(560, 520))
axa = Axis(figa[1, 1]; title="Langmuir cells: w near surface", xlabel="x (m)", ylabel="y (m)", aspect=1)
obs = Observable(first(w_xy_frames))
wlima = maximum(maximum.(abs, w_xy_frames)) + eps()
hma = heatmap!(axa, xc, yc, obs; colormap=:balance, colorrange=(-wlima, wlima))
Colorbar(figa[1, 2], hma)
CairoMakie.record(figa, "langmuir_two_way.mp4", eachindex(w_xy_frames); framerate=6) do i
    obs[] = w_xy_frames[i]
end
@info "LANGMUIR DONE"
