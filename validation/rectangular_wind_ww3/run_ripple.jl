##### Rectangular-wind WW3 comparison — Ripple side.
#####
##### Reads wind fields from data/wind/TestXX_2D.nc (prescribed spatially
##### varying U10), runs Ripple on the same 2-D domain, and writes hourly
##### Hs(x,y,t) to output/TestXX_2D_ripple.nc for comparison with the WW3
##### reference in data/reference/TestXX_2D.nc.
#####
##### Run with:
#####   julia --project=. validation/rectangular_wind_ww3/run_ripple.jl

using Ripple
using Oceananigans: compute!
using NCDatasets
using Printf

include(joinpath(@__DIR__, "gridded_wind.jl"))

const DIR      = @__DIR__
const WIND_DIR = joinpath(DIR, "data", "wind")
const OUT_DIR  = joinpath(DIR, "output")
mkpath(OUT_DIR)

# WW3-standard spectral grid: 25 log-spaced frequencies × 24 directions.
const NFREQ = 25
const NDIR  = 24
const F0    = 0.04118
const FR    = 1.1
const G     = 9.81

const FREQ_CENTERS = [F0 * FR^(k-1) for k in 1:NFREQ]
const DIR_CENTERS  = collect(range(0, 2π * (NDIR - 1) / NDIR; length=NDIR))

# Maximum deep-water group velocity (at lowest frequency f0): cg = g/(4π f0).
const CG_MAX = G / (4π * F0)

# ─── helpers ────────────────────────────────────────────────────────────────

# Config is read from NC global attributes (mirrors the companion .json files).
function load_case_config(nc_path)
    NCDataset(nc_path) do ds
        a = ds.attrib
        return (
            Lx        = Float64(a["Lx"]),
            Ly        = Float64(a["Ly"]),
            Nx        = Int(a["Nx"]),
            Ny        = Int(a["Ny"]),
            dx        = Float64(a["dx"]),
            dy        = Float64(a["dy"]),
            T         = Float64(a["T"]),
            long_name = String(a["long_name"]),
        )
    end
end

function run_case(case_id)
    nc_path = joinpath(WIND_DIR, "$(case_id).nc")
    cfg = load_case_config(nc_path)
    Lx      = cfg.Lx
    Ly      = cfg.Ly
    Nx      = cfg.Nx
    Ny      = cfg.Ny
    dx      = cfg.dx
    dy      = cfg.dy
    T_total = cfg.T

    @printf "─── %s: %s ───\n" case_id (cfg.long_name)
    @printf "  Grid: %d×%d, dx=%.0f m, dy=%.0f m, T=%.0f h\n" Nx Ny dx dy (T_total/3600)

    # Physical grid: cell centers match the 51×51 wind data grid exactly.
    # Faces are placed ±dx/2 outside the data boundary so that cell centers
    # land on the data grid points 0, dx, …, (Nx-1)*dx.
    grid = RectilinearGrid(CPU();
                           size  = (Nx, Ny, 1),
                           x     = (-dx/2, Lx + dx/2),
                           y     = (-dy/2, Ly + dy/2),
                           z     = (0.0, 1.0),
                           halo  = (3, 3, 3),
                           topology = (Bounded, Bounded, Bounded))

    spectral_grid = FrequencyDirectionGrid(; frequency=FREQ_CENTERS, φ=DIR_CENTERS)

    wind = GriddedWind(nc_path)

    wind_input  = PressureCorrelationInput(; drag=BulkWindDrag(:linear), wind)
    dissipation = LocalSaturationDissipation(; B_r=1.05e-2, σ_power=1.0)
    nonlinear   = DiscreteInteractionApproximation(SymmetricQuadruplet())
    physics     = PrecomputedSources(; wind_input, dissipation, nonlinear)

    model = SpectralWaveModel(grid, spectral_grid;
                              sources     = physics,
                              timestepper = :LowStorageRK3)

    set!(model, N=1e-6)

    # Simulation timestep: 70% of the advective CFL limit at the maximum
    # deep-water group velocity (cg at the lowest frequency).
    dt_sim = 0.7 * min(dx, dy) / CG_MAX
    dt_sim = max(dt_sim, 1.0)   # never below 1 s

    @printf "  dt_sim = %.0f s\n" dt_sim

    OUTPUT_INTERVAL = 3600.0   # hourly snapshots to match WW3 reference
    n_out = round(Int, T_total / OUTPUT_INTERVAL)

    xs = xnodes(grid)
    ys = ynodes(grid)
    hs_out = zeros(Float64, Nx, Ny, n_out)
    t_out  = zeros(Float64, n_out)

    Hs_field = significant_wave_height(model.action)

    next_out = OUTPUT_INTERVAL
    out_idx  = 0

    while model.clock.time < T_total - dt_sim/2
        time_step!(model, dt_sim)

        if model.clock.time >= next_out - dt_sim/2
            out_idx += 1
            compute!(Hs_field)
            hs_out[:, :, out_idx] = Float64.(interior(Hs_field)[:, :, 1])
            t_out[out_idx] = model.clock.time
            @printf "  t=%6.1f h  max(Hs)=%5.3f m\n" (model.clock.time/3600) maximum(hs_out[:,:,out_idx])
            next_out += OUTPUT_INTERVAL
            out_idx >= n_out && break
        end
    end

    # Trim if any outputs were missed.
    n_written = out_idx
    hs_out = hs_out[:, :, 1:n_written]
    t_out  = t_out[1:n_written]

    out_path = joinpath(OUT_DIR, "$(case_id)_ripple.nc")
    _write_output(out_path, xs, ys, t_out, hs_out, case_id, cfg.long_name)

    @printf "  Wrote %s\n\n" out_path
    return nothing
end

function _write_output(path, xs, ys, t_sec, hs, case_id, long_name)
    Nx, Ny, Nt = size(hs)
    NCDataset(path, "c") do ds
        ds.attrib["case_id"]   = case_id
        ds.attrib["long_name"] = long_name
        ds.attrib["model"]     = "Ripple"

        defDim(ds, "x", Nx)
        defDim(ds, "y", Ny)
        defDim(ds, "time", Nt)

        vx = defVar(ds, "x", Float64, ("x",))
        vx.attrib["units"] = "m"
        vx[:] = xs

        vy = defVar(ds, "y", Float64, ("y",))
        vy.attrib["units"] = "m"
        vy[:] = ys

        vt = defVar(ds, "time", Float64, ("time",))
        vt.attrib["units"] = "seconds since simulation start"
        vt[:] = t_sec

        vh = defVar(ds, "hs", Float64, ("x", "y", "time"))
        vh.attrib["long_name"] = "significant wave height"
        vh.attrib["units"]     = "m"
        vh[:, :, :] = hs
    end
end

# ─── main ───────────────────────────────────────────────────────────────────

for i in 1:7
    case_id = @sprintf "Test%02d_2D" i
    run_case(case_id)
end

println("All cases complete. Outputs in $OUT_DIR")
