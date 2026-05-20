##### Rectangular-wind WW3 comparison — Ripple vs WW3.
#####
##### Reads:
#####   output/TestXX_2D_ripple.nc  — Hs(x,y,t) from run_ripple.jl
#####   data/reference/TestXX_2D.nc — Hs(x,y,t) from WW3 reference run
#####
##### Writes plots to output/compare/:
#####   hs_timeseries.png  — spatial-mean Hs(t) for all 7 cases
#####   TestXX_2D_maps.png — final-time Hs spatial maps, one file per case
#####
##### Run after run_ripple.jl has produced output.

using NCDatasets
using Dates
using CairoMakie
using Printf
using Statistics

const DIR     = @__DIR__
const REF_DIR = joinpath(DIR, "data", "reference")
const OUT_DIR = joinpath(DIR, "output")
const PLT_DIR = joinpath(OUT_DIR, "compare")
mkpath(PLT_DIR)

# ─── loaders ────────────────────────────────────────────────────────────────

function load_ripple(case_id)
    path = joinpath(OUT_DIR, "$(case_id)_ripple.nc")
    NCDataset(path) do ds
        x   = Float64.(ds["x"][:])
        y   = Float64.(ds["y"][:])
        t   = Float64.(ds["time"][:])
        hs  = Float64.(ds["hs"][:, :, :])
        return (; x, y, t, hs)
    end
end

function load_ww3(case_id)
    path = joinpath(REF_DIR, "$(case_id).nc")
    NCDataset(path) do ds
        x  = Float64.(ds["x"][:])
        y  = Float64.(ds["y"][:])
        t_raw = ds["time"][:]
        t = [Float64(Dates.value(ti - t_raw[1])) / 1000.0 for ti in t_raw]
        hs_raw = ds["hs"][:, :, :]
        hs = Float64.(coalesce.(hs_raw, NaN))
        return (; x, y, t, hs)
    end
end

function spatial_mean_hs(hs)
    [mean(filter(!isnan, vec(hs[:, :, ti]))) for ti in axes(hs, 3)]
end

# ─── metrics ────────────────────────────────────────────────────────────────

function compare_case(case_id)
    rip = load_ripple(case_id)
    ww3 = load_ww3(case_id)

    rip_mean = spatial_mean_hs(rip.hs)
    ww3_mean = spatial_mean_hs(ww3.hs)

    # Align in time: Ripple outputs at exact integer hours, WW3 similarly.
    n = min(length(rip_mean), length(ww3_mean))
    rip_t_h = rip.t[1:n] ./ 3600
    rip_m   = rip_mean[1:n]
    ww3_m   = ww3_mean[1:n]

    rmse = sqrt(mean((rip_m .- ww3_m).^2))
    bias = mean(rip_m .- ww3_m)

    return (; rip, ww3, rip_t_h, rip_m, ww3_m, rmse, bias, n, case_id)
end

# ─── plots ──────────────────────────────────────────────────────────────────

function plot_timeseries(results)
    n = length(results)
    fig = Figure(size=(1200, 160 * n))
    for (row, res) in enumerate(results)
        ax = Axis(fig[row, 1];
                  xlabel    = row == n ? "time (h)" : "",
                  ylabel    = "⟨Hs⟩ (m)",
                  title     = "$(res.case_id): $(res.rip.hs |> size |> first) WW3 ref vs Ripple",
                  titlesize = 12)
        lines!(ax, res.rip_t_h, res.rip_m; label="Ripple", linewidth=2)
        ww3_t_h = res.ww3.t[1:res.n] ./ 3600
        lines!(ax, ww3_t_h, res.ww3_m; label="WW3 ref", linewidth=2, linestyle=:dash)
        axislegend(ax; position=:rb, labelsize=10)
        text!(ax, 0.02, 0.92; text=@sprintf("RMSE=%.3f m  bias=%.3f m", res.rmse, res.bias),
              space=:relative, fontsize=10)
    end
    save(joinpath(PLT_DIR, "hs_timeseries.png"), fig; px_per_unit=2)
    return fig
end

function plot_maps(res)
    rip_final = res.rip.hs[:, :, end]
    ww3_final = replace(res.ww3.hs[:, :, end], NaN => 0.0)

    vmax = max(maximum(rip_final), maximum(filter(!isnan, vec(res.ww3.hs[:, :, end]))))
    vmax = vmax > 0 ? vmax : 1.0

    fig = Figure(size=(900, 380))
    ax1 = Axis(fig[1, 1]; title="Ripple  t=$(res.rip_t_h[end]|>round|>Int) h",
               xlabel="x (km)", ylabel="y (km)", aspect=DataAspect())
    ax2 = Axis(fig[1, 2]; title="WW3 ref  t=$(round(Int, res.ww3.t[end]/3600)) h",
               xlabel="x (km)", ylabel="y (km)", aspect=DataAspect())

    hm1 = heatmap!(ax1, res.rip.x ./ 1e3, res.rip.y ./ 1e3, rip_final;
                   colorrange=(0, vmax), colormap=:viridis)
    hm2 = heatmap!(ax2, res.ww3.x ./ 1e3, res.ww3.y ./ 1e3, ww3_final;
                   colorrange=(0, vmax), colormap=:viridis)
    Colorbar(fig[1, 3], hm1; label="Hs (m)")
    save(joinpath(PLT_DIR, "$(res.case_id)_maps.png"), fig; px_per_unit=2)
end

# ─── main ───────────────────────────────────────────────────────────────────

results = []
for i in 1:7
    case_id = @sprintf "Test%02d_2D" i
    rip_path = joinpath(OUT_DIR, "$(case_id)_ripple.nc")
    if !isfile(rip_path)
        @warn "Missing Ripple output: $rip_path (skipping)"
        continue
    end
    try
        res = compare_case(case_id)
        push!(results, res)
        plot_maps(res)
        @printf "%-12s  RMSE=%5.3f m  bias=%+.3f m  final Hs(Ripple)=%.3f m  final Hs(WW3)=%.3f m\n" \
            case_id res.rmse res.bias res.rip_m[end] res.ww3_m[end]
    catch e
        @warn "Failed to compare $case_id: $e"
    end
end

isempty(results) && error("No results to plot — run run_ripple.jl first")

plot_timeseries(results)
println("\nPlots saved to $PLT_DIR")
