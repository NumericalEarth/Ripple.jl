#####
##### Compare Ripple vs WW3 for the duration-limited test.
#####
##### Reads:
#####   - output/ripple_hs.tsv              (Hs(t), Ripple)
#####   - output/ripple_spectrum_final.tsv  (E(f), Ripple)
#####   - output/ww3/ww3.196806_spec.nc     (E(θ,f) per (station,t), WW3)
#####
##### Writes plots to output/compare/:
#####   - hs_timeseries.png
#####   - spectrum_final.png
#####
##### Run after ./run_ww3.sh and run_ripple.jl have both produced output.

using DelimitedFiles
using Printf
using NCDatasets
using Dates
using CairoMakie

const DIR    = @__DIR__
const OUTDIR = joinpath(DIR, "output")
const PLOTDIR = joinpath(OUTDIR, "compare")
mkpath(PLOTDIR)

# ─── Ripple side ────────────────────────────────────────────────────────────
function _read_tsv(path)
    raw = readdlm(path, '\t'; header=true)
    return Float64.(raw[1])
end

rdata = _read_tsv(joinpath(OUTDIR, "ripple_hs.tsv"))
ripple_t  = rdata[:, 1]
ripple_hs = rdata[:, 2]

sdata = _read_tsv(joinpath(OUTDIR, "ripple_spectrum_final.tsv"))
ripple_f = sdata[:, 1]
ripple_E = sdata[:, 2]

# ─── WW3 side ───────────────────────────────────────────────────────────────
ww3_nc = first(filter(p -> endswith(p, ".nc"), readdir(joinpath(OUTDIR, "ww3"); join=true)))
ds = NCDataset(ww3_nc)

_t = ds["time"][:]
ww3_time = Float64[Dates.value(t - _t[1]) / 1000 for t in _t]    # ms→s
ww3_freq = Float64.(ds["frequency"][:])
ww3_f1   = Float64.(ds["frequency1"][:])    # lower face
ww3_f2   = Float64.(ds["frequency2"][:])    # upper face
ww3_df   = ww3_f2 .- ww3_f1
ww3_dir  = Float64.(ds["direction"][:])

# efth dims: (direction, frequency, station, time). Energy density per (f, θ).
efth = Float64.(ds["efth"][:, :, 1, :])
Nθ, Nf, Nt = size(efth)
Δθ = 2pi / Nθ
ww3_hs = zeros(Float64, Nt)
for ti in 1:Nt
    m0 = 0.0
    for fi in 1:Nf
        ef = sum(efth[:, fi, ti]) * Δθ      # 1-D E(f) at this time
        m0 += ef * ww3_df[fi]
    end
    ww3_hs[ti] = 4 * sqrt(max(m0, 0.0))
end

ww3_E_final = zeros(Float64, Nf)
for fi in 1:Nf
    ww3_E_final[fi] = sum(efth[:, fi, end]) * Δθ
end

close(ds)

# ─── Plot ───────────────────────────────────────────────────────────────────
fig1 = Figure(size=(750, 420))
ax1  = Axis(fig1[1, 1]; xlabel="time (h)", ylabel="Hs (m)",
            title="Duration-limited growth, U10 = 17 m/s")
lines!(ax1, ripple_t ./ 3600, ripple_hs;
       label="Ripple (ST3-eq + DIA)", linewidth=2)
lines!(ax1, ww3_time ./ 3600, ww3_hs;
       label="WW3 v6 (ST4 + DIA)", linewidth=2, linestyle=:dash)
Hs_PM = 0.246 * 17.0^2 / 9.81     # empirical PM Hs ≈ 7.25 m for U10=17 m/s
hlines!(ax1, [Hs_PM]; color=:black, linestyle=:dot, label="PM equilibrium")
axislegend(ax1; position=:rb)
save(joinpath(PLOTDIR, "hs_timeseries.png"), fig1; px_per_unit=2)

fig2 = Figure(size=(750, 420))
ax2  = Axis(fig2[1, 1]; xlabel="frequency (Hz)", ylabel="E(f) (m²/Hz)",
            title="1-D energy spectrum at t_final",
            xscale=log10, yscale=log10)
lines!(ax2, ripple_f, max.(ripple_E, 1e-20); label="Ripple", linewidth=2)
lines!(ax2, ww3_freq, max.(ww3_E_final, 1e-20);
       label="WW3", linewidth=2, linestyle=:dash)
axislegend(ax2; position=:lb)
save(joinpath(PLOTDIR, "spectrum_final.png"), fig2; px_per_unit=2)

# ─── Summary ────────────────────────────────────────────────────────────────
@printf "\nFinal Hs:\n"
@printf "  Ripple : %.3f m (t = %.1f h)\n" ripple_hs[end] (ripple_t[end] / 3600)
@printf "  WW3    : %.3f m (t = %.1f h)\n" ww3_hs[end] (ww3_time[end] / 3600)
@printf "  PM eq. : %.3f m\n" Hs_PM
@printf "  Relative error: %.1f%%\n" 100 * abs(ripple_hs[end] - ww3_hs[end]) / max(ww3_hs[end], eps())
@printf "Plots: %s\n" PLOTDIR
