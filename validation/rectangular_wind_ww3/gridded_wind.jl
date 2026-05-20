using NCDatasets, Dates
import Ripple: wind_speed, wind_angle

struct GriddedWind{FT}
    x :: Vector{FT}
    y :: Vector{FT}
    t :: Vector{FT}
    u :: Array{FT, 3}   # u10m[ix, iy, it]
    v :: Array{FT, 3}   # v10m[ix, iy, it]
end

function GriddedWind(nc_path::AbstractString)
    NCDataset(nc_path) do ds
        x = Float64.(ds["x"][:])
        y = Float64.(ds["y"][:])
        datetimes = ds["time"][:]
        t = [Dates.value(dt - datetimes[1]) / 1000.0 for dt in datetimes]
        u = Float64.(ds["u10m"][:, :, :])
        v = Float64.(ds["v10m"][:, :, :])
        return GriddedWind{Float64}(x, y, t, u, v)
    end
end

function _interp_uv(wind::GriddedWind, x, y, t)
    xs, ys, ts = wind.x, wind.y, wind.t

    ix = clamp(searchsortedfirst(xs, x) - 1, 1, length(xs) - 1)
    iy = clamp(searchsortedfirst(ys, y) - 1, 1, length(ys) - 1)
    it = clamp(searchsortedfirst(ts, t) - 1, 1, length(ts) - 1)

    wx = clamp((x - xs[ix]) / (xs[ix+1] - xs[ix]), 0.0, 1.0)
    wy = clamp((y - ys[iy]) / (ys[iy+1] - ys[iy]), 0.0, 1.0)
    it2 = min(it + 1, length(ts))
    wt = it2 > it ? clamp((t - ts[it]) / (ts[it2] - ts[it]), 0.0, 1.0) : 0.0

    ix2 = ix + 1
    iy2 = iy + 1

    function trilin(arr)
        v000 = arr[ix,  iy,  it ]
        v100 = arr[ix2, iy,  it ]
        v010 = arr[ix,  iy2, it ]
        v110 = arr[ix2, iy2, it ]
        v001 = arr[ix,  iy,  it2]
        v101 = arr[ix2, iy,  it2]
        v011 = arr[ix,  iy2, it2]
        v111 = arr[ix2, iy2, it2]
        return ((v000*(1-wx) + v100*wx)*(1-wy) + (v010*(1-wx) + v110*wx)*wy) * (1-wt) +
               ((v001*(1-wx) + v101*wx)*(1-wy) + (v011*(1-wx) + v111*wx)*wy) * wt
    end

    return trilin(wind.u), trilin(wind.v)
end

function wind_speed(w::GriddedWind, x, y, t)
    u, v = _interp_uv(w, x, y, t)
    return hypot(u, v)
end

function wind_angle(w::GriddedWind, x, y, t)
    u, v = _interp_uv(w, x, y, t)
    return atan(v, u)
end
