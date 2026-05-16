struct Clockwise end
struct Counterclockwise end

struct LinearStormTrack{T, C}
    times :: T
    centers :: C
end

function LinearStormTrack(times::AbstractVector, centers::AbstractVector)
    t = collect(float.(times))
    c = collect(centers)
    length(t) == length(c) ||
        throw(ArgumentError("storm-track times and centers must have the same length"))
    length(t) > 0 ||
        throw(ArgumentError("storm-track times and centers must be non-empty"))
    for n in 2:length(t)
        t[n] > t[n-1] ||
            throw(ArgumentError("storm-track times must be strictly increasing"))
    end
    for center in c
        length(center) == 2 ||
            throw(ArgumentError("storm-track centers must be two-component coordinates"))
    end
    normalized_centers = [(float(center[1]), float(center[2])) for center in c]
    return LinearStormTrack{typeof(t), typeof(normalized_centers)}(t, normalized_centers)
end

function (track::LinearStormTrack)(t)
    times = track.times
    centers = track.centers

    t <= first(times) && return first(centers)
    t >= last(times) && return last(centers)

    upper = searchsortedfirst(times, t)
    lower = upper - 1
    weight = (t - times[lower]) / (times[upper] - times[lower])
    x = (1 - weight) * centers[lower][1] + weight * centers[upper][1]
    y = (1 - weight) * centers[lower][2] + weight * centers[upper][2]
    return (x, y)
end

struct StationaryVortexWind{C, FT, R}
    center :: C
    diameter :: FT
    speed :: FT
    radial_width :: FT
    rotation :: R
end

function StationaryVortexWind(; center=(0.0, 0.0),
                                diameter=1.0,
                                speed=1.0,
                                radial_width=0.35,
                                rotation=Counterclockwise())
    diameter > 0 || throw(ArgumentError("stationary vortex diameter must be positive"))
    speed >= 0 || throw(ArgumentError("stationary vortex speed must be nonnegative"))
    radial_width > 0 || throw(ArgumentError("stationary vortex radial_width must be positive"))
    FT = promote_type(typeof(float(diameter)), typeof(float(speed)), typeof(float(radial_width)))
    return StationaryVortexWind(center, FT(diameter), FT(speed), FT(radial_width), rotation)
end

struct IdealizedHurricaneWind{C, B, FT, R}
    center :: C
    vmax :: FT
    rmax :: FT
    radius :: FT
    inflow_angle :: FT
    background :: B
    rotation :: R
end

function IdealizedHurricaneWind(; center=(0.0, 0.0),
                                  vmax=1.0,
                                  rmax=1.0,
                                  radius=4.0,
                                  inflow_angle=0.0,
                                  background=(0.0, 0.0),
                                  rotation=Counterclockwise())
    vmax >= 0 || throw(ArgumentError("hurricane vmax must be nonnegative"))
    rmax > 0 || throw(ArgumentError("hurricane rmax must be positive"))
    radius > rmax || throw(ArgumentError("hurricane radius must exceed rmax"))
    FT = promote_type(typeof(float(vmax)), typeof(float(rmax)),
                      typeof(float(radius)), typeof(float(inflow_angle)))
    return IdealizedHurricaneWind(center, FT(vmax), FT(rmax), FT(radius),
                                  FT(inflow_angle), background, rotation)
end

struct HollandHurricaneWind{C, B, FT, R}
    center :: C
    vmax :: FT
    rmax :: FT
    radius :: FT
    shape_parameter :: FT
    inflow_angle :: FT
    background :: B
    rotation :: R
end

function HollandHurricaneWind(; center=(0.0, 0.0),
                                vmax=1.0,
                                rmax=1.0,
                                radius=4.0,
                                shape_parameter=1.5,
                                inflow_angle=0.0,
                                background=(0.0, 0.0),
                                rotation=Counterclockwise())
    vmax >= 0 || throw(ArgumentError("Holland hurricane vmax must be nonnegative"))
    rmax > 0 || throw(ArgumentError("Holland hurricane rmax must be positive"))
    radius > rmax || throw(ArgumentError("Holland hurricane radius must exceed rmax"))
    shape_parameter > 0 || throw(ArgumentError("Holland hurricane shape_parameter must be positive"))
    FT = promote_type(typeof(float(vmax)), typeof(float(rmax)),
                      typeof(float(radius)), typeof(float(shape_parameter)),
                      typeof(float(inflow_angle)))
    return HollandHurricaneWind(center, FT(vmax), FT(rmax), FT(radius),
                                FT(shape_parameter), FT(inflow_angle),
                                background, rotation)
end

rotation_sign(::Counterclockwise) = 1
rotation_sign(::Clockwise) = -1

wind_center(center::Tuple, t) = center
wind_center(center, t) = center(t)

function stationary_vortex_speed(w::StationaryVortexWind, r)
    radius = w.diameter / 2
    peak_radius = 0.45radius
    width = w.radial_width * radius
    eye = 1 - exp(-(r / max(width, eps(width)))^2)
    ring = exp(-((r - peak_radius) / max(width, eps(width)))^2)
    return w.speed * eye * ring
end

function hurricane_speed(w::IdealizedHurricaneWind, r)
    if r <= w.rmax
        return w.vmax * r / w.rmax
    else
        decay_scale = max(w.radius - w.rmax, eps(w.radius))
        return w.vmax * exp(-(r - w.rmax) / decay_scale)
    end
end

function hurricane_speed(w::HollandHurricaneWind, r)
    r == 0 && return zero(w.vmax)
    scaled_radius = w.rmax / r
    shape_term = scaled_radius^w.shape_parameter
    speed = w.vmax * sqrt(max(shape_term * exp(1 - shape_term), zero(shape_term)))
    if r > w.radius
        decay_scale = max(w.radius - w.rmax, eps(w.radius))
        speed *= exp(-(r - w.radius) / decay_scale)
    end
    return speed
end

function wind_velocity(w::StationaryVortexWind, x, y, t=0)
    cx, cy = wind_center(w.center, t)
    dx, dy = x - cx, y - cy
    r = hypot(dx, dy)
    r == 0 && return (zero(w.speed), zero(w.speed))

    speed = stationary_vortex_speed(w, r)
    s = rotation_sign(w.rotation)
    tx, ty = -s * dy / r, s * dx / r
    return (speed * tx, speed * ty)
end

function wind_velocity(w::IdealizedHurricaneWind, x, y, t=0)
    cx, cy = wind_center(w.center, t)
    dx, dy = x - cx, y - cy
    r = hypot(dx, dy)
    bx, by = w.background
    r == 0 && return (bx, by)

    speed = hurricane_speed(w, r)
    s = rotation_sign(w.rotation)
    tx, ty = -s * dy / r, s * dx / r
    rx, ry = dx / r, dy / r
    vx = speed * (cos(w.inflow_angle) * tx - sin(w.inflow_angle) * rx) + bx
    vy = speed * (cos(w.inflow_angle) * ty - sin(w.inflow_angle) * ry) + by
    return (vx, vy)
end

function wind_velocity(w::HollandHurricaneWind, x, y, t=0)
    cx, cy = wind_center(w.center, t)
    dx, dy = x - cx, y - cy
    r = hypot(dx, dy)
    bx, by = w.background
    r == 0 && return (bx, by)

    speed = hurricane_speed(w, r)
    s = rotation_sign(w.rotation)
    tx, ty = -s * dy / r, s * dx / r
    rx, ry = dx / r, dy / r
    vx = speed * (cos(w.inflow_angle) * tx - sin(w.inflow_angle) * rx) + bx
    vy = speed * (cos(w.inflow_angle) * ty - sin(w.inflow_angle) * ry) + by
    return (vx, vy)
end

wind_speed(w, x, y, t=0) = hypot(wind_velocity(w, x, y, t)...)
function wind_angle(w, x, y, t=0)
    vx, vy = wind_velocity(w, x, y, t)
    return atan(vy, vx)
end

(w::StationaryVortexWind)(x, y, t=0) = wind_speed(w, x, y, t)
(w::IdealizedHurricaneWind)(x, y, t=0) = wind_speed(w, x, y, t)
(w::HollandHurricaneWind)(x, y, t=0) = wind_speed(w, x, y, t)
