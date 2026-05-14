import Oceananigans.Advection: AbstractAdvectionScheme
import Oceananigans.Advection: AbstractCenteredAdvectionScheme
import Oceananigans.Advection: AbstractUpwindBiasedAdvectionScheme
import Oceananigans.Advection: div_Uc, materialize_advection
import Oceananigans.Advection: Centered, UpwindBiased, WENO, FluxFormAdvection
import Oceananigans.Fields: ConstantField, ZeroField
import Oceananigans.Grids: halo_size, required_halo_size_x, required_halo_size_y

is_tracer_direction_advection(::Nothing) = true
is_tracer_direction_advection(::AbstractCenteredAdvectionScheme) = true
is_tracer_direction_advection(::AbstractUpwindBiasedAdvectionScheme) = true
is_tracer_direction_advection(advection) = false

is_tracer_advection(advection) = is_tracer_direction_advection(advection)
is_tracer_advection(advection::FluxFormAdvection) =
    is_tracer_direction_advection(advection.x) &&
    is_tracer_direction_advection(advection.y) &&
    is_tracer_direction_advection(advection.z)

canonical_model_advection(::Nothing) = nothing
canonical_model_advection(advection::AbstractAdvectionScheme) = advection

function canonical_model_advection(advection::NamedTuple)
    haskey(advection, :N) ||
        throw(ArgumentError("Ripple has one prognostic tracer, `N`; named advection tuples must include `N`"))
    return canonical_model_advection(advection.N)
end

canonical_model_advection(advection) =
    throw(ArgumentError("advection must be nothing or an Oceananigans tracer advection scheme; got $(typeof(advection))"))

validate_model_advection(::Nothing, grid, spectral_grid) = nothing

function validate_model_advection(advection::AbstractAdvectionScheme, grid, spectral_grid)
    materialized = materialize_advection(advection, grid)
    is_tracer_advection(materialized) ||
        throw(ArgumentError("advection must be an Oceananigans tracer advection scheme for `N` (`Centered`, `UpwindBiased`, `WENO`, or `FluxFormAdvection` composed from those schemes and `nothing`); got $(summary(materialized))"))

    Hx, Hy, _ = halo_size(grid)
    Hx_required = required_halo_size_x(materialized)
    Hy_required = required_halo_size_y(materialized)

    Hx >= Hx_required && Hy >= Hy_required ||
        throw(ArgumentError("grid horizontal halo $(halo_size(grid)) is too small for $(summary(materialized)); required at least ($Hx_required, $Hy_required, _)"))

    return materialized
end

function deep_water_group_velocity(cgrid, m, n; gravity=9.81)
    kx, ky = k_components(cgrid, m, n)
    k = radial_wavenumber(cgrid, m, n)
    iszero(k) && return (zero(float(kx)), zero(float(ky)))
    cg = deep_water_intrinsic_group_speed(cgrid, m, n; gravity)
    return cg * kx / k, cg * ky / k
end

function deep_water_group_velocity(cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid},
                                   m, n; gravity=9.81)
    k1, k2 = cgrid.kappa_faces[m], cgrid.kappa_faces[m+1]
    theta1, theta2 = cgrid.theta_faces[n], cgrid.theta_faces[n+1]
    radial = (k2^(3 / 2) - k1^(3 / 2)) / (3 / 2)
    angular_x = sin(theta2) - sin(theta1)
    angular_y = cos(theta1) - cos(theta2)
    scale = sqrt(gravity) / (2 * spectral_cell_measure(cgrid, m, n))
    return scale * radial * angular_x, scale * radial * angular_y
end

active_physical_k(f::ProductField) = first(axes(physical_field(f, 1, 1).data, 3))

transport_velocity(model, m, n) =
    deep_water_group_velocity(model.spectral_grid, m, n)

advection_x_component(advection) = advection
advection_y_component(advection) = advection
advection_x_component(advection::FluxFormAdvection) = advection.x
advection_y_component(advection::FluxFormAdvection) = advection.y

function horizontal_advection(advection, u, v)
    x_advection = iszero(u) ? nothing : advection_x_component(advection)
    y_advection = iszero(v) ? nothing : advection_y_component(advection)
    H = max(required_halo_size_x(x_advection), required_halo_size_y(y_advection))
    FT = eltype(advection)
    return FluxFormAdvection{H, FT}(x_advection, y_advection, nothing)
end

transport_velocity_fields(model, m, n) = begin
    u, v = transport_velocity(model, m, n)
    FT = eltype(model.action)
    return (u=ConstantField(convert(FT, u)),
            v=ConstantField(convert(FT, v)),
            w=ZeroField(FT))
end

transport_tendency(::Nothing, model, c, i, j, m, n) = zero(eltype(model.action))

function transport_tendency(advection::AbstractAdvectionScheme, model, c, i, j, m, n)
    k = active_physical_k(model.action)
    u, v = transport_velocity(model, m, n)
    horizontal = horizontal_advection(advection, u, v)
    U = transport_velocity_fields(model, m, n)
    return -div_Uc(i, j, k, model.grid, horizontal, U, c)
end
