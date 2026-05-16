import Oceananigans.Advection: AbstractAdvectionScheme
import Oceananigans.Advection: AbstractCenteredAdvectionScheme
import Oceananigans.Advection: AbstractUpwindBiasedAdvectionScheme
import Oceananigans.Advection: div_Uc, materialize_advection
import Oceananigans.Advection: Centered, UpwindBiased, WENO, FluxFormAdvection
import KernelAbstractions: @kernel, @index
import Oceananigans.Architectures: architecture
import Oceananigans.Fields: CenterField, ConstantField, Field, ZeroField, fill_halo_regions!, interior
import Oceananigans.Utils: launch!
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
    k1, k2 = cgrid.κ_faces[m], cgrid.κ_faces[m+1]
    φ1, φ2 = cgrid.φ_faces[n], cgrid.φ_faces[n+1]
    radial = (k2^(3 / 2) - k1^(3 / 2)) / (3 / 2)
    angular_x = sin(φ2) - sin(φ1)
    angular_y = cos(φ1) - cos(φ2)
    scale = sqrt(gravity) / (2 * spectral_cell_measure(cgrid, m, n))
    return scale * radial * angular_x, scale * radial * angular_y
end

function finite_depth_group_velocity(cgrid, m, n, depth; gravity=9.81)
    kx, ky = k_components(cgrid, m, n)
    k = radial_wavenumber(cgrid, m, n)
    iszero(k) && return (zero(float(kx)), zero(float(ky)))
    h = float(depth)
    h > 0 || throw(ArgumentError("finite-depth dispersion requires positive depth"))
    μ = k * h
    phase_speed = sqrt(gravity * tanh(μ) / k)
    group_factor = (one(μ) + 2μ / sinh(2μ)) / 2
    cg = group_factor * phase_speed
    return cg * kx / k, cg * ky / k
end

intrinsic_group_velocity(cgrid, m, n, depth::InfiniteDepth; gravity=9.81) =
    deep_water_group_velocity(cgrid, m, n; gravity)

intrinsic_group_velocity(cgrid, m, n, depth::Number; gravity=9.81) =
    finite_depth_group_velocity(cgrid, m, n, depth; gravity)

intrinsic_group_velocity(cgrid, m, n, depth::ConstantField; gravity=9.81) =
    finite_depth_group_velocity(cgrid, m, n, depth.constant; gravity)

intrinsic_group_velocity(cgrid, m, n, depth::AbstractArray; gravity=9.81) =
    finite_depth_group_velocity(cgrid, m, n, depth[1, 1]; gravity)

intrinsic_group_velocity(cgrid, m, n, depth::Field; gravity=9.81) =
    finite_depth_group_velocity(cgrid, m, n,
                                finite_depth_values(depth, depth.grid; name="model `depth`")[1, 1];
                                gravity)

is_spatially_varying_depth(depth) =
    !(depth isa InfiniteDepth || depth isa Number || depth isa ConstantField)

@inline intrinsic_velocity_component(table::AbstractArray{T, 2}, i, j, m, n) where T =
    @inbounds table[m, n]

@inline intrinsic_velocity_component(table::AbstractArray{T, 4}, i, j, m, n) where T =
    @inbounds table[i, j, m, n]

function intrinsic_group_velocity_tables(cgrid, depth, grid, ::Type{FT}) where FT
    Nκ, Nφ = coordinate_size(cgrid)

    if is_spatially_varying_depth(depth)
        Nx, Ny = horizontal_size(grid)
        depth_values = finite_depth_values(depth, grid; name="model `depth`")
        depth_host = Array(depth_values)
        cg_x = zeros(FT, Nx, Ny, Nκ, Nφ)
        cg_y = zeros(FT, Nx, Ny, Nκ, Nφ)

        @inbounds for n in 1:Nφ, m in 1:Nκ, j in 1:Ny, i in 1:Nx
            u, v = finite_depth_group_velocity(cgrid, m, n, depth_host[i, j])
            cg_x[i, j, m, n] = u
            cg_y[i, j, m, n] = v
        end
    else
        cg_x = zeros(FT, Nκ, Nφ)
        cg_y = zeros(FT, Nκ, Nφ)

        @inbounds for n in 1:Nφ, m in 1:Nκ
            u, v = intrinsic_group_velocity(cgrid, m, n, depth)
            cg_x[m, n] = u
            cg_y[m, n] = v
        end
    end

    arch = architecture(grid)
    return on_architecture(arch, cg_x), on_architecture(arch, cg_y)
end

active_physical_k(f::ProductField) = first(axes(physical_field(f, 1, 1).data, 3))

transport_velocity(model, m, n) =
    intrinsic_group_velocity(model.spectral_grid, m, n, model.depth)

advection_x_component(advection) = advection
advection_y_component(advection) = advection
advection_x_component(advection::FluxFormAdvection) = advection.x
advection_y_component(advection::FluxFormAdvection) = advection.y

function bin_horizontal_advection(advection, u, v)
    x_advection = iszero(u) ? nothing : advection_x_component(advection)
    y_advection = iszero(v) ? nothing : advection_y_component(advection)
    H = max(required_halo_size_x(x_advection), required_halo_size_y(y_advection))
    FT = eltype(advection)
    return FluxFormAdvection{H, FT}(x_advection, y_advection, nothing)
end

transport_velocity_fields(model, m, n) =
    transport_velocity_fields(model.coupling, model, m, n)

function transport_velocity_fields(::Any, model, m, n)
    u, v = transport_velocity(model, m, n)
    FT = eltype(model.action)
    return (u=ConstantField(convert(FT, u)),
            v=ConstantField(convert(FT, v)),
            w=ZeroField(FT))
end

@kernel function _doppler_shift_velocity_fields!(u, v, cg_x, cg_y, Ux, Uy, m)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = cg_x + Ux[i, j, m]
    @inbounds v[i, j, k] = cg_y + Uy[i, j, m]
end

function transport_velocity_fields(coupling::CWCMPrescribedCurrentCoupling, model, m, n)
    cg_x, cg_y = transport_velocity(model, m, n)
    FT = eltype(model.action)
    grid = model.grid
    if coupling.u_transport_scratch === nothing
        coupling.u_transport_scratch = CenterField(grid)
        coupling.v_transport_scratch = CenterField(grid)
    end
    u_field = coupling.u_transport_scratch
    v_field = coupling.v_transport_scratch
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _doppler_shift_velocity_fields!,
            u_field, v_field, convert(FT, cg_x), convert(FT, cg_y),
            coupling.Ux, coupling.Uy, m)
    fill_halo_regions!(u_field)
    fill_halo_regions!(v_field)
    return (u=u_field, v=v_field, w=ZeroField(FT))
end

transport_tendency(::Nothing, model, c, i, j, m, n) = zero(eltype(model.action))

function transport_tendency(advection::AbstractAdvectionScheme, model, c, i, j, m, n)
    k = active_physical_k(model.action)
    u, v = transport_velocity(model, m, n)
    horizontal = bin_horizontal_advection(advection, u, v)
    U = transport_velocity_fields(model, m, n)
    return -div_Uc(i, j, k, model.grid, horizontal, U, c)
end
