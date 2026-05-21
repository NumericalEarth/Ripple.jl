import Oceananigans
import Oceananigans: AbstractModel, fields, prognostic_fields
import Oceananigans.Advection: WENO
import Oceananigans.Architectures: architecture, device, on_architecture
import Oceananigans.BoundaryConditions: DefaultBoundaryCondition, FieldBoundaryConditions, fill_halo_regions!
import Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
import Oceananigans.Fields: Field, CenterField, interior, set!
import Oceananigans.Grids: Center, Flat, Periodic
import Oceananigans.Operators: Δxᶜᵃᵃ, Δyᵃᶜᵃ
import Oceananigans.TimeSteppers: Clock, RungeKutta3TimeStepper, time_step!, update_state!, tick!
import KernelAbstractions
import KernelAbstractions: @kernel, @index

const MONOBANDED_PROGNOSTIC_NAMES = (:A, :AKx, :AKy)

const MONOBANDED_DIAGNOSTIC_NAMES =
    (:Kx, :Ky, :κ, :uᴰx, :uᴰy, :Hx, :Hy, :Cx, :Cy,
     :Γxx, :Γyx, :Γxy, :Γyy, :Ω, :ZK, :Ĉx, :Ĉy)

mutable struct MonobandedExplicitTimeStepper{Tendencies, PreviousTendencies}
    name :: Symbol
    Gⁿ :: Tendencies
    G⁻ :: PreviousTendencies
end

struct MonobandedPrescribedCurrentCoupling{Current, QT} <: AbstractCurrentCoupling
    current :: Current
    qtransform :: QT
end

mutable struct MonobandedPseudomomentumCoupling{QT, D, PX, PY} <: AbstractCurrentCoupling
    qtransform :: QT
    depth :: D
    px :: PX
    py :: PY
end

mutable struct MonobandedWaveModel{TS, Arch, G, C, A, M, D, Adv, Sources, Coupling, GA, MA, MK} <: AbstractModel{TS, Arch}
    architecture :: Arch
    grid :: G
    clock :: C
    action :: A
    wavenumber_moment :: M
    diagnostics :: D
    advection :: Adv
    sources :: Sources
    coupling :: Coupling
    timestepper :: TS
    previous_tendencies_ready :: Bool
    gravitational_acceleration :: GA
    minimum_action :: MA
    minimum_wavenumber :: MK
end

monobanded_surface_indices(grid) = (:, :, grid.Nz:grid.Nz)

monobanded_default_gravity(::Type{FT}) where FT = FT(981) / FT(100)

function regularize_monobanded_field_boundary_conditions(grid, bcs::FieldBoundaryConditions, name)
    slab_bcs = FieldBoundaryConditions(monobanded_surface_indices(grid), bcs)
    loc = (Center(), Center(), Center())
    return regularize_field_boundary_conditions(slab_bcs, grid, loc, MONOBANDED_PROGNOSTIC_NAMES, name)
end

monobanded_override_boundary_condition(override::DefaultBoundaryCondition, embedded) = embedded
monobanded_override_boundary_condition(override, embedded) = override

function merge_monobanded_field_boundary_conditions(embedded::FieldBoundaryConditions,
                                                    override::FieldBoundaryConditions)
    return FieldBoundaryConditions(monobanded_override_boundary_condition(override.west, embedded.west),
                                   monobanded_override_boundary_condition(override.east, embedded.east),
                                   monobanded_override_boundary_condition(override.south, embedded.south),
                                   monobanded_override_boundary_condition(override.north, embedded.north),
                                   monobanded_override_boundary_condition(override.bottom, embedded.bottom),
                                   monobanded_override_boundary_condition(override.top, embedded.top),
                                   monobanded_override_boundary_condition(override.immersed, embedded.immersed))
end

function monobanded_field_boundary_conditions(grid, boundary_conditions, name,
                                              embedded_bcs::FieldBoundaryConditions=FieldBoundaryConditions())
    override_bcs = haskey(boundary_conditions, name) ? boundary_conditions[name] : FieldBoundaryConditions()
    override_bcs isa FieldBoundaryConditions ||
        throw(ArgumentError("boundary_conditions.$name must be an Oceananigans FieldBoundaryConditions; got $(typeof(override_bcs))"))

    indices = monobanded_surface_indices(grid)
    embedded_bcs = FieldBoundaryConditions(indices, embedded_bcs)
    override_bcs = FieldBoundaryConditions(indices, override_bcs)
    bcs = merge_monobanded_field_boundary_conditions(embedded_bcs, override_bcs)
    return regularize_monobanded_field_boundary_conditions(grid, bcs, name)
end

function validate_monobanded_boundary_condition_names(boundary_conditions)
    for name in keys(boundary_conditions)
        name in MONOBANDED_PROGNOSTIC_NAMES ||
            throw(ArgumentError("unknown MonobandedWaveModel boundary condition `$name`; valid names are `A`, `AKx`, and `AKy`"))
    end

    return boundary_conditions
end

# The transport and gradient kernels hard-code no-flux at non-periodic edges
# and never read field halos. Any BC other than the default (NoFlux on
# bounded sides, Periodic on periodic sides) would be silently ignored.
# Reject up front so users notice rather than getting a different model
# than they asked for.
monobanded_bc_is_supported(::Oceananigans.BoundaryConditions.DefaultBoundaryCondition) = true
function monobanded_bc_is_supported(bc::Oceananigans.BoundaryConditions.BoundaryCondition)
    bc.condition === nothing || return false
    cls = typeof(bc).parameters[1]
    return cls <: Oceananigans.BoundaryConditions.Flux ||
           cls <: Oceananigans.BoundaryConditions.Periodic
end
monobanded_bc_is_supported(::Nothing) = true
monobanded_bc_is_supported(bc) = false

function validate_monobanded_field_boundary_conditions(field, name)
    bcs = field.boundary_conditions
    bcs isa Oceananigans.BoundaryConditions.FieldBoundaryConditions || return field
    for side in (:west, :east, :south, :north, :bottom, :top, :immersed)
        bc = getproperty(bcs, side)
        monobanded_bc_is_supported(bc) ||
            throw(ArgumentError(string("MonobandedWaveModel currently only supports default ",
                                       "(NoFlux / Periodic) boundary conditions on prognostic fields; ",
                                       "`$name.$side` is $(typeof(bc)). The transport and gradient ",
                                       "kernels hardcode no-flux at non-periodic boundaries, so ",
                                       "non-default BCs would have no numerical effect — implementing ",
                                       "BC-driven boundary fluxes is tracked separately.")))
    end
    return field
end

function validate_uniform_horizontal_grid(grid)
    topology = Oceananigans.Grids.topology(grid)
    if topology[1] !== Flat
        Δx = xspacings(grid)
        length(Δx) > 0 && is_uniform_spacing(Δx) ||
            throw(ArgumentError("MonobandedWaveModel requires uniform x-spacing on the model grid; got Δx = $(collect(Δx))"))
    end
    if topology[2] !== Flat
        Δy = yspacings(grid)
        length(Δy) > 0 && is_uniform_spacing(Δy) ||
            throw(ArgumentError("MonobandedWaveModel requires uniform y-spacing on the model grid; got Δy = $(collect(Δy))"))
    end
    return grid
end

function is_uniform_spacing(spacings)
    Δ₀ = first(spacings)
    tol = eps(typeof(Δ₀)) * 1024
    for Δ in spacings
        isapprox(Δ, Δ₀; rtol=tol, atol=tol * abs(Δ₀)) || return false
    end
    return true
end

function monobanded_center_field(grid, ::Type{FT}, boundary_conditions, name) where FT
    return CenterField(grid, FT;
                       indices=monobanded_surface_indices(grid),
                       boundary_conditions=monobanded_field_boundary_conditions(grid, boundary_conditions, name))
end

function materialize_monobanded_field(::Nothing, grid, ::Type{FT}, boundary_conditions, name) where FT
    return monobanded_center_field(grid, FT, boundary_conditions, name)
end

function validate_monobanded_field(field::Field, grid, name)
    compatible_model_physical_grid(field.grid, grid) ||
        throw(ArgumentError("provided $name field is not on the model physical grid"))
    location(field) == (Center, Center, Center) ||
        throw(ArgumentError("provided $name field must be located at (Center, Center, Center)"))
    length(axes(field.data, 3)) == 1 ||
        throw(ArgumentError("provided $name field must be a surface slab with one active z index"))
    first(axes(field.data, 3)) == grid.Nz ||
        throw(ArgumentError("provided $name field must live on the top surface index grid.Nz"))
    return field
end

validate_monobanded_field(field, grid, name) =
    throw(ArgumentError("$name must be an Oceananigans Field; got $(typeof(field))"))

function materialize_monobanded_field(field::Field, grid, ::Type{FT}, boundary_conditions, name) where FT
    field = validate_monobanded_field(field, grid, string(name))

    haskey(boundary_conditions, name) || return field

    return CenterField(grid, eltype(field);
                       indices=monobanded_surface_indices(grid),
                       data=field.data,
                       boundary_conditions=monobanded_field_boundary_conditions(grid, boundary_conditions, name,
                                                                               field.boundary_conditions))
end

materialize_monobanded_field(field, grid, ::Type{FT}, boundary_conditions, name) where FT =
    validate_monobanded_field(field, grid, string(name))

function validate_monobanded_wavenumber_moment(wavenumber_moment, grid, ::Type{FT}, boundary_conditions) where FT
    if wavenumber_moment === nothing
        return (x=materialize_monobanded_field(nothing, grid, FT, boundary_conditions, :AKx),
                y=materialize_monobanded_field(nothing, grid, FT, boundary_conditions, :AKy))
    end

    wavenumber_moment isa NamedTuple && haskey(wavenumber_moment, :x) && haskey(wavenumber_moment, :y) ||
        throw(ArgumentError("wavenumber_moment must be a NamedTuple with fields `x` and `y`"))

    return (x=materialize_monobanded_field(wavenumber_moment.x, grid, FT, boundary_conditions, :AKx),
            y=materialize_monobanded_field(wavenumber_moment.y, grid, FT, boundary_conditions, :AKy))
end

function monobanded_diagnostic_fields(grid, ::Type{FT}) where FT
    fields = ntuple(_ -> CenterField(grid, FT; indices=monobanded_surface_indices(grid)),
                    length(MONOBANDED_DIAGNOSTIC_NAMES))
    return NamedTuple{MONOBANDED_DIAGNOSTIC_NAMES}(fields)
end

monobanded_supported_timestepper(timestepper::Symbol) =
    timestepper === :ForwardEuler ||
    timestepper === :AB2 ||
    timestepper === :RungeKutta3 ||
    timestepper === :RK3 ||
    timestepper === :SSPRungeKutta3 ||
    is_low_storage_rk3(timestepper)

function canonical_monobanded_timestepper(timestepper::Symbol)
    timestepper === :RK3 && return :RungeKutta3
    timestepper === :SSPRungeKutta3 && return :RungeKutta3
    timestepper === :QuasiAdamsBashforth2 && return :AB2

    monobanded_supported_timestepper(timestepper) ||
        throw(ArgumentError("unsupported MonobandedWaveModel timestepper $timestepper"))
    is_low_storage_rk3(timestepper) && return :RungeKutta3
    return timestepper
end

canonical_monobanded_timestepper(timestepper) =
    throw(ArgumentError("timestepper must be a Symbol; got $(typeof(timestepper))"))

monobanded_validate_scalar_source_parameter(value::Number, name) = nothing
monobanded_validate_scalar_source_parameter(::Nothing, name) = nothing
monobanded_validate_scalar_source_parameter(value, name) =
    throw(ArgumentError("MonobandedWaveModel currently supports only scalar source parameter `$name`; got $(typeof(value))"))

validate_monobanded_sources(::Nothing) = nothing

function validate_monobanded_sources(source::LinearWindInput)
    monobanded_validate_scalar_source_parameter(source.rate, "rate")
    return source
end

function validate_monobanded_sources(source::BottomFriction)
    monobanded_validate_scalar_source_parameter(source.rate, "rate")
    monobanded_validate_scalar_source_parameter(source.depth, "depth")
    source.reference_depth > 0 || throw(ArgumentError("bottom-friction reference depth must be positive"))
    source.minimum_depth > 0 || throw(ArgumentError("bottom-friction minimum depth must be positive"))
    source.reference_wavenumber > 0 || throw(ArgumentError("bottom-friction reference wavenumber must be positive"))
    source.depth_power >= 0 || throw(ArgumentError("bottom-friction depth power must be nonnegative"))
    source.wavenumber_power >= 0 || throw(ArgumentError("bottom-friction wavenumber power must be nonnegative"))
    return source
end

function validate_monobanded_sources(sources::SourceTermSet)
    for source in sources
        validate_monobanded_sources(source)
    end

    return sources
end

validate_monobanded_sources(source::AbstractSourceTerm) =
    throw(ArgumentError("MonobandedWaveModel currently supports action-only sources `LinearWindInput`, `BottomFriction`, and `SourceTermSet` combinations of those; got $(typeof(source))"))

validate_monobanded_sources(sources) =
    throw(ArgumentError("MonobandedWaveModel sources must be `nothing` or a supported action-only source term; got $(typeof(sources))"))

monobanded_prescribed_q_grid(v::PrescribedVelocities, model_grid) = begin
    u_grid = velocity_grid(v.u)
    v_grid = velocity_grid(v.v)
    q_grid = v.q_grid !== nothing ? v.q_grid :
             u_grid !== nothing ? u_grid :
             v_grid !== nothing ? v_grid :
             model_grid

    validate_q_grid(q_grid, model_grid)
    validate_velocity_field_grid(u_grid, q_grid, "u")
    validate_velocity_field_grid(v_grid, q_grid, "v")
    return q_grid
end

build_monobanded_coupling(::ZeroVelocities, grid, ::Type{FT}) where FT = nothing

function build_monobanded_coupling(v::PrescribedVelocities, grid, ::Type{FT}) where FT
    q_grid = monobanded_prescribed_q_grid(v, grid)
    current = PrescribedLagrangianMeanCurrent(u=v.u, v=v.v, depth=grid_depth(q_grid))
    qtransform = QTransform(QKernel(FT), q_grid)
    return MonobandedPrescribedCurrentCoupling(current, qtransform)
end

function build_monobanded_coupling(v::PseudomomentumVelocities, grid, ::Type{FT}) where FT
    has_flat_vertical_topology(grid) && v.q_grid === nothing &&
        throw(ArgumentError("Flat MonobandedWaveModel grids require `q_grid` for PseudomomentumVelocities"))

    q_grid = v.q_grid === nothing ? grid : v.q_grid
    validate_q_grid(q_grid, grid)
    qtransform = QTransform(QKernel(FT), q_grid)
    depth = grid_depth(q_grid)
    px = pseudomomentum_field(q_grid; eltype=FT)
    py = pseudomomentum_field(q_grid; eltype=FT)
    return MonobandedPseudomomentumCoupling(qtransform, depth, px, py)
end

function build_monobanded_coupling(nt::NamedTuple, grid, ::Type{FT}) where FT
    haskey(nt, :u) && haskey(nt, :v) ||
        throw(ArgumentError("velocities NamedTuple must contain `u` and `v` (got keys $(keys(nt)))"))
    haskey(nt, :depth) &&
        throw(ArgumentError("`depth` is derived from the velocity grid; do not pass it in `velocities`"))
    haskey(nt, :q_grid) && haskey(nt, :vertical_grid) &&
        throw(ArgumentError("velocities NamedTuple cannot contain both `q_grid` and `vertical_grid`"))

    q_grid = haskey(nt, :q_grid) ? nt.q_grid : (haskey(nt, :vertical_grid) ? nt.vertical_grid : nothing)
    Nz = haskey(nt, :Nz) ? nt.Nz : nothing
    surface_spacing = haskey(nt, :surface_spacing) ? nt.surface_spacing : nothing
    vertical_stretching = haskey(nt, :vertical_stretching) ? nt.vertical_stretching : 2

    velocities = PrescribedVelocities(nt.u, nt.v, q_grid, Nz, surface_spacing, vertical_stretching)
    return build_monobanded_coupling(velocities, grid, FT)
end

build_monobanded_coupling(velocities, grid, ::Type{FT}) where FT =
    throw(ArgumentError("MonobandedWaveModel velocities must be `ZeroVelocities`, `PrescribedVelocities`, `PseudomomentumVelocities`, or a NamedTuple `(; u, v)`; got $(typeof(velocities))"))

function validate_monobanded_coupling(velocities, coupling, grid, ::Type{FT}) where FT
    velocities === nothing || coupling === nothing ||
        throw(ArgumentError("pass either `velocities` or `coupling`, not both"))

    velocities !== nothing && return build_monobanded_coupling(velocities, grid, FT)

    coupling = canonical_model_coupling(coupling)
    coupling === nothing && return nothing
    coupling isa Union{MonobandedPrescribedCurrentCoupling, MonobandedPseudomomentumCoupling} ||
        throw(ArgumentError("MonobandedWaveModel coupling must be `nothing`, `MonobandedPrescribedCurrentCoupling`, or `MonobandedPseudomomentumCoupling`; got $(typeof(coupling))"))
    compatible_horizontal_grid(coupling.qtransform.grid, grid) ||
        throw(ArgumentError("monobanded Q-transform grid must match the model grid horizontally"))
    return coupling
end

function monobanded_tendency_fields(prognostics)
    return map(similar, prognostics)
end

function materialize_monobanded_timestepper(timestepper::Symbol, grid, prognostics,
                                            tendencies, previous_tendencies)
    timestepper === :RungeKutta3 &&
        return RungeKutta3TimeStepper(grid, prognostics; Gⁿ=tendencies, G⁻=previous_tendencies)
    return MonobandedExplicitTimeStepper(timestepper, tendencies, previous_tendencies)
end

function MonobandedWaveModel(grid;
                             action=nothing,
                             wavenumber_moment=nothing,
                             advection=WENO(),
                             sources=nothing,
                             velocities=nothing,
                             coupling=nothing,
                             boundary_conditions=NamedTuple(),
                             timestepper=:RungeKutta3,
                             clock=nothing,
                             gravitational_acceleration=nothing,
                             minimum_action=nothing,
                             minimum_wavenumber=nothing)
    grid = validate_model_physical_grid(adapt_physical_grid(grid))
    grid = validate_uniform_horizontal_grid(grid)
    FT = grid_float_type(grid)

    boundary_conditions isa NamedTuple ||
        throw(ArgumentError("boundary_conditions must be a NamedTuple keyed by `A`, `AKx`, and `AKy`"))
    boundary_conditions = validate_monobanded_boundary_condition_names(boundary_conditions)

    action = materialize_monobanded_field(action, grid, FT, boundary_conditions, :A)
    validate_monobanded_field_boundary_conditions(action, :A)
    wavenumber_moment = validate_monobanded_wavenumber_moment(wavenumber_moment, grid, FT, boundary_conditions)
    validate_monobanded_field_boundary_conditions(wavenumber_moment.x, :AKx)
    validate_monobanded_field_boundary_conditions(wavenumber_moment.y, :AKy)
    diagnostics = monobanded_diagnostic_fields(grid, FT)
    sources = validate_monobanded_sources(canonical_model_sources(sources))
    coupling = validate_monobanded_coupling(velocities, coupling, grid, FT)
    advection = validate_model_advection(canonical_model_advection(advection), grid, nothing)
    timestepper_name = canonical_monobanded_timestepper(timestepper)
    clock = clock === nothing ? Clock(time=zero(FT)) : validate_model_clock(clock)

    g = gravitational_acceleration === nothing ? monobanded_default_gravity(FT) :
                                                 convert(FT, gravitational_acceleration)
    g > zero(FT) || throw(ArgumentError("gravitational_acceleration must be positive"))

    # `minimum_action` floors `K = AK/A` to keep the diagnostic group velocity
    # bounded in cells where action is vanishingly small. Defaulting to
    # `cbrt(eps(FT))` is large enough to prevent the
    # K → ∞ cascade that triggers a CFL violation when WENO overshoot drives A
    # near zero, and small enough not to interfere with physical wave fields
    # (which carry A of order unity).
    minimum_action = minimum_action === nothing ? cbrt(eps(FT)) : convert(FT, minimum_action)
    minimum_wavenumber = minimum_wavenumber === nothing ? sqrt(eps(FT)) : convert(FT, minimum_wavenumber)
    minimum_action > zero(FT) || throw(ArgumentError("minimum_action must be positive"))
    minimum_wavenumber > zero(FT) || throw(ArgumentError("minimum_wavenumber must be positive"))

    prognostics = (A=action, AKx=wavenumber_moment.x, AKy=wavenumber_moment.y)
    tendencies = monobanded_tendency_fields(prognostics)
    previous_tendencies = monobanded_tendency_fields(prognostics)
    timestepper = materialize_monobanded_timestepper(timestepper_name, grid, prognostics,
                                                     tendencies, previous_tendencies)

    arch = architecture(grid)
    model = MonobandedWaveModel{typeof(timestepper), typeof(arch), typeof(grid), typeof(clock),
                                typeof(action), typeof(wavenumber_moment), typeof(diagnostics),
                                typeof(advection), typeof(sources), typeof(coupling), typeof(g),
                                typeof(minimum_action), typeof(minimum_wavenumber)}(arch, grid, clock,
                                                                                     action, wavenumber_moment,
                                                                                     diagnostics, advection,
                                                                                     sources, coupling,
                                                                                     timestepper, false, g,
                                                                                     minimum_action,
                                                                                     minimum_wavenumber)

    update_monobanded_diagnostics!(model)
    return model
end

prognostic_fields(model::MonobandedWaveModel) =
    (A=model.action, AKx=model.wavenumber_moment.x, AKy=model.wavenumber_moment.y)

fields(model::MonobandedWaveModel) = merge(prognostic_fields(model), model.diagnostics)
Base.eltype(model::MonobandedWaveModel) = eltype(model.action)
architecture(model::MonobandedWaveModel) = model.architecture

monobanded_timestepper_name(timestepper::RungeKutta3TimeStepper) = :RungeKutta3
monobanded_timestepper_name(timestepper::MonobandedExplicitTimeStepper) = timestepper.name

# `velocities(model)` returns the (u, v) pair the wave model's coupling
# treats as the Lagrangian-mean current. For prescribed currents this is
# the supplied user current; for pseudomomentum self-coupling it is the
# wave-induced pseudomomentum cell averages (px, py).
velocities(model::MonobandedWaveModel) = velocities(model.coupling)
velocities(::Nothing) = nothing
velocities(c::MonobandedPrescribedCurrentCoupling) = (u=c.current.u, v=c.current.v)
velocities(c::MonobandedPseudomomentumCoupling) = (u=c.px, v=c.py)

monobanded_coupling_summary(::Nothing) = "none"
monobanded_coupling_summary(::MonobandedPrescribedCurrentCoupling) = "prescribed velocities"
monobanded_coupling_summary(::MonobandedPseudomomentumCoupling) = "pseudomomentum velocities"

monobanded_sources_summary(::Nothing) = "none"
monobanded_sources_summary(source) = string(nameof(typeof(source)))

function Base.summary(model::MonobandedWaveModel)
    Nx, Ny = horizontal_size(model.grid)
    return string("MonobandedWaveModel{", eltype(model), "} on a ",
                  Nx, "×", Ny, " surface grid")
end

function Base.show(io::IO, model::MonobandedWaveModel)
    println(io, summary(model))
    println(io, "├── grid: ", summary(model.grid))
    println(io, "├── prognostic fields: A, AKx, AKy")
    println(io, "├── diagnostics: Kx, Ky, κ, uᴰx, uᴰy, Hx, Hy, Cx, Cy, Γxx, Γyx, Γxy, Γyy, Ω, ZK, Ĉx, Ĉy")
    println(io, "├── advection: ", model.advection === nothing ? "none" : nameof(typeof(model.advection)))
    println(io, "├── coupling: ", monobanded_coupling_summary(model.coupling))
    println(io, "├── sources: ", monobanded_sources_summary(model.sources))
    println(io, "├── timestepper: ", monobanded_timestepper_name(model.timestepper))
    println(io, "├── clock: time=", model.clock.time, ", iteration=", model.clock.iteration)
    print(io,   "└── safeguards: minimum_action=", model.minimum_action,
                ", minimum_wavenumber=", model.minimum_wavenumber)
end

active_monobanded_k(field) = first(axes(field.data, 3))
monobanded_data_offsets(field) = ntuple(d -> first(axes(field.data, d)) - 1, 3)

function launch_monobanded_kernel!(kernel!, reference::Field, args...)
    Nx, Ny = horizontal_size(reference.grid)
    k = active_monobanded_k(reference)
    Ox, Oy, Oz = monobanded_data_offsets(reference)
    arch = architecture(reference.grid)
    kernel = kernel!(device(arch), (16, 16), (Nx, Ny))
    kernel(args..., reference.grid, Nx, Ny, k, Ox, Oy, Oz)
    KernelAbstractions.synchronize(device(arch))
    return nothing
end

monobanded_parent(field) = parent(field.data)

@inline monobanded_data_index(i, offset) = i - offset
@inline monobanded_left_index(i, N, periodic) = ifelse(i == 1, ifelse(periodic, N, 1), i - 1)
@inline monobanded_right_index(i, N, periodic) = ifelse(i == N, ifelse(periodic, 1, N), i + 1)
@inline monobanded_stencil_index(i, N, periodic) = ifelse(periodic, _periodic(i, N), _clamp_idx(i, N))

@inline function monobanded_x_flux(q, Cx, iₗ, iᵣ, j, k, Nx, Ox, Oy, Oz, xperiodic, use_weno)
    ixₗ = monobanded_data_index(iₗ, Ox)
    ixᵣ = monobanded_data_index(iᵣ, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)
    half = inv(eltype(q)(2))
    u = half * (Cx[ixₗ, jy, kz] + Cx[ixᵣ, jy, kz])
    q_upwind = ifelse(u >= zero(u), q[ixₗ, jy, kz], q[ixᵣ, jy, kz])

    im2 = monobanded_data_index(monobanded_stencil_index(iₗ - 2, Nx, xperiodic), Ox)
    im1 = monobanded_data_index(monobanded_stencil_index(iₗ - 1, Nx, xperiodic), Ox)
    i0  = monobanded_data_index(monobanded_stencil_index(iₗ,     Nx, xperiodic), Ox)
    ip1 = monobanded_data_index(monobanded_stencil_index(iₗ + 1, Nx, xperiodic), Ox)
    ip2 = monobanded_data_index(monobanded_stencil_index(iₗ + 2, Nx, xperiodic), Ox)
    ip3 = monobanded_data_index(monobanded_stencil_index(iₗ + 3, Nx, xperiodic), Ox)
    has_stencil = ifelse(xperiodic, true, (iₗ - 2 >= 1) & (iₗ + 3 <= Nx))
    q_weno = weno5_face_iphalf(q[im2, jy, kz], q[im1, jy, kz],
                               q[i0, jy, kz], q[ip1, jy, kz],
                               q[ip2, jy, kz], q[ip3, jy, kz],
                               u, has_stencil)

    return u * ifelse(use_weno, q_weno, q_upwind)
end

@inline function monobanded_y_flux(q, Cy, i, jₗ, jᵣ, k, Ny, Ox, Oy, Oz, yperiodic, use_weno)
    ix = monobanded_data_index(i, Ox)
    jyₗ = monobanded_data_index(jₗ, Oy)
    jyᵣ = monobanded_data_index(jᵣ, Oy)
    kz = monobanded_data_index(k, Oz)
    half = inv(eltype(q)(2))
    v = half * (Cy[ix, jyₗ, kz] + Cy[ix, jyᵣ, kz])
    q_upwind = ifelse(v >= zero(v), q[ix, jyₗ, kz], q[ix, jyᵣ, kz])

    jm2 = monobanded_data_index(monobanded_stencil_index(jₗ - 2, Ny, yperiodic), Oy)
    jm1 = monobanded_data_index(monobanded_stencil_index(jₗ - 1, Ny, yperiodic), Oy)
    j0  = monobanded_data_index(monobanded_stencil_index(jₗ,     Ny, yperiodic), Oy)
    jp1 = monobanded_data_index(monobanded_stencil_index(jₗ + 1, Ny, yperiodic), Oy)
    jp2 = monobanded_data_index(monobanded_stencil_index(jₗ + 2, Ny, yperiodic), Oy)
    jp3 = monobanded_data_index(monobanded_stencil_index(jₗ + 3, Ny, yperiodic), Oy)
    has_stencil = ifelse(yperiodic, true, (jₗ - 2 >= 1) & (jₗ + 3 <= Ny))
    q_weno = weno5_face_iphalf(q[ix, jm2, kz], q[ix, jm1, kz],
                               q[ix, j0, kz], q[ix, jp1, kz],
                               q[ix, jp2, kz], q[ix, jp3, kz],
                               v, has_stencil)

    return v * ifelse(use_weno, q_weno, q_upwind)
end

@inline function monobanded_transport_divergence(q, Cx, Cy, grid, i, j, k,
                                                 Nx, Ny, Ox, Oy, Oz,
                                                 xperiodic, yperiodic, xflat, yflat, use_weno)
    i₋ = monobanded_left_index(i, Nx, xperiodic)
    i₊ = monobanded_right_index(i, Nx, xperiodic)
    j₋ = monobanded_left_index(j, Ny, yperiodic)
    j₊ = monobanded_right_index(j, Ny, yperiodic)

    Fᵢ₊ = monobanded_x_flux(q, Cx, i, i₊, j, k, Nx, Ox, Oy, Oz, xperiodic, use_weno)
    Fᵢ₋ = monobanded_x_flux(q, Cx, i₋, i, j, k, Nx, Ox, Oy, Oz, xperiodic, use_weno)
    Fⱼ₊ = monobanded_y_flux(q, Cy, i, j, j₊, k, Ny, Ox, Oy, Oz, yperiodic, use_weno)
    Fⱼ₋ = monobanded_y_flux(q, Cy, i, j₋, j, k, Ny, Ox, Oy, Oz, yperiodic, use_weno)

    Fᵢ₋ = ifelse(xflat | ((i == 1) & !xperiodic), zero(Fᵢ₋), Fᵢ₋)
    Fᵢ₊ = ifelse(xflat | ((i == Nx) & !xperiodic), zero(Fᵢ₊), Fᵢ₊)
    Fⱼ₋ = ifelse(yflat | ((j == 1) & !yperiodic), zero(Fⱼ₋), Fⱼ₋)
    Fⱼ₊ = ifelse(yflat | ((j == Ny) & !yperiodic), zero(Fⱼ₊), Fⱼ₊)

    Δx = Δxᶜᵃᵃ(i, j, k, grid)
    Δy = Δyᵃᶜᵃ(i, j, k, grid)
    safe_Δx = ifelse(xflat, one(Δx), Δx)
    safe_Δy = ifelse(yflat, one(Δy), Δy)
    x_divergence = ifelse(xflat, zero(Fᵢ₊), (Fᵢ₊ - Fᵢ₋) / safe_Δx)
    y_divergence = ifelse(yflat, zero(Fⱼ₊), (Fⱼ₊ - Fⱼ₋) / safe_Δy)

    return x_divergence + y_divergence
end

@inline function monobanded_centered_gradient_x(q, grid, i, j, k, Nx, Ox, Oy, Oz, xperiodic, xflat)
    i₋ = monobanded_left_index(i, Nx, xperiodic)
    i₊ = monobanded_right_index(i, Nx, xperiodic)
    ix₋ = monobanded_data_index(i₋, Ox)
    ix₊ = monobanded_data_index(i₊, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)
    numerator = q[ix₊, jy, kz] - q[ix₋, jy, kz]
    Δx = Δxᶜᵃᵃ(i, j, k, grid)
    denominator = (one(eltype(q)) + one(eltype(q))) * ifelse(xflat, one(Δx), Δx)
    gradient = numerator / denominator
    return ifelse(xflat, zero(gradient),
                  ifelse((i == 1) & !xperiodic, zero(gradient),
                         ifelse((i == Nx) & !xperiodic, zero(gradient), gradient)))
end

@inline function monobanded_centered_gradient_y(q, grid, i, j, k, Ny, Ox, Oy, Oz, yperiodic, yflat)
    j₋ = monobanded_left_index(j, Ny, yperiodic)
    j₊ = monobanded_right_index(j, Ny, yperiodic)
    ix = monobanded_data_index(i, Ox)
    jy₋ = monobanded_data_index(j₋, Oy)
    jy₊ = monobanded_data_index(j₊, Oy)
    kz = monobanded_data_index(k, Oz)
    numerator = q[ix, jy₊, kz] - q[ix, jy₋, kz]
    Δy = Δyᵃᶜᵃ(i, j, k, grid)
    denominator = (one(eltype(q)) + one(eltype(q))) * ifelse(yflat, one(Δy), Δy)
    gradient = numerator / denominator
    return ifelse(yflat, zero(gradient),
                  ifelse((j == 1) & !yperiodic, zero(gradient),
                         ifelse((j == Ny) & !yperiodic, zero(gradient), gradient)))
end

function update_monobanded_local_diagnostics!(model::MonobandedWaveModel, ::Nothing)
    diagnostics = model.diagnostics
    launch_monobanded_kernel!(_monobanded_local_diagnostics!, model.action,
                              monobanded_parent(model.action),
                              monobanded_parent(model.wavenumber_moment.x),
                              monobanded_parent(model.wavenumber_moment.y),
                              monobanded_parent(diagnostics.Kx),
                              monobanded_parent(diagnostics.Ky),
                              monobanded_parent(diagnostics.κ),
                              monobanded_parent(diagnostics.uᴰx),
                              monobanded_parent(diagnostics.uᴰy),
                              monobanded_parent(diagnostics.Hx),
                              monobanded_parent(diagnostics.Hy),
                              monobanded_parent(diagnostics.Cx),
                              monobanded_parent(diagnostics.Cy),
                              monobanded_parent(diagnostics.Ω),
                              monobanded_parent(diagnostics.Ĉx),
                              monobanded_parent(diagnostics.Ĉy),
                              model.gravitational_acceleration,
                              model.minimum_action,
                              model.minimum_wavenumber)
    return model
end

function update_monobanded_prescribed_current_local_diagnostics!(model::MonobandedWaveModel,
                                                                 uᴸ, vᴸ, depth, qtransform)
    diagnostics = model.diagnostics
    Nx, Ny, Nz = size(uᴸ)
    size(vᴸ) == size(uᴸ) || throw(ArgumentError("u and v current fields must have matching size"))
    horizontal_size(model.grid) == (Nx, Ny) ||
        throw(ArgumentError("current fields must match the monobanded model grid horizontally"))

    faces = vertical_faces(qtransform)
    length(faces) == Nz + 1 ||
        throw(ArgumentError("Q-transform vertical grid does not match velocity fields"))

    arch = architecture(qtransform.grid)
    faces = on_architecture(arch, faces)
    depth = q_depth_on_architecture(arch, depth)

    launch_monobanded_kernel!(_monobanded_prescribed_current_local_diagnostics!, model.action,
                              monobanded_parent(model.action),
                              monobanded_parent(model.wavenumber_moment.x),
                              monobanded_parent(model.wavenumber_moment.y),
                              monobanded_parent(diagnostics.Kx),
                              monobanded_parent(diagnostics.Ky),
                              monobanded_parent(diagnostics.κ),
                              monobanded_parent(diagnostics.uᴰx),
                              monobanded_parent(diagnostics.uᴰy),
                              monobanded_parent(diagnostics.Hx),
                              monobanded_parent(diagnostics.Hy),
                              monobanded_parent(diagnostics.Cx),
                              monobanded_parent(diagnostics.Cy),
                              monobanded_parent(diagnostics.Ω),
                              monobanded_parent(diagnostics.Ĉx),
                              monobanded_parent(diagnostics.Ĉy),
                              uᴸ, vᴸ, depth, faces,
                              qtransform.kernel,
                              OnTheFlyQ(),
                              Nz,
                              model.gravitational_acceleration,
                              model.minimum_action,
                              model.minimum_wavenumber)
    return model
end

function update_monobanded_local_diagnostics!(model::MonobandedWaveModel,
                                              coupling::MonobandedPrescribedCurrentCoupling)
    current = coupling.current
    return update_monobanded_prescribed_current_local_diagnostics!(model,
                                                                  current_data(current.u),
                                                                  current_data(current.v),
                                                                  current.depth,
                                                                  coupling.qtransform)
end

function update_monobanded_local_diagnostics!(model::MonobandedWaveModel,
                                              coupling::MonobandedPseudomomentumCoupling)
    # PseudomomentumVelocities feeds the wave-induced pseudomomentum back as
    # the Lagrangian-mean "current" seen by the wave field:
    #
    #  1. Compute κ via the no-current diagnostic (κ only depends on A, AK).
    #  2. Project the moment AK onto Q^κ to get cell-averaged pseudomomentum
    #     p_x, p_y stored on `coupling.px`, `coupling.py`.
    #  3. Reuse the prescribed-current diagnostic kernel with (p_x, p_y) in
    #     the role of (uᴸ, vᴸ). This computes u^Dκ, H, C, Ω, Ĉ self-consistently.
    update_monobanded_local_diagnostics!(model, nothing)
    compute_pseudomomentum_cell_averages!(coupling.px, coupling.py,
                                          model, coupling.depth, coupling.qtransform;
                                          update_diagnostics=false)
    return update_monobanded_prescribed_current_local_diagnostics!(model,
                                                                  field_storage(coupling.px),
                                                                  field_storage(coupling.py),
                                                                  coupling.depth,
                                                                  coupling.qtransform)
end

update_coupling!(coupling::MonobandedPrescribedCurrentCoupling, model::MonobandedWaveModel) =
    coupling

function update_coupling!(coupling::MonobandedPseudomomentumCoupling, model::MonobandedWaveModel)
    update_monobanded_diagnostics!(model)
    return coupling
end

function update_monobanded_diagnostics!(model::MonobandedWaveModel)
    diagnostics = model.diagnostics
    update_monobanded_local_diagnostics!(model, model.coupling)

    fill_halo_regions!((Kx=diagnostics.Kx, Ky=diagnostics.Ky, κ=diagnostics.κ,
                        uᴰx=diagnostics.uᴰx, uᴰy=diagnostics.uᴰy,
                        Hx=diagnostics.Hx, Hy=diagnostics.Hy))

    topology = Oceananigans.Grids.topology(model.grid)
    xperiodic = topology[1] === Periodic
    yperiodic = topology[2] === Periodic
    xflat = topology[1] === Flat
    yflat = topology[2] === Flat

    launch_monobanded_kernel!(_monobanded_gradient_diagnostics!, model.action,
                              monobanded_parent(diagnostics.Kx),
                              monobanded_parent(diagnostics.Ky),
                              monobanded_parent(diagnostics.κ),
                              monobanded_parent(diagnostics.uᴰx),
                              monobanded_parent(diagnostics.uᴰy),
                              monobanded_parent(diagnostics.Hx),
                              monobanded_parent(diagnostics.Hy),
                              monobanded_parent(diagnostics.Γxx),
                              monobanded_parent(diagnostics.Γyx),
                              monobanded_parent(diagnostics.Γxy),
                              monobanded_parent(diagnostics.Γyy),
                              monobanded_parent(diagnostics.ZK),
                              xperiodic, yperiodic, xflat, yflat)
    return model
end

function monobanded_pseudomomentum_context(model::MonobandedWaveModel, ::Nothing)
    qtransform = QTransform(QKernel(eltype(model)), model.grid)
    return qtransform, grid_depth(model.grid)
end

monobanded_pseudomomentum_context(model::MonobandedWaveModel,
                                  coupling::MonobandedPrescribedCurrentCoupling) =
    coupling.qtransform, coupling.current.depth

monobanded_pseudomomentum_context(model::MonobandedWaveModel,
                                  coupling::MonobandedPseudomomentumCoupling) =
    coupling.qtransform, coupling.depth

function compute_pseudomomentum_cell_averages!(px, py,
                                               model::MonobandedWaveModel,
                                               depth,
                                               qtransform::QTransform;
                                               update_diagnostics=true)
    update_diagnostics && update_monobanded_diagnostics!(model)

    px_data = field_storage(px)
    py_data = field_storage(py)
    Nx, Ny, Nz = size(px_data)
    size(py_data) == (Nx, Ny, Nz) ||
        throw(ArgumentError("monobanded pseudomomentum fields must have matching size"))
    horizontal_size(model.grid) == (Nx, Ny) ||
        throw(ArgumentError("monobanded pseudomomentum fields must match the model grid horizontally"))
    qtransform.grid.Nz == Nz ||
        throw(ArgumentError("monobanded pseudomomentum fields must use the Q-transform vertical grid"))

    arch = architecture(px)
    faces = on_architecture(arch, vertical_faces(qtransform))
    depth = q_depth_on_architecture(arch, depth)
    k = active_monobanded_k(model.action)
    Ox, Oy, Oz = monobanded_data_offsets(model.action)

    kernel = _monobanded_pseudomomentum_cells_kernel!(device(arch), (8, 8, 1), (Nx, Ny, Nz))
    kernel(px_data, py_data,
           monobanded_parent(model.wavenumber_moment.x),
           monobanded_parent(model.wavenumber_moment.y),
           monobanded_parent(model.diagnostics.κ),
           depth, faces, qtransform.kernel, OnTheFlyQ(),
           k, Ox, Oy, Oz)
    KernelAbstractions.synchronize(device(arch))

    return px, py
end

function pseudomomentum_fields(model::MonobandedWaveModel;
                               location=(Center, Center, Center))
    qtransform, depth = monobanded_pseudomomentum_context(model, model.coupling)
    px = pseudomomentum_field(qtransform.grid; location, eltype=eltype(model))
    py = pseudomomentum_field(qtransform.grid; location, eltype=eltype(model))
    compute_pseudomomentum_cell_averages!(px, py, model, depth, qtransform)
    return px, py
end

@kernel function _monobanded_pseudomomentum_cells_kernel!(px, py, AKx, AKy, κ,
                                                          depth, faces, qkernel, qpolicy,
                                                          surface_k, Ox, Oy, Oz)
    i, j, k = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(surface_k, Oz)
    d = q_depth_at(depth, i, j)
    z₁ = faces[k]
    z₂ = faces[k+1]

    @inbounds begin
        qΔz = q_cell_weight_kernel(qpolicy, qkernel, i, j, k, 1, κ[ix, jy, kz], z₁, z₂, d)
        scale = inv(abs(z₂ - z₁))
        px[i, j, k] = AKx[ix, jy, kz] * qΔz * scale
        py[i, j, k] = AKy[ix, jy, kz] * qΔz * scale
    end
end

@kernel function _monobanded_local_diagnostics!(A, AKx, AKy,
                                                Kx, Ky, κ,
                                                uᴰx, uᴰy, Hx, Hy, Cx, Cy,
                                                Ω, Ĉx, Ĉy,
                                                g, minimum_action, minimum_wavenumber,
                                                grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds begin
        # K = AK/A regularized by `max(A, minimum_action)`. The previous
        # ifelse(abs(A) > min_A, A, min_A) form let the denominator flip sign
        # when A was transiently negative (e.g. WENO overshoot before the
        # positivity clamp lands), which silently flipped K.
        Aᵢ = A[ix, jy, kz]
        denominator = max(Aᵢ, minimum_action)
        Kxᵢ = AKx[ix, jy, kz] / denominator
        Kyᵢ = AKy[ix, jy, kz] / denominator
        κ_raw = sqrt(Kxᵢ * Kxᵢ + Kyᵢ * Kyᵢ)
        κᵢ = ifelse(κ_raw > minimum_wavenumber, κ_raw, minimum_wavenumber)
        fourth = (one(g) + one(g)) * (one(g) + one(g))
        cg_over_κ = sqrt(g / (fourth * κᵢ * κᵢ * κᵢ))

        Kx[ix, jy, kz] = Kxᵢ
        Ky[ix, jy, kz] = Kyᵢ
        κ[ix, jy, kz] = κᵢ

        # No-current branch: uᴰ and H vanish. Γ is overwritten by the gradient
        # kernel later, so we do not zero it here.
        uᴰx[ix, jy, kz] = zero(Aᵢ)
        uᴰy[ix, jy, kz] = zero(Aᵢ)
        Hx[ix, jy, kz] = zero(Aᵢ)
        Hy[ix, jy, kz] = zero(Aᵢ)

        Cxᵢ = cg_over_κ * Kxᵢ
        Cyᵢ = cg_over_κ * Kyᵢ
        Cx[ix, jy, kz] = Cxᵢ
        Cy[ix, jy, kz] = Cyᵢ
        Ω[ix, jy, kz] = sqrt(g * κᵢ)
        Ĉx[ix, jy, kz] = Cxᵢ
        Ĉy[ix, jy, kz] = Cyᵢ
    end
end

@kernel function _monobanded_prescribed_current_local_diagnostics!(A, AKx, AKy,
                                                                  Kx, Ky, κ,
                                                                  uᴰx, uᴰy, Hx, Hy,
                                                                  Cx, Cy, Ω, Ĉx, Ĉy,
                                                                  uᴸ, vᴸ, depth, faces,
                                                                  qkernel, qpolicy, Nz,
                                                                  g, minimum_action, minimum_wavenumber,
                                                                  grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)
    d = q_depth_at(depth, i, j)

    @inbounds begin
        Aᵢ = A[ix, jy, kz]
        denominator = max(Aᵢ, minimum_action)
        Kxᵢ = AKx[ix, jy, kz] / denominator
        Kyᵢ = AKy[ix, jy, kz] / denominator
        κ_raw = sqrt(Kxᵢ * Kxᵢ + Kyᵢ * Kyᵢ)
        κᵢ = ifelse(κ_raw > minimum_wavenumber, κ_raw, minimum_wavenumber)
        fourth = (one(g) + one(g)) * (one(g) + one(g))
        cg_over_κ = sqrt(g / (fourth * κᵢ * κᵢ * κᵢ))

        uᴰxᵢ = zero(Aᵢ)
        uᴰyᵢ = zero(Aᵢ)
        Hxᵢ = zero(Aᵢ)
        Hyᵢ = zero(Aᵢ)

        for ℓ in 1:Nz
            qΔz = q_cell_weight_kernel(qpolicy, qkernel, i, j, ℓ, 1, κᵢ, faces[ℓ], faces[ℓ+1], d)
            dqΔz = q_cell_weight_kappa_derivative_kernel(qpolicy, qkernel, i, j, ℓ, 1, κᵢ, faces[ℓ], faces[ℓ+1], d)
            uᵢ = uᴸ[i, j, ℓ]
            vᵢ = vᴸ[i, j, ℓ]
            uᴰxᵢ += uᵢ * qΔz
            uᴰyᵢ += vᵢ * qΔz
            Hxᵢ += uᵢ * dqΔz
            Hyᵢ += vᵢ * dqΔz
        end

        KH_over_κ = (Kxᵢ * Hxᵢ + Kyᵢ * Hyᵢ) / κᵢ
        Ĉxᵢ = cg_over_κ * Kxᵢ + uᴰxᵢ
        Ĉyᵢ = cg_over_κ * Kyᵢ + uᴰyᵢ
        Cxᵢ = Ĉxᵢ + KH_over_κ * Kxᵢ
        Cyᵢ = Ĉyᵢ + KH_over_κ * Kyᵢ

        Kx[ix, jy, kz] = Kxᵢ
        Ky[ix, jy, kz] = Kyᵢ
        κ[ix, jy, kz] = κᵢ
        uᴰx[ix, jy, kz] = uᴰxᵢ
        uᴰy[ix, jy, kz] = uᴰyᵢ
        Hx[ix, jy, kz] = Hxᵢ
        Hy[ix, jy, kz] = Hyᵢ
        Cx[ix, jy, kz] = Cxᵢ
        Cy[ix, jy, kz] = Cyᵢ
        Ω[ix, jy, kz] = sqrt(g * κᵢ) + Kxᵢ * uᴰxᵢ + Kyᵢ * uᴰyᵢ
        Ĉx[ix, jy, kz] = Ĉxᵢ
        Ĉy[ix, jy, kz] = Ĉyᵢ
    end
end

@kernel function _monobanded_gradient_diagnostics!(Kx, Ky, κ, uᴰx, uᴰy, Hx, Hy,
                                                   Γxx, Γyx, Γxy, Γyy, ZK,
                                                   xperiodic, yperiodic, xflat, yflat,
                                                   grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds begin
        ∂xKy = monobanded_centered_gradient_x(Ky, grid, i, j, k, Nx, Ox, Oy, Oz, xperiodic, xflat)
        ∂yKx = monobanded_centered_gradient_y(Kx, grid, i, j, k, Ny, Ox, Oy, Oz, yperiodic, yflat)
        ∂xκ = monobanded_centered_gradient_x(κ, grid, i, j, k, Nx, Ox, Oy, Oz, xperiodic, xflat)
        ∂yκ = monobanded_centered_gradient_y(κ, grid, i, j, k, Ny, Ox, Oy, Oz, yperiodic, yflat)
        ∂x_uᴰx = monobanded_centered_gradient_x(uᴰx, grid, i, j, k, Nx, Ox, Oy, Oz, xperiodic, xflat)
        ∂y_uᴰx = monobanded_centered_gradient_y(uᴰx, grid, i, j, k, Ny, Ox, Oy, Oz, yperiodic, yflat)
        ∂x_uᴰy = monobanded_centered_gradient_x(uᴰy, grid, i, j, k, Nx, Ox, Oy, Oz, xperiodic, xflat)
        ∂y_uᴰy = monobanded_centered_gradient_y(uᴰy, grid, i, j, k, Ny, Ox, Oy, Oz, yperiodic, yflat)

        Hxᵢ = Hx[ix, jy, kz]
        Hyᵢ = Hy[ix, jy, kz]
        Γxx[ix, jy, kz] = ∂x_uᴰx - Hxᵢ * ∂xκ
        Γyx[ix, jy, kz] = ∂x_uᴰy - Hyᵢ * ∂xκ
        Γxy[ix, jy, kz] = ∂y_uᴰx - Hxᵢ * ∂yκ
        Γyy[ix, jy, kz] = ∂y_uᴰy - Hyᵢ * ∂yκ
        ZK[ix, jy, kz] = ∂xKy - ∂yKx
    end
end

function zero_monobanded_tendencies!(G)
    set!(G.A, zero(eltype(G.A)))
    set!(G.AKx, zero(eltype(G.AKx)))
    set!(G.AKy, zero(eltype(G.AKy)))
    return G
end

function compute_monobanded_transport_tendency!(G, model)
    diagnostics = model.diagnostics
    topology = Oceananigans.Grids.topology(model.grid)
    xperiodic = topology[1] === Periodic
    yperiodic = topology[2] === Periodic
    xflat = topology[1] === Flat
    yflat = topology[2] === Flat
    use_weno = model.advection isa WENO

    launch_monobanded_kernel!(_monobanded_transport_tendency!, model.action,
                              monobanded_parent(G.A),
                              monobanded_parent(G.AKx),
                              monobanded_parent(G.AKy),
                              monobanded_parent(model.action),
                              monobanded_parent(model.wavenumber_moment.x),
                              monobanded_parent(model.wavenumber_moment.y),
                              monobanded_parent(diagnostics.Cx),
                              monobanded_parent(diagnostics.Cy),
                              xperiodic, yperiodic, xflat, yflat, use_weno)
    return G
end

@kernel function _monobanded_transport_tendency!(GA, GAKx, GAKy,
                                                 A, AKx, AKy, Cx, Cy,
                                                 xperiodic, yperiodic, xflat, yflat, use_weno,
                                                 grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds begin
        GA[ix, jy, kz] = -monobanded_transport_divergence(A, Cx, Cy, grid, i, j, k,
                                                          Nx, Ny, Ox, Oy, Oz,
                                                          xperiodic, yperiodic, xflat, yflat, use_weno)
        GAKx[ix, jy, kz] = -monobanded_transport_divergence(AKx, Cx, Cy, grid, i, j, k,
                                                            Nx, Ny, Ox, Oy, Oz,
                                                            xperiodic, yperiodic, xflat, yflat, use_weno)
        GAKy[ix, jy, kz] = -monobanded_transport_divergence(AKy, Cx, Cy, grid, i, j, k,
                                                            Nx, Ny, Ox, Oy, Oz,
                                                            xperiodic, yperiodic, xflat, yflat, use_weno)
    end
end

# Hamiltonian refraction tendency on the moment equations: −A·K_β·Γ_{β,α}.
# Runs unconditionally; it is not a transport flux and must remain active
# when `model.advection === nothing`.
function compute_monobanded_refraction_tendency!(G, model)
    diagnostics = model.diagnostics
    launch_monobanded_kernel!(_monobanded_refraction_tendency!, model.action,
                              monobanded_parent(G.AKx),
                              monobanded_parent(G.AKy),
                              monobanded_parent(model.action),
                              monobanded_parent(diagnostics.Kx),
                              monobanded_parent(diagnostics.Ky),
                              monobanded_parent(diagnostics.Γxx),
                              monobanded_parent(diagnostics.Γyx),
                              monobanded_parent(diagnostics.Γxy),
                              monobanded_parent(diagnostics.Γyy))
    return G
end

@kernel function _monobanded_refraction_tendency!(GAKx, GAKy,
                                                  A, Kx, Ky,
                                                  Γxx, Γyx, Γxy, Γyy,
                                                  grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds begin
        Aᵢ = A[ix, jy, kz]
        Kxᵢ = Kx[ix, jy, kz]
        Kyᵢ = Ky[ix, jy, kz]
        GAKx[ix, jy, kz] -= Aᵢ * (Kxᵢ * Γxx[ix, jy, kz] + Kyᵢ * Γyx[ix, jy, kz])
        GAKy[ix, jy, kz] -= Aᵢ * (Kxᵢ * Γxy[ix, jy, kz] + Kyᵢ * Γyy[ix, jy, kz])
    end
end

add_monobanded_sources!(G, model::MonobandedWaveModel, ::Nothing) = G

function add_monobanded_sources!(G, model::MonobandedWaveModel, sources)
    launch_monobanded_kernel!(_monobanded_source_tendency!, model.action,
                              monobanded_parent(G.A),
                              monobanded_parent(G.AKx),
                              monobanded_parent(G.AKy),
                              monobanded_parent(model.action),
                              monobanded_parent(model.diagnostics.Kx),
                              monobanded_parent(model.diagnostics.Ky),
                              monobanded_parent(model.diagnostics.κ),
                              sources)
    return G
end

@inline monobanded_source_rate(source::LinearWindInput, A, κ, i, j) =
    source.rate

@inline monobanded_bottom_depth_factor(source::BottomFriction{Rate, Nothing}, i, j) where Rate =
    one(source.reference_depth)

@inline function monobanded_bottom_depth_factor(source::BottomFriction{Rate, Depth}, i, j) where {Rate, Depth<:Number}
    depth = max(source.depth, source.minimum_depth)
    return (source.reference_depth / depth)^source.depth_power
end

@inline function monobanded_source_rate(source::BottomFriction, A, κ, i, j)
    wavenumber_factor = (κ / source.reference_wavenumber)^source.wavenumber_power
    damping = source.rate * monobanded_bottom_depth_factor(source, i, j) * wavenumber_factor
    return -damping
end

@inline function monobanded_source_rate(sources::SourceTermSet, A, κ, i, j)
    rate = zero(A)
    for source in sources.terms
        rate += monobanded_source_rate(source, A, κ, i, j)
    end
    return rate
end

@kernel function _monobanded_source_tendency!(GA, GAKx, GAKy,
                                              A, Kx, Ky, κ, sources,
                                              grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds begin
        Aᵢ = A[ix, jy, kz]
        Kxᵢ = Kx[ix, jy, kz]
        Kyᵢ = Ky[ix, jy, kz]
        source_A = monobanded_source_rate(sources, Aᵢ, κ[ix, jy, kz], i, j) * Aᵢ
        GA[ix, jy, kz] += source_A
        GAKx[ix, jy, kz] += Kxᵢ * source_A
        GAKy[ix, jy, kz] += Kyᵢ * source_A
    end
end

function compute_tendencies!(G::NamedTuple, model::MonobandedWaveModel)
    update_monobanded_diagnostics!(model)
    if model.advection === nothing
        zero_monobanded_tendencies!(G)
    else
        compute_monobanded_transport_tendency!(G, model)
    end
    compute_monobanded_refraction_tendency!(G, model)
    return add_monobanded_sources!(G, model, model.sources)
end

compute_tendencies!(model::MonobandedWaveModel) = compute_tendencies!(model.timestepper.Gⁿ, model)

function cfl(model::MonobandedWaveModel)
    FT = eltype(model)
    model.advection === nothing && return zero(FT)

    update_monobanded_diagnostics!(model)

    Δt = convert(FT, model.clock.last_Δt)
    topology = Oceananigans.Grids.topology(model.grid)
    xflat = topology[1] === Flat
    yflat = topology[2] === Flat

    Cx_data = interior(model.diagnostics.Cx)
    Cy_data = interior(model.diagnostics.Cy)

    inv_Δx = xflat ? zero(FT) : inv(convert(FT, first(xspacings(model.grid))))
    inv_Δy = yflat ? zero(FT) : inv(convert(FT, first(yspacings(model.grid))))

    # Per-cell CFL: max_i (|Cx_i|/Δx + |Cy_i|/Δy)·Δt, rather than the looser
    # bound max|Cx|/Δx + max|Cy|/Δy, which over-reports when Cx and Cy peak in
    # different cells.
    ratio = @. abs(Cx_data) * inv_Δx + abs(Cy_data) * inv_Δy
    return convert(FT, maximum(ratio))::FT * Δt
end

function set!(model::MonobandedWaveModel; A=nothing, AKx=nothing, AKy=nothing)
    A === nothing || set!(model.action, A)
    AKx === nothing || set!(model.wavenumber_moment.x, AKx)
    AKy === nothing || set!(model.wavenumber_moment.y, AKy)

    if A !== nothing || AKx !== nothing || AKy !== nothing
        model.previous_tendencies_ready = false
        update_monobanded_diagnostics!(model)
    end

    return model
end

function monobanded_update_field!(field, tendency, dt, clamp_nonnegative)
    launch_monobanded_kernel!(_monobanded_update_field!, field,
                              monobanded_parent(field),
                              monobanded_parent(tendency),
                              convert(eltype(field), dt),
                              clamp_nonnegative)
    return field
end

@kernel function _monobanded_update_field!(field, tendency, dt, clamp_nonnegative,
                                           grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds begin
        value = field[ix, jy, kz] + dt * tendency[ix, jy, kz]
        field[ix, jy, kz] = ifelse(clamp_nonnegative, max(zero(value), value), value)
    end
end

function monobanded_update_ab2_field!(field, tendency, previous_tendency, dt, clamp_nonnegative)
    launch_monobanded_kernel!(_monobanded_update_ab2_field!, field,
                              monobanded_parent(field),
                              monobanded_parent(tendency),
                              monobanded_parent(previous_tendency),
                              convert(eltype(field), dt),
                              clamp_nonnegative)
    return field
end

@kernel function _monobanded_update_ab2_field!(field, tendency, previous_tendency, dt, clamp_nonnegative,
                                               grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds begin
        half = one(dt) / (one(dt) + one(dt))
        increment = (3 * half * tendency[ix, jy, kz] - half * previous_tendency[ix, jy, kz]) * dt
        value = field[ix, jy, kz] + increment
        field[ix, jy, kz] = ifelse(clamp_nonnegative, max(zero(value), value), value)
    end
end

function monobanded_copy_field!(dest, src)
    launch_monobanded_kernel!(_monobanded_copy_field!, dest,
                              monobanded_parent(dest),
                              monobanded_parent(src))
    return dest
end

@kernel function _monobanded_copy_field!(dest, src, grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds dest[ix, jy, kz] = src[ix, jy, kz]
end

function monobanded_copy_state!(dest, src)
    monobanded_copy_field!(dest.A, src.A)
    monobanded_copy_field!(dest.AKx, src.AKx)
    monobanded_copy_field!(dest.AKy, src.AKy)
    return dest
end

# When `A` drops below `minimum_action` in a cell, the moment fields are zeroed
# alongside the action so that `K = AK/A` stays well-defined and the state-
# dependent group velocity does not blow up. This is non-conservative by
# design (it bleeds mass at vanishing-action cells), per the plan's optional
# cleanup step. The clamp on its own (positivity-only on A) is not enough:
# once A is clamped to 0 but AKx, AKy survive, the next diagnostic update
# produces huge K and triggers a CFL violation cascade.
function monobanded_clamp_low_action!(state, minimum_action)
    launch_monobanded_kernel!(_monobanded_clamp_low_action!, state.A,
                              monobanded_parent(state.A),
                              monobanded_parent(state.AKx),
                              monobanded_parent(state.AKy),
                              convert(eltype(state.A), minimum_action))
    return state
end

@kernel function _monobanded_clamp_low_action!(A, AKx, AKy, minimum_action,
                                                grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds begin
        below = A[ix, jy, kz] < minimum_action
        A[ix, jy, kz]   = ifelse(below, zero(eltype(A)),   A[ix, jy, kz])
        AKx[ix, jy, kz] = ifelse(below, zero(eltype(AKx)), AKx[ix, jy, kz])
        AKy[ix, jy, kz] = ifelse(below, zero(eltype(AKy)), AKy[ix, jy, kz])
    end
end

function monobanded_update_state!(state, G, dt, minimum_action)
    monobanded_update_field!(state.A, G.A, dt, true)
    monobanded_update_field!(state.AKx, G.AKx, dt, false)
    monobanded_update_field!(state.AKy, G.AKy, dt, false)
    monobanded_clamp_low_action!(state, minimum_action)
    return state
end

function monobanded_add_scaled_runge_kutta_3_field!(field, G, G⁻, dt, γ, ζ, clamp_nonnegative)
    FT = eltype(field)
    launch_monobanded_kernel!(_monobanded_add_scaled_runge_kutta_3_field!, field,
                              monobanded_parent(field),
                              monobanded_parent(G),
                              monobanded_parent(G⁻),
                              convert(FT, dt),
                              convert(FT, γ),
                              convert(FT, ζ),
                              clamp_nonnegative)
    return field
end

function monobanded_add_scaled_runge_kutta_3_field!(field, G, G⁻, dt, γ, ::Nothing, clamp_nonnegative)
    FT = eltype(field)
    launch_monobanded_kernel!(_monobanded_add_scaled_runge_kutta_3_first_stage_field!, field,
                              monobanded_parent(field),
                              monobanded_parent(G),
                              convert(FT, dt),
                              convert(FT, γ),
                              clamp_nonnegative)
    return field
end

@kernel function _monobanded_add_scaled_runge_kutta_3_field!(field, G, G⁻, dt, γ, ζ,
                                                             clamp_nonnegative,
                                                             grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds begin
        value = field[ix, jy, kz] + dt * (γ * G[ix, jy, kz] + ζ * G⁻[ix, jy, kz])
        field[ix, jy, kz] = ifelse(clamp_nonnegative, max(zero(value), value), value)
    end
end

@kernel function _monobanded_add_scaled_runge_kutta_3_first_stage_field!(field, G, dt, γ,
                                                                         clamp_nonnegative,
                                                                         grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds begin
        value = field[ix, jy, kz] + dt * γ * G[ix, jy, kz]
        field[ix, jy, kz] = ifelse(clamp_nonnegative, max(zero(value), value), value)
    end
end

function monobanded_add_scaled_runge_kutta_3_state!(state, G, G⁻, dt, γ, ζ, minimum_action)
    monobanded_add_scaled_runge_kutta_3_field!(state.A, G.A, G⁻.A, dt, γ, ζ, true)
    monobanded_add_scaled_runge_kutta_3_field!(state.AKx, G.AKx, G⁻.AKx, dt, γ, ζ, false)
    monobanded_add_scaled_runge_kutta_3_field!(state.AKy, G.AKy, G⁻.AKy, dt, γ, ζ, false)
    monobanded_clamp_low_action!(state, minimum_action)
    return state
end

function monobanded_update_ab2_state!(state, G, Gprevious, dt, minimum_action)
    monobanded_update_ab2_field!(state.A, G.A, Gprevious.A, dt, true)
    monobanded_update_ab2_field!(state.AKx, G.AKx, Gprevious.AKx, dt, false)
    monobanded_update_ab2_field!(state.AKy, G.AKy, Gprevious.AKy, dt, false)
    monobanded_clamp_low_action!(state, minimum_action)
    return state
end

function monobanded_combine_field!(dest, a, A, b, B, clamp_nonnegative)
    FT = eltype(dest)
    launch_monobanded_kernel!(_monobanded_combine_field!, dest,
                              monobanded_parent(dest),
                              convert(FT, a),
                              monobanded_parent(A),
                              convert(FT, b),
                              monobanded_parent(B),
                              clamp_nonnegative)
    return dest
end

@kernel function _monobanded_combine_field!(dest, a, A, b, B, clamp_nonnegative,
                                            grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds begin
        value = a * A[ix, jy, kz] + b * B[ix, jy, kz]
        dest[ix, jy, kz] = ifelse(clamp_nonnegative, max(zero(value), value), value)
    end
end

function monobanded_combine_state!(dest, a, A, b, B, minimum_action)
    monobanded_combine_field!(dest.A, a, A.A, b, B.A, true)
    monobanded_combine_field!(dest.AKx, a, A.AKx, b, B.AKx, false)
    monobanded_combine_field!(dest.AKy, a, A.AKy, b, B.AKy, false)
    monobanded_clamp_low_action!(dest, minimum_action)
    return dest
end

function monobanded_combine_with_increment_field!(dest, a, A, b, dt, G, clamp_nonnegative)
    FT = eltype(dest)
    launch_monobanded_kernel!(_monobanded_combine_with_increment_field!, dest,
                              monobanded_parent(dest),
                              convert(FT, a),
                              monobanded_parent(A),
                              convert(FT, b),
                              convert(FT, dt),
                              monobanded_parent(G),
                              clamp_nonnegative)
    return dest
end

@kernel function _monobanded_combine_with_increment_field!(dest, a, A, b, dt, G, clamp_nonnegative,
                                                           grid, Nx, Ny, k, Ox, Oy, Oz)
    i, j = @index(Global, NTuple)
    ix = monobanded_data_index(i, Ox)
    jy = monobanded_data_index(j, Oy)
    kz = monobanded_data_index(k, Oz)

    @inbounds begin
        stage_value = dest[ix, jy, kz] + dt * G[ix, jy, kz]
        stage_value = ifelse(clamp_nonnegative, max(zero(stage_value), stage_value), stage_value)
        value = a * A[ix, jy, kz] + b * stage_value
        dest[ix, jy, kz] = ifelse(clamp_nonnegative, max(zero(value), value), value)
    end
end

function monobanded_combine_state_with_increment!(dest, a, A, b, dt, G, minimum_action)
    monobanded_combine_with_increment_field!(dest.A, a, A.A, b, dt, G.A, true)
    monobanded_combine_with_increment_field!(dest.AKx, a, A.AKx, b, dt, G.AKx, false)
    monobanded_combine_with_increment_field!(dest.AKy, a, A.AKy, b, dt, G.AKy, false)
    monobanded_clamp_low_action!(dest, minimum_action)
    return dest
end

function time_step!(model::MonobandedWaveModel, dt; callbacks=[])
    dt > 0 || throw(ArgumentError("time step must be positive"))

    state = prognostic_fields(model)
    timestepper = model.timestepper

    if timestepper isa RungeKutta3TimeStepper
        model.previous_tendencies_ready = false

        compute_tendencies!(model)
        monobanded_add_scaled_runge_kutta_3_state!(state, timestepper.Gⁿ, timestepper.G⁻,
                                                   dt, timestepper.γ¹, nothing, model.minimum_action)
        monobanded_copy_state!(timestepper.G⁻, timestepper.Gⁿ)

        compute_tendencies!(model)
        monobanded_add_scaled_runge_kutta_3_state!(state, timestepper.Gⁿ, timestepper.G⁻,
                                                   dt, timestepper.γ², timestepper.ζ², model.minimum_action)
        monobanded_copy_state!(timestepper.G⁻, timestepper.Gⁿ)

        compute_tendencies!(model)
        monobanded_add_scaled_runge_kutta_3_state!(state, timestepper.Gⁿ, timestepper.G⁻,
                                                   dt, timestepper.γ³, timestepper.ζ³, model.minimum_action)
        monobanded_copy_state!(timestepper.G⁻, timestepper.Gⁿ)
    elseif timestepper.name === :ForwardEuler
        compute_tendencies!(model)
        monobanded_update_state!(state, timestepper.Gⁿ, dt, model.minimum_action)
        model.previous_tendencies_ready = false
    elseif timestepper.name === :AB2
        compute_tendencies!(model)
        if model.previous_tendencies_ready
            monobanded_update_ab2_state!(state, timestepper.Gⁿ, timestepper.G⁻, dt, model.minimum_action)
        else
            monobanded_update_state!(state, timestepper.Gⁿ, dt, model.minimum_action)
        end
        monobanded_copy_state!(timestepper.G⁻, timestepper.Gⁿ)
        model.previous_tendencies_ready = true
    else
        throw(ArgumentError("unsupported timestepper $(timestepper)"))
    end

    update_monobanded_diagnostics!(model)
    tick!(model.clock, dt)
    return model
end

update_state!(model::MonobandedWaveModel; callbacks=[], kwargs...) =
    (update_monobanded_diagnostics!(model); model)
