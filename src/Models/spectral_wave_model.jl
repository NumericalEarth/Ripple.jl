import Oceananigans
import Oceananigans: AbstractModel, fields, prognostic_fields
import Oceananigans.Architectures: architecture
import Oceananigans.Advection: WENO
import Oceananigans.TimeSteppers: Clock, RungeKutta3TimeStepper

validate_model_clock(clock::Clock) = clock
validate_model_clock(clock) =
    throw(ArgumentError("clock must be a Clock; got $(typeof(clock))"))

canonical_model_sources(sources) = sources
canonical_model_sources(::NoSource) = nothing
canonical_model_sources(sources::SourceTermSet) = isempty(sources) ? nothing : sources

validate_model_sources(::Nothing) = nothing

function validate_model_sources(sources::SourceTermSet)
    for source in sources
        source isa AbstractSourceTerm ||
            throw(ArgumentError("SourceTermSet terms must be AbstractSourceTerm instances; got $(typeof(source))"))
    end

    return sources
end

function validate_model_sources(sources)
    sources isa AbstractSourceTerm ||
        throw(ArgumentError("sources must be nothing or an AbstractSourceTerm; got $(typeof(sources))"))
    return sources
end

canonical_model_coupling(coupling) = coupling
canonical_model_coupling(::NoCurrentCoupling) = nothing

validate_model_physical_grid(grid::AbstractGrid) = grid
validate_model_physical_grid(grid) =
    throw(ArgumentError("grid must be an Oceananigans grid; got $(typeof(grid))"))

validate_model_spectral_grid(spectral_grid::AbstractSpectralGrid) = spectral_grid
validate_model_spectral_grid(spectral_grid) =
    throw(ArgumentError("spectral_grid must be an AbstractSpectralGrid; got $(typeof(spectral_grid))"))

compatible_model_physical_grid(a, b) =
    a === b || (horizontal_size(a) == horizontal_size(b) &&
                vertical_size(a) == vertical_size(b) &&
                xnodes(a) == xnodes(b) &&
                ynodes(a) == ynodes(b) &&
                znodes(a) == znodes(b) &&
                xfaces(a) == xfaces(b) &&
                yfaces(a) == yfaces(b) &&
                zfaces(a) == zfaces(b) &&
                OceanGrids.topology(a) == OceanGrids.topology(b))

compatible_model_spectral_grid(a, b) =
    a === b || (typeof(a) === typeof(b) &&
                coordinate_size(a) == coordinate_size(b) &&
                coordinate_centers(a, 1) == coordinate_centers(b, 1) &&
                coordinate_centers(a, 2) == coordinate_centers(b, 2) &&
                coordinate_faces(a, 1) == coordinate_faces(b, 1) &&
                coordinate_faces(a, 2) == coordinate_faces(b, 2) &&
                spectral_weights(a) == spectral_weights(b) &&
                a.boundary_conditions == b.boundary_conditions)

function validate_model_action(action::ProductField, grid, spectral_grid)
    compatible_model_physical_grid(physical_grid(action), grid) ||
        throw(ArgumentError("provided action field is not on the model physical grid"))
    compatible_model_spectral_grid(coordinate_grid(action), spectral_grid) ||
        throw(ArgumentError("provided action field is not on the model spectral grid"))
    return action
end

validate_model_action(action, grid, spectral_grid) =
    throw(ArgumentError("action must be a ProductField; got $(typeof(action))"))

function validate_cwcm_coupling_cache_shape(coupling, name, cache, expected_size)
    size(cache) == expected_size ||
        throw(ArgumentError("CWCM current coupling $name cache has size $(size(cache)); expected $expected_size from the model grid and spectral grid"))
    return nothing
end

resolve_coupling(::Nothing, coupling, grid, spectral_grid, depth) = coupling
function resolve_coupling(velocities, coupling, grid, spectral_grid, depth)
    coupling === nothing ||
        throw(ArgumentError("pass either `velocities` or `coupling`, not both"))
    FT = promote_type(grid_float_type(grid), coordinate_float_type(spectral_grid))
    return build_coupling(velocities, grid, spectral_grid, depth; FT)
end

validate_model_coupling(::Nothing, grid, spectral_grid) = nothing

function validate_model_coupling(coupling::AbstractCWCMCurrentCoupling, grid, spectral_grid)
    spectral_grid isa PolarWaveVectorGrid ||
        throw(ArgumentError("CWCM current coupling requires a PolarWaveVectorGrid"))

    compatible_horizontal_grid(coupling.qtransform.grid, grid) ||
        throw(ArgumentError("CWCM Q-transform grid must match the model grid horizontally"))

    spectral_kappa = collect(float.(spectral_grid.κ))
    coupling.kappa == spectral_kappa ||
        throw(ArgumentError("CWCM current coupling kappa does not match the model spectral grid"))

    u_expected_size = cgrid_velocity_cache_size(grid, :x, length(spectral_kappa))
    v_expected_size = cgrid_velocity_cache_size(grid, :y, length(spectral_kappa))
    validate_cwcm_coupling_cache_shape(coupling, "uᴰx", coupling.uᴰx, u_expected_size)
    validate_cwcm_coupling_cache_shape(coupling, "uᴰy", coupling.uᴰy, v_expected_size)
    validate_cwcm_coupling_cache_shape(coupling, "duᴰxdκ", coupling.duᴰxdκ, u_expected_size)
    validate_cwcm_coupling_cache_shape(coupling, "duᴰydκ", coupling.duᴰydκ, v_expected_size)
    return coupling
end

function validate_model_coupling(coupling, grid, spectral_grid)
    coupling isa AbstractCurrentCoupling ||
        throw(ArgumentError("coupling must be nothing or an AbstractCurrentCoupling; got $(typeof(coupling))"))
    return coupling
end

fused_refraction_supported_coordinate_bcs(bcs) =
    bcs[1] isa NoFlux && bcs[2] isa Oceananigans.Grids.Periodic

function validate_regular_full_period_direction_grid(spectral_grid)
    hasproperty(spectral_grid, :φ_faces) || return nothing

    φ_faces = collect(Array(spectral_grid.φ_faces))
    φ = collect(Array(spectral_grid.φ))
    FT = eltype(φ_faces)
    Δφ = diff(φ_faces)
    period = FT(2pi)
    tolerance = sqrt(eps(FT))

    isapprox(last(φ_faces) - first(φ_faces), period; rtol=tolerance, atol=tolerance) ||
        throw(ArgumentError("fused spectral refraction requires φ faces to span exactly 2π"))

    all(δ -> isapprox(δ, first(Δφ); rtol=tolerance, atol=tolerance), Δφ) ||
        throw(ArgumentError("fused spectral refraction currently requires uniformly-spaced φ faces"))

    for n in eachindex(φ)
        midpoint = (φ_faces[n] + φ_faces[n+1]) / 2
        isapprox(φ[n], midpoint; rtol=tolerance, atol=tolerance) ||
            throw(ArgumentError("fused spectral refraction requires φ centers at cell midpoints; φ[$n]=$(φ[n]) but midpoint is $midpoint"))
    end

    return nothing
end

function validate_fused_refraction_configuration(coupling, spectral_advection, spectral_grid, boundary_conditions)
    coupling isa AbstractCWCMCurrentCoupling || return nothing
    spectral_advection === nothing && return nothing

    fused_refraction_supported_coordinate_bcs(Tuple(spectral_grid.boundary_conditions)) ||
        throw(ArgumentError("fused CWCM spectral refraction supports only NoFlux radial and Periodic directional spectral-grid boundary conditions"))

    fused_refraction_supported_coordinate_bcs(Tuple(boundary_conditions.coordinate)) ||
        throw(ArgumentError("fused CWCM spectral refraction supports only NoFlux radial and Periodic directional model boundary conditions"))

    validate_regular_full_period_direction_grid(spectral_grid)
    return nothing
end

supported_model_timestepper(timestepper::Symbol) =
    timestepper === :ForwardEuler ||
    timestepper === :SemiImplicitEuler ||
    timestepper === :AB2 ||
    timestepper === :RungeKutta3 ||
    timestepper === :RK3 ||
    is_low_storage_rk3(timestepper)

function canonical_model_timestepper(timestepper::Symbol)
    supported_model_timestepper(timestepper) ||
        throw(ArgumentError("unsupported timestepper $timestepper"))
    timestepper === :RK3 && return :RungeKutta3
    is_low_storage_rk3(timestepper) && return :RungeKutta3
    return timestepper
end

canonical_model_timestepper(timestepper) =
    throw(ArgumentError("timestepper must be a Symbol; got $(typeof(timestepper))"))

function materialize_model_timestepper(timestepper::Symbol, grid, action, tendencies, previous_tendencies)
    timestepper === :RungeKutta3 &&
        return RungeKutta3TimeStepper(grid, action; Gⁿ=tendencies, G⁻=previous_tendencies)
    return timestepper
end

mutable struct SpectralWaveModel{Arch, G, SG, Depth, A, HAdv, SAdv, Sources, Coupling, GA, BCs, TS, Tend, PrevTend, C} <: AbstractModel{Nothing, Arch}
    grid :: G
    spectral_grid :: SG
    depth :: Depth
    action :: A
    horizontal_advection :: HAdv
    spectral_advection :: SAdv
    sources :: Sources
    coupling :: Coupling
    propagation_smoothing :: GA
    boundary_conditions :: BCs
    timestepper :: TS
    tendencies :: Tend
    previous_tendencies :: PrevTend
    previous_tendencies_ready :: Bool
    clock :: C
    intrinsic_transport_workspace :: Any  # lazy cache for the fused source-free transport kernel
end

# Marker sentinel so we can detect when the user did not pass `advection=...`.
const _ADVECTION_UNSET = Base.RefValue{Any}(nothing)

function SpectralWaveModel(grid, spectral_grid;
                           action=nothing,
                           horizontal_advection=WENO(),
                           spectral_advection=WENO(),
                           advection=_ADVECTION_UNSET,
                           sources=nothing,
                           depth=InfiniteDepth(),
                           velocities=nothing,
                           coupling=nothing,
                           propagation_smoothing=nothing,
                           boundary_conditions=nothing,
                           timestepper=:ForwardEuler,
                           clock=Clock(time=0.0))
    grid = validate_model_physical_grid(adapt_physical_grid(grid))
    spectral_grid = validate_model_spectral_grid(spectral_grid)
    depth = validate_model_depth(depth, grid)

    if advection !== _ADVECTION_UNSET
        horizontal_advection = advection
        spectral_advection = advection
    end

    action = action === nothing ? WaveActionField(grid, spectral_grid) :
                                  validate_model_action(action, grid, spectral_grid)
    sources = validate_model_sources(canonical_model_sources(sources))
    horizontal_advection = validate_model_advection(canonical_model_advection(horizontal_advection), grid, spectral_grid)
    spectral_advection = validate_model_spectral_advection(spectral_advection)
    coupling = resolve_coupling(velocities, coupling, grid, spectral_grid, depth)
    coupling = canonical_model_coupling(coupling)
    coupling = validate_model_coupling(coupling, grid, spectral_grid)
    timestepper = canonical_model_timestepper(timestepper)
    clock = validate_model_clock(clock)
    boundary_conditions = boundary_conditions === nothing ?
                          default_wave_action_bcs(grid, spectral_grid) :
                          validate_model_boundary_conditions(boundary_conditions, grid, spectral_grid)
    validate_fused_refraction_configuration(coupling, spectral_advection, spectral_grid, boundary_conditions)

    if coupling isa AbstractCWCMCurrentCoupling && spectral_advection !== nothing &&
       horizontal_advection !== nothing
        @info "SpectralWaveModel: CWCM coupling with `spectral_advection` set; the fused refraction kernel handles physical transport, so `horizontal_advection` is ignored."
    end

    tendencies = similar(action)
    previous_tendencies = similar(action)
    timestepper = materialize_model_timestepper(timestepper, grid, action, tendencies, previous_tendencies)
    Arch = typeof(architecture(grid))
    model = SpectralWaveModel{Arch, typeof(grid), typeof(spectral_grid), typeof(depth), typeof(action),
                              typeof(horizontal_advection), typeof(spectral_advection),
                              typeof(sources), typeof(coupling),
                              typeof(propagation_smoothing),
                              typeof(boundary_conditions),
                              typeof(timestepper),
                              typeof(tendencies), typeof(previous_tendencies), typeof(clock)}(
        grid, spectral_grid, depth, action, horizontal_advection, spectral_advection, sources, coupling,
        propagation_smoothing, boundary_conditions,
        timestepper, tendencies, previous_tendencies, false, clock,
        nothing)
    update_coupling!(model)
    return model
end

validate_model_boundary_conditions(bcs::ProductBoundaryConditions, grid, spectral_grid) = bcs
validate_model_boundary_conditions(bcs, grid, spectral_grid) =
    throw(ArgumentError("boundary_conditions must be a ProductBoundaryConditions; got $(typeof(bcs))"))

# Validate the spectral_advection kwarg. nothing disables kinematic refraction;
# WENO() (or another AbstractAdvectionScheme) enables the fused kernel when the
# coupling is CWCM. Other types are rejected.
validate_model_spectral_advection(::Nothing) = nothing
validate_model_spectral_advection(advection::Oceananigans.Advection.AbstractAdvectionScheme) = advection
validate_model_spectral_advection(advection) =
    throw(ArgumentError("spectral_advection must be nothing or an Oceananigans advection scheme; got $(typeof(advection))"))

fields(model::SpectralWaveModel) = (N=model.action, G=model.tendencies)
prognostic_fields(model::SpectralWaveModel) = (N=model.action,)
Base.eltype(model::SpectralWaveModel) = eltype(model.action)
architecture(model::SpectralWaveModel) = architecture(model.grid)

spectral_coupling_summary(::Nothing) = "none"
spectral_coupling_summary(::CWCMPrescribedCurrentCoupling) = "CWCM prescribed velocities"
spectral_coupling_summary(::CWCMPseudomomentumCoupling) = "CWCM pseudomomentum velocities"
spectral_coupling_summary(c) = string(nameof(typeof(c)))

# `velocities(model)` returns the (u, v) Lagrangian-mean current the wave
# model is coupled to (same contract as `MonobandedWaveModel`).
velocities(model::SpectralWaveModel) = velocities(model.coupling)
velocities(c::CWCMPrescribedCurrentCoupling) = (u=c.current.u, v=c.current.v)
velocities(c::CWCMPseudomomentumCoupling) = nothing  # self-coupled; no externally-set u/v

# `pseudomomentum_fields(model::SpectralWaveModel; location)` mirrors the
# monobanded API: returns center-by-default fields px, py constructed from
# the model's coupling Q-transform.
function pseudomomentum_fields(model::SpectralWaveModel; location=(Center, Center, Center))
    coupling = model.coupling
    coupling isa AbstractCWCMCurrentCoupling ||
        throw(ArgumentError("pseudomomentum_fields(::SpectralWaveModel) requires a CWCM coupling that owns a Q-transform; got $(typeof(coupling))"))
    return pseudomomentum_fields(model.action, model.depth, coupling.qtransform; location)
end

# Project the analytic action tendency `model.tendencies` onto Q(z) to return
# (∂t uˢ, ∂t vˢ). The caller is expected to have called
# `compute_tendencies!(model)` so that `model.tendencies` reflects the desired
# tendency operator (transport + refraction + sources).
function pseudomomentum_tendency_fields(model::SpectralWaveModel;
                                        location=(Center, Center, Center))
    coupling = model.coupling
    coupling isa AbstractCWCMCurrentCoupling ||
        throw(ArgumentError("pseudomomentum_tendency_fields(::SpectralWaveModel) requires a CWCM coupling that owns a Q-transform; got $(typeof(coupling))"))
    return pseudomomentum_fields(model.tendencies, model.depth, coupling.qtransform; location)
end

spectral_sources_summary(::Nothing) = "none"
spectral_sources_summary(s) = string(nameof(typeof(s)))

spectral_timestepper_name(ts::RungeKutta3TimeStepper) = :RungeKutta3
spectral_timestepper_name(ts) = ts isa Symbol ? ts : nameof(typeof(ts))

function Base.show(io::IO, model::SpectralWaveModel)
    println(io, summary(model))
    println(io, "├── grid: ", summary(model.grid))
    println(io, "├── spectral grid: ", summary(model.spectral_grid))
    println(io, "├── prognostic fields: N")
    println(io, "├── horizontal advection: ",
                model.horizontal_advection === nothing ? "none" : nameof(typeof(model.horizontal_advection)))
    println(io, "├── spectral advection: ",
                model.spectral_advection === nothing ? "none" : nameof(typeof(model.spectral_advection)))
    println(io, "├── coupling: ", spectral_coupling_summary(model.coupling))
    println(io, "├── sources: ", spectral_sources_summary(model.sources))
    println(io, "├── propagation smoothing: ",
                model.propagation_smoothing === nothing ? "none" : nameof(typeof(model.propagation_smoothing)))
    println(io, "├── timestepper: ", spectral_timestepper_name(model.timestepper))
    print(io,   "└── clock: time=", model.clock.time, ", iteration=", model.clock.iteration)
end
