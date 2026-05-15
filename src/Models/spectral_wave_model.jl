import Oceananigans
import Oceananigans: AbstractModel, fields, prognostic_fields
import Oceananigans.Architectures: architecture
import Oceananigans.Advection: WENO
import Oceananigans.TimeSteppers: Clock

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
                a.topology == b.topology)

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
        throw(ArgumentError("CWCMPrescribedCurrentCoupling $name cache has size $(size(cache)); expected $expected_size from the model grid and spectral grid"))
    return nothing
end

resolve_coupling(::Nothing, coupling, grid, spectral_grid) = coupling
function resolve_coupling(velocities, coupling, grid, spectral_grid)
    coupling === nothing ||
        throw(ArgumentError("pass either `velocities` or `coupling`, not both"))
    return build_coupling(velocities, grid, spectral_grid; FT=Float64)
end

validate_model_coupling(::Nothing, grid, spectral_grid) = nothing

function validate_model_coupling(coupling::CWCMPrescribedCurrentCoupling, grid, spectral_grid)
    spectral_grid isa PolarWaveVectorGrid ||
        throw(ArgumentError("CWCMPrescribedCurrentCoupling requires a PolarWaveVectorGrid"))

    spectral_kappa = collect(float.(spectral_grid.κ))
    coupling.kappa == spectral_kappa ||
        throw(ArgumentError("CWCMPrescribedCurrentCoupling kappa does not match the model spectral grid"))

    Nx, Ny = horizontal_size(grid)
    expected_size = (Nx, Ny, length(spectral_kappa))
    validate_cwcm_coupling_cache_shape(coupling, "Ux", coupling.Ux, expected_size)
    validate_cwcm_coupling_cache_shape(coupling, "Uy", coupling.Uy, expected_size)
    validate_cwcm_coupling_cache_shape(coupling, "dUxdkappa", coupling.dUxdkappa, expected_size)
    validate_cwcm_coupling_cache_shape(coupling, "dUydkappa", coupling.dUydkappa, expected_size)
    return coupling
end

function validate_model_coupling(coupling, grid, spectral_grid)
    coupling isa AbstractCurrentCoupling ||
        throw(ArgumentError("coupling must be nothing or an AbstractCurrentCoupling; got $(typeof(coupling))"))
    return coupling
end

supported_model_timestepper(timestepper::Symbol) =
    timestepper === :ForwardEuler ||
    timestepper === :SemiImplicitEuler ||
    timestepper === :AB2 ||
    timestepper === :RK3 ||
    is_low_storage_rk3(timestepper)

function canonical_model_timestepper(timestepper::Symbol)
    supported_model_timestepper(timestepper) ||
        throw(ArgumentError("unsupported timestepper $timestepper"))
    return timestepper
end

canonical_model_timestepper(timestepper) =
    throw(ArgumentError("timestepper must be a Symbol; got $(typeof(timestepper))"))

mutable struct SpectralWaveModel{Arch, G, SG, A, HAdv, SAdv, Sources, Coupling, Tend, PrevTend, C} <: AbstractModel{Nothing, Arch}
    grid :: G
    spectral_grid :: SG
    action :: A
    horizontal_advection :: HAdv
    spectral_advection :: SAdv
    sources :: Sources
    coupling :: Coupling
    timestepper :: Symbol
    tendencies :: Tend
    previous_tendencies :: PrevTend
    previous_tendencies_ready :: Bool
    clock :: C
end

# Marker sentinel so we can detect when the user did not pass `advection=...`.
const _ADVECTION_UNSET = Base.RefValue{Any}(nothing)

function SpectralWaveModel(grid, spectral_grid;
                           action=nothing,
                           horizontal_advection=WENO(),
                           spectral_advection=WENO(),
                           advection=_ADVECTION_UNSET,
                           sources=nothing,
                           velocities=nothing,
                           coupling=nothing,
                           timestepper=:ForwardEuler,
                           clock=Clock(time=0.0))
    grid = validate_model_physical_grid(adapt_physical_grid(grid))
    spectral_grid = validate_model_spectral_grid(spectral_grid)

    if advection !== _ADVECTION_UNSET
        horizontal_advection = advection
        spectral_advection = advection
    end

    action = action === nothing ? WaveActionField(grid, spectral_grid) :
                                  validate_model_action(action, grid, spectral_grid)
    sources = validate_model_sources(canonical_model_sources(sources))
    horizontal_advection = validate_model_advection(canonical_model_advection(horizontal_advection), grid, spectral_grid)
    spectral_advection = validate_model_spectral_advection(spectral_advection)
    coupling = resolve_coupling(velocities, coupling, grid, spectral_grid)
    coupling = canonical_model_coupling(coupling)
    coupling = validate_model_coupling(coupling, grid, spectral_grid)
    timestepper = canonical_model_timestepper(timestepper)
    clock = validate_model_clock(clock)

    if coupling isa CWCMPrescribedCurrentCoupling && spectral_advection !== nothing &&
       horizontal_advection !== nothing
        @info "SpectralWaveModel: CWCM coupling with `spectral_advection` set; the fused refraction kernel handles physical transport, so `horizontal_advection` is ignored."
    end

    tendencies = similar(action)
    previous_tendencies = similar(action)
    Arch = typeof(architecture(grid))
    model = SpectralWaveModel{Arch, typeof(grid), typeof(spectral_grid), typeof(action),
                              typeof(horizontal_advection), typeof(spectral_advection),
                              typeof(sources), typeof(coupling),
                              typeof(tendencies), typeof(previous_tendencies), typeof(clock)}(
        grid, spectral_grid, action, horizontal_advection, spectral_advection, sources, coupling,
        timestepper, tendencies, previous_tendencies, false, clock)
    update_coupling!(model)
    return model
end

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
