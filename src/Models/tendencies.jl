import Oceananigans

# Hooks for source-term bundles that precompute per-grid-point state (e.g. a
# wave-supported-stress cap or a 4-D nonlinear-transfer field) once per
# tendency pass and read it back per spectral cell. See
# `src/Physics/Packages/precomputed_sources.jl` for the bundle that uses
# these. Defaults are no-ops; only bundle types override.
prepare_sources(::Any, model) = nothing
source_tendency(s, ::Nothing, model, i, j, m, n) =
    source_tendency(s, model, i, j, m, n)
source_tendency(::Nothing, ::Nothing, model, i, j, m, n) =
    zero(eltype(model.action))
source_split(s, ::Nothing, model, i, j, m, n) =
    source_split(s, model, i, j, m, n)
source_split(::Nothing, ::Nothing, model, i, j, m, n) =
    (zero(eltype(model.action)), zero(eltype(model.action)))

function compute_tendencies!(G::ProductField, model::SpectralWaveModel)
    return compute_tendencies!(G, model, model.coupling)
end

# No-coupling + multi-bin WENO horizontal advection → fused KA kernel. Avoids the
# 480 × Nx × Ny FluxFormAdvection / ConstantField allocations the legacy
# bin loop costs per step. The single-bin path stays on Oceananigans' exact
# tracer-advection fallback for validation against analytical transport tests;
# bounded physical topologies also stay on Oceananigans for boundary fluxes.
# Sources, if any, are added in a second pass.
function compute_tendencies!(G::ProductField, model::SpectralWaveModel, coupling::Nothing)
    if intrinsic_transport_kernel_enabled(model)
        compute_intrinsic_transport_tendency!(G, model.action, model)
        if model.sources !== nothing
            state = prepare_sources(model.sources, model)
            Nx, Ny, Nκ, Nφ = size(model.action)
            @inbounds for n in 1:Nφ, m in 1:Nκ, j in 1:Ny, i in 1:Nx
                G[i, j, m, n] += source_tendency(model.sources, state, model, i, j, m, n)
            end
        end
        return G
    end
    return _per_bin_tendencies!(G, model)
end

function intrinsic_transport_kernel_enabled(model)
    model.horizontal_advection isa WENO || return false
    Nκ, Nφ = coordinate_size(model.spectral_grid)
    Nκ * Nφ > 1 || return false
    topology = Oceananigans.Grids.topology(model.grid)
    return topology[1] === Oceananigans.Grids.Periodic &&
           topology[2] === Oceananigans.Grids.Periodic
end

# Generic per-bin fallback (used for non-WENO horizontal advection schemes
# and any coupling type that doesn't match a specialized dispatch).
function compute_tendencies!(G::ProductField, model::SpectralWaveModel, coupling)
    return _per_bin_tendencies!(G, model)
end

function _per_bin_tendencies!(G::ProductField, model::SpectralWaveModel)
    Nx, Ny, Nxi, Neta = size(model.action)
    if model.sources === nothing && model.horizontal_advection === nothing
        set!(G, zero(eltype(G)))
        return G
    end

    fill_halo_regions!(model.action)
    state = prepare_sources(model.sources, model)
    for n in 1:Neta, m in 1:Nxi
        Nmn = physical_field(model.action, m, n)
        for j in 1:Ny, i in 1:Nx
            source = source_tendency(model.sources, state, model, i, j, m, n)
            transport = transport_tendency(model.horizontal_advection, model, Nmn, i, j, m, n)
            G[i, j, m, n] = source + transport
        end
    end
    return G
end

# CWCM coupling + spectral advection on -> use the fused KA kernel that does
# Doppler-shifted physical transport AND kinematic spectral refraction in
# a single pass. The fused kernel always includes physical transport when
# active; `horizontal_advection` is ignored in this branch.
function cwcm_tendencies!(G::ProductField, model::SpectralWaveModel, coupling)
    if model.spectral_advection isa Nothing
        # spectral refraction disabled -> fall back to the default per-bin path.
        return invoke(compute_tendencies!, Tuple{ProductField, SpectralWaveModel, Any}, G, model, coupling)
    end
    compute_wave_current_refraction_tendency!(G, model.action, coupling, model)
    if model.sources !== nothing
        state = prepare_sources(model.sources, model)
        Nx, Ny, Nκ, Nφ = size(model.action)
        for n in 1:Nφ, m in 1:Nκ, j in 1:Ny, i in 1:Nx
            @inbounds G[i, j, m, n] += source_tendency(model.sources, state, model, i, j, m, n)
        end
    end
    return G
end

function compute_tendencies!(G::ProductField, model::SpectralWaveModel,
                             coupling::CWCMPrescribedCurrentCoupling)
    return cwcm_tendencies!(G, model, coupling)
end

function compute_tendencies!(G::ProductField, model::SpectralWaveModel,
                             coupling::CWCMPseudomomentumCoupling)
    update_coupling!(coupling, model)
    return cwcm_tendencies!(G, model, coupling)
end

compute_tendencies!(model::SpectralWaveModel) = compute_tendencies!(model.tendencies, model)
