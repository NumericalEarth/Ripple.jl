function compute_tendencies!(G::ProductField, model::SpectralWaveModel)
    return compute_tendencies!(G, model, model.coupling)
end

# Default path (no current coupling, or coupling that doesn't enable the
# fused kernel): per-bin transport + sources loop.
function compute_tendencies!(G::ProductField, model::SpectralWaveModel, coupling)
    Nx, Ny, Nxi, Neta = size(model.action)
    fill_halo_regions!(model.action)
    for n in 1:Neta, m in 1:Nxi
        Nmn = physical_field(model.action, m, n)
        for j in 1:Ny, i in 1:Nx
            source = source_tendency(model.sources, model, i, j, m, n)
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
function compute_tendencies!(G::ProductField, model::SpectralWaveModel,
                             coupling::CWCMPrescribedCurrentCoupling)
    if model.spectral_advection isa Nothing
        # spectral refraction disabled -> fall back to the default per-bin path.
        return invoke(compute_tendencies!, Tuple{ProductField, SpectralWaveModel, Any}, G, model, coupling)
    end
    compute_wave_current_refraction_tendency!(G, model.action, coupling, model)
    if model.sources !== nothing
        Nx, Ny, Nκ, Nφ = size(model.action)
        for n in 1:Nφ, m in 1:Nκ, j in 1:Ny, i in 1:Nx
            @inbounds G[i, j, m, n] += source_tendency(model.sources, model, i, j, m, n)
        end
    end
    return G
end

compute_tendencies!(model::SpectralWaveModel) = compute_tendencies!(model.tendencies, model)
