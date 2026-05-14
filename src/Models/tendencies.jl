function compute_tendencies!(G::ProductField, model::SpectralWaveModel)
    Nx, Ny, Nxi, Neta = size(model.action)
    fill_halo_regions!(model.action)
    for n in 1:Neta, m in 1:Nxi
        Nmn = physical_field(model.action, m, n)
        for j in 1:Ny, i in 1:Nx
            source = source_tendency(model.sources, model, i, j, m, n)
            transport = transport_tendency(model.advection, model, Nmn, i, j, m, n)
            G[i, j, m, n] = source + transport
        end
    end
    return G
end

compute_tendencies!(model::SpectralWaveModel) = compute_tendencies!(model.tendencies, model)
