struct CartesianWaveVectorGrid{FT, KX, KY, KXF, KYF, W, Topo} <: AbstractSpectralGrid
    kx :: KX
    ky :: KY
    kx_faces :: KXF
    ky_faces :: KYF
    weights :: W
    topology :: Topo
end

function CartesianWaveVectorGrid(::Type{FT}=Float64; kx, ky,
                                 kx_faces=centers_to_faces(collect(FT, kx)),
                                 ky_faces=centers_to_faces(collect(FT, ky)),
                                 topology=(Bounded(), Bounded())) where FT
    topology = canonical_topology_tuple(topology, 2, "CartesianWaveVectorGrid")
    kxc = collect(FT, kx)
    kyc = collect(FT, ky)
    kxf = collect(FT, kx_faces)
    kyf = collect(FT, ky_faces)
    validate_coordinate_topology("kx", kxc, kxf)
    validate_coordinate_topology("ky", kyc, kyf)
    wx = diff(kxf)
    wy = diff(kyf)
    weights = [wx[m] * wy[n] for m in eachindex(kxc), n in eachindex(kyc)]
    return CartesianWaveVectorGrid{FT, typeof(kxc), typeof(kyc), typeof(kxf), typeof(kyf), typeof(weights), typeof(topology)}(
        kxc, kyc, kxf, kyf, weights, topology)
end

coordinate_centers(g::CartesianWaveVectorGrid, dim) = dim == 1 ? g.kx : g.ky
coordinate_faces(g::CartesianWaveVectorGrid, dim) = dim == 1 ? g.kx_faces : g.ky_faces
coordinate_float_type(::CartesianWaveVectorGrid{FT}) where FT = FT

@inline k_components(g::CartesianWaveVectorGrid, m, n) = (g.kx[m], g.ky[n])
@inline radial_wavenumber(g::CartesianWaveVectorGrid, m, n) = hypot(g.kx[m], g.ky[n])
@inline metric_jacobian(::CartesianWaveVectorGrid, m, n) = 1

@inline function spectral_first_moment_measures(g::CartesianWaveVectorGrid, m, n)
    x₁, x₂ = g.kx_faces[m], g.kx_faces[m+1]
    y₁, y₂ = g.ky_faces[n], g.ky_faces[n+1]
    return (x₂^2 - x₁^2) * (y₂ - y₁) / 2,
           (x₂ - x₁) * (y₂^2 - y₁^2) / 2
end

@inline function spectral_second_moment_measures(g::CartesianWaveVectorGrid, m, n)
    x₁, x₂ = g.kx_faces[m], g.kx_faces[m+1]
    y₁, y₂ = g.ky_faces[n], g.ky_faces[n+1]
    return (x₂^3 - x₁^3) * (y₂ - y₁) / 3,
           (x₂^2 - x₁^2) * (y₂^2 - y₁^2) / 4,
           (x₂ - x₁) * (y₂^3 - y₁^3) / 3
end
