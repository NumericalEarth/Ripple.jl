struct CartesianWaveVectorGrid{FT, A, KX, KY, KXF, KYF, W, BCs} <: AbstractSpectralGrid
    architecture :: A
    kx :: KX
    ky :: KY
    kx_faces :: KXF
    ky_faces :: KYF
    weights :: W
    boundary_conditions :: BCs
end

# Both wavenumber components are physically bounded; boundary conditions
# at the faces live in `boundary_conditions`.
spectral_topology(::CartesianWaveVectorGrid) = (Bounded(), Bounded())

CartesianWaveVectorGrid(FT::DataType; kwargs...) = CartesianWaveVectorGrid(CPU(), FT; kwargs...)

function CartesianWaveVectorGrid(arch::AbstractArchitecture = CPU(),
                                 FT::DataType = Float64;
                                 kx, ky,
                                 kx_faces = nothing,
                                 ky_faces = nothing,
                                 boundary_conditions = (NoFlux(), NoFlux()))
    boundary_conditions = canonical_bc_tuple(boundary_conditions, 2, "CartesianWaveVectorGrid")
    kxc_host = collect(FT, kx)
    kyc_host = collect(FT, ky)
    kxf_host = kx_faces === nothing ? centers_to_faces(kxc_host) : collect(FT, kx_faces)
    kyf_host = ky_faces === nothing ? centers_to_faces(kyc_host) : collect(FT, ky_faces)
    validate_coordinate_topology("kx", kxc_host, kxf_host)
    validate_coordinate_topology("ky", kyc_host, kyf_host)
    wx = diff(kxf_host)
    wy = diff(kyf_host)
    weights_host = [wx[m] * wy[n] for m in eachindex(kxc_host), n in eachindex(kyc_host)]
    kxc = on_architecture(arch, kxc_host)
    kyc = on_architecture(arch, kyc_host)
    kxf = on_architecture(arch, kxf_host)
    kyf = on_architecture(arch, kyf_host)
    weights = on_architecture(arch, weights_host)
    return CartesianWaveVectorGrid{FT, typeof(arch), typeof(kxc), typeof(kyc),
                                   typeof(kxf), typeof(kyf), typeof(weights), typeof(boundary_conditions)}(
        arch, kxc, kyc, kxf, kyf, weights, boundary_conditions)
end

architecture(g::CartesianWaveVectorGrid) = g.architecture

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

function Base.summary(g::CartesianWaveVectorGrid)
    return string(spectral_size_string(g),
                  " CartesianWaveVectorGrid", spectral_type_signature(g),
                  " on ", summary(architecture(g)))
end

function Base.show(io::IO, g::CartesianWaveVectorGrid)
    println(io, summary(g))
    topo = spectral_topology(g)
    bcs  = g.boundary_conditions
    _print_spectral_axes(io, (axis_summary(topo[1], g.kx_faces, "kx"; bc = bcs[1]),
                              axis_summary(topo[2], g.ky_faces, "ky"; bc = bcs[2])))
end
