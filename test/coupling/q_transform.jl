struct MockOceanField{A}
    data :: A
end

Base.parent(field::MockOceanField) = field.data
Base.size(field::MockOceanField) = size(parent(field))
Base.axes(field::MockOceanField) = axes(parent(field))

struct MockBackendArray{T, N} <: AbstractArray{T, N}
    data :: Array{T, N}
end

MockBackendArray(data::Array{T, N}) where {T, N} = MockBackendArray{T, N}(data)
Base.IndexStyle(::Type{<:MockBackendArray}) = IndexCartesian()
Base.size(a::MockBackendArray) = size(a.data)
Base.axes(a::MockBackendArray) = axes(a.data)
Base.getindex(a::MockBackendArray, I...) = getindex(a.data, I...)
Base.setindex!(a::MockBackendArray, value, I...) = setindex!(a.data, value, I...)
Base.similar(::MockBackendArray, ::Type{T}, dims::Dims) where T = MockBackendArray(zeros(T, dims))

@testset "QTransform uses RectilinearGrid vertical cells" begin
    grid = RectilinearGrid(CPU();
                           size=(1, 1, 4),
                           x=[0.0, 1.0],
                           y=[0.0, 1.0],
                           z=[-1.0, -0.73, -0.31, -0.08, 0.0],
                           topology=(Bounded, Bounded, Bounded))
    q = QKernel(Float64)
    qt = QTransform(q, grid)
    faces = zfaces(grid)

    @test vertical_nodes(qt) == znodes(grid)
    @test vertical_faces(qt) == faces

    for kappa in (1e-8, 0.5, 5.0)
        integral = sum(q_cell_integral(q, kappa, faces[k], faces[k+1], 1.0)
                       for k in 1:vertical_size(grid))
        derivative_integral = sum(q_cell_integral_kappa_derivative(q, kappa, faces[k], faces[k+1], 1.0)
                                  for k in 1:vertical_size(grid))
        @test integral ≈ 1 atol=1e-14
        @test derivative_integral ≈ 0 atol=1e-12
    end

    @test q_value(q, 1e-10, -0.3, 2.0) ≈ 0.5
    @test q_cell_integral(q, 0.7, -10.0, 10.0, 1.0) ≈ 1 atol=1e-14
    @test q_cell_integral(q, 0.7, 10.0, -10.0, 1.0) ≈ 1 atol=1e-14
    @test q_cell_integral(q, 0.7, -2.0, -1.0, 1.0) ≈ 0 atol=1e-14
    @test q_cell_integral(q, 0.7, 0.0, 2.0, 1.0) ≈ 0 atol=1e-14
    @test q_cell_integral_kappa_derivative(q, 0.7, -10.0, 10.0, 1.0) ≈ 0 atol=1e-14

    u = reshape([0.1, -0.4, 0.8, 1.6], 1, 1, 4)
    v = reshape([1.2, 0.7, -0.5, 0.3], 1, 1, 4)
    Ux = zeros(1, 1, 2)
    Uy = zeros(1, 1, 2)
    dUxdkappa = zeros(1, 1, 2)
    dUydkappa = zeros(1, 1, 2)
    compute_doppler_velocity!(Ux, Uy, u, v, 1.0, [0.2, 1.0], qt)
    compute_doppler_velocity_derivative!(dUxdkappa, dUydkappa, u, v, 1.0, [0.2, 1.0], qt)
    expected_Ux = sum(u[1, 1, k] * q_cell_integral(q, 0.2, faces[k], faces[k+1], 1.0)
                      for k in axes(u, 3))
    expected_Uy = sum(v[1, 1, k] * q_cell_integral(q, 1.0, faces[k], faces[k+1], 1.0)
                      for k in axes(v, 3))
    expected_dUxdkappa = sum(u[1, 1, k] * q_cell_integral_kappa_derivative(q, 0.2, faces[k], faces[k+1], 1.0)
                             for k in axes(u, 3))
    expected_dUydkappa = sum(v[1, 1, k] * q_cell_integral_kappa_derivative(q, 1.0, faces[k], faces[k+1], 1.0)
                             for k in axes(v, 3))
    @test Ux[1, 1, 1] ≈ expected_Ux atol=1e-14
    @test Uy[1, 1, 2] ≈ expected_Uy atol=1e-14
    @test dUxdkappa[1, 1, 1] ≈ expected_dUxdkappa atol=1e-14
    @test dUydkappa[1, 1, 2] ≈ expected_dUydkappa atol=1e-14

    precomputed = PrecomputeQWeights(q, grid, [0.2, 1.0], 1.0)
    @test size(precomputed.weights) == (vertical_size(grid), 2)
    @test sum(precomputed.weights[:, 1]) ≈ 1 atol=1e-14
    cached_qt = QTransform(q, grid, precomputed)
    cached_Ux = zeros(1, 1, 2)
    cached_Uy = zeros(1, 1, 2)
    compute_doppler_velocity!(cached_Ux, cached_Uy, u, v, 1.0, [0.2, 1.0], cached_qt)
    @test cached_Ux ≈ Ux atol=1e-14
    @test cached_Uy ≈ Uy atol=1e-14

    cgrid = PolarWaveVectorGrid(; κ=[0.5, 1.0], φ=range(0, 2pi; length=9)[1:8])
    N = WaveActionField(grid, cgrid)
    set!(N, (x, y, kx, ky) -> 1 + kx)
    px, py = compute_pseudomomentum_cell_integrals(N, 1.0, qt)
    spectral_cached_qt = QTransform(q, grid, PrecomputeQWeights(qt, cgrid.κ, 1.0))
    cached_px, cached_py = compute_pseudomomentum_cell_integrals(N, 1.0, spectral_cached_qt)
    @test cached_px ≈ px atol=1e-14
    @test cached_py ≈ py atol=1e-14

    px_average, py_average = compute_pseudomomentum_cell_averages(N, 1.0, qt)
    dz = abs.(zspacings(grid))
    for k in axes(px, 3)
        @test px_average[1, 1, k] * dz[k] ≈ px[1, 1, k] atol=1e-14
        @test py_average[1, 1, k] * dz[k] ≈ py[1, 1, k] atol=1e-14
    end

    px_field, py_field = pseudomomentum_fields(N, 1.0, qt)
    @test location(px_field) == (Center, Center, Center)
    @test location(py_field) == (Center, Center, Center)
    @test Ripple.grid(px_field) === grid
    @test Ripple.field_storage(px_field) == px_average
    @test Ripple.field_storage(py_field) == py_average
    mx, my = first_moment(N)
    @test vertical_integral(px_field)[1, 1] ≈ mx[1, 1] atol=1e-12
    @test vertical_integral(py_field)[1, 1] ≈ my[1, 1] atol=1e-12

    ocean_px = pseudomomentum_field(grid)
    ocean_py = pseudomomentum_field(grid)
    compute_pseudomomentum_cell_averages!(ocean_px, ocean_py, N, 1.0, qt)
    @test Ripple.field_storage(ocean_px) == px_average
    @test Ripple.field_storage(ocean_py) == py_average
end

@testset "Prescribed CWCM current coupling caches" begin
    grid = RectilinearGrid(CPU();
                           size=(2, 2, 32),
                           x=(0, 2),
                           y=(0, 2),
                           z=(-1, 0))
    qt = QTransform(QKernel(Float64), grid)
    cgrid = PolarWaveVectorGrid(;
                                κ=[0.5, 1.0],
                                φ=[0.0, pi/2, pi, 3pi/2])
    u = ones(2, 2, vertical_size(grid))
    v = 2 .* ones(2, 2, vertical_size(grid))
    current = PrescribedLagrangianMeanCurrent(u=u, v=v, depth=1.0)
    coupling = CWCMPrescribedCurrentCoupling(current, qt, cgrid.κ)

    @test coupling.Ux[1, 1, 1] ≈ 1 atol=1e-12
    @test coupling.Uy[2, 2, 2] ≈ 2 atol=1e-12
    @test maximum(abs.(coupling.dUxdkappa)) < 1e-10

    model = SpectralWaveModel(; grid, spectral_grid=cgrid, coupling, advection=nothing)
    @test model.coupling === coupling
    time_step!(model, 0.001)
    @test all(interior(model.action) .>= 0)

    backend_u = MockBackendArray(ones(2, 2, vertical_size(grid)))
    backend_v = MockBackendArray(2 .* ones(2, 2, vertical_size(grid)))
    backend_current = PrescribedLagrangianMeanCurrent(u=backend_u, v=backend_v, depth=1.0)
    backend_coupling = CWCMPrescribedCurrentCoupling(backend_current, qt, cgrid.κ)
    @test backend_coupling.Ux isa MockBackendArray
    @test backend_coupling.Uy isa MockBackendArray
    @test backend_coupling.Ux[1, 1, 1] ≈ 1 atol=1e-12
    @test backend_coupling.Uy[1, 1, 1] ≈ 2 atol=1e-12
end
