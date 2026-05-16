#####
##### MeanSpectrumPhysics — ST3-equivalent bundle, GPU-compatible.
#####
##### Composes pressure-correlation wind input (Janssen), mean-spectrum or
##### local-saturation whitecapping, nonlinear quadruplet transfer (DIA), and
##### any extra terms.
#####
##### `prepare_physics(::MeanSpectrumPhysics, model)` precomputes per-grid-point
##### state via KernelAbstractions kernels:
#####   - (m₀, k̄, σ̄) bulk moments for `MeanSpectrumWhitecapping`
#####   - `stress_factor` cap for `PressureCorrelationInput`
#####   - 4D DIA `transfer` field for `HasselmannDIA`
#####
##### All kernels are scalar/inline and run on whatever backend Oceananigans
##### picked for the action field (`KernelAbstractions.get_backend(N_data)`).

import KernelAbstractions
import KernelAbstractions: @kernel, @index
import Oceananigans.Architectures: architecture, device

struct MeanSpectrumPhysics{I, D, NL, Ex} <: AbstractPhysicsBundle
    wind_input  :: I
    dissipation :: D
    nonlinear   :: NL
    extras      :: Ex
end

function MeanSpectrumPhysics(; wind_input=nothing,
                               dissipation=nothing,
                               nonlinear=nothing,
                               extras=())
    MeanSpectrumPhysics(wind_input, dissipation, nonlinear, tuple(extras...))
end

#####
##### MeanSpectrumWhitecapping bulk moments — one thread per (i, j).
##### Inner loop over (m, n) for spectral integration.
#####
@kernel function _mean_spectrum_state_kernel!(
        m₀_field, k̄_field, σ̄_field,
        N_data, @Const(k_centers), @Const(weights),
        Hx, Hy, iz, Nκ, Nφ, p, g)
    i, j = @index(Global, NTuple)
    ix = i + Hx; jy = j + Hy

    m₀ = zero(eltype(N_data))
    kp = zero(eltype(N_data))
    σp = zero(eltype(N_data))
    @inbounds for n in 1:Nφ, m in 1:Nκ
        action = N_data[ix, jy, iz, m, n]
        if action > 0
            w = weights[m, n]
            k = k_centers[m]
            if k > 0
                σ = sqrt(g * k)
                m₀ += action * w
                kp += action * w * k^p
                σp += action * w * σ^p
            end
        end
    end
    @inbounds begin
        m₀_field[i, j] = m₀
        if m₀ > 0
            k̄_field[i, j] = (kp / m₀)^(1 / p)
            σ̄_field[i, j] = (σp / m₀)^(1 / p)
        else
            k̄_field[i, j] = zero(eltype(N_data))
            σ̄_field[i, j] = zero(eltype(N_data))
        end
    end
end

function _compute_mean_spectrum_state(diss::MeanSpectrumWhitecapping, model)
    grid = model.grid
    arch = architecture(grid)
    N = model.action
    N_data = flat_data(N)
    Nx, Ny, Nκ, Nφ = size(N)
    FT = eltype(N)
    cgrid = model.spectral_grid

    m₀_field = KernelAbstractions.zeros(device(arch), FT, Nx, Ny)
    k̄_field  = KernelAbstractions.zeros(device(arch), FT, Nx, Ny)
    σ̄_field  = KernelAbstractions.zeros(device(arch), FT, Nx, Ny)

    weights   = spectral_weights(cgrid)
    k_centers = cgrid.κ
    Hx, Hy, _ = halo_size_3d(grid)
    iz        = data_z_index(N)

    kernel = _mean_spectrum_state_kernel!(device(arch), (8, 8), (Nx, Ny))
    kernel(m₀_field, k̄_field, σ̄_field,
           N_data, k_centers, weights,
           Hx, Hy, iz, Nκ, Nφ, FT(diss.p), FT(diss.gravity))
    KernelAbstractions.synchronize(device(arch))

    return (m₀ = m₀_field, k̄ = k̄_field, σ̄ = σ̄_field)
end

_compute_mean_spectrum_state(::Any, ::Any) = nothing
_compute_mean_spectrum_state(::Nothing, ::Any) = nothing

#####
##### Wind-input stress cap state — KA kernel over (i, j).
##### Computes τ_w along-wind from the *current* spectrum and stress_factor =
##### min(1, ε·τ_max/τ_w_eff). Inner loop over (m, n) integrates the raw Sin
##### contribution. The raw-rate kernel `_jp_rate` is `@inline` and pure, so it
##### compiles into the kernel cleanly.
#####
@inline function _jp_rate(k, σ, C, θ, θ_u, u_star, ρ_air_over_water,
                          β_max, z_α, p_in, α₀, κ_vk, g)
    cos_θu = cos(θ - θ_u)
    cos_θu > 0 || return zero(σ)
    z₁ = α₀ * u_star^2 / g
    z₁ > 0 || return zero(σ)
    inv_age = u_star / C + z_α
    Z = log(k * z₁) + κ_vk / (inv_age * cos_θu)
    Z >= 0 && return zero(σ)
    return ρ_air_over_water * (β_max / κ_vk^2) * exp(Z) * Z^4 * inv_age^2 *
           cos_θu^p_in * σ
end

@kernel function _wind_input_state_kernel!(
        stress_factor, N_data,
        @Const(k_centers), @Const(φ_centers), @Const(weights),
        @Const(U10_field), @Const(θ_u_field), @Const(u_star_field),
        Hx, Hy, iz, Nκ, Nφ,
        ρ_w_g, ρ_air, ρ_air_over_water, ε, g,
        β_max, z_α, p_in, α₀, κ_vk)
    i, j = @index(Global, NTuple)
    ix = i + Hx; jy = j + Hy

    @inbounds U10 = U10_field[i, j]
    @inbounds u_star = u_star_field[i, j]
    if U10 > 0 && u_star > 0
        @inbounds θ_u = θ_u_field[i, j]
        # Both τ_w (from ρ_w·g·∫ mflux) and τ_max are in Pa.
        τ_max = ρ_air * u_star^2

        τ_wx = zero(U10)
        τ_wy = zero(U10)
        @inbounds for n in 1:Nφ, m in 1:Nκ
            action = N_data[ix, jy, iz, m, n]
            if action > 0
                k = k_centers[m]
                if k > 0
                    σ = sqrt(g * k); C = σ / k
                    θ = φ_centers[n]
                    rate = _jp_rate(k, σ, C, θ, θ_u, u_star,
                                    ρ_air_over_water,
                                    β_max, z_α, p_in, α₀, κ_vk, g)
                    if rate > 0
                        mflux = rate * action / C * weights[m, n] * ρ_w_g
                        τ_wx += mflux * cos(θ)
                        τ_wy += mflux * sin(θ)
                    end
                end
            end
        end
        τ_w_par = τ_wx * cos(θ_u) + τ_wy * sin(θ_u)
        τ_w_eff = max(τ_w_par, zero(τ_w_par))
        @inbounds stress_factor[i, j] = (τ_w_eff > ε * τ_max && τ_max > 0) ?
                                         ε * τ_max / τ_w_eff : one(τ_max)
    else
        @inbounds stress_factor[i, j] = one(τ_max)
    end
end

function _compute_wind_input_state(inp::PressureCorrelationInput, model)
    grid = model.grid
    arch = architecture(grid)
    N = model.action
    N_data = flat_data(N)
    Nx, Ny, Nκ, Nφ = size(N)
    FT = eltype(N)
    cgrid = model.spectral_grid

    # Per-(i,j) wind diagnostics on host (wind structs may be non-GPU-friendly).
    U10_h    = Array{FT}(undef, Nx, Ny)
    θ_u_h    = Array{FT}(undef, Nx, Ny)
    u_star_h = Array{FT}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        U10 = pressure_correlation_wind_speed(inp.wind, model, i, j)
        θ_u = pressure_correlation_wind_dir(inp.wind, inp.direction, model, i, j)
        U10_h[i, j]    = FT(U10)
        θ_u_h[i, j]    = FT(θ_u)
        u_star_h[i, j] = FT(friction_velocity(inp.drag, U10))
    end
    backend = device(arch)
    U10_field    = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(U10_field, U10_h)
    θ_u_field    = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(θ_u_field, θ_u_h)
    u_star_field = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(u_star_field, u_star_h)
    stress_factor = KernelAbstractions.ones(backend, FT, Nx, Ny)

    weights   = spectral_weights(cgrid)
    k_centers = cgrid.κ
    φ_centers = cgrid.φ
    Hx, Hy, _ = halo_size_3d(grid)
    iz        = data_z_index(N)
    ε  = FT(0.4)

    kernel = _wind_input_state_kernel!(backend, (8, 8), (Nx, Ny))
    kernel(stress_factor, N_data,
           k_centers, φ_centers, weights,
           U10_field, θ_u_field, u_star_field,
           Hx, Hy, iz, Nκ, Nφ,
           FT(inp.ρ_water * inp.gravity),
           FT(inp.ρ_air),
           FT(inp.ρ_air / inp.ρ_water),
           ε, FT(inp.gravity),
           FT(inp.β_max), FT(inp.z_α), FT(inp.p_in), FT(inp.α₀), FT(inp.von_karman))
    KernelAbstractions.synchronize(backend)

    return (stress_factor = stress_factor,)
end

_compute_wind_input_state(::Any, ::Any) = nothing
_compute_wind_input_state(::Nothing, ::Any) = nothing

function prepare_physics(b::MeanSpectrumPhysics, model)
    wind_state = _compute_wind_input_state(b.wind_input, model)
    diss_state = _compute_mean_spectrum_state(b.dissipation, model)
    nl_state   = prepare_physics(b.nonlinear, model)
    return (wind_input  = wind_state,
            dissipation = diss_state,
            nonlinear   = nl_state,
            extras      = map(t -> prepare_physics(t, model), b.extras))
end

function source_split(b::MeanSpectrumPhysics, state::NamedTuple, model, i, j, m, n)
    FT = eltype(model.action)
    positive = zero(FT)
    damping  = zero(FT)
    if b.wind_input !== nothing
        p, λ = source_split(b.wind_input, state.wind_input, model, i, j, m, n)
        positive += p; damping += λ
    end
    if b.dissipation !== nothing
        p, λ = source_split(b.dissipation, state.dissipation, model, i, j, m, n)
        positive += p; damping += λ
    end
    if b.nonlinear !== nothing
        p, λ = source_split(b.nonlinear, state.nonlinear, model, i, j, m, n)
        positive += p; damping += λ
    end
    for (term, slot) in zip(b.extras, state.extras)
        p, λ = source_split(term, slot, model, i, j, m, n)
        positive += p; damping += λ
    end
    return positive, damping
end

function source_split(b::MeanSpectrumPhysics, model, i, j, m, n)
    FT = eltype(model.action)
    positive = zero(FT)
    damping  = zero(FT)
    for term in (b.wind_input, b.dissipation, b.nonlinear, b.extras...)
        term === nothing && continue
        p, λ = source_split(term, model, i, j, m, n)
        positive += p; damping += λ
    end
    return positive, damping
end

function source_tendency(b::MeanSpectrumPhysics, model, i, j, m, n)
    positive, damping = source_split(b, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function source_tendency(b::MeanSpectrumPhysics, state::NamedTuple, model, i, j, m, n)
    positive, damping = source_split(b, state, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end
