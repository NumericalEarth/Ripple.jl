import Oceananigans.TimeSteppers: time_step!, update_state!, tick!
import Oceananigans.Fields: set!, interior
import Oceananigans.BoundaryConditions: fill_halo_regions!

#####
##### SSP-RK3 time stepping for the reconstituted-amplitude pair (Gʳ, Gⁱ)
#####
##### The stepper never touches the mass matrix: each stage evaluates the
##### tendency ∂ₜG (which first diagnoses A = M⁻¹G) and advances G with the
##### three-stage strong-stability-preserving weights. Amplitudes are signed, so
##### there is no positivity clamp (unlike the action model's RK3).

# Field arithmetic on interiors (GPU-portable broadcasts, no host loops).
@inline _add_scaled!(u, dt, ut) = (interior(u) .+= dt .* interior(ut); u)
@inline _combine!(u, a, u0, b)  = (interior(u) .= a .* interior(u0) .+ b .* interior(u); u)
@inline _copy_interior!(dest, src) = (interior(dest) .= interior(src); dest)

function time_step!(model::NarrowBandWaveModel, Δt; callbacks=[])
    Δt > 0 || throw(ArgumentError("time step must be positive"))
    FT = eltype(model.grid)
    dt = convert(FT, Δt)

    # Save the stage-0 state.
    _copy_interior!(model.G0r, model.Gr)
    _copy_interior!(model.G0i, model.Gi)

    # Stage 1: G ← G0 + Δt R(G0)
    compute_tendencies!(model)
    _add_scaled!(model.Gr, dt, model.Gr_tendency)
    _add_scaled!(model.Gi, dt, model.Gi_tendency)

    # Stage 2: G ← 3/4 G0 + 1/4 (G + Δt R(G))
    compute_tendencies!(model)
    _add_scaled!(model.Gr, dt, model.Gr_tendency)
    _add_scaled!(model.Gi, dt, model.Gi_tendency)
    _combine!(model.Gr, convert(FT, 3//4), model.G0r, convert(FT, 1//4))
    _combine!(model.Gi, convert(FT, 3//4), model.G0i, convert(FT, 1//4))

    # Stage 3: G ← 1/3 G0 + 2/3 (G + Δt R(G))
    compute_tendencies!(model)
    _add_scaled!(model.Gr, dt, model.Gr_tendency)
    _add_scaled!(model.Gi, dt, model.Gi_tendency)
    _combine!(model.Gr, convert(FT, 1//3), model.G0r, convert(FT, 2//3))
    _combine!(model.Gi, convert(FT, 1//3), model.G0i, convert(FT, 2//3))

    # Refresh the diagnostic amplitude to match the advanced G.
    update_state!(model)
    tick!(model.clock, dt)
    return model
end

"""
    update_state!(model::NarrowBandWaveModel)

Diagnose the amplitude pair `(Aʳ, Aⁱ)` from the prognostic reconstituted
amplitude `(Gʳ, Gⁱ)`, so diagnostics and Stokes-drift feedback see a current
`A`. Called at the end of each step and after `set!`.
"""
function update_state!(model::NarrowBandWaveModel; callbacks=[], kwargs...)
    solve_amplitude!(model.Ar, model.helmholtz_solver, model.Gr)
    solve_amplitude!(model.Ai, model.helmholtz_solver, model.Gi)
    return model
end

#####
##### Initial conditions
#####

_real_part(f::Function) = (x, y) -> real(f(x, y))
_imag_part(f::Function) = (x, y) -> imag(f(x, y))
_real_part(a::AbstractArray) = real.(a)
_imag_part(a::AbstractArray) = imag.(a)
_real_part(z::Number) = real(z)
_imag_part(z::Number) = imag(z)

_set_component!(field, spec::Function) = set!(field, spec)
_set_component!(field, spec::Number) = set!(field, spec)
function _set_component!(field, spec::AbstractArray)
    isize = size(interior(field))
    set!(field, size(spec) == isize ? spec : reshape(spec, isize))
    return field
end

"""
    set!(model::NarrowBandWaveModel; A=nothing, G=nothing)

Initialize the model from either the amplitude `A` or the reconstituted
amplitude `G` (not both). Each may be a complex-valued function `(x, y) -> A`, a
complex array over the horizontal grid, or a complex `Number`. Setting `A`
computes `G = [1 + α(∇ₕ² + κ²)]A` via [`reconstitute!`](@ref); the diagnostic
amplitude is then refreshed so `A` and `G` are mutually consistent.
"""
function set!(model::NarrowBandWaveModel; A=nothing, G=nothing)
    A === nothing && G === nothing && return model
    A !== nothing && G !== nothing &&
        throw(ArgumentError("set! the narrow-band model with either `A` or `G`, not both"))

    if A !== nothing
        _set_component!(model.Ar, _real_part(A))
        _set_component!(model.Ai, _imag_part(A))
        fill_halo_regions!(model.Ar)
        fill_halo_regions!(model.Ai)
        reconstitute!(model.Gr, model.helmholtz_solver, model.Ar)
        reconstitute!(model.Gi, model.helmholtz_solver, model.Ai)
    else
        _set_component!(model.Gr, _real_part(G))
        _set_component!(model.Gi, _imag_part(G))
    end

    update_state!(model)
    return model
end
