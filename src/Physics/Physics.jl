abstract type AbstractPhysicsTerm end

# Category-aligned subtypes. Lets the tendency dispatcher tell wind-input from
# dissipation from nonlinear interactions, and lets package bundles constrain
# what they accept.
abstract type AbstractWindInput   <: AbstractPhysicsTerm end
abstract type AbstractDissipation <: AbstractPhysicsTerm end
abstract type AbstractNonlinear   <: AbstractPhysicsTerm end

# A `PhysicsPackage` (= `AbstractPhysicsBundle`) groups terms whose evaluation
# can share per-grid-point state (bulk moments, drag, wave-supported stress).
# `GenericPhysics` is the tuple wrapper that user-supplied `(t1, t2, ...)`
# auto-converts into; it has no shared state.
abstract type AbstractPhysicsBundle <: AbstractPhysicsTerm end

struct NoPhysics <: AbstractPhysicsTerm end

struct GenericPhysics{Terms} <: AbstractPhysicsBundle
    terms :: Terms
end

GenericPhysics(terms::Tuple=()) = GenericPhysics{typeof(terms)}(terms)
GenericPhysics(term::AbstractPhysicsTerm, terms::AbstractPhysicsTerm...) =
    GenericPhysics((term, terms...))

Base.length(s::GenericPhysics) = length(s.terms)
Base.isempty(s::GenericPhysics) = isempty(s.terms)
Base.iterate(s::GenericPhysics) = iterate(s.terms)
Base.iterate(s::GenericPhysics, state) = iterate(s.terms, state)
Base.getindex(s::GenericPhysics, i::Int) = getindex(s.terms, i)

# Shared infrastructure — drag, diagnostic tail, dynamic substep limiter.
include("shared/drag.jl")
include("shared/parametric_tail.jl")
include("shared/substep_limiter.jl")

struct RelaxationToSpectrum{F, FT} <: AbstractPhysicsTerm
    target :: F
    timescale :: FT
end

RelaxationToSpectrum(target; timescale) = RelaxationToSpectrum(target, float(timescale))

struct LinearWindInput{Rate} <: AbstractPhysicsTerm
    rate :: Rate
end

LinearWindInput(; rate=0.0) = LinearWindInput(source_parameter(rate))

struct ExponentialWindInput{Rate, Direction, Power} <: AbstractPhysicsTerm
    rate :: Rate
    direction :: Direction
    spreading_power :: Power
end

ExponentialWindInput(; rate=0.0, direction=0.0, spreading_power=2.0) =
    ExponentialWindInput(source_parameter(rate), direction, float(spreading_power))

struct PowerLawWindInput{Rate, Speed, Direction, FT} <: AbstractPhysicsTerm
    rate :: Rate
    speed :: Speed
    direction :: Direction
    reference_speed :: FT
    speed_power :: FT
    spreading_power :: FT
end

function PowerLawWindInput(; rate=0.0,
                             speed=nothing,
                             wind=nothing,
                             direction=nothing,
                             reference_speed=1.0,
                             speed_power=1.0,
                             spreading_power=2.0)
    speed_value = wind === nothing ? (speed === nothing ? 1.0 : speed) : wind
    direction_value = direction === nothing ? (wind === nothing ? 0.0 : wind) : direction
    return PowerLawWindInput(source_parameter(rate),
                             source_parameter(speed_value),
                             direction_value,
                             float(reference_speed),
                             float(speed_power),
                             float(spreading_power))
end

struct WaveAgeWindInput{Rate, Speed, Direction, FT} <: AbstractPhysicsTerm
    rate :: Rate
    speed :: Speed
    direction :: Direction
    inverse_wave_age_threshold :: FT
    power :: FT
    spreading_power :: FT
    gravity :: FT
end

function WaveAgeWindInput(; rate=0.0,
                            speed=nothing,
                            wind=nothing,
                            direction=nothing,
                            inverse_wave_age_threshold=0.83,
                            power=1.0,
                            spreading_power=1.0,
                            gravity=9.81)
    speed_value = wind === nothing ? (speed === nothing ? 10.0 : speed) : wind
    direction_value = direction === nothing ? (wind === nothing ? 0.0 : wind) : direction
    return WaveAgeWindInput(source_parameter(rate),
                            source_parameter(speed_value),
                            direction_value,
                            float(inverse_wave_age_threshold),
                            float(power),
                            float(spreading_power),
                            float(gravity))
end

struct SaturationDissipation{Rate, FT} <: AbstractPhysicsTerm
    rate :: Rate
    threshold :: FT
    power :: FT
end

SaturationDissipation(; rate=0.0, threshold=1.0, power=1.0) =
    SaturationDissipation(source_parameter(rate), float(threshold), float(power))

struct WhitecappingDissipation{Rate, FT} <: AbstractPhysicsTerm
    rate :: Rate
    saturation_threshold :: FT
    saturation_power :: FT
    wavenumber_power :: FT
    reference_wavenumber :: FT
end

WhitecappingDissipation(; rate=0.0,
                          saturation_threshold=1.0,
                          saturation_power=1.0,
                          wavenumber_power=1.0,
                          reference_wavenumber=1.0) =
    WhitecappingDissipation(source_parameter(rate),
                            float(saturation_threshold),
                            float(saturation_power),
                            float(wavenumber_power),
                            float(reference_wavenumber))

struct MeanFrequencyDissipation{Rate, FT} <: AbstractPhysicsTerm
    rate :: Rate
    reference_frequency :: FT
    power :: FT
end

MeanFrequencyDissipation(; rate=0.0, reference_frequency=1.0, power=1.0) =
    MeanFrequencyDissipation(source_parameter(rate), float(reference_frequency), float(power))

struct PeakFrequencyDissipation{Rate, FT} <: AbstractPhysicsTerm
    rate :: Rate
    reference_frequency :: FT
    power :: FT
end

PeakFrequencyDissipation(; rate=0.0, reference_frequency=1.0, power=1.0) =
    PeakFrequencyDissipation(source_parameter(rate), float(reference_frequency), float(power))

struct FrequencyDissipation{Rate, FT} <: AbstractPhysicsTerm
    rate :: Rate
    reference_frequency :: FT
    power :: FT
end

FrequencyDissipation(; rate=0.0, reference_frequency=1.0, power=1.0) =
    FrequencyDissipation(source_parameter(rate), float(reference_frequency), float(power))

struct WavenumberDissipation{Rate, FT} <: AbstractPhysicsTerm
    rate :: Rate
    reference_wavenumber :: FT
    power :: FT
end

WavenumberDissipation(; rate=0.0, reference_wavenumber=1.0, power=1.0) =
    WavenumberDissipation(source_parameter(rate), float(reference_wavenumber), float(power))

struct MeanSquareWavenumberDissipation{Rate, FT} <: AbstractPhysicsTerm
    rate :: Rate
    reference_wavenumber :: FT
    power :: FT
end

MeanSquareWavenumberDissipation(; rate=0.0, reference_wavenumber=1.0, power=1.0) =
    MeanSquareWavenumberDissipation(source_parameter(rate), float(reference_wavenumber), float(power))

struct PeakWavenumberDissipation{Rate, FT} <: AbstractPhysicsTerm
    rate :: Rate
    reference_wavenumber :: FT
    power :: FT
end

PeakWavenumberDissipation(; rate=0.0, reference_wavenumber=1.0, power=1.0) =
    PeakWavenumberDissipation(source_parameter(rate), float(reference_wavenumber), float(power))

struct DepthLimitedBreaking{Rate, Depth, FT} <: AbstractPhysicsTerm
    rate :: Rate
    depth :: Depth
    gamma :: FT
    power :: FT
    minimum_depth :: FT
end

DepthLimitedBreaking(; rate=0.0, depth=1.0, gamma=0.78, power=2.0, minimum_depth=1e-6) =
    DepthLimitedBreaking(source_parameter(rate),
                         source_parameter(depth),
                         float(gamma),
                         float(power),
                         float(minimum_depth))

struct BottomFriction{Rate, Depth, FT} <: AbstractPhysicsTerm
    rate :: Rate
    depth :: Depth
    reference_depth :: FT
    depth_power :: FT
    minimum_depth :: FT
    wavenumber_power :: FT
    reference_wavenumber :: FT
end

function BottomFriction(; rate=0.0,
                          depth=nothing,
                          reference_depth=1.0,
                          depth_power=nothing,
                          minimum_depth=1e-6,
                          wavenumber_power=0.0,
                          reference_wavenumber=1.0)
    power = depth_power === nothing ? (depth === nothing ? 0.0 : 1.0) : depth_power
    return BottomFriction(source_parameter(rate),
                          source_parameter(depth),
                          float(reference_depth),
                          float(power),
                          float(minimum_depth),
                          float(wavenumber_power),
                          float(reference_wavenumber))
end

struct IceDamping{Rate, Concentration, FT} <: AbstractPhysicsTerm
    rate :: Rate
    concentration :: Concentration
    wavenumber_power :: FT
    reference_wavenumber :: FT
end

IceDamping(; rate=0.0,
             concentration=1.0,
             wavenumber_power=0.0,
             reference_wavenumber=1.0) =
    IceDamping(source_parameter(rate),
               source_parameter(concentration),
               float(wavenumber_power),
               float(reference_wavenumber))

struct SwellDissipation{Rate, Direction, FT} <: AbstractPhysicsTerm
    rate :: Rate
    direction :: Direction
    spreading_power :: FT
    wavenumber_power :: FT
    reference_wavenumber :: FT
end

SwellDissipation(; rate=0.0,
                   direction=0.0,
                   spreading_power=2.0,
                   wavenumber_power=0.0,
                   reference_wavenumber=1.0) =
    SwellDissipation(source_parameter(rate),
                     direction,
                     float(spreading_power),
                     float(wavenumber_power),
                     float(reference_wavenumber))

struct MeanDirectionDissipation{Rate, FT} <: AbstractPhysicsTerm
    rate :: Rate
    power :: FT
end

MeanDirectionDissipation(; rate=0.0, power=1.0) =
    MeanDirectionDissipation(source_parameter(rate), float(power))

struct DirectionalDiffusion{Rate} <: AbstractPhysicsTerm
    rate :: Rate
end

DirectionalDiffusion(; rate=0.0) = DirectionalDiffusion(source_parameter(rate))

struct DirectionalAdvection{Velocity} <: AbstractPhysicsTerm
    velocity :: Velocity
end

function DirectionalAdvection(; velocity=0.0, angular_velocity=nothing)
    value = angular_velocity === nothing ? velocity : angular_velocity
    return DirectionalAdvection(source_parameter(value))
end

struct RadialDiffusion{Rate} <: AbstractPhysicsTerm
    rate :: Rate
end

RadialDiffusion(; rate=0.0) = RadialDiffusion(source_parameter(rate))

struct RadialAdvection{Velocity} <: AbstractPhysicsTerm
    velocity :: Velocity
end

RadialAdvection(; velocity=0.0) = RadialAdvection(source_parameter(velocity))

struct SpectralTransferInteraction{Rate}
    from_m :: Int
    from_n :: Int
    to_m :: Int
    to_n :: Int
    rate :: Rate
end

function SpectralTransferInteraction(from::Tuple{Int, Int},
                                     to::Tuple{Int, Int};
                                     rate=0.0)
    return SpectralTransferInteraction(from[1], from[2], to[1], to[2], source_parameter(rate))
end

struct NonlinearSpectralTransfer{Interactions, FT} <: AbstractPhysicsTerm
    interactions :: Interactions
    power :: FT
end

function NonlinearSpectralTransfer(interactions=(); power=2.0)
    return NonlinearSpectralTransfer(tuple(interactions...), float(power))
end

struct SpectralTransferStencil{Bins, Coefficients, Rate}
    bins :: Bins
    coefficients :: Coefficients
    rate :: Rate
end

function SpectralTransferStencil(bins, coefficients; rate=0.0)
    length(bins) == length(coefficients) ||
        throw(ArgumentError("spectral transfer stencil bins and coefficients must have the same length"))
    normalized_bins = tuple(bins...)
    normalized_coefficients = tuple((float(coefficient) for coefficient in coefficients)...)
    length(normalized_bins) > 1 ||
        throw(ArgumentError("spectral transfer stencil requires at least two bins"))
    abs(sum(normalized_coefficients)) <= 32eps(Float64) ||
        throw(ArgumentError("spectral transfer stencil coefficients must sum to zero to conserve action"))
    any(coefficient -> coefficient < 0, normalized_coefficients) ||
        throw(ArgumentError("spectral transfer stencil requires at least one donor coefficient"))
    any(coefficient -> coefficient > 0, normalized_coefficients) ||
        throw(ArgumentError("spectral transfer stencil requires at least one receiver coefficient"))
    return SpectralTransferStencil(normalized_bins, normalized_coefficients, source_parameter(rate))
end

struct NonlinearInvariantTransfer{Stencils, FT} <: AbstractPhysicsTerm
    stencils :: Stencils
    power :: FT
end

function NonlinearInvariantTransfer(stencils=(); power=1.0)
    return NonlinearInvariantTransfer(tuple(stencils...), float(power))
end

struct TriadTransferInteraction{Rate}
    parent1_m :: Int
    parent1_n :: Int
    parent2_m :: Int
    parent2_n :: Int
    child_m :: Int
    child_n :: Int
    rate :: Rate
end

function TriadTransferInteraction(parent1::Tuple{Int, Int},
                                  parent2::Tuple{Int, Int},
                                  child::Tuple{Int, Int};
                                  rate=0.0)
    return TriadTransferInteraction(parent1[1], parent1[2],
                                    parent2[1], parent2[2],
                                    child[1], child[2],
                                    source_parameter(rate))
end

struct TriadSpectralTransfer{Interactions, FT} <: AbstractPhysicsTerm
    interactions :: Interactions
    power :: FT
    frequency_tolerance :: FT
end

function TriadSpectralTransfer(interactions=(); power=1.0, frequency_tolerance=1e-12)
    return TriadSpectralTransfer(tuple(interactions...), float(power), float(frequency_tolerance))
end

struct QuadrupletTransferInteraction{Rate}
    donor1_m :: Int
    donor1_n :: Int
    donor2_m :: Int
    donor2_n :: Int
    receiver1_m :: Int
    receiver1_n :: Int
    receiver2_m :: Int
    receiver2_n :: Int
    rate :: Rate
end

function QuadrupletTransferInteraction(donor1::Tuple{Int, Int},
                                       donor2::Tuple{Int, Int},
                                       receiver1::Tuple{Int, Int},
                                       receiver2::Tuple{Int, Int};
                                       rate=0.0)
    return QuadrupletTransferInteraction(donor1[1], donor1[2],
                                         donor2[1], donor2[2],
                                         receiver1[1], receiver1[2],
                                         receiver2[1], receiver2[2],
                                         source_parameter(rate))
end

struct DiscreteInteractionApproximation{Interactions, FT} <: AbstractPhysicsTerm
    interactions :: Interactions
    power :: FT
    resonance_tolerance :: FT
end

function DiscreteInteractionApproximation(interactions=(); power=1.0, resonance_tolerance=1e-12)
    return DiscreteInteractionApproximation(tuple(interactions...), float(power), float(resonance_tolerance))
end

split_growth_rate(rate, action_value) =
    rate >= 0 ? (rate * action_value, zero(rate)) : (zero(rate * action_value), -rate)

split_damping_rate(rate, action_value) =
    rate >= 0 ? (zero(rate * action_value), rate) : (-rate * action_value, zero(rate))

source_split(::NoPhysics, model, i, j, m, n) = (zero(eltype(model.action)), zero(eltype(model.action)))
source_tendency(::NoPhysics, model, i, j, m, n) = zero(eltype(model.action))
source_split(::Nothing, model, i, j, m, n) = (zero(eltype(model.action)), zero(eltype(model.action)))
source_tendency(::Nothing, model, i, j, m, n) = zero(eltype(model.action))
implicit_source_rate(source::AbstractPhysicsTerm, model, i, j, m, n) = source_split(source, model, i, j, m, n)[2]
implicit_source_rate(::Nothing, model, i, j, m, n) = zero(eltype(model.action))

function source_tendency(s::GenericPhysics, model, i, j, m, n)
    total = zero(eltype(model.action))
    for term in s.terms
        total += source_tendency(term, model, i, j, m, n)
    end
    return total
end

function source_split(s::GenericPhysics, model, i, j, m, n)
    positive = zero(eltype(model.action))
    damping = zero(eltype(model.action))
    for term in s.terms
        term_positive, term_damping = source_split(term, model, i, j, m, n)
        positive += term_positive
        damping += term_damping
    end
    return positive, damping
end

function relaxation_target_value(s::RelaxationToSpectrum, model, i, j, m, n)
    xs, ys = xnodes(model.grid), ynodes(model.grid)
    kx, ky = k_components(model.spectral_grid, m, n)
    return s.target(xs[i], ys[j], kx, ky)
end

function source_split(s::RelaxationToSpectrum, model, i, j, m, n)
    s.timescale > 0 || throw(ArgumentError("relaxation timescale must be positive"))
    damping = inv(s.timescale)
    positive = relaxation_target_value(s, model, i, j, m, n) * damping
    return positive, damping
end

function source_tendency(s::RelaxationToSpectrum, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

source_parameter(::Nothing) = nothing
source_parameter(value::Number) = float(value)
source_parameter(value) = value

source_split(s::LinearWindInput, model, i, j, m, n) =
    split_growth_rate(source_value(s.rate, model, i, j), model.action[i, j, m, n])

function source_tendency(s::LinearWindInput, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function source_split(s::ExponentialWindInput, model, i, j, m, n)
    rate = source_value(s.rate, model, i, j) *
           wind_directional_weight(model.spectral_grid, m, n, wind_direction(s.direction, model, i, j), s.spreading_power)
    return split_growth_rate(rate, model.action[i, j, m, n])
end

function source_tendency(s::ExponentialWindInput, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function source_split(s::PowerLawWindInput, model, i, j, m, n)
    s.reference_speed > 0 || throw(ArgumentError("power-law wind-input reference speed must be positive"))
    s.speed_power >= 0 || throw(ArgumentError("power-law wind-input speed power must be nonnegative"))
    s.spreading_power >= 0 || throw(ArgumentError("power-law wind-input spreading power must be nonnegative"))

    speed = source_value(s.speed, model, i, j)
    speed >= 0 || throw(ArgumentError("power-law wind-input speed must be nonnegative"))
    direction = wind_direction(s.direction, model, i, j)
    speed_factor = (speed / s.reference_speed)^s.speed_power
    direction_factor = wind_directional_weight(model.spectral_grid, m, n, direction, s.spreading_power)
    rate = source_value(s.rate, model, i, j) * speed_factor * direction_factor
    return split_growth_rate(rate, model.action[i, j, m, n])
end

function source_tendency(s::PowerLawWindInput, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function wave_age_wind_alignment(cgrid, m, n, direction)
    kx, ky = k_components(cgrid, m, n)
    k = hypot(kx, ky)
    k == 0 && return zero(k)
    return max((kx * cos(direction) + ky * sin(direction)) / k, zero(k))
end

function source_split(s::WaveAgeWindInput, model, i, j, m, n)
    s.inverse_wave_age_threshold >= 0 ||
        throw(ArgumentError("wave-age wind-input threshold must be nonnegative"))
    s.power >= 0 || throw(ArgumentError("wave-age wind-input power must be nonnegative"))
    s.spreading_power >= 0 || throw(ArgumentError("wave-age wind-input spreading power must be nonnegative"))
    s.gravity > 0 || throw(ArgumentError("wave-age wind-input gravity must be positive"))

    k = radial_wavenumber(model.spectral_grid, m, n)
    k <= 0 && return (zero(eltype(model.action)), zero(eltype(model.action)))

    direction = wind_direction(s.direction, model, i, j)
    alignment = wave_age_wind_alignment(model.spectral_grid, m, n, direction)
    alignment == 0 && return (zero(eltype(model.action)), zero(eltype(model.action)))

    speed = source_value(s.speed, model, i, j)
    phase_speed = sqrt(s.gravity / k)
    inverse_wave_age = speed * alignment / phase_speed
    excess = max(inverse_wave_age - s.inverse_wave_age_threshold, zero(inverse_wave_age))
    rate = source_value(s.rate, model, i, j) * excess^s.power * alignment^s.spreading_power
    return split_growth_rate(rate, model.action[i, j, m, n])
end

function source_tendency(s::WaveAgeWindInput, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function source_tendency(s::SaturationDissipation, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function source_split(s::SaturationDissipation, model, i, j, m, n)
    s.threshold > 0 || throw(ArgumentError("saturation threshold must be positive"))
    saturation = local_zeroth_moment(model, i, j)
    excess = max(saturation / s.threshold - 1, zero(saturation))
    damping_rate = source_value(s.rate, model, i, j) * excess^s.power
    return split_damping_rate(damping_rate, model.action[i, j, m, n])
end

function radial_power_source_factor(cgrid, m, n, power, reference_wavenumber)
    reference_wavenumber > 0 || throw(ArgumentError("reference wavenumber must be positive"))
    power >= 0 || throw(ArgumentError("wavenumber power must be nonnegative"))
    power == 0 && return one(reference_wavenumber)
    return spectral_radial_power_average(cgrid, m, n, power) / reference_wavenumber^power
end

function frequency_power_source_factor(cgrid, m, n, power, reference_frequency)
    reference_frequency > 0 || throw(ArgumentError("reference frequency must be positive"))
    power >= 0 || throw(ArgumentError("frequency power must be nonnegative"))
    power == 0 && return one(reference_frequency)
    return spectral_frequency_power_average(cgrid, m, n, power) / reference_frequency^power
end

function source_tendency(s::WhitecappingDissipation, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function source_split(s::WhitecappingDissipation, model, i, j, m, n)
    s.saturation_threshold > 0 || throw(ArgumentError("whitecapping saturation threshold must be positive"))
    s.reference_wavenumber > 0 || throw(ArgumentError("whitecapping reference wavenumber must be positive"))
    saturation = local_zeroth_moment(model, i, j)
    excess = max(saturation / s.saturation_threshold - 1, zero(saturation))
    spectral_weighting = radial_power_source_factor(model.spectral_grid, m, n,
                                                    s.wavenumber_power,
                                                    s.reference_wavenumber)
    damping_rate = source_value(s.rate, model, i, j) * excess^s.saturation_power * spectral_weighting
    return split_damping_rate(damping_rate, model.action[i, j, m, n])
end

function local_mean_frequency(model, i, j)
    N = model.action
    cgrid = model.spectral_grid
    _, _, Nxi, Neta = size(N)
    numerator = zero(eltype(N))
    denominator = zero(eltype(N))
    for n in 1:Neta, m in 1:Nxi
        action = N[i, j, m, n]
        numerator += action * spectral_frequency_power_measure(cgrid, m, n, 1)
        denominator += action * spectral_weight(cgrid, m, n)
    end
    return denominator == 0 ? zero(eltype(N)) : numerator / denominator
end

function local_peak_frequency(model, i, j)
    N = model.action
    cgrid = model.spectral_grid
    cgrid isa FrequencyDirectionGrid ||
        throw(ArgumentError("peak-frequency dissipation requires a FrequencyDirectionGrid"))

    _, _, Nxi, Neta = size(N)
    best = -Inf
    best_frequency = zero(eltype(N))

    for m in 1:Nxi
        band = zero(eltype(N))
        for n in 1:Neta
            band += max(N[i, j, m, n], zero(eltype(N))) * spectral_weight(cgrid, m, n)
        end

        if band > best
            best = band
            best_frequency = spectral_frequency(cgrid, m, 1)
        end
    end

    return best <= 0 ? zero(eltype(N)) : best_frequency
end

function local_mean_square_wavenumber(model, i, j)
    N = model.action
    cgrid = model.spectral_grid
    _, _, Nxi, Neta = size(N)
    numerator = zero(eltype(N))
    denominator = zero(eltype(N))
    for n in 1:Neta, m in 1:Nxi
        action_weight = N[i, j, m, n]
        weight = spectral_weight(cgrid, m, n)
        xx_measure, _, yy_measure = spectral_second_moment_measures(cgrid, m, n)
        numerator += action_weight * (xx_measure + yy_measure)
        denominator += action_weight * weight
    end
    return denominator == 0 ? zero(eltype(N)) : numerator / denominator
end

function local_peak_wavenumber(model, i, j, cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid})
    N = model.action
    _, _, Nxi, Neta = size(N)
    best = -Inf
    best_wavenumber = zero(eltype(N))

    for m in 1:Nxi
        band = zero(eltype(N))
        for n in 1:Neta
            band += max(N[i, j, m, n], zero(eltype(N))) * spectral_weight(cgrid, m, n)
        end

        if band > best
            best = band
            best_wavenumber = radial_wavenumber(cgrid, m, 1)
        end
    end

    return best <= 0 ? zero(eltype(N)) : best_wavenumber
end

function local_peak_wavenumber(model, i, j, cgrid)
    N = model.action
    _, _, Nxi, Neta = size(N)
    best = -Inf
    best_wavenumber = zero(eltype(N))

    for n in 1:Neta, m in 1:Nxi
        value = max(N[i, j, m, n], zero(eltype(N))) * spectral_weight(cgrid, m, n)
        if value > best
            best = value
            best_wavenumber = radial_wavenumber(cgrid, m, n)
        end
    end

    return best <= 0 ? zero(eltype(N)) : best_wavenumber
end

function source_split(s::MeanFrequencyDissipation, model, i, j, m, n)
    s.reference_frequency > 0 || throw(ArgumentError("mean-frequency reference frequency must be positive"))
    fmean = local_mean_frequency(model, i, j)
    damping_rate = source_value(s.rate, model, i, j) * (fmean / s.reference_frequency)^s.power
    return split_damping_rate(damping_rate, model.action[i, j, m, n])
end

function source_tendency(s::MeanFrequencyDissipation, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function source_split(s::PeakFrequencyDissipation, model, i, j, m, n)
    s.reference_frequency > 0 || throw(ArgumentError("peak-frequency reference frequency must be positive"))
    s.power >= 0 || throw(ArgumentError("peak-frequency dissipation power must be nonnegative"))
    fpeak = local_peak_frequency(model, i, j)
    damping_rate = source_value(s.rate, model, i, j) * (fpeak / s.reference_frequency)^s.power
    return split_damping_rate(damping_rate, model.action[i, j, m, n])
end

function source_tendency(s::PeakFrequencyDissipation, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function source_split(s::FrequencyDissipation, model, i, j, m, n)
    s.reference_frequency > 0 || throw(ArgumentError("frequency dissipation reference frequency must be positive"))
    s.power >= 0 || throw(ArgumentError("frequency dissipation power must be nonnegative"))
    spectral_factor = frequency_power_source_factor(model.spectral_grid, m, n,
                                                    s.power,
                                                    s.reference_frequency)
    damping_rate = source_value(s.rate, model, i, j) * spectral_factor
    return split_damping_rate(damping_rate, model.action[i, j, m, n])
end

function source_tendency(s::FrequencyDissipation, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function source_split(s::WavenumberDissipation, model, i, j, m, n)
    s.reference_wavenumber > 0 || throw(ArgumentError("wavenumber dissipation reference wavenumber must be positive"))
    s.power >= 0 || throw(ArgumentError("wavenumber dissipation power must be nonnegative"))
    spectral_factor = radial_power_source_factor(model.spectral_grid, m, n,
                                                 s.power,
                                                 s.reference_wavenumber)
    damping_rate = source_value(s.rate, model, i, j) * spectral_factor
    return split_damping_rate(damping_rate, model.action[i, j, m, n])
end

function source_tendency(s::WavenumberDissipation, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function source_split(s::MeanSquareWavenumberDissipation, model, i, j, m, n)
    s.reference_wavenumber > 0 || throw(ArgumentError("mean-square-wavenumber reference wavenumber must be positive"))
    s.power >= 0 || throw(ArgumentError("mean-square-wavenumber dissipation power must be nonnegative"))
    mean_square_k = local_mean_square_wavenumber(model, i, j)
    rms_k = sqrt(max(mean_square_k, zero(mean_square_k)))
    damping_rate = source_value(s.rate, model, i, j) * (rms_k / s.reference_wavenumber)^s.power
    return split_damping_rate(damping_rate, model.action[i, j, m, n])
end

function source_tendency(s::MeanSquareWavenumberDissipation, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function source_split(s::PeakWavenumberDissipation, model, i, j, m, n)
    s.reference_wavenumber > 0 || throw(ArgumentError("peak-wavenumber reference wavenumber must be positive"))
    s.power >= 0 || throw(ArgumentError("peak-wavenumber dissipation power must be nonnegative"))
    kpeak = local_peak_wavenumber(model, i, j, model.spectral_grid)
    damping_rate = source_value(s.rate, model, i, j) * (kpeak / s.reference_wavenumber)^s.power
    return split_damping_rate(damping_rate, model.action[i, j, m, n])
end

function source_tendency(s::PeakWavenumberDissipation, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function depth_limited_breaking_rate(s::DepthLimitedBreaking, model, i, j)
    s.gamma > 0 || throw(ArgumentError("depth-limited breaking gamma must be positive"))
    s.minimum_depth > 0 || throw(ArgumentError("depth-limited breaking minimum depth must be positive"))
    depth = max(source_value(s.depth, model, i, j), s.minimum_depth)
    saturation_height = s.gamma * depth
    Hs = 4sqrt(max(local_zeroth_moment(model, i, j), zero(eltype(model.action))))
    excess = max(Hs / saturation_height - 1, zero(Hs))
    return source_value(s.rate, model, i, j) * excess^s.power
end

source_split(s::DepthLimitedBreaking, model, i, j, m, n) =
    split_damping_rate(depth_limited_breaking_rate(s, model, i, j), model.action[i, j, m, n])

function source_tendency(s::DepthLimitedBreaking, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function bottom_friction_rate(s::BottomFriction, model, i, j, m, n)
    s.reference_depth > 0 || throw(ArgumentError("bottom-friction reference depth must be positive"))
    s.minimum_depth > 0 || throw(ArgumentError("bottom-friction minimum depth must be positive"))
    s.reference_wavenumber > 0 || throw(ArgumentError("bottom-friction reference wavenumber must be positive"))

    rate = source_value(s.rate, model, i, j)
    depth_factor = if s.depth === nothing || s.depth_power == 0
        one(rate)
    else
        depth = max(source_value(s.depth, model, i, j), s.minimum_depth)
        (s.reference_depth / depth)^s.depth_power
    end

    spectral_factor = radial_power_source_factor(model.spectral_grid, m, n,
                                                 s.wavenumber_power,
                                                 s.reference_wavenumber)
    return rate * depth_factor * spectral_factor
end

source_split(s::BottomFriction, model, i, j, m, n) =
    split_damping_rate(bottom_friction_rate(s, model, i, j, m, n), model.action[i, j, m, n])

function source_tendency(s::BottomFriction, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function ice_damping_rate(s::IceDamping, model, i, j, m, n)
    s.reference_wavenumber > 0 || throw(ArgumentError("ice-damping reference wavenumber must be positive"))
    spectral_factor = radial_power_source_factor(model.spectral_grid, m, n,
                                                 s.wavenumber_power,
                                                 s.reference_wavenumber)
    concentration = clamp(source_value(s.concentration, model, i, j), 0.0, 1.0)
    return source_value(s.rate, model, i, j) * concentration * spectral_factor
end

source_split(s::IceDamping, model, i, j, m, n) =
    split_damping_rate(ice_damping_rate(s, model, i, j, m, n), model.action[i, j, m, n])

function source_tendency(s::IceDamping, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function swell_dissipation_rate(s::SwellDissipation, model, i, j, m, n)
    s.reference_wavenumber > 0 || throw(ArgumentError("swell-dissipation reference wavenumber must be positive"))
    spectral_factor = radial_power_source_factor(model.spectral_grid, m, n,
                                                 s.wavenumber_power,
                                                 s.reference_wavenumber)
    wind_sea_weight = wind_directional_weight(model.spectral_grid, m, n,
                                              wind_direction(s.direction, model, i, j),
                                              s.spreading_power)
    swell_weight = max(1 - wind_sea_weight, 0.0)
    return source_value(s.rate, model, i, j) * swell_weight * spectral_factor
end

source_split(s::SwellDissipation, model, i, j, m, n) =
    split_damping_rate(swell_dissipation_rate(s, model, i, j, m, n), model.action[i, j, m, n])

function source_tendency(s::SwellDissipation, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function local_mean_direction_unit(model, i, j)
    N = model.action
    cgrid = model.spectral_grid
    _, _, Nxi, Neta = size(N)
    mx = zero(eltype(N))
    my = zero(eltype(N))
    scale = zero(eltype(N))

    for n in 1:Neta, m in 1:Nxi
        kx_measure, ky_measure = spectral_first_moment_measures(cgrid, m, n)
        action = N[i, j, m, n]
        mx += action * kx_measure
        my += action * ky_measure
        scale += abs(action) * hypot(kx_measure, ky_measure)
    end

    magnitude = hypot(mx, my)
    threshold = sqrt(eps(float(max(scale, one(scale))))) * max(scale, one(scale))
    return magnitude <= threshold ? (zero(mx), zero(my)) : (mx / magnitude, my / magnitude)
end

function mean_direction_dissipation_rate(s::MeanDirectionDissipation, model, i, j, m, n)
    s.power >= 0 || throw(ArgumentError("mean-direction dissipation power must be nonnegative"))

    k = radial_wavenumber(model.spectral_grid, m, n)
    k == 0 && return zero(eltype(model.action))

    ux, uy = local_mean_direction_unit(model, i, j)
    ux == 0 && uy == 0 && return zero(eltype(model.action))

    kx, ky = k_components(model.spectral_grid, m, n)
    alignment = clamp((kx * ux + ky * uy) / k, -one(k), one(k))
    mismatch = max(1 - alignment, zero(alignment))
    return source_value(s.rate, model, i, j) * mismatch^s.power
end

source_split(s::MeanDirectionDissipation, model, i, j, m, n) =
    split_damping_rate(mean_direction_dissipation_rate(s, model, i, j, m, n), model.action[i, j, m, n])

function source_tendency(s::MeanDirectionDissipation, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function directional_diffusion_spacing(cgrid)
    cgrid isa Union{PolarWaveVectorGrid, FrequencyDirectionGrid} ||
        throw(ArgumentError("DirectionalDiffusion requires a direction-coordinate spectral grid"))
    cgrid.topology[2] isa Periodic ||
        throw(ArgumentError("DirectionalDiffusion requires periodic directional topology"))
    spacings = coordinate_spacings(cgrid, 2)
    first_spacing = first(spacings)
    all(Δ -> isapprox(Δ, first_spacing; rtol=1e-12, atol=1e-14), spacings) ||
        throw(ArgumentError("DirectionalDiffusion currently requires uniform directional spacing"))
    return first_spacing
end

function source_split(s::DirectionalDiffusion, model, i, j, m, n)
    rate = source_value(s.rate, model, i, j)
    rate >= 0 || throw(ArgumentError("directional diffusion rate must be nonnegative"))
    _, Neta = coordinate_size(model.spectral_grid)
    Δφ = directional_diffusion_spacing(model.spectral_grid)
    left = periodic_index(n - 1, Neta)
    right = periodic_index(n + 1, Neta)
    coefficient = rate / Δφ^2
    positive = coefficient * (model.action[i, j, m, left] + model.action[i, j, m, right])
    damping = 2coefficient
    return positive, damping
end

function source_tendency(s::DirectionalDiffusion, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function directional_advection_face_measure(cgrid, m, nleft, nright)
    spacings = coordinate_spacings(cgrid, 2)
    left_measure_density = spectral_weight(cgrid, m, nleft) / spacings[nleft]
    right_measure_density = spectral_weight(cgrid, m, nright) / spacings[nright]
    left_measure_density > 0 || throw(ArgumentError("DirectionalAdvection requires positive spectral cell measures"))
    right_measure_density > 0 || throw(ArgumentError("DirectionalAdvection requires positive spectral cell measures"))
    return sqrt(left_measure_density * right_measure_density)
end

function source_split(s::DirectionalAdvection, model, i, j, m, n)
    model.spectral_grid isa Union{PolarWaveVectorGrid, FrequencyDirectionGrid} ||
        throw(ArgumentError("DirectionalAdvection requires a direction-coordinate spectral grid"))
    model.spectral_grid.topology[2] isa Periodic ||
        throw(ArgumentError("DirectionalAdvection requires periodic directional topology"))

    velocity = source_value(s.velocity, model, i, j)
    cgrid = model.spectral_grid
    _, Neta = coordinate_size(cgrid)
    Neta == 1 && return (zero(eltype(model.action)), zero(eltype(model.action)))

    weight = spectral_weight(cgrid, m, n)
    weight > 0 || throw(ArgumentError("DirectionalAdvection requires positive spectral cell measures"))

    left = periodic_index(n - 1, Neta)
    right = periodic_index(n + 1, Neta)
    left_coefficient = abs(velocity) * directional_advection_face_measure(cgrid, m, left, n) / weight
    right_coefficient = abs(velocity) * directional_advection_face_measure(cgrid, m, n, right) / weight

    positive = zero(eltype(model.action))
    damping = zero(eltype(model.action))

    if velocity > 0
        positive += left_coefficient * model.action[i, j, m, left]
        damping += right_coefficient
    elseif velocity < 0
        damping += left_coefficient
        positive += right_coefficient * model.action[i, j, m, right]
    end

    return positive, damping
end

function source_tendency(s::DirectionalAdvection, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function radial_diffusion_coefficient(rate, cgrid, m, neighbor, n)
    centers = coordinate_centers(cgrid, 1)
    distance = abs(centers[m] - centers[neighbor])
    distance > 0 || throw(ArgumentError("RadialDiffusion requires distinct radial coordinate centers"))
    weight = spectral_weight(cgrid, m, n)
    neighbor_weight = spectral_weight(cgrid, neighbor, n)
    conductance = rate * sqrt(weight * neighbor_weight) / distance^2
    return conductance / weight
end

function source_split(s::RadialDiffusion, model, i, j, m, n)
    model.spectral_grid isa Union{PolarWaveVectorGrid, FrequencyDirectionGrid} ||
        throw(ArgumentError("RadialDiffusion requires a radial-coordinate spectral grid"))
    model.spectral_grid.topology[1] isa Periodic &&
        throw(ArgumentError("RadialDiffusion uses no-flux radial boundaries and does not support periodic radial topology"))

    rate = source_value(s.rate, model, i, j)
    rate >= 0 || throw(ArgumentError("radial diffusion rate must be nonnegative"))
    cgrid = model.spectral_grid
    Nxi, _ = coordinate_size(cgrid)

    positive = zero(eltype(model.action))
    damping = zero(eltype(model.action))

    if m > 1
        coefficient = radial_diffusion_coefficient(rate, cgrid, m, m - 1, n)
        positive += coefficient * model.action[i, j, m - 1, n]
        damping += coefficient
    end

    if m < Nxi
        coefficient = radial_diffusion_coefficient(rate, cgrid, m, m + 1, n)
        positive += coefficient * model.action[i, j, m + 1, n]
        damping += coefficient
    end

    return positive, damping
end

function source_tendency(s::RadialDiffusion, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function radial_advection_face_measure(cgrid, mleft, mright, n)
    spacings = coordinate_spacings(cgrid, 1)
    left_measure_density = spectral_weight(cgrid, mleft, n) / spacings[mleft]
    right_measure_density = spectral_weight(cgrid, mright, n) / spacings[mright]
    left_measure_density > 0 || throw(ArgumentError("RadialAdvection requires positive spectral cell measures"))
    right_measure_density > 0 || throw(ArgumentError("RadialAdvection requires positive spectral cell measures"))
    return sqrt(left_measure_density * right_measure_density)
end

function source_split(s::RadialAdvection, model, i, j, m, n)
    model.spectral_grid isa Union{PolarWaveVectorGrid, FrequencyDirectionGrid} ||
        throw(ArgumentError("RadialAdvection requires a radial-coordinate spectral grid"))
    model.spectral_grid.topology[1] isa Periodic &&
        throw(ArgumentError("RadialAdvection uses no-flux radial boundaries and does not support periodic radial topology"))

    velocity = source_value(s.velocity, model, i, j)
    cgrid = model.spectral_grid
    Nxi, _ = coordinate_size(cgrid)
    weight = spectral_weight(cgrid, m, n)
    weight > 0 || throw(ArgumentError("RadialAdvection requires positive spectral cell measures"))

    positive = zero(eltype(model.action))
    damping = zero(eltype(model.action))

    if m > 1
        coefficient = abs(velocity) * radial_advection_face_measure(cgrid, m - 1, m, n) / weight
        if velocity > 0
            positive += coefficient * model.action[i, j, m - 1, n]
        elseif velocity < 0
            damping += coefficient
        end
    end

    if m < Nxi
        coefficient = abs(velocity) * radial_advection_face_measure(cgrid, m, m + 1, n) / weight
        if velocity > 0
            damping += coefficient
        elseif velocity < 0
            positive += coefficient * model.action[i, j, m + 1, n]
        end
    end

    return positive, damping
end

function source_tendency(s::RadialAdvection, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function check_spectral_transfer_indices(interaction::SpectralTransferInteraction, cgrid)
    Nxi, Neta = coordinate_size(cgrid)
    1 <= interaction.from_m <= Nxi ||
        throw(ArgumentError("spectral transfer donor radial index $(interaction.from_m) is outside 1:$Nxi"))
    1 <= interaction.from_n <= Neta ||
        throw(ArgumentError("spectral transfer donor directional index $(interaction.from_n) is outside 1:$Neta"))
    1 <= interaction.to_m <= Nxi ||
        throw(ArgumentError("spectral transfer receiver radial index $(interaction.to_m) is outside 1:$Nxi"))
    1 <= interaction.to_n <= Neta ||
        throw(ArgumentError("spectral transfer receiver directional index $(interaction.to_n) is outside 1:$Neta"))
    return nothing
end

function source_split(s::NonlinearSpectralTransfer, model, i, j, m, n)
    s.power >= 1 || throw(ArgumentError("nonlinear spectral-transfer power must be at least one"))

    cgrid = model.spectral_grid
    weight = spectral_weight(cgrid, m, n)
    weight > 0 || throw(ArgumentError("nonlinear spectral transfer requires positive spectral cell measures"))

    positive = zero(eltype(model.action))
    damping = zero(eltype(model.action))

    for interaction in s.interactions
        check_spectral_transfer_indices(interaction, cgrid)
        rate = source_value(interaction.rate, model, i, j)
        rate >= 0 || throw(ArgumentError("nonlinear spectral-transfer rates must be nonnegative"))

        donor_action = model.action[i, j, interaction.from_m, interaction.from_n]
        donor_weight = spectral_weight(cgrid, interaction.from_m, interaction.from_n)
        donor_weight > 0 || throw(ArgumentError("nonlinear spectral transfer requires positive donor cell measures"))
        transfer = rate * donor_action^s.power

        if m == interaction.to_m && n == interaction.to_n
            positive += transfer * donor_weight / weight
        end

        if m == interaction.from_m && n == interaction.from_n
            damping += rate * donor_action^(s.power - 1)
        end
    end

    return positive, damping
end

function source_tendency(s::NonlinearSpectralTransfer, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function check_spectral_stencil_indices(stencil::SpectralTransferStencil, cgrid)
    Nxi, Neta = coordinate_size(cgrid)
    for (m, n) in stencil.bins
        1 <= m <= Nxi ||
            throw(ArgumentError("spectral transfer stencil radial index $m is outside 1:$Nxi"))
        1 <= n <= Neta ||
            throw(ArgumentError("spectral transfer stencil directional index $n is outside 1:$Neta"))
    end
    return nothing
end

function source_split(s::NonlinearInvariantTransfer, model, i, j, m, n)
    s.power > 0 || throw(ArgumentError("nonlinear invariant-transfer power must be positive"))

    cgrid = model.spectral_grid
    weight = spectral_weight(cgrid, m, n)
    weight > 0 || throw(ArgumentError("nonlinear invariant transfer requires positive spectral cell measures"))

    positive = zero(eltype(model.action))
    damping = zero(eltype(model.action))

    for stencil in s.stencils
        check_spectral_stencil_indices(stencil, cgrid)
        rate = source_value(stencil.rate, model, i, j)
        rate >= 0 || throw(ArgumentError("nonlinear invariant-transfer rates must be nonnegative"))

        flux = rate
        for ((donor_m, donor_n), coefficient) in zip(stencil.bins, stencil.coefficients)
            if coefficient < 0
                donor_action = model.action[i, j, donor_m, donor_n]
                flux *= max(donor_action, zero(donor_action))^s.power
            end
        end

        flux == 0 && continue

        for ((bin_m, bin_n), coefficient) in zip(stencil.bins, stencil.coefficients)
            if m == bin_m && n == bin_n
                density_tendency = coefficient * flux / weight
                if coefficient > 0
                    positive += density_tendency
                elseif coefficient < 0
                    action_value = model.action[i, j, m, n]
                    action_value > 0 && (damping += -density_tendency / action_value)
                end
            end
        end
    end

    return positive, damping
end

function source_tendency(s::NonlinearInvariantTransfer, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function check_triad_transfer_indices(interaction::TriadTransferInteraction, cgrid)
    cgrid isa FrequencyDirectionGrid ||
        throw(ArgumentError("TriadSpectralTransfer requires a FrequencyDirectionGrid"))

    Nxi, Neta = coordinate_size(cgrid)
    bins = ((interaction.parent1_m, interaction.parent1_n),
            (interaction.parent2_m, interaction.parent2_n),
            (interaction.child_m, interaction.child_n))

    for (m, n) in bins
        1 <= m <= Nxi ||
            throw(ArgumentError("triad transfer radial index $m is outside 1:$Nxi"))
        1 <= n <= Neta ||
            throw(ArgumentError("triad transfer directional index $n is outside 1:$Neta"))
    end

    child = (interaction.child_m, interaction.child_n)
    child != (interaction.parent1_m, interaction.parent1_n) &&
        child != (interaction.parent2_m, interaction.parent2_n) ||
        throw(ArgumentError("triad transfer child bin must differ from parent bins"))

    return nothing
end

function check_triad_frequency_resonance(interaction::TriadTransferInteraction, cgrid, tolerance)
    parent1_frequency = spectral_frequency(cgrid, interaction.parent1_m, interaction.parent1_n)
    parent2_frequency = spectral_frequency(cgrid, interaction.parent2_m, interaction.parent2_n)
    child_frequency = spectral_frequency(cgrid, interaction.child_m, interaction.child_n)
    parent_sum = parent1_frequency + parent2_frequency
    scale = max(abs(parent_sum), abs(child_frequency), eps(Float64))
    error = abs(child_frequency - parent_sum)
    error <= tolerance * scale ||
        throw(ArgumentError("triad transfer requires child frequency to match the sum of parent frequencies within tolerance"))
    return nothing
end

triad_bin_matches(m, n, bin_m, bin_n) = m == bin_m && n == bin_n

function source_split(s::TriadSpectralTransfer, model, i, j, m, n)
    s.power > 0 || throw(ArgumentError("triad transfer power must be positive"))
    s.frequency_tolerance >= 0 || throw(ArgumentError("triad transfer frequency tolerance must be nonnegative"))

    cgrid = model.spectral_grid
    cgrid isa FrequencyDirectionGrid ||
        throw(ArgumentError("TriadSpectralTransfer requires a FrequencyDirectionGrid"))

    weight = spectral_weight(cgrid, m, n)
    weight > 0 || throw(ArgumentError("triad transfer requires positive spectral cell measures"))

    positive = zero(eltype(model.action))
    damping = zero(eltype(model.action))

    for interaction in s.interactions
        check_triad_transfer_indices(interaction, cgrid)
        check_triad_frequency_resonance(interaction, cgrid, s.frequency_tolerance)

        rate = source_value(interaction.rate, model, i, j)
        rate >= 0 || throw(ArgumentError("triad transfer rates must be nonnegative"))

        parent1_action = max(model.action[i, j, interaction.parent1_m, interaction.parent1_n],
                             zero(eltype(model.action)))
        parent2_action = max(model.action[i, j, interaction.parent2_m, interaction.parent2_n],
                             zero(eltype(model.action)))
        action_flux = rate * parent1_action^s.power * parent2_action^s.power
        action_flux == 0 && continue

        if triad_bin_matches(m, n, interaction.child_m, interaction.child_n)
            positive += action_flux / weight
        end

        if triad_bin_matches(m, n, interaction.parent1_m, interaction.parent1_n)
            action_value = model.action[i, j, m, n]
            action_value > 0 && (damping += action_flux / (weight * action_value))
        end

        if triad_bin_matches(m, n, interaction.parent2_m, interaction.parent2_n)
            action_value = model.action[i, j, m, n]
            action_value > 0 && (damping += action_flux / (weight * action_value))
        end
    end

    return positive, damping
end

function source_tendency(s::TriadSpectralTransfer, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

quadruplet_bins(interaction::QuadrupletTransferInteraction) =
    ((interaction.donor1_m, interaction.donor1_n),
     (interaction.donor2_m, interaction.donor2_n),
     (interaction.receiver1_m, interaction.receiver1_n),
     (interaction.receiver2_m, interaction.receiver2_n))

function check_quadruplet_transfer_indices(interaction::QuadrupletTransferInteraction, cgrid)
    cgrid isa FrequencyDirectionGrid ||
        throw(ArgumentError("DiscreteInteractionApproximation requires a FrequencyDirectionGrid"))

    Nxi, Neta = coordinate_size(cgrid)
    for (m, n) in quadruplet_bins(interaction)
        1 <= m <= Nxi ||
            throw(ArgumentError("quadruplet transfer radial index $m is outside 1:$Nxi"))
        1 <= n <= Neta ||
            throw(ArgumentError("quadruplet transfer directional index $n is outside 1:$Neta"))
    end

    donors = ((interaction.donor1_m, interaction.donor1_n),
              (interaction.donor2_m, interaction.donor2_n))
    receivers = ((interaction.receiver1_m, interaction.receiver1_n),
                 (interaction.receiver2_m, interaction.receiver2_n))
    sort(collect(donors)) != sort(collect(receivers)) ||
        throw(ArgumentError("quadruplet transfer donor and receiver bin pairs must differ"))

    return nothing
end

function check_quadruplet_resonance(interaction::QuadrupletTransferInteraction, cgrid, tolerance)
    tolerance >= 0 || throw(ArgumentError("quadruplet resonance tolerance must be nonnegative"))

    donor1 = (interaction.donor1_m, interaction.donor1_n)
    donor2 = (interaction.donor2_m, interaction.donor2_n)
    receiver1 = (interaction.receiver1_m, interaction.receiver1_n)
    receiver2 = (interaction.receiver2_m, interaction.receiver2_n)

    donor_frequency = spectral_frequency(cgrid, donor1...) + spectral_frequency(cgrid, donor2...)
    receiver_frequency = spectral_frequency(cgrid, receiver1...) + spectral_frequency(cgrid, receiver2...)
    frequency_scale = max(abs(donor_frequency), abs(receiver_frequency), eps(Float64))
    frequency_error = abs(receiver_frequency - donor_frequency)
    frequency_error <= tolerance * frequency_scale ||
        throw(ArgumentError("quadruplet transfer requires receiver frequencies to match donor frequencies within tolerance"))

    donor1_kx, donor1_ky = k_components(cgrid, donor1...)
    donor2_kx, donor2_ky = k_components(cgrid, donor2...)
    receiver1_kx, receiver1_ky = k_components(cgrid, receiver1...)
    receiver2_kx, receiver2_ky = k_components(cgrid, receiver2...)
    donor_kx = donor1_kx + donor2_kx
    donor_ky = donor1_ky + donor2_ky
    receiver_kx = receiver1_kx + receiver2_kx
    receiver_ky = receiver1_ky + receiver2_ky
    vector_scale = max(hypot(donor_kx, donor_ky), hypot(receiver_kx, receiver_ky),
                       maximum(abs, (donor1_kx, donor1_ky, donor2_kx, donor2_ky,
                                     receiver1_kx, receiver1_ky, receiver2_kx, receiver2_ky)),
                       eps(Float64))
    vector_error = hypot(receiver_kx - donor_kx, receiver_ky - donor_ky)
    vector_error <= tolerance * vector_scale ||
        throw(ArgumentError("quadruplet transfer requires receiver wavevectors to match donor wavevectors within tolerance"))

    return nothing
end

quadruplet_bin_matches(m, n, bin_m, bin_n) = m == bin_m && n == bin_n

function source_split(s::DiscreteInteractionApproximation, model, i, j, m, n)
    s.power > 0 || throw(ArgumentError("DIA transfer power must be positive"))

    cgrid = model.spectral_grid
    cgrid isa FrequencyDirectionGrid ||
        throw(ArgumentError("DiscreteInteractionApproximation requires a FrequencyDirectionGrid"))

    weight = spectral_weight(cgrid, m, n)
    weight > 0 || throw(ArgumentError("DIA transfer requires positive spectral cell measures"))

    positive = zero(eltype(model.action))
    damping = zero(eltype(model.action))

    for interaction in s.interactions
        check_quadruplet_transfer_indices(interaction, cgrid)
        check_quadruplet_resonance(interaction, cgrid, s.resonance_tolerance)

        rate = source_value(interaction.rate, model, i, j)
        rate >= 0 || throw(ArgumentError("DIA transfer rates must be nonnegative"))

        donor1_action = max(model.action[i, j, interaction.donor1_m, interaction.donor1_n],
                            zero(eltype(model.action)))
        donor2_action = max(model.action[i, j, interaction.donor2_m, interaction.donor2_n],
                            zero(eltype(model.action)))
        action_flux = rate * donor1_action^s.power * donor2_action^s.power
        action_flux == 0 && continue

        if quadruplet_bin_matches(m, n, interaction.receiver1_m, interaction.receiver1_n)
            positive += action_flux / weight
        end

        if quadruplet_bin_matches(m, n, interaction.receiver2_m, interaction.receiver2_n)
            positive += action_flux / weight
        end

        if quadruplet_bin_matches(m, n, interaction.donor1_m, interaction.donor1_n)
            action_value = model.action[i, j, m, n]
            action_value > 0 && (damping += action_flux / (weight * action_value))
        end

        if quadruplet_bin_matches(m, n, interaction.donor2_m, interaction.donor2_n)
            action_value = model.action[i, j, m, n]
            action_value > 0 && (damping += action_flux / (weight * action_value))
        end
    end

    return positive, damping
end

function source_tendency(s::DiscreteInteractionApproximation, model, i, j, m, n)
    positive, damping = source_split(s, model, i, j, m, n)
    return positive - damping * model.action[i, j, m, n]
end

function nonnegative_integer_spreading_power(spreading_power)
    spreading_power >= 0 ||
        throw(ArgumentError("directional spreading power must be nonnegative"))
    isinteger(spreading_power) ||
        throw(ArgumentError("finite-volume directional spreading currently supports nonnegative integer powers; got $spreading_power"))
    return Int(spreading_power)
end

function cosine_power_antiderivative(α, power::Int)
    power == 0 && return α
    power == 1 && return sin(α)
    return sin(α) * cos(α)^(power - 1) / power +
           (power - 1) / power * cosine_power_antiderivative(α, power - 2)
end

function positive_cosine_power_integral(α₁, α₂, power::Int)
    α₂ < α₁ && return -positive_cosine_power_integral(α₂, α₁, power)
    power == 0 && return α₂ - α₁

    period = 2π
    lower_period = floor(Int, (α₁ - π / 2) / period) - 1
    upper_period = ceil(Int, (α₂ + π / 2) / period) + 1
    integral = zero(float(α₁ + α₂))
    F = α -> cosine_power_antiderivative(α, power)

    for q in lower_period:upper_period
        shift = q * period
        left = max(α₁, shift - π / 2)
        right = min(α₂, shift + π / 2)
        right > left || continue
        integral += F(right - shift) - F(left - shift)
    end

    return integral
end

function wind_directional_weight(cgrid::Union{PolarWaveVectorGrid, FrequencyDirectionGrid},
                                 m, n, direction, spreading_power)
    power = nonnegative_integer_spreading_power(spreading_power)
    power == 0 && return one(float(spreading_power))

    faces = coordinate_faces(cgrid, 2)
    φ₁ = faces[n]
    φ₂ = faces[n+1]
    Δφ = φ₂ - φ₁
    Δφ > 0 || throw(ArgumentError("directional cell faces must be increasing"))
    return positive_cosine_power_integral(φ₁ - direction, φ₂ - direction, power) / Δφ
end

function wind_directional_weight(cgrid, m, n, direction, spreading_power)
    spreading_power >= 0 ||
        throw(ArgumentError("directional spreading power must be nonnegative"))
    kx, ky = k_components(cgrid, m, n)
    k = hypot(kx, ky)
    k == 0 && return zero(k)
    alignment = (kx * cos(direction) + ky * sin(direction)) / k
    return max(alignment, zero(alignment))^spreading_power
end

wind_direction(direction::Number, model, i, j) = direction
wind_direction(w::Union{StationaryVortexWind, IdealizedHurricaneWind, HollandHurricaneWind}, model, i, j) =
    wind_angle(w, xnodes(model.grid)[i], ynodes(model.grid)[j], model.clock.time)

source_value(value::Number, model, i, j) = value
source_value(value::AbstractArray, model, i, j) = value[i, j]

function source_value(value, model, i, j)
    x = xnodes(model.grid)[i]
    y = ynodes(model.grid)[j]
    t = model.clock.time

    if applicable(value, x, y, t)
        return value(x, y, t)
    elseif applicable(value, x, y)
        return value(x, y)
    elseif applicable(value, t)
        return value(t)
    else
        throw(ArgumentError("callable source parameters must accept (x, y, t), (x, y), or (t)"))
    end
end

wind_direction(direction, model, i, j) = source_value(direction, model, i, j)

function local_zeroth_moment(model, i, j)
    N = model.action
    cgrid = model.spectral_grid
    _, _, Nxi, Neta = size(N)
    total = zero(eltype(N))
    for n in 1:Neta, m in 1:Nxi
        total += N[i, j, m, n] * spectral_weight(cgrid, m, n)
    end
    return total
end

# Bundle interface — prepare_physics + state-aware source_split fallbacks.
include("bundle_interface.jl")

# Operational physics packages.
include("WindInput/pressure_correlation.jl")
include("Dissipation/mean_spectrum.jl")
include("Packages/mean_spectrum_physics.jl")
include("NonlinearInteractions/hasselmann_dia.jl")
include("Dissipation/local_saturation.jl")
