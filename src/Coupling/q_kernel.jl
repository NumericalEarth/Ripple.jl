struct QKernel{FT}
    small_mu :: FT
end

QKernel(::Type{FT}=Float64; small_mu=FT(1e-6)) where FT = QKernel{FT}(small_mu)

function q_value(kernel::QKernel, kappa, z, depth)
    depth <= 0 && throw(ArgumentError("depth must be positive"))
    mu = kappa * depth
    abs(mu) < kernel.small_mu && return inv(depth)
    s = (z + depth) / depth
    b = 2mu
    a = b * s
    ratio = (exp(a - b) + exp(-a - b)) / (1 - exp(-2b))
    return (2mu / depth) * ratio
end

function q_cdf(kernel::QKernel, kappa, z, depth)
    depth <= 0 && throw(ArgumentError("depth must be positive"))
    mu = kappa * depth
    s = clamp((z + depth) / depth, zero(depth), one(depth))
    abs(mu) < kernel.small_mu && return s

    a = 2 * mu * s
    b = 2 * mu
    return (exp(a - b) - exp(-a - b)) / (1 - exp(-2b))
end

function q_cell_integral(kernel::QKernel, kappa, z₁, z₂, depth)
    zlow, zhigh = minmax(z₁, z₂)
    return q_cdf(kernel, kappa, zhigh, depth) - q_cdf(kernel, kappa, zlow, depth)
end

function q_cdf_kappa_derivative(kernel::QKernel, kappa, z, depth)
    depth <= 0 && throw(ArgumentError("depth must be positive"))
    mu = kappa * depth
    s = clamp((z + depth) / depth, zero(depth), one(depth))

    if abs(mu) < kernel.small_mu
        return depth * (4 * mu * s * (s^2 - 1) / 3)
    end

    a = exp(-2 * mu * (1 - s))
    b = exp(-2 * mu * (1 + s))
    e = exp(-4 * mu)
    numerator = a - b
    denominator = 1 - e
    dnumerator_dmu = -2 * (1 - s) * a + 2 * (1 + s) * b
    ddenominator_dmu = 4e
    dF_dmu = (dnumerator_dmu * denominator - numerator * ddenominator_dmu) / denominator^2
    return depth * dF_dmu
end

function q_cell_integral_kappa_derivative(kernel::QKernel, kappa, z₁, z₂, depth)
    zlow, zhigh = minmax(z₁, z₂)
    return q_cdf_kappa_derivative(kernel, kappa, zhigh, depth) -
           q_cdf_kappa_derivative(kernel, kappa, zlow, depth)
end
