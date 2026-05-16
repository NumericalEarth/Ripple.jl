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

@inline function _kernel_safe_mu(mu, small_mu)
    sign = ifelse(mu < zero(mu), -one(mu), one(mu))
    return ifelse(abs(mu) < small_mu, sign * small_mu, mu)
end

@inline function q_value_kernel(kernel::QKernel, kappa, z, depth)
    mu = kappa * depth
    μ = _kernel_safe_mu(mu, kernel.small_mu)
    s = (z + depth) / depth
    b = 2μ
    a = b * s
    ratio = (exp(a - b) + exp(-a - b)) / (one(b) - exp(-2b))
    asymptotic = inv(depth)
    finite_depth = (2μ / depth) * ratio
    return ifelse(abs(mu) < kernel.small_mu, asymptotic, finite_depth)
end

@inline function q_cdf_kernel(kernel::QKernel, kappa, z, depth)
    mu = kappa * depth
    μ = _kernel_safe_mu(mu, kernel.small_mu)
    s = clamp((z + depth) / depth, zero(depth), one(depth))
    a = 2 * μ * s
    b = 2μ
    finite_depth = (exp(a - b) - exp(-a - b)) / (one(b) - exp(-2b))
    return ifelse(abs(mu) < kernel.small_mu, s, finite_depth)
end

@inline function q_cell_integral_kernel(kernel::QKernel, kappa, z₁, z₂, depth)
    zlow = min(z₁, z₂)
    zhigh = max(z₁, z₂)
    return q_cdf_kernel(kernel, kappa, zhigh, depth) -
           q_cdf_kernel(kernel, kappa, zlow, depth)
end

@inline function q_cdf_kappa_derivative_kernel(kernel::QKernel, kappa, z, depth)
    mu = kappa * depth
    μ = _kernel_safe_mu(mu, kernel.small_mu)
    s = clamp((z + depth) / depth, zero(depth), one(depth))
    small_mu_value = depth * (4 * mu * s * (s^2 - one(s)) / 3)

    a = exp(-2 * μ * (one(μ) - s))
    b = exp(-2 * μ * (one(μ) + s))
    e = exp(-4 * μ)
    numerator = a - b
    denominator = one(e) - e
    dnumerator_dmu = -2 * (one(μ) - s) * a + 2 * (one(μ) + s) * b
    ddenominator_dmu = 4e
    finite_depth = depth * (dnumerator_dmu * denominator - numerator * ddenominator_dmu) / denominator^2
    return ifelse(abs(mu) < kernel.small_mu, small_mu_value, finite_depth)
end

@inline function q_cell_integral_kappa_derivative_kernel(kernel::QKernel, kappa, z₁, z₂, depth)
    zlow = min(z₁, z₂)
    zhigh = max(z₁, z₂)
    return q_cdf_kappa_derivative_kernel(kernel, kappa, zhigh, depth) -
           q_cdf_kappa_derivative_kernel(kernel, kappa, zlow, depth)
end
