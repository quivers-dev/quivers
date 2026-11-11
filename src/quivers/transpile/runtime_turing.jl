using Distributions
using Random

struct HalfStudentT{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    df::T
    scale::T
end

HalfStudentT(df::Real, scale::Real) = HalfStudentT(promote(df, scale)...)

Distributions.minimum(::HalfStudentT) = 0.0

Distributions.maximum(::HalfStudentT{T}) where {T} = T(Inf)

Distributions.insupport(::HalfStudentT, x::Real) = x >= 0

function Distributions.logpdf(d::HalfStudentT, x::Real)
    if x < 0
        return -Inf
    end
    return Distributions.logpdf(Distributions.TDist(d.df), x / d.scale) - log(d.scale) + log(2)
end

function Distributions.rand(rng::Random.AbstractRNG, d::HalfStudentT)
    return abs(rand(rng, Distributions.TDist(d.df))) * d.scale
end

struct ContinuousBernoulli{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    probs::T
end

ContinuousBernoulli(probs::Real) = ContinuousBernoulli{typeof(float(probs))}(float(probs))

Distributions.minimum(::ContinuousBernoulli) = 0.0

Distributions.maximum(::ContinuousBernoulli) = 1.0

Distributions.insupport(::ContinuousBernoulli, x::Real) = 0 <= x <= 1

function _continuous_bernoulli_log_norm(probs::Real)
    if abs(probs - 0.5) < 1e-4
        delta = probs - 0.5
        return log(2) + 2 * delta^2 + (4 / 3) * delta^4
    end
    return log(abs(2 * atanh(1 - 2 * probs))) - log(abs(1 - 2 * probs))
end

function Distributions.logpdf(d::ContinuousBernoulli, x::Real)
    if x < 0 || x > 1
        return -Inf
    end
    return x * log(d.probs) + (1 - x) * log1p(-d.probs) + _continuous_bernoulli_log_norm(d.probs)
end

function Distributions.rand(rng::Random.AbstractRNG, d::ContinuousBernoulli)
    u = rand(rng)
    p = d.probs
    if abs(p - 0.5) < 1e-4
        return u
    end
    return log1p((2 * p - 1) * u / (1 - p)) / log(p / (1 - p))
end

using LinearAlgebra

function _qvr_rbf_kernel(x::AbstractVector, length_scale::Real, jitter::Real)
    n = length(x)
    K = [exp(-0.5 * (x[i] - x[j])^2 / length_scale^2) for i in 1:n, j in 1:n]
    return K + jitter * Matrix{Float64}(I, n, n)
end

# A QVR Categorical draws class codes counting from zero, and
# Distributions.jl's `Categorical` counts from one. The zero-based
# family is the same distribution over the support `0:K-1`, so an
# observed code and a drawn class are scored and returned as QVR
# codes, and a subscript reads them one higher.
qvr_categorical(probs::AbstractVector) =
    Distributions.DiscreteNonParametric(0:(length(probs) - 1), collect(probs))

# One probability vector per observation, whether the table is a
# matrix (one row each) or already a vector of vectors.
qvr_rows(table::AbstractMatrix) = eachrow(table)
qvr_rows(table::AbstractVector) = table
