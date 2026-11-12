using Gen
using Distributions

struct TruncatedNormalDist <: Gen.Distribution{Float64} end

const truncated_normal = TruncatedNormalDist()

function Gen.random(::TruncatedNormalDist, loc::Real, scale::Real, low::Real, high::Real)
    return rand(Distributions.truncated(Distributions.Normal(loc, scale), low, high))
end

function Gen.logpdf(::TruncatedNormalDist, x::Real, loc::Real, scale::Real, low::Real, high::Real)
    return Distributions.logpdf(Distributions.truncated(Distributions.Normal(loc, scale), low, high), x)
end

function Gen.logpdf_grad(::TruncatedNormalDist, x::Real, loc::Real, scale::Real, low::Real, high::Real)
    return (nothing, nothing, nothing, nothing, nothing)
end

Gen.has_output_grad(::TruncatedNormalDist) = false

Gen.has_argument_grads(::TruncatedNormalDist) = (false, false, false, false)

(::TruncatedNormalDist)(loc, scale, low, high) = Gen.random(TruncatedNormalDist(), loc, scale, low, high)

struct LogisticDist <: Gen.Distribution{Float64} end

const logistic = LogisticDist()

function Gen.random(::LogisticDist, loc::Real, scale::Real)
    return rand(Distributions.Logistic(loc, scale))
end

function Gen.logpdf(::LogisticDist, x::Real, loc::Real, scale::Real)
    return Distributions.logpdf(Distributions.Logistic(loc, scale), x)
end

function Gen.logpdf_grad(::LogisticDist, x::Real, loc::Real, scale::Real)
    return (nothing, nothing, nothing)
end

Gen.has_output_grad(::LogisticDist) = false

Gen.has_argument_grads(::LogisticDist) = (false, false)

(::LogisticDist)(loc, scale) = Gen.random(LogisticDist(), loc, scale)

struct BetaBinomialDist <: Gen.Distribution{Int} end

const beta_binomial = BetaBinomialDist()

function Gen.random(::BetaBinomialDist, total_count::Int, concentration1::Real, concentration0::Real)
    return rand(Distributions.BetaBinomial(total_count, concentration1, concentration0))
end

function Gen.logpdf(::BetaBinomialDist, x::Int, total_count::Int, concentration1::Real, concentration0::Real)
    return Distributions.logpdf(Distributions.BetaBinomial(total_count, concentration1, concentration0), x)
end

function Gen.logpdf_grad(::BetaBinomialDist, x::Int, total_count::Int, concentration1::Real, concentration0::Real)
    return (nothing, nothing, nothing, nothing)
end

Gen.has_output_grad(::BetaBinomialDist) = false

Gen.has_argument_grads(::BetaBinomialDist) = (false, false, false)

(::BetaBinomialDist)(total_count, concentration1, concentration0) = Gen.random(BetaBinomialDist(), total_count, concentration1, concentration0)

struct HalfStudentTDist <: Gen.Distribution{Float64} end

const half_student_t = HalfStudentTDist()

function Gen.random(::HalfStudentTDist, df::Real, scale::Real)
    return abs(rand(Distributions.TDist(df))) * scale
end

function Gen.logpdf(::HalfStudentTDist, x::Real, df::Real, scale::Real)
    if x < 0
        return -Inf
    end
    return Distributions.logpdf(Distributions.TDist(df), x / scale) - log(scale) + log(2)
end

function Gen.logpdf_grad(::HalfStudentTDist, x::Real, df::Real, scale::Real)
    return (nothing, nothing, nothing)
end

Gen.has_output_grad(::HalfStudentTDist) = false

Gen.has_argument_grads(::HalfStudentTDist) = (false, false)

(::HalfStudentTDist)(df, scale) = Gen.random(HalfStudentTDist(), df, scale)

struct HalfCauchyDist <: Gen.Distribution{Float64} end

const half_cauchy = HalfCauchyDist()

function Gen.random(::HalfCauchyDist, scale::Real)
    return abs(rand(Distributions.Cauchy(0, scale)))
end

function Gen.logpdf(::HalfCauchyDist, x::Real, scale::Real)
    if x < 0
        return -Inf
    end
    return Distributions.logpdf(Distributions.Cauchy(0, scale), x) + log(2)
end

function Gen.logpdf_grad(::HalfCauchyDist, x::Real, scale::Real)
    return (nothing, nothing)
end

Gen.has_output_grad(::HalfCauchyDist) = false

Gen.has_argument_grads(::HalfCauchyDist) = (false,)

(::HalfCauchyDist)(scale) = Gen.random(HalfCauchyDist(), scale)

struct KumaraswamyDist <: Gen.Distribution{Float64} end

const kumaraswamy = KumaraswamyDist()

function Gen.random(::KumaraswamyDist, concentration1::Real, concentration0::Real)
    u = rand()
    return (1.0 - (1.0 - u)^(1.0 / concentration0))^(1.0 / concentration1)
end

function Gen.logpdf(::KumaraswamyDist, x::Real, concentration1::Real, concentration0::Real)
    if x <= 0 || x >= 1
        return -Inf
    end
    return log(concentration1) + log(concentration0) +
           (concentration1 - 1) * log(x) +
           (concentration0 - 1) * log1p(-x^concentration1)
end

function Gen.logpdf_grad(::KumaraswamyDist, x::Real, concentration1::Real, concentration0::Real)
    return (nothing, nothing, nothing)
end

Gen.has_output_grad(::KumaraswamyDist) = false

Gen.has_argument_grads(::KumaraswamyDist) = (false, false)

(::KumaraswamyDist)(concentration1, concentration0) = Gen.random(KumaraswamyDist(), concentration1, concentration0)

struct ContinuousBernoulliDist <: Gen.Distribution{Float64} end

const continuous_bernoulli = ContinuousBernoulliDist()

function _cb_log_norm(p::Real)
    if abs(p - 0.5) < 1e-4
        return log(2.0) + (4.0 / 3.0) * (p - 0.5)^2
    end
    return log(abs(2.0 * atanh(1.0 - 2.0 * p))) - log(abs(1.0 - 2.0 * p))
end

function Gen.random(::ContinuousBernoulliDist, probs::Real)
    u = rand()
    if abs(probs - 0.5) < 1e-4
        return u
    end
    return log1p(u * (2.0 * probs - 1.0) / (1.0 - probs)) / log(probs / (1.0 - probs))
end

function Gen.logpdf(::ContinuousBernoulliDist, x::Real, probs::Real)
    if x <= 0 || x >= 1
        return -Inf
    end
    return x * log(probs) + (1.0 - x) * log1p(-probs) + _cb_log_norm(probs)
end

function Gen.logpdf_grad(::ContinuousBernoulliDist, x::Real, probs::Real)
    return (nothing, nothing)
end

Gen.has_output_grad(::ContinuousBernoulliDist) = false

Gen.has_argument_grads(::ContinuousBernoulliDist) = (false,)

(::ContinuousBernoulliDist)(probs) = Gen.random(ContinuousBernoulliDist(), probs)

struct LogNormalDist <: Gen.Distribution{Float64} end

const lognormal = LogNormalDist()

function Gen.random(::LogNormalDist, loc::Real, scale::Real)
    return rand(Distributions.LogNormal(loc, scale))
end

function Gen.logpdf(::LogNormalDist, x::Real, loc::Real, scale::Real)
    return Distributions.logpdf(Distributions.LogNormal(loc, scale), x)
end

function Gen.logpdf_grad(::LogNormalDist, x::Real, loc::Real, scale::Real)
    return (nothing, nothing, nothing)
end

Gen.has_output_grad(::LogNormalDist) = false

Gen.has_argument_grads(::LogNormalDist) = (false, false)

(::LogNormalDist)(loc, scale) = Gen.random(LogNormalDist(), loc, scale)

struct WeibullDist <: Gen.Distribution{Float64} end

const weibull = WeibullDist()

function Gen.random(::WeibullDist, scale::Real, concentration::Real)
    return rand(Distributions.Weibull(concentration, scale))
end

function Gen.logpdf(::WeibullDist, x::Real, scale::Real, concentration::Real)
    return Distributions.logpdf(Distributions.Weibull(concentration, scale), x)
end

function Gen.logpdf_grad(::WeibullDist, x::Real, scale::Real, concentration::Real)
    return (nothing, nothing, nothing)
end

Gen.has_output_grad(::WeibullDist) = false

Gen.has_argument_grads(::WeibullDist) = (false, false)

(::WeibullDist)(scale, concentration) = Gen.random(WeibullDist(), scale, concentration)

using LinearAlgebra: LowerTriangular, Cholesky

struct LKJCholeskyDist <: Gen.Distribution{Matrix{Float64}} end

const lkj_cholesky = LKJCholeskyDist()

function Gen.random(::LKJCholeskyDist, dim::Int, concentration::Real)
    c = rand(Distributions.LKJCholesky(dim, concentration, :L))
    return Matrix(c.L)
end

function Gen.logpdf(::LKJCholeskyDist, x::AbstractMatrix, dim::Int, concentration::Real)
    L = LowerTriangular(x)
    c = Cholesky(L)
    return Distributions.logpdf(Distributions.LKJCholesky(dim, concentration, :L), c)
end

function Gen.logpdf_grad(::LKJCholeskyDist, x::AbstractMatrix, dim::Int, concentration::Real)
    return (nothing, nothing, nothing)
end

Gen.has_output_grad(::LKJCholeskyDist) = false

Gen.has_argument_grads(::LKJCholeskyDist) = (false, false)

(::LKJCholeskyDist)(dim, concentration) = Gen.random(LKJCholeskyDist(), dim, concentration)

struct MatrixNormalDist <: Gen.Distribution{Matrix{Float64}} end

const matrix_normal = MatrixNormalDist()

function Gen.random(::MatrixNormalDist, loc::AbstractMatrix, row_covariance::AbstractMatrix, column_covariance::AbstractMatrix)
    return rand(Distributions.MatrixNormal(Matrix{Float64}(loc), Matrix{Float64}(row_covariance), Matrix{Float64}(column_covariance)))
end

function Gen.logpdf(::MatrixNormalDist, x::AbstractMatrix, loc::AbstractMatrix, row_covariance::AbstractMatrix, column_covariance::AbstractMatrix)
    return Distributions.logpdf(Distributions.MatrixNormal(Matrix{Float64}(loc), Matrix{Float64}(row_covariance), Matrix{Float64}(column_covariance)), Matrix{Float64}(x))
end

function Gen.logpdf_grad(::MatrixNormalDist, x::AbstractMatrix, loc::AbstractMatrix, row_covariance::AbstractMatrix, column_covariance::AbstractMatrix)
    return (nothing, nothing, nothing, nothing)
end

Gen.has_output_grad(::MatrixNormalDist) = false

Gen.has_argument_grads(::MatrixNormalDist) = (false, false, false)

(::MatrixNormalDist)(loc, row_covariance, column_covariance) = Gen.random(MatrixNormalDist(), loc, row_covariance, column_covariance)

using LinearAlgebra

function _qvr_rbf_kernel(x::AbstractVector, length_scale::Real, jitter::Real)
    n = length(x)
    K = [exp(-0.5 * (x[i] - x[j])^2 / length_scale^2) for i in 1:n, j in 1:n]
    return K + jitter * Matrix{Float64}(I, n, n)
end

struct QvrFactorDist <: Gen.Distribution{Float64} end

# A scored weight as a traced choice: a one-point distribution whose
# log density at its point is the weight, so a `score` step and a
# helper's scored weight enter the trace's score the way a draw does.
const _qvr_qiec_factor = QvrFactorDist()

Gen.random(::QvrFactorDist, weight::Real) = 0.0

Gen.logpdf(::QvrFactorDist, x::Real, weight::Real) = weight

Gen.logpdf_grad(::QvrFactorDist, x::Real, weight::Real) = (nothing, 1.0)

Gen.has_output_grad(::QvrFactorDist) = false

Gen.has_argument_grads(::QvrFactorDist) = (true,)

Gen.is_discrete(::QvrFactorDist) = true

(::QvrFactorDist)(weight) = Gen.random(QvrFactorDist(), weight)

# A Bernoulli given by its log-odds `l`. Each outcome scores by the
# log-sigmoid of the signed log-odds, `log p(1) = log sigmoid(l)` and
# `log p(0) = log sigmoid(-l)`, written `min(v, 0) - log1p(exp(-|v|))`
# so it stays exact where the probability is within an ulp of zero or
# one. A draw is a Boolean, as Gen's `bernoulli` draws, and an observed
# outcome is either a Boolean or a QVR code, 0 or 1.
struct QvrBernoulliLogitDist <: Gen.Distribution{Bool} end

const qvr_bernoulli_logit = QvrBernoulliLogitDist()

_qvr_log_sigmoid(v::Real) = min(v, 0.0) - log1p(exp(-abs(v)))

function Gen.random(::QvrBernoulliLogitDist, logit::Real)
    return rand() < 1.0 / (1.0 + exp(-logit))
end

function Gen.logpdf(::QvrBernoulliLogitDist, x::Real, logit::Real)
    return x > 0.5 ? _qvr_log_sigmoid(logit) : _qvr_log_sigmoid(-logit)
end

function Gen.logpdf_grad(::QvrBernoulliLogitDist, x::Real, logit::Real)
    return (nothing, nothing)
end

Gen.has_output_grad(::QvrBernoulliLogitDist) = false
Gen.has_argument_grads(::QvrBernoulliLogitDist) = (false,)
Gen.is_discrete(::QvrBernoulliLogitDist) = true

(::QvrBernoulliLogitDist)(logit) = Gen.random(QvrBernoulliLogitDist(), logit)

# A QVR Categorical draws class codes counting from zero, and Gen's
# built-in `categorical` counts from one. This family is the same
# distribution over the support `0:K-1`, so an observed code and a
# traced class are QVR codes, and a subscript reads them one higher.
struct QvrCategoricalDist <: Gen.Distribution{Int} end

const qvr_categorical = QvrCategoricalDist()

function Gen.random(::QvrCategoricalDist, probs::AbstractVector)
    return Gen.random(Gen.categorical, probs) - 1
end

function Gen.logpdf(::QvrCategoricalDist, x::Int, probs::AbstractVector)
    return Gen.logpdf(Gen.categorical, x + 1, probs)
end

function Gen.logpdf_grad(::QvrCategoricalDist, x::Int, probs::AbstractVector)
    return (nothing, nothing)
end

Gen.has_output_grad(::QvrCategoricalDist) = false
Gen.has_argument_grads(::QvrCategoricalDist) = (false,)
Gen.is_discrete(::QvrCategoricalDist) = true

(::QvrCategoricalDist)(probs) = Gen.random(QvrCategoricalDist(), probs)
