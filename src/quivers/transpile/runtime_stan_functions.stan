functions {
  int qvr_observation_status(int x) {
    if (x < -1 || x > 2) reject("observation_status requires integer codes -1, 0, 1, or 2");
    return x;
  }
  real qvr_interval_upper(real lower_endpoint, real upper_endpoint) {
    if (is_nan(lower_endpoint) || is_nan(upper_endpoint) || is_inf(upper_endpoint) || upper_endpoint <= lower_endpoint) reject("interval_upper requires a finite upper endpoint greater than its lower endpoint");
    return upper_endpoint;
  }

  real qvr_positive_parameter(real x) {
    if (is_nan(x) || is_inf(x) || x <= 0) reject("positive_parameter requires a finite strictly positive real");
    return x;
  }


  real qvr_log_probability(real x) {
    if (is_nan(x) || is_inf(x) || x < 0 || x > 1) reject("log_probability requires a finite real in [0,1]");
    return x == 0 ? negative_infinity() : log(x);
  }
  real qvr_log1mexp(real x) {
    if (is_nan(x) || x > 0) reject("log1mexp requires a nonpositive real");
    return x == 0 ? negative_infinity() : log1m_exp(x);
  }
  real qvr_log_ndtr(real x) {
    if (is_nan(x)) reject("log_ndtr requires a real other than NaN");
    if (x == positive_infinity()) return 0;
    if (x == negative_infinity()) return negative_infinity();
    if (abs(x) > 2.5) {
      real magnitude = abs(x);
      real denominator = magnitude;
      real tail;
      for (j in 1:96) {
        denominator = magnitude + (97 - j) / denominator;
      }
      tail = -0.5 * magnitude * magnitude - 0.9189385332046727 - log(denominator);
      return x < 0 ? tail : log1m_exp(tail);
    }
    return std_normal_lcdf(x);
  }

  real kumaraswamy_lpdf(real y, real a, real b) {
    return log(a) + log(b) + (a - 1) * log(y) + (b - 1) * log1m(y ^ a);
  }
  real kumaraswamy_rng(real a, real b) {
    real u = uniform_rng(0, 1);
    return (1 - (1 - u) ^ (1 / b)) ^ (1 / a);
  }
  // Continuous Bernoulli (Loaiza-Ganem & Cunningham, NeurIPS 2019,
  // https://arxiv.org/abs/1907.06845). Stan ships no built-in, so
  // the renderer grafts these helpers when an IR sample / observe /
  // marginalize uses the ContinuousBernoulli family.
  //
  // The log normalizing constant C(p) = 2 * atanh(1 - 2p) / (1 - 2p)
  // is unstable near p = 0.5; the implementation mirrors PyTorch's
  // `ContinuousBernoulli._cont_bern_log_norm`: a fourth-order Taylor
  // expansion around 0.5 takes over when |p - 0.5| <= 0.001, and the
  // analytic form is used otherwise. The closed-form ``log_norm``
  // expression below matches the PyTorch one term-for-term.
  real continuous_bernoulli_log_norm(real p) {
    real half_gap = p - 0.5;
    if (abs(half_gap) <= 0.001) {
      real x2 = half_gap * half_gap;
      return log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x2) * x2;
    }
    real abs_log_diff = abs(log1m(p) - log(p));
    if (p <= 0.5) {
      return log(abs_log_diff) - log1m(2.0 * p);
    }
    return log(abs_log_diff) - log(2.0 * p - 1.0);
  }
  real continuous_bernoulli_lpdf(real y, real p) {
    return continuous_bernoulli_log_norm(p)
      + y * log(p) + (1 - y) * log1m(p);
  }
  // Sample via the inverse-CDF. The Taylor branch around p = 0.5
  // degenerates to ``u`` itself (the uniform on (0, 1) is the
  // limiting distribution at p = 0.5); the analytic branch matches
  // PyTorch's `icdf`.
  real continuous_bernoulli_rng(real p) {
    real u = uniform_rng(0, 1);
    if (abs(p - 0.5) <= 0.001) {
      return u;
    }
    return (log1p(-p + u * (2.0 * p - 1.0)) - log1m(p))
      / (log(p) - log1m(p));
  }
  // Logit-normal density on (0, 1): Y = inv_logit(X) with
  // X ~ Normal(mu, sigma). Stan ships no built-in logit_normal, so
  // the renderer grafts these helpers when an IR sample / observe /
  // marginalize uses the LogitNormal family. The change-of-variables
  // Jacobian d/dy logit(y) = 1 / (y (1 - y)) contributes the
  // -log(y) - log1m(y) term.
  real logit_normal_lpdf(real y, real mu, real sigma) {
    return normal_lpdf(logit(y) | mu, sigma) - log(y) - log1m(y);
  }
  real logit_normal_rng(real mu, real sigma) {
    return inv_logit(normal_rng(mu, sigma));
  }
  // Log-Gamma density on the real line: Y = log(G) with
  // G ~ Gamma(a, r), so log p(y) = a (log r + y) - r exp(y) - lgamma(a),
  // exact for every finite y. The sampler stays in log space through
  // the shape augmentation G = G' U^(1 / a) with G' ~ Gamma(a + 1, 1)
  // and U ~ Uniform(0, 1) (Marsaglia and Tsang 2000), so a small shape
  // never forms a Gamma draw that underflows.
  real log_gamma_lpdf(real y, real a, real r) {
    return a * (log(r) + y) - exp(y + log(r)) - lgamma(a);
  }
  real log_gamma_rng(real a, real r) {
    return log(gamma_rng(a + 1, 1)) + log(uniform_rng(0, 1)) / a - log(r);
  }
  // Logit-Beta density on the real line: X = logit(P) with
  // P ~ Beta(a, b), so log p(x) = a log_inv_logit(x)
  // + b log1m_inv_logit(x) - lbeta(a, b), exact for every finite x
  // where scoring inv_logit(x) under a Beta saturates. A draw is the
  // difference of two unit-rate log-Gamma draws.
  real logit_beta_lpdf(real x, real a, real b) {
    return a * log_inv_logit(x) + b * log1m_inv_logit(x) - lbeta(a, b);
  }
  real logit_beta_rng(real a, real b) {
    return log_gamma_rng(a, 1) - log_gamma_rng(b, 1);
  }
  // Matrix-normal density for X in R^{p x n} with mean M, row
  // covariance U (p x p, SPD), and column covariance V (n x n, SPD):
  //
  //   log p(X | M, U, V) = -0.5 * (
  //       n * p * log(2 pi)
  //     + n * log|U|
  //     + p * log|V|
  //     + tr(V^{-1} (X - M)' U^{-1} (X - M))
  //   )
  //
  // Equivalent to vec(X) ~ MultiNormal(vec(M), V (x) U). Stan ships
  // no built-in matrix-variate normal, so the renderer grafts this
  // helper when an IR sample / observe uses the MatrixNormal family.
  real matrix_normal_lpdf(matrix X, matrix M, matrix U, matrix V) {
    int p = rows(X);
    int n = cols(X);
    matrix[p, n] D = X - M;
    matrix[p, n] U_inv_D = mdivide_left_spd(U, D);
    matrix[n, n] quad = D' * U_inv_D;
    real trace_term = trace(mdivide_left_spd(V, quad));
    real log_det_U = log_determinant_spd(U);
    real log_det_V = log_determinant_spd(V);
    return -0.5 * (n * p * log(2 * pi())
                   + n * log_det_U
                   + p * log_det_V
                   + trace_term);
  }
  // Sample via the Cholesky-factor reconstruction
  //   X = M + L_U Z L_V'
  // where Z is p x n with iid standard-normal entries and L_U L_U' = U,
  // L_V L_V' = V.
  matrix matrix_normal_rng(matrix M, matrix U, matrix V) {
    int p = rows(M);
    int n = cols(M);
    matrix[p, p] L_U = cholesky_decompose(U);
    matrix[n, n] L_V = cholesky_decompose(V);
    matrix[p, n] Z;
    for (i in 1:p) {
      for (j in 1:n) {
        Z[i, j] = std_normal_rng();
      }
    }
    return M + L_U * Z * L_V';
  }
}
