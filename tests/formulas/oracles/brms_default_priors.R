# Generates brms_default_priors.json, pinned by tests/formulas/test_brms_priors.py.
# brms oracle for the default prior table: get_prior() (default_prior() in
# brms 2.21 and later, with the same table) over a spread of
# families, links, distributional, grouped, smooth, ordinal and joint models,
# the intercept rows under center = FALSE and 0 + Intercept, the flat
# defaults of conditional and covariance ARMA coefficients, mo(), me()
# (meanme, sdme) and mi() terms, and validate_prior() for user priors with
# bounds.
# Run from this directory: Rscript brms_default_priors.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(
  y = c(0.31, -1.24, 0.85, 1.93, -0.42, 0.13, 1.41, -0.73, 0.62, 2.21,
        -1.52, 0.94, 0.27, -0.66, 1.18, 0.05, -0.91, 1.67, 0.44, -0.18),
  y2 = c(1.2, 0.4, -0.3, 2.2, 0.9, 1.1, -0.8, 0.2, 1.7, 0.6,
         -0.1, 1.4, 0.8, 0.3, 2.6, -0.5, 1.0, 0.7, 1.9, 0.0),
  pos = c(1.3, 0.4, 2.7, 5.1, 0.9, 1.6, 3.3, 0.7, 2.2, 6.4,
          0.3, 2.9, 1.1, 0.6, 3.8, 1.4, 0.5, 4.6, 1.9, 1.2),
  count = c(2, 0, 5, 9, 1, 3, 6, 0, 4, 12, 0, 5, 2, 1, 7, 2, 0, 8, 3, 2),
  n = rep(12, 20),
  b = c(1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0),
  prop = c(0.31, 0.12, 0.55, 0.83, 0.22, 0.41, 0.66, 0.18, 0.49, 0.91,
           0.07, 0.58, 0.36, 0.25, 0.72, 0.33, 0.15, 0.79, 0.44, 0.29),
  x = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8,
        0.4, -0.3, 1.1, -0.6, 0.2, 0.8, -1.2, 1.5, -0.1, 0.7),
  z = c(-0.2, 0.4, 1.1, -0.9, 0.3, -0.5, 0.8, 1.4, -1.3, 0.2,
        0.6, -0.8, -0.1, 1.2, -0.6, 0.5, 0.9, -1.1, 0.0, 0.7),
  g = rep(c("a", "b", "c", "d"), 5)
)
d$wide <- 10 * d$y + 40
d$big <- 25 * d$count + c(3, 7, 0, 11, 5, 2, 9, 4, 1, 6, 8, 0, 3, 5, 2, 7, 1, 4, 6, 9)
d$off <- log(d$n) + 0.1 * d$x
d$t <- seq_len(20)
d$sdx <- rep(0.2, 20)
d$xm <- d$x
d$xm[c(3, 7)] <- NA
d$xo <- factor(rep(c(1, 2, 3), length.out = 20), ordered = TRUE)
d$r <- factor(c(1, 0, 2, 3, 1, 1, 2, 0, 2, 3, 0, 2, 1, 0, 2, 1, 0, 3, 1, 1),
              levels = 0:3, ordered = TRUE)
table_of <- function(p) {
  p <- as.data.frame(p)
  data.frame(prior = p$prior, class = p$class, coef = p$coef, group = p$group,
             resp = p$resp, dpar = p$dpar, nlpar = p$nlpar,
             lb = p$lb, ub = p$ub,
             tag = if ("tag" %in% names(p)) p$tag else rep("", nrow(p)),
             source = p$source)
}
case <- function(name, formula, family, ...) {
  list(name = name, formula = deparse(formula[[1]]),
       family = family$family, link = family$link,
       table = table_of(get_prior(formula, data = d, family = family, ...)))
}
cases <- list(
  case("gaussian", bf(y ~ x + z), gaussian()),
  case("gaussian_groups", bf(y ~ x + (1 + x | g)), gaussian()),
  case("gaussian_uncentered", bf(y ~ x, center = FALSE), gaussian()),
  case("gaussian_intercept_term", bf(y ~ 0 + Intercept + x), gaussian()),
  case("gaussian_sigma", bf(y ~ x, sigma ~ z), gaussian()),
  case("student", bf(y ~ x), student()),
  case("poisson_groups", bf(count ~ x + (1 | g)), poisson()),
  case("negbinomial", bf(count ~ x), negbinomial()),
  case("binomial", bf(count | trials(n) ~ x), binomial()),
  case("bernoulli", bf(b ~ x), bernoulli()),
  case("gamma_log", bf(pos ~ x), Gamma(link = "log")),
  case("lognormal", bf(pos ~ x), lognormal()),
  case("weibull", bf(pos ~ x), weibull()),
  case("beta", bf(prop ~ x), Beta()),
  case("zero_inflated_poisson", bf(count ~ x), zero_inflated_poisson()),
  case("hurdle_gamma", bf(pos ~ x), hurdle_gamma()),
  case("cumulative", bf(r ~ x), cumulative()),
  case("smooth", bf(y ~ s(x, k = 5)), gaussian()),
  case("gaussian_wide_groups", bf(wide ~ x + (1 | g)), gaussian()),
  case("poisson_big", bf(big ~ x), poisson()),
  case("poisson_offset", bf(count ~ x + offset(off)), poisson()),
  case("lognormal_wide", bf(big ~ x), lognormal()),
  case("gamma_inverse", bf(pos ~ x), Gamma(link = "inverse")),
  case("gaussian_wide_sigma", bf(wide ~ x, sigma ~ z), gaussian()),
  case("arma_conditional", bf(y ~ x + arma(time = t, p = 1, q = 1)), gaussian()),
  case("ar_covariance", bf(y ~ x + ar(time = t, p = 1, cov = TRUE)), gaussian()),
  case("monotonic", bf(y ~ mo(xo)), gaussian()),
  case("measurement_error", bf(y ~ me(x, sdx)), gaussian()),
  case("missing_predictor",
       bf(y ~ mi(xm)) + bf(xm | mi() ~ z) + set_rescor(FALSE), gaussian())
)
mv <- get_prior(bf(y ~ x) + bf(y2 ~ x) + set_rescor(TRUE), data = d)
user <- prior(normal(0, 2), class = b, lb = 0) +
  prior(normal(1, 0.5), class = b, coef = x) +
  prior(student_t(3, 0, 1), class = sigma)
validated <- validate_prior(user, y ~ x + z, data = d)
coef_bound <- tryCatch(
  validate_prior(prior(normal(1, 0.5), class = b, coef = x, lb = 0),
                 y ~ x + z, data = d),
  error = function(e) conditionMessage(e))
round_cases <- list(
  median_log = round(median(log(d$pos)), 1),
  mad_log = round(mad(log(d$pos)), 1),
  mad_y = round(mad(d$y), 1),
  median_y = round(median(d$y), 1)
)
# R's round() at one digit on halves and near-halves, which brms applies to
# the medians and MADs of its default priors.
grid <- c(seq(-2.05, 2.05, by = 0.1), (0:40) / 40 - 0.5, 0.45, 0.15, 2.675)
# Written as 17 significant digits so the doubles round-trip exactly.
rounding_grid <- list(values = sprintf("%.17g", grid),
                      rounded = sprintf("%.17g", round(grid, 1)))
writeLines(
  toJSON(list(brms = as.character(packageVersion("brms")), data = d,
              cases = cases, multivariate = table_of(mv),
              validated = table_of(validated), coef_bound = coef_bound,
              rounding = round_cases, rounding_grid = rounding_grid),
         auto_unbox = TRUE, pretty = TRUE, digits = NA),
  "brms_default_priors.json"
)
