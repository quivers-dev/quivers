# Generates brms_mixture.json, pinned by tests/formulas/test_mixture.py.
# brms oracle for finite mixtures: Stan code of the identification and mixing
# proportion constructions, default prior classes, and log_lik / pp_mixture /
# posterior_epred values at fixed posterior draws.
# Run from this directory: Rscript brms_mixture.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
set.seed(7)
n <- 12
d <- data.frame(
  y = c(rnorm(8, 0, 1), rnorm(4, 5, 1)),
  x = round(rnorm(n), 3),
  z = rep(c(0, 1), n / 2)
)
d$y <- round(d$y, 3)
d$count <- rpois(n, c(rep(2, 8), rep(9, 4)))
lines <- function(code, pattern) grep(pattern, strsplit(code, "\n")[[1]], value = TRUE)
# Meanfield output draws suffice: they are only evaluation points.
quick <- function(formula, family, data) {
  for (seed in 1:20) {
    fit <- tryCatch(suppressMessages(brm(formula, data = data, family = family,
      algorithm = "meanfield", output_samples = 3, iter = 2000, refresh = 0,
      seed = seed, init = 0)), error = function(e) NULL)
    if (!is.null(fit)) return(fit)
  }
  stop("no variational fit")
}
mix_gg <- mixture(gaussian, gaussian)
mix_gs <- mixture(gaussian, student)
mix_pp <- mixture(poisson, poisson)
codes <- list(
  ordered = make_stancode(bf(y ~ x), d, family = mix_gg),
  predicted = make_stancode(bf(y ~ x, theta2 ~ z), d, family = mix_gg),
  fixed = make_stancode(bf(y ~ x, theta1 = 1, theta2 = 3), d, family = mix_gg),
  equal = make_stancode(bf(y ~ x, sigma2 = "sigma1"), d, family = mix_gg),
  hetero = make_stancode(bf(y ~ x), d, family = mix_gs)
)
priors <- function(formula, family) {
  p <- get_prior(formula, d, family = family)
  unique(paste(p$class, p$dpar, sep = "|"))
}
fit_gs <- quick(bf(y ~ x, theta2 ~ z), mix_gs, d)
fit_pp <- quick(bf(count ~ x), mix_pp, d)
draws_of <- function(fit) {
  out <- as.data.frame(as_draws_df(fit))[1:3, ]
  out[, setdiff(names(out), c(".chain", ".iteration", ".draw", "lp__", "lprior"))]
}
result <- list(
  ordered_stan = lines(codes$ordered, "ordered|Intercept_mu|simplex|dirichlet|log_sum_exp|ps\\["),
  predicted_stan = lines(codes$predicted, "theta|log_sum_exp"),
  fixed_stan = lines(codes$fixed, "theta"),
  equal_stan = lines(codes$equal, "sigma2"),
  hetero_stan = lines(codes$hetero, "ordered|ps\\[|nu2"),
  order_default_same = mix_gg$order,
  order_default_hetero = mix_gs$order,
  prior_classes_ordered = priors(bf(y ~ x), mix_gg),
  prior_classes_predicted = priors(bf(y ~ x, theta2 ~ z), mix_gg),
  prior_classes_hetero = priors(bf(y ~ x), mix_gs),
  gs_draws = draws_of(fit_gs),
  gs_log_lik = log_lik(fit_gs, draw_ids = 1:3),
  gs_pp_mixture = pp_mixture(fit_gs, summary = FALSE, draw_ids = 1:3),
  gs_epred = posterior_epred(fit_gs, draw_ids = 1:3),
  pp_draws = draws_of(fit_pp),
  pp_log_lik = log_lik(fit_pp, draw_ids = 1:3),
  pp_epred = posterior_epred(fit_pp, draw_ids = 1:3),
  data = d
)
writeLines(toJSON(result, digits = 15, auto_unbox = TRUE), "brms_mixture.json")
