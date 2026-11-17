# Generates brms_ordinal_mixture.json, pinned by
# tests/formulas/test_ordinal_mixture.py: brms Stan threshold declarations,
# prior rows, variable names, and log_lik / posterior_epred / pp_mixture at
# brms's own posterior draws for mixtures of ordinal families, with separate
# thresholds (order = "none") and with thresholds fixed across components
# (order = "mu").
# Run from this directory: Rscript brms_ordinal_mixture.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
set.seed(31)
n <- 30
d <- data.frame(
  y = sample(1:4, n, TRUE),
  x = round(rnorm(n), 3),
  z = round(rnorm(n), 3)
)
drop <- c(".chain", ".iteration", ".draw", "lp__", "lprior")
run <- function(form, family) {
  code <- strsplit(make_stancode(form, d, family = family), "\n")[[1]]
  prior <- get_prior(form, d, family = family)
  fit <- suppressMessages(brm(form, data = d, family = family, algorithm = "meanfield",
    output_samples = 3, iter = 2000, refresh = 0, seed = 1, init = 0))
  draws <- as.data.frame(as_draws_df(fit))[1:3, ]
  list(
    order = family$order,
    stan = trimws(grep("Intercept|ps\\[", code, value = TRUE)),
    prior = prior[, c("prior", "class", "coef", "dpar")],
    variables = setdiff(variables(fit), c("lprior", "lp__")),
    draws = draws[, setdiff(names(draws), drop)],
    log_lik = log_lik(fit, draw_ids = 1:3),
    epred = posterior_epred(fit, draw_ids = 1:3),
    pp_mixture = pp_mixture(fit, draw_ids = 1:3, summary = FALSE)
  )
}
result <- list(
  brms = as.character(packageVersion("brms")),
  data = d,
  separate = run(bf(y ~ x, mu2 ~ z), mixture(cumulative, sratio)),
  fixed = run(bf(y ~ x, mu2 ~ z), mixture(sratio, cumulative, order = "mu")),
  ratio = run(bf(y ~ x), mixture(acat, cratio)),
  mixed_error = tryCatch({mixture(cumulative, poisson); ""}, error = function(e) conditionMessage(e)),
  equidistant_error = tryCatch(
    {make_stancode(bf(y ~ x), d, family = mixture(cumulative(threshold = "equidistant"), cumulative, order = "mu")); ""},
    error = function(e) conditionMessage(e))
)
writeLines(toJSON(result, digits = 15, auto_unbox = TRUE, dataframe = "columns"),
  "brms_ordinal_mixture.json")
