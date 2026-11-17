# Generates brms_category_formulas.json, pinned by
# tests/formulas/test_category_formulas.py: brms names, prior classes and
# Stan code of category-specific formulas (muB ~ x, muC ~ z), and log_lik /
# posterior_epred of a categorical model at brms's own posterior draws.
# Run from this directory: Rscript brms_category_formulas.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
set.seed(11)
n <- 15
d <- data.frame(
  y = factor(sample(c("A", "B", "C"), n, TRUE), levels = c("A", "B", "C")),
  x = round(rnorm(n), 3),
  z = round(rnorm(n), 3)
)
form <- bf(y ~ x, muC ~ z)
code <- make_stancode(form, d, family = categorical())
prior <- get_prior(form, d, family = categorical())
fit <- suppressMessages(brm(form, data = d, family = categorical(), algorithm = "meanfield", output_samples = 3,
  iter = 2000, refresh = 0, seed = 1, init = 0))
draws <- as.data.frame(as_draws_df(fit))[1:3, ]
draws <- draws[, setdiff(names(draws), c(".chain", ".iteration", ".draw", "lp__", "lprior"))]
refcat_code <- make_stancode(bf(y ~ x, muA ~ z), d, family = categorical(refcat = "C"))
result <- list(
  stan = grep("mu[ABC]|categorical", strsplit(code, "\n")[[1]], value = TRUE),
  refcat_stan = grep("mu[ABC]|categorical", strsplit(refcat_code, "\n")[[1]], value = TRUE),
  prior_classes = unique(paste(prior$class, prior$coef, prior$dpar, sep = "|")),
  variables = setdiff(variables(fit), c("lprior", "lp__")),
  draws = draws,
  log_lik = log_lik(fit, draw_ids = 1:3),
  epred = posterior_epred(fit, draw_ids = 1:3),
  data = list(y = as.integer(d$y) - 1L, x = d$x, z = d$z)
)
writeLines(toJSON(result, digits = 15, auto_unbox = TRUE), "brms_category_formulas.json")
