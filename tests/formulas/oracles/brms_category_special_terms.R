# Generates brms_category_special_terms.json, pinned by
# tests/formulas/test_category_special_terms.py: brms variable names, prior
# rows and Stan code of mo(), gp(), mm() and gr(by =) terms in the per-category
# predictors of a categorical model, with log_lik / posterior_epred at brms's
# own posterior draws.
# Run from this directory: Rscript brms_category_special_terms.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
set.seed(23)
n <- 24
d <- data.frame(
  y = factor(sample(c("A", "B", "C"), n, TRUE), levels = c("A", "B", "C")),
  x = round(rnorm(n), 3),
  z = round(rnorm(n), 3),
  k = sample(1:4, n, TRUE),
  g1 = sample(c("p", "q", "r"), n, TRUE),
  g2 = sample(c("p", "q", "r"), n, TRUE),
  stringsAsFactors = FALSE
)
d$gb <- ifelse(d$g1 == "r", "v", "u")
drop <- c(".chain", ".iteration", ".draw", "lp__", "lprior")
run <- function(form) {
  code <- strsplit(make_stancode(form, d, family = categorical()), "\n")[[1]]
  prior <- get_prior(form, d, family = categorical())
  fit <- suppressMessages(brm(form, data = d, family = categorical(), algorithm = "meanfield",
    output_samples = 3, iter = 2000, refresh = 0, seed = 1, init = 0))
  draws <- as.data.frame(as_draws_df(fit))[1:3, ]
  list(
    stan = trimws(grep("mu[BC]|simo|sdgp|lscale|zgp|categorical|sd_[0-9]|z_[0-9]|r_[0-9]", code, value = TRUE)),
    prior = prior[, c("prior", "class", "coef", "group", "dpar")],
    variables = setdiff(variables(fit), c("lprior", "lp__")),
    draws = draws[, setdiff(names(draws), drop)],
    log_lik = log_lik(fit, draw_ids = 1:3),
    epred = posterior_epred(fit, draw_ids = 1:3)
  )
}
result <- list(
  brms = as.character(packageVersion("brms")),
  data = d,
  separate = run(bf(y ~ x, muB ~ mo(k) + (1 | mm(g1, g2)),
    muC ~ gp(z, k = 5, c = 5 / 4) + (1 | gr(g1, by = gb)))),
  copied = run(bf(y ~ mo(k) + gp(z)))
)
writeLines(toJSON(result, digits = 15, auto_unbox = TRUE, dataframe = "columns"),
  "brms_category_special_terms.json")
