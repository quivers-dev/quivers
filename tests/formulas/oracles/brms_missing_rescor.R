# Generates brms_missing_rescor.json, pinned by
# tests/formulas/test_missing_rescor.py: brms's Stan statements for mi()
# responses under estimated residual correlations, and log_lik /
# posterior_epred / posterior_linpred values at fixed posterior draws.
# Run from this directory: Rscript brms_missing_rescor.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
options(digits = 17)
d <- data.frame(
  y = c(0.125, 0.375, -0.25, NA, 0.5, 0.0, 0.75, -0.125, 0.25, 0.625, -0.375, 0.25),
  x = c(1.0, NA, 0.5, 0.75, 2.0, 1.5, NA, 0.25, 0.875, 1.25, 0.375, 1.75),
  z = c(0.25, 0.125, -0.25, 0.375, 0.0, 0.25, -0.25, 0.625, 0.5, -0.375, 0.125, -0.125),
  s = c(0.25, 0.125, 0.5, 0.25, 0.375, 0.25, 0.125, 0.5, 0.25, 0.375, 0.125, 0.25),
  w = c(1, 2, 0.5, 1, 1.5, 1, 2, 0.5, 1, 1, 2, 1)
)
statements <- function(formula, family = gaussian()) {
  tryCatch({
    lines <- trimws(strsplit(make_stancode(formula, data = d, family = family), "\n")[[1]])
    keep <- grepl("^target \\+= (weights|multi|normal_lpdf)|Yl|Ymi|LSigma =|Sigma =|mu_y\\[n\\] \\+=|real<lower=1> nu", lines)
    list(status = "ok", lines = lines[keep])
  }, error = function(e) list(status = "refused", message = conditionMessage(e)))
}
plain <- bf(y | mi() ~ mi(x) + z) + bf(x | mi() ~ z) + set_rescor(TRUE)
noisy <- bf(y ~ mi(x) + z) + bf(x | mi(s) ~ z) + set_rescor(TRUE)
cases <- list(
  plain = statements(plain),
  noisy = statements(noisy),
  student = statements(plain, student()),
  weighted = statements(bf(y | mi() + weights(w) ~ mi(x) + z) +
                          bf(x | mi() + weights(w) ~ z) + set_rescor(TRUE)),
  cyclic = statements(bf(y | mi() ~ mi(x) + z) + bf(x | mi() ~ mi(y) + z) + set_rescor(TRUE)),
  censored = statements(bf(y | mi() ~ mi(x) + z) + bf(x | mi() + cens(w) ~ z) + set_rescor(TRUE))
)
sdata <- make_standata(plain, data = d)
standata <- list(Jmi_y = as.integer(sdata$Jmi_y), Jmi_x = as.integer(sdata$Jmi_x),
                 N = sdata$N, nresp = sdata$nresp)

# Meanfield output draws suffice: they are only evaluation points. The latent
# true values are saved, so brms predicts from their draws and not from fresh
# draws of the measurement model.
points <- function(formula, family = gaussian(), newdata = NULL) {
  fit <- NULL
  for (seed in 1:20) {
    fit <- tryCatch(suppressMessages(brm(formula, data = d, family = family,
      algorithm = "meanfield", output_samples = 3, iter = 2000,
      save_pars = save_pars(latent = TRUE),
      refresh = 0, seed = seed, init = 0)), error = function(e) NULL)
    if (!is.null(fit)) break
  }
  draws <- as.data.frame(as_draws_df(fit))[1:3, ]
  draws <- draws[, !grepl("^(lp__|lprior|\\.chain|\\.iteration|\\.draw)$", names(draws))]
  # A brms prediction that fails is recorded by its message.
  attempt <- function(expr) tryCatch(expr, error = function(e) list(error = conditionMessage(e)))
  numbers <- function(value) if (is.list(value)) value else {
    text <- array(as.character(signif(value, 17)), dim(value))
    text[is.na(value)] <- "NA"
    text
  }
  out <- list(draws = draws,
              log_lik = numbers(attempt(log_lik(fit, draw_ids = 1:3))),
              log_lik_y = numbers(attempt(log_lik(fit, resp = "y", draw_ids = 1:3))),
              epred = numbers(attempt(posterior_epred(fit, draw_ids = 1:3))),
              linpred = numbers(attempt(posterior_linpred(fit, draw_ids = 1:3))))
  if (!is.null(newdata)) {
    out$new_log_lik <- numbers(attempt(log_lik(fit, newdata = newdata, draw_ids = 1:3)))
    out$new_epred <- numbers(attempt(posterior_epred(fit, newdata = newdata, draw_ids = 1:3)))
  }
  out
}
fresh <- d[c(1, 3, 5, 6, 9), ]
values <- list(
  plain = points(plain, newdata = fresh),
  noisy = points(noisy, newdata = fresh),
  student = points(plain, student())
)
writeLines(toJSON(list(data = d, fresh = fresh, cases = cases, standata = standata,
                       values = values),
                  auto_unbox = TRUE, digits = NA, dataframe = "columns", na = "null"),
           "brms_missing_rescor.json")
