# Generates brms_autocor_combinations.json, pinned by
# tests/formulas/test_autocor_combinations.py: brms's messages and Stan
# likelihood statements for autocorrelation terms combined with se(), cens(),
# trunc(), weights(), and each other; the standata row order of an ARMA term
# beside a SAR matrix; and log_lik / posterior_epred / posterior_linpred values
# at fixed posterior draws of the admitted combinations.
# Run from this directory: Rscript brms_autocor_combinations.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
options(digits = 17)
# Rows sorted by group, then time, so brms's sorted rows are the rows as given.
d <- data.frame(
  y = c(0.31, -0.42, 0.85, 1.42, 0.18, -0.77, 0.52, 1.9, -0.25, 0.64, 1.11, -0.9),
  x = c(1.0, 0.5, -0.2, 0.7, 1.3, -0.9, 0.1, 0.4, -1.1, 0.3, 0.8, -0.6),
  t = rep(1:4, 3), g = rep(c("a", "b", "c"), each = 4),
  s = c(0.2, 0.35, 0.15, 0.4, 0.25, 0.3, 0.45, 0.2, 0.1, 0.5, 0.3, 0.25),
  c = c("none", "right", "none", "none", "left", "none", "interval", "none",
        "none", "right", "none", "none"),
  w = c(1, 2, 0.5, 1, 1.5, 1, 2, 0.5, 1, 1, 2, 1)
)
d$y2 <- d$y + 0.6
n <- nrow(d)
W <- matrix(0, n, n)
for (i in 1:(n - 1)) { W[i, i + 1] <- 1; W[i + 1, i] <- 1 }
W <- W / rowSums(W)
V <- 0.4 * diag(n) + 0.6 * exp(-abs(outer(1:n, 1:n, "-")) / 3)
data2 <- list(W = W, V = V)

outcome <- function(formula, family = gaussian(), parse = TRUE) {
  tryCatch({
    code <- make_stancode(formula, data = d, family = family, data2 = data2,
                          parse = parse)
    lines <- strsplit(code, "\n")[[1]]
    keep <- grepl("target \\+= (weights|normal|student)|^\\s+(normal|student)_t?.*l(c|cc)df|mu\\[n\\] \\+=|err\\[n\\] =|if \\(cens|else if \\(cens",
                  lines) & !grepl("lprior", lines)
    list(status = "ok", likelihood = trimws(lines[keep]))
  }, error = function(e) {
    message <- conditionMessage(e)
    compiled <- grepl("Semantic error|Ill-typed|undeclared identifier", message)
    list(status = if (compiled) "stan_does_not_compile" else "refused",
         message = if (compiled) regmatches(message, regexpr("(Ill-typed arguments supplied to function '[a-z_]+'|undeclared identifier '[a-z_]+')", message))
                   else message)
  })
}
cases <- list(
  se_ar_cov_default_sigma = outcome(y | se(s) ~ x + ar(t, g, cov = TRUE)),
  se_ar_cov = outcome(y | se(s, sigma = TRUE) ~ x + ar(t, g, cov = TRUE)),
  se_arma_cov_student = outcome(y | se(s, sigma = TRUE) ~ x + arma(t, g, cov = TRUE), student()),
  se_cosy = outcome(y | se(s, sigma = TRUE) ~ x + cosy(t, g)),
  se_unstr = outcome(y | se(s, sigma = TRUE) ~ x + unstr(t, g)),
  se_cosy_het = outcome(bf(y | se(s, sigma = TRUE) ~ x + cosy(t, g), sigma ~ x)),
  se_ar = outcome(y | se(s, sigma = TRUE) ~ x + ar(t, g)),
  se_fcor = outcome(y | se(s, sigma = TRUE) ~ x + fcor(V)),
  se_sar = outcome(y | se(s, sigma = TRUE) ~ x + sar(W)),
  se_sar_error_student = outcome(y | se(s, sigma = TRUE) ~ x + sar(W, type = "error"), student()),
  cens_ar = outcome(y | cens(c, y2) ~ x + ar(t, g)),
  trunc_arma = outcome(y | trunc(lb = -3, ub = 4) ~ x + arma(t, g, 2, 1)),
  cens_trunc_ma_student = outcome(y | cens(c, y2) + trunc(lb = -3, ub = 4) ~ x + ma(t, g), student()),
  weights_ar = outcome(y | weights(w) ~ x + ar(t, g)),
  cens_ar_cov = outcome(y | cens(c, y2) ~ x + ar(t, g, cov = TRUE)),
  trunc_cosy = outcome(y | trunc(lb = -3) ~ x + cosy(t, g)),
  weights_unstr = outcome(y | weights(w) ~ x + unstr(t, g)),
  cens_fcor = outcome(y | cens(c, y2) ~ x + fcor(V)),
  cens_sar = outcome(y | cens(c, y2) ~ x + sar(W)),
  weights_sar = outcome(y | weights(w) ~ x + sar(W)),
  trunc_sar = outcome(y | trunc(lb = -3) ~ x + sar(W)),
  ar_sar_lag = outcome(y ~ x + ar(t, g) + sar(W)),
  arma_sar_error = outcome(y ~ x + arma(t, g) + sar(W, type = "error")),
  ar_fcor = outcome(y ~ x + ar(t, g, p = 2) + fcor(V)),
  ma_fcor_student = outcome(y ~ x + ma(t, g) + fcor(V), student()),
  ar_fcor_het = outcome(bf(y ~ x + ar(t, g) + fcor(V), sigma ~ x)),
  ar_cov_sar = outcome(y ~ x + ar(t, g, cov = TRUE) + sar(W)),
  cosy_fcor = outcome(y ~ x + cosy(t, g) + fcor(V)),
  sar_fcor = outcome(y ~ x + sar(W) + fcor(V)),
  ar_sar_poisson = outcome(round(exp(y)) ~ x + ar(t, g) + sar(W), poisson()),
  ar_fcor_poisson = outcome(round(exp(y)) ~ x + ar(t, g) + fcor(V), poisson()),
  ar_sar_sigma = outcome(bf(y ~ x + ar(t, g) + sar(W), sigma ~ x)),
  cens_ar_sar = outcome(y | cens(c, y2) ~ x + ar(t, g) + sar(W)),
  cens_ar_fcor = outcome(y | cens(c, y2) ~ x + ar(t, g) + fcor(V))
)

# brms sorts rows by group and time and leaves the SAR matrix as given.
shuffled <- d[c(7, 2, 11, 4, 9, 1, 12, 5, 3, 10, 6, 8), ]
sd_order <- make_standata(y ~ x + ar(t, g) + sar(W), data = shuffled, data2 = data2)
row_order <- list(given_y = shuffled$y, standata_y = as.numeric(sd_order$Y),
                  matrix_unpermuted = isTRUE(all.equal(sd_order$Msar, W, check.attributes = FALSE)))

# Meanfield output draws suffice: they are only evaluation points.
points <- function(formula, family = gaussian(), newdata = NULL, newdata2 = NULL) {
  fit <- NULL
  for (seed in 1:20) {
    fit <- tryCatch(suppressMessages(brm(formula, data = d, family = family,
      data2 = data2, algorithm = "meanfield", output_samples = 3, iter = 2000,
      refresh = 0, seed = seed, init = 0)), error = function(e) NULL)
    if (!is.null(fit)) break
  }
  draws <- as.data.frame(as_draws_df(fit))[1:3, ]
  draws <- draws[, !grepl("^(lp__|lprior|\\.chain|\\.iteration|\\.draw)$", names(draws))]
  # A brms prediction that fails is recorded by its message.
  attempt <- function(expr) tryCatch(expr, error = function(e) list(error = conditionMessage(e)))
  out <- list(draws = draws,
              log_lik = attempt(log_lik(fit, draw_ids = 1:3)),
              epred = attempt(posterior_epred(fit, draw_ids = 1:3)),
              linpred = attempt(posterior_linpred(fit, draw_ids = 1:3)))
  if (!is.null(newdata)) {
    out$new_log_lik <- attempt(log_lik(fit, newdata = newdata, newdata2 = newdata2, draw_ids = 1:3))
    out$new_epred <- attempt(posterior_epred(fit, newdata = newdata, newdata2 = newdata2, draw_ids = 1:3))
  }
  out
}
fresh <- d[1:8, ]
fresh$g <- rep(c("p", "q"), each = 4)
fresh2 <- list(W = W[1:8, 1:8] / rowSums(W[1:8, 1:8]), V = V[1:8, 1:8])
values <- list(
  se_cosy = points(y | se(s, sigma = TRUE) ~ x + cosy(t, g), newdata = fresh),
  se_ar_cov_student = points(y | se(s, sigma = TRUE) ~ x + ar(t, g, cov = TRUE), student(), newdata = fresh),
  cens_ar = points(y | cens(c, y2) ~ x + ar(t, g), newdata = fresh),
  trunc_arma = points(y | trunc(lb = -3, ub = 4) ~ x + arma(t, g)),
  ar_sar_lag = points(y ~ x + ar(t, g) + sar(W), newdata = fresh, newdata2 = fresh2),
  ma_sar_error_student = points(y ~ x + ma(t, g) + sar(W, type = "error"), student()),
  ar_fcor = points(y ~ x + ar(t, g) + fcor(V), newdata = fresh, newdata2 = fresh2)
)
writeLines(toJSON(list(data = d, W = W, V = V, fresh = fresh, fresh_W = fresh2$W,
                       fresh_V = fresh2$V, cases = cases, row_order = row_order,
                       values = values),
                  auto_unbox = TRUE, digits = NA, dataframe = "columns"),
           "brms_autocor_combinations.json")
