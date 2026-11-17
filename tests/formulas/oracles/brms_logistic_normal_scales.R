# Generates brms_logistic_normal_scales.json, pinned by
# tests/formulas/test_logistic_normal_scales.py: brms's Stan statements, prior
# rows, and messages for per-category logistic-normal scale formulas
# (sigma<cat> ~ ...), and variables, log_lik, and posterior_linpred at fixed
# posterior draws on the fitted and on fresh rows.
# Run from this directory: Rscript brms_logistic_normal_scales.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
options(digits = 17)
d <- data.frame(
  x = c(0.25, -0.5, 1.0, 1.5, -0.25, 0.75, -1.0, 2.0, 0.0, 0.5, 1.25, -1.5,
        0.125, 0.875, -0.75, 0.375),
  a = c(0.20, 0.35, 0.10, 0.25, 0.40, 0.15, 0.30, 0.05, 0.45, 0.22, 0.18, 0.50,
        0.33, 0.12, 0.28, 0.38),
  b = c(0.30, 0.25, 0.50, 0.15, 0.35, 0.45, 0.20, 0.60, 0.25, 0.48, 0.32, 0.10,
        0.27, 0.58, 0.42, 0.22)
)
d$c <- 1 - d$a - d$b
frame <- function(rows) {
  out <- data.frame(x = rows$x)
  out$Y <- as.matrix(rows[, c("a", "b", "c")])
  out
}
train <- frame(d)
fresh_rows <- d[c(2, 5, 11), ]
fresh_rows$x <- fresh_rows$x + 0.5
fresh <- frame(fresh_rows)

models <- list(
  one = list(formula = bf(Y ~ x, sigmab ~ x), family = logistic_normal()),
  both = list(formula = bf(Y ~ x, sigmab ~ x, sigmac ~ 1), family = logistic_normal()),
  refcat = list(formula = bf(Y ~ x, sigmaa ~ x), family = logistic_normal(refcat = "b"))
)
structure_of <- function(formula, family) {
  tryCatch({
    lines <- trimws(strsplit(make_stancode(formula, data = train, family = family), "\n")[[1]])
    prior <- as.data.frame(get_prior(formula, train, family = family))
    list(status = "ok",
         lines = lines[grepl("sigma", lines) & !grepl("^//|^\\*|means_X|Xc_|X_sigma|K_sigma|Kc_sigma", lines)],
         prior = prior[grepl("sigma", prior$class) | grepl("sigma", prior$dpar),
                       c("prior", "class", "coef", "dpar", "lb")])
  }, error = function(e) list(status = "refused", message = conditionMessage(e)))
}
cases <- lapply(models, function(m) structure_of(m$formula, m$family))
cases$plain <- structure_of(bf(Y ~ x), logistic_normal())
cases$shared <- structure_of(bf(Y ~ x, sigma ~ x), logistic_normal())
cases$reference <- structure_of(bf(Y ~ x, sigmaa ~ x), logistic_normal())

# Meanfield output draws suffice: they are only evaluation points.
points <- function(model, dpar) {
  fit <- suppressMessages(brm(model$formula, data = train, family = model$family,
    algorithm = "meanfield", output_samples = 3, iter = 2000, refresh = 0,
    seed = 1, init = 0))
  draws <- as.data.frame(as_draws_df(fit))[1:3, ]
  draws <- draws[, !grepl("^(lp__|lprior|\\.chain|\\.iteration|\\.draw)$", names(draws))]
  attempt <- function(expr) tryCatch(expr, error = function(e) list(error = conditionMessage(e)))
  list(variables = names(draws), draws = draws,
       log_lik = attempt(log_lik(fit, draw_ids = 1:3)),
       epred = attempt(posterior_epred(fit, draw_ids = 1:3)),
       linpred = attempt(posterior_linpred(fit, draw_ids = 1:3)),
       scale_linpred = attempt(posterior_linpred(fit, dpar = dpar, draw_ids = 1:3)),
       scale = attempt(posterior_linpred(fit, dpar = dpar, transform = TRUE, draw_ids = 1:3)),
       new_log_lik = attempt(log_lik(fit, newdata = fresh, draw_ids = 1:3)),
       new_scale = attempt(posterior_linpred(fit, dpar = dpar, transform = TRUE,
                                             newdata = fresh, draw_ids = 1:3)))
}
values <- list(
  one = points(models$one, "sigmab"),
  both = points(models$both, "sigmab"),
  refcat = points(models$refcat, "sigmaa")
)
writeLines(toJSON(list(data = d, fresh = fresh_rows, cases = cases, values = values),
                  auto_unbox = TRUE, digits = NA, dataframe = "columns", na = "null"),
           "brms_logistic_normal_scales.json")
