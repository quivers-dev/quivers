# Generates mgcv_factor_smooths.json, pinned by
# tests/formulas/test_mgcv_factor_smooths.py: brms's design for factor-smooth
# interactions s(x, g, bs = "fs"). Each entry follows brms's data_sm() path:
# smoothCon(absorb.cons = TRUE, diagonal.penalty = TRUE), smooth2random(type = 2)
# on the training rows, and brms:::s2rPred() on fresh rows, one of which has a
# factor level absent from training. The training matrices are checked against
# make_standata(). The base smooth's penalty rank and null-space dimension,
# the penalty index of every coefficient, brms's prior rows, and brms's
# handling of the unseen level in posterior_linpred are recorded.
#
# Run from this directory: Rscript mgcv_factor_smooths.R
suppressMessages({
  library(brms)
  library(mgcv)
  library(jsonlite)
})
n <- 40
i <- seq_len(n)
train <- data.frame(
  x = (i * 0.618034) %% 1,
  z = (i * 0.414214) %% 1 * 2 - 0.5,
  g = factor(c("a", "b", "c", "d")[(i * 3) %% 4 + 1])
)
train$y <- sin(3 * train$x) + train$z^2
fresh <- data.frame(
  x = c(-0.1, 0.05, 0.33, 0.5, 0.77, 0.99, 1.2),
  z = c(-0.6, -0.2, 0.1, 0.4, 0.9, 1.4, 1.6),
  g = factor(c("a", "b", "c", "d", "a", "b", "e"), levels = c("a", "b", "c", "d", "e"))
)
specs <- c(
  "s(x, g, bs = 'fs')", "s(x, g, bs = 'fs', k = 6, m = 1)",
  "s(x, g, bs = 'fs', k = 5)", "s(g, x, bs = 'fs', k = 7, m = 3)",
  "s(x, z, g, bs = 'fs', k = 8)"
)
matrix_rows <- function(m) {
  m <- as.matrix(m)
  lapply(seq_len(nrow(m)), function(r) unname(m[r, ]))
}
entry <- function(spec) {
  sm <- smoothCon(eval(parse(text = spec)), data = train, knots = NULL,
                  absorb.cons = TRUE, diagonal.penalty = TRUE)[[1]]
  re <- smooth2random(sm, names(train), type = 2)
  stopifnot(ncol(re$Xf) == 0)
  new <- brms:::s2rPred(sm, re, fresh)
  standata <- make_standata(as.formula(paste("y ~", spec)), data = train)
  for (b in seq_along(re$rand)) {
    stopifnot(max(abs(standata[[paste0("Zs_1_", b)]] - as.matrix(re$rand[[b]]))) == 0)
  }
  prior <- get_prior(as.formula(paste("y ~", spec)), data = train)
  list(
    spec = spec,
    rank = sm$base$rank,
    null_dim = sm$base$bs.dim - sm$base$rank,
    levels = sm$flev,
    pen_ind = re$pen.ind,
    random = unname(lapply(re$rand, matrix_rows)),
    fresh_random = unname(lapply(new$rand, matrix_rows)),
    sds = prior$coef[prior$class == "sds" & nzchar(prior$coef)],
    stan = trimws(grep("sds_|zs_|s_1_", strsplit(make_stancode(as.formula(paste("y ~", spec)), data = train), "\n")[[1]], value = TRUE))
  )
}
fit <- suppressMessages(brm(y ~ s(x, g, bs = "fs", k = 5, m = 1), data = train,
  algorithm = "meanfield", output_samples = 3, iter = 2000, refresh = 0, seed = 1))
message_of <- function(expr) tryCatch({expr; ""}, error = function(e) conditionMessage(e))
draws <- as.data.frame(as_draws_df(fit))[1:3, ]
out <- list(
  generator = "mgcv_factor_smooths.R",
  brms = as.character(packageVersion("brms")),
  mgcv = as.character(packageVersion("mgcv")),
  train = as.list(transform(train, g = as.character(g))),
  fresh = as.list(transform(fresh, g = as.character(g))),
  entries = lapply(specs, entry),
  fit = list(
    variables = setdiff(variables(fit), c("lprior", "lp__")),
    design = lapply(c("Zs_1_1", "Zs_1_2"), function(name) matrix_rows(standata(fit)[[name]])),
    draws = draws[, setdiff(names(draws), c(".chain", ".iteration", ".draw", "lp__", "lprior"))],
    linpred = posterior_linpred(fit, draw_ids = 1:3),
    new_level_error = message_of(posterior_linpred(fit, newdata = fresh, draw_ids = 1:3)),
    new_level_allowed_error = message_of(posterior_linpred(fit, newdata = fresh, draw_ids = 1:3, allow_new_levels = TRUE)),
    fresh_linpred = tryCatch(posterior_linpred(fit, newdata = fresh, draw_ids = 1:3, allow_new_levels = TRUE), error = function(e) NULL),
    seen_linpred = posterior_linpred(fit, newdata = droplevels(fresh[1:6, ]), draw_ids = 1:3)
  )
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, dataframe = "columns"), "mgcv_factor_smooths.json")
