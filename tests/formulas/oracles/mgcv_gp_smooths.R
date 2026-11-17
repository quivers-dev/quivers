# Generates mgcv_gp_smooths.json, pinned by tests/formulas/test_mgcv_gp_smooths.py:
# brms's Stan data for mgcv's Gaussian process smooths s(x, bs = "gp") with
# each covariance choice of m (spherical, power exponential, Matern 1.5, 2.5
# and 3.5, given range and power, and the stationary form of a negative type).
# Each entry follows brms's data_sm() path as in mgcv_smooths.R: smoothCon with
# absorb.cons = TRUE and diagonal.penalty = TRUE, smooth2random(type = 2) on the
# training rows, brms:::s2rPred() on fresh rows, and a check against
# make_standata().
#
# Run from this directory: Rscript mgcv_gp_smooths.R
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
  v = (i * 0.732051) %% 1,
  g = factor(c("a", "b", "c", "d")[(i * 3) %% 4 + 1]),
  w = 0.5 + (i * 0.236068) %% 1
)
train$y <- sin(3 * train$x) + train$z^2
fresh <- data.frame(
  x = c(-0.1, 0.05, 0.33, 0.5, 0.77, 0.99, 1.2),
  z = c(-0.6, -0.2, 0.1, 0.4, 0.9, 1.4, 1.6),
  v = c(0.1, 0.9, 0.5, 0.2, 0.3, 0.7, 1.1),
  g = factor(c("a", "b", "c", "d", "a", "b", "c"), levels = levels(train$g)),
  w = c(0.6, 1.1, 1.4, 0.7, 0.9, 1.2, 0.55)
)

specs <- c(
  "s(x, bs = 'gp')", "s(x, bs = 'gp', k = 7, m = 1)",
  "s(x, bs = 'gp', k = 8, m = c(2, 0.7, 1.5))", "s(x, bs = 'gp', k = 6, m = 4)",
  "s(x, bs = 'gp', k = 6, m = c(5, 0.4))", "s(x, bs = 'gp', k = 6, m = c(-3, 0.5))",
  "s(x, bs = 'gp', k = 5, m = -2)", "s(x, z, bs = 'gp', k = 12)",
  "s(x, z, bs = 'gp', k = 9, m = c(-4, 1.2))", "s(x, bs = 'gp', k = 6, by = g)",
  "s(x, bs = 'gp', k = 6, by = w)", "s(x, bs = 'gp', k = 6, fx = TRUE)"
)

matrix_rows <- function(m) {
  m <- as.matrix(m)
  lapply(seq_len(nrow(m)), function(r) unname(m[r, ]))
}

entry <- function(spec, knots = NULL) {
  sm <- smoothCon(eval(parse(text = spec)), data = train, knots = knots,
                  absorb.cons = TRUE, diagonal.penalty = TRUE)
  tensor <- startsWith(spec, "t2(")
  fit_sm <- if (tensor) {
    smoothCon(eval(parse(text = spec)), data = train, knots = knots,
              absorb.cons = TRUE, diagonal.penalty = TRUE, modCon = 3)
  } else sm
  blocks <- lapply(seq_along(sm), function(j) {
    one <- sm[[j]]
    re <- smooth2random(one, names(train), type = 2)
    stopifnot(max(abs(fit_sm[[j]]$X - one$X)) == 0)
    replay <- brms:::s2rPred(fit_sm[[j]], re, train)
    for (b in seq_along(re$rand)) {
      stopifnot(max(abs(replay$rand[[b]] - re$rand[[b]])) < 1e-10)
    }
    new <- brms:::s2rPred(fit_sm[[j]], re, fresh)
    list(
      fixed = matrix_rows(re$Xf),
      random = unname(lapply(re$rand, matrix_rows)),
      fresh_fixed = matrix_rows(new$Xf),
      fresh_random = unname(lapply(new$rand, matrix_rows))
    )
  })
  formula <- as.formula(paste("y ~", spec))
  standata <- make_standata(formula, data = train, knots = knots)
  Xs <- do.call(cbind, lapply(sm, function(one) {
    smooth2random(one, names(train), type = 2)$Xf
  }))
  if (NCOL(Xs)) stopifnot(max(abs(standata$Xs - Xs)) == 0)
  for (j in seq_along(blocks)) {
    for (b in seq_along(blocks[[j]]$random)) {
      brms_z <- standata[[paste0("Zs_", j, "_", b)]]
      ours <- do.call(rbind, blocks[[j]]$random[[b]])
      stopifnot(max(abs(brms_z - ours)) == 0)
    }
  }
  out <- list(spec = spec, blocks = blocks)
  if (!is.null(knots)) out$knots <- knots
  out
}

out <- list(
  generator = "mgcv_gp_smooths.R",
  brms = as.character(packageVersion("brms")),
  mgcv = as.character(packageVersion("mgcv")),
  train = as.list(transform(train, g = as.character(g))),
  fresh = as.list(transform(fresh, g = as.character(g))),
  entries = lapply(specs, entry)
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), "mgcv_gp_smooths.json")
