# Generates mgcv_smooths.json: brms 2.20.4 smooth design matrices from mgcv.
#
# Each entry reproduces brms's data_sm() path exactly: smoothCon with
# absorb.cons = TRUE and diagonal.penalty = TRUE, smooth2random(type = 2) for
# the training rows, and brms:::s2rPred() for fresh rows. The training
# matrices are also checked against make_standata() so the oracle is brms's
# own Stan data, not a reimplementation of it.
#
# brms predicts a t2() smooth on fresh rows through mgcv's prediction
# constraint Cp, which differs from the constraint absorbed in the training
# design, so its fresh-row matrices do not reproduce its own training rows.
# For t2() the fresh rows here use the training constraint (modCon = 3, which
# leaves the training design unchanged; checked below), the parameterization
# QVR retains for prediction.
#
# Values are written at full precision: the covariates must round-trip
# exactly, because mgcv's Lanczos stopping point (and with it a small-k
# thin-plate basis) moves under perturbations of the training values.
#
# brm(knots = ) entries pass user knots to the same constructors.
#
# Run from this directory: Rscript mgcv_smooths.R
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
  "s(x)", "s(x, k = 5)", "s(x, bs = 'tp', m = 3, k = 8)",
  "s(x, z)", "s(x, z, v, k = 15)", "s(x, bs = 'ts')",
  "s(x, bs = 'cr')", "s(x, bs = 'cs', k = 6)", "s(x, bs = 'cc', k = 7)",
  "s(x, bs = 'ps')", "s(x, bs = 'ps', k = 7, m = 1)", "s(x, bs = 'bs')",
  "s(x, bs = 'bs', k = 8, m = c(3, 1))", "s(g, bs = 're')",
  "s(x, by = g)", "s(x, by = w)", "s(x, bs = 'cr', k = 6, by = g)",
  "t2(x, z)", "t2(x, z, k = 4, bs = 'tp')", "t2(x, z, v, k = 4)",
  "t2(x, z, k = c(5, 4), bs = c('cr', 'ps'))", "t2(x, z, by = g, k = 4)"
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

# brm(knots = ) passes user knots to mgcv's constructors: cr and cc take k
# knots (cc also two end points), ps and bs two end points or the full knot
# sequence (nk + 2 m[1] + 2 for ps, nk + 2 m[1] for bs), and bs four knots
# (outer end points and inner range).
knot_cases <- list(
  list("s(x, bs = 'cr', k = 5)", list(x = c(0, 0.2, 0.5, 0.8, 1))),
  list("s(x, bs = 'cs', k = 5)", list(x = c(-0.1, 0.3, 0.5, 0.6, 1.05))),
  list("s(x, bs = 'cc', k = 6)", list(x = c(-0.05, 1.05))),
  list("s(x, bs = 'cc', k = 5)", list(x = c(0, 0.3, 0.5, 0.7, 1))),
  list("s(x, bs = 'ps', k = 8)", list(x = c(-0.1, 1.1))),
  list("s(x, bs = 'ps', k = 6)", list(x = seq(-1.1, 2.1, length.out = 10))),
  list("s(x, bs = 'bs', k = 6)", list(x = c(-0.1, 1.1))),
  list("s(x, bs = 'bs', k = 8)", list(x = c(0, 0.2, 0.8, 1))),
  list("s(x, bs = 'bs', k = 6)", list(x = seq(-1.2, 2.2, length.out = 10))),
  list("s(x, bs = 'cr', k = 6, by = g)", list(x = c(0, 0.1, 0.3, 0.6, 0.9, 1)))
)

out <- list(
  generator = "mgcv_smooths.R",
  brms = as.character(packageVersion("brms")),
  mgcv = as.character(packageVersion("mgcv")),
  train = as.list(transform(train, g = as.character(g))),
  fresh = as.list(transform(fresh, g = as.character(g))),
  entries = lapply(specs, entry),
  knot_entries = lapply(knot_cases, function(case) entry(case[[1]], case[[2]]))
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), "mgcv_smooths.json")
