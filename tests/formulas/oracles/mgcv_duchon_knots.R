# Generates mgcv_duchon_knots.json, pinned by
# tests/formulas/test_mgcv_duchon_knots.py: brms's Stan data for mgcv's Duchon
# splines s(..., bs = "ds") (penalty order m and Fourier weight s, eigenbasis
# and full knot basis, by, fx, user knots, and a t2 margin), and for user knots
# (brm(knots = )) of thin-plate bases (tp, ts, a dropped null space,
# multivariate s(), by) and of t2() margins. Each entry follows brms's
# data_sm() path as in mgcv_smooths.R: smoothCon with absorb.cons = TRUE and
# diagonal.penalty = TRUE, smooth2random(type = 2) on the training rows,
# brms:::s2rPred() on fresh rows, and a check against make_standata().
#
# Run from this directory: Rscript mgcv_duchon_knots.R
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
  out <- list(spec = spec, blocks = blocks)
  if (!is.null(knots)) out$knots <- knots
  out
}

duchon_specs <- c(
  "s(x, z, bs = 'ds')", "s(x, z, bs = 'ds', k = 12, m = c(1, 0.5))",
  "s(x, z, bs = 'ds', k = 10, m = c(2, -0.5))", "s(x, bs = 'ds', k = 8)",
  "s(x, bs = 'ds', k = 7, m = 1)", "s(x, bs = 'ds', k = 9, m = 3)",
  "s(x, z, v, bs = 'ds', k = 15, m = c(2, 0.5))",
  "s(x, z, v, bs = 'ds', k = 12, m = c(1, 1))", "s(x, z, bs = 'ds', k = 40)",
  "s(x, z, bs = 'ds', k = 10, by = g)", "s(x, z, bs = 'ds', k = 10, by = w)",
  "s(x, z, bs = 'ds', k = 9, fx = TRUE)",
  "t2(x, z, v, d = c(2, 1), bs = c('ds', 'cr'), k = c(8, 4))"
)

kx <- c(0.02, 0.11, 0.2, 0.31, 0.39, 0.5, 0.58, 0.69, 0.8, 0.9, 0.97, 0.15, 0.45, 0.75)
kz <- c(-0.4, 1.3, 0.2, 0.9, -0.1, 0.5, 1.1, -0.3, 0.7, 0.1, 1.4, 0.8, -0.45, 0.4)
kv <- c(0.05, 0.35, 0.65, 0.95)
knot_cases <- list(
  # Duchon splines on user knots: eigenbasis of the knots, and one basis
  # function per knot.
  list("s(x, z, bs = 'ds', k = 10, m = c(1, 0.5))", list(x = kx, z = kz)),
  list("s(x, z, bs = 'ds', k = 14)", list(x = kx, z = kz)),
  list("s(x, bs = 'ds', k = 6)", list(x = kx[1:11])),
  # Thin-plate bases on user knots.
  list("s(x, k = 6)", list(x = kx[1:11])),
  list("s(x, k = 6)", list(x = kx[c(1, 3, 5, 7, 9, 11)])),
  list("s(x, bs = 'ts', k = 6)", list(x = kx[1:9])),
  list("s(x, k = 7, m = 3)", list(x = kx[1:11])),
  list("s(x, k = 6, m = c(2, 0))", list(x = kx[1:11])),
  list("s(x, z, k = 8)", list(x = kx, z = kz)),
  list("s(x, z, k = 14)", list(x = kx, z = kz)),
  list("s(x, z, bs = 'ts', k = 9)", list(x = kx, z = kz)),
  list("s(x, k = 6, by = g)", list(x = kx[1:11])),
  list("s(x, k = 6, by = w)", list(x = kx[1:11])),
  # Knots of t2() margins: each margin reads the knots of its covariates.
  list("t2(x, z, bs = 'cr', k = 4)", list(x = c(0, 0.3, 0.7, 1), z = c(-0.5, 0.2, 0.9, 1.5))),
  list("t2(x, z, bs = 'cr', k = 4)", list(x = c(0, 0.3, 0.7, 1))),
  list("t2(x, z, bs = c('tp', 'cr'), k = c(5, 4))", list(x = kx[1:11], z = c(-0.5, 0.2, 0.9, 1.5))),
  list("t2(x, z, v, d = c(2, 1), bs = c('tp', 'cr'), k = c(8, 4))", list(x = kx, z = kz, v = kv)),
  list("t2(x, z, v, d = c(2, 1), bs = c('ds', 'cr'), k = c(8, 4))", list(x = kx, z = kz))
)

out <- list(
  generator = "mgcv_duchon_knots.R",
  brms = as.character(packageVersion("brms")),
  mgcv = as.character(packageVersion("mgcv")),
  train = as.list(transform(train, g = as.character(g))),
  fresh = as.list(transform(fresh, g = as.character(g))),
  entries = lapply(duchon_specs, entry),
  knot_entries = lapply(knot_cases, function(case) entry(case[[1]], case[[2]]))
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), "mgcv_duchon_knots.json")
