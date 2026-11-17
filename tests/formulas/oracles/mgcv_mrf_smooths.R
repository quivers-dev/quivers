# Generates mgcv_mrf_smooths.json, pinned by
# tests/formulas/test_mgcv_mrf_smooths.py: brms's design for Markov random
# field smooths s(region, bs = "mrf", xt = list(penalty = P)) and
# xt = list(nb = nb). Each entry is make_standata() on the training rows,
# checked against smoothCon(absorb.cons = TRUE, diagonal.penalty = TRUE) and
# smooth2random(type = 2), with brms:::s2rPred() on fresh rows. One model is
# fitted for brms's variable names, posterior_linpred at its own draws, and
# its refusal of an unknown region.
#
# brms evaluates xt in the global environment, so the structures below are
# global variables. The JSON records each penalty and neighbourhood matrix.
#
# Run from this directory: Rscript mgcv_mrf_smooths.R
suppressMessages({
  library(brms)
  library(mgcv)
  library(jsonlite)
})
regions <- paste0("r", 1:6)
graph <- function(edges) {
  a <- matrix(0, 6, 6, dimnames = list(regions, regions))
  for (e in seq_len(nrow(edges))) {
    a[edges[e, 1], edges[e, 2]] <- 1
    a[edges[e, 2], edges[e, 1]] <- 1
  }
  a
}
neighbours <- function(a) {
  nb <- lapply(regions, function(r) regions[a[r, ] > 0])
  names(nb) <- regions
  nb
}
# An irregular connected graph (distinct penalty eigenvalues), a ring
# (repeated eigenvalues), and a graph with two components.
A_irregular <- graph(rbind(c(1, 2), c(2, 3), c(3, 4), c(4, 5), c(5, 6), c(1, 3), c(2, 5)))
A_ring <- graph(rbind(c(1, 2), c(2, 3), c(3, 4), c(4, 5), c(5, 6), c(6, 1)))
A_split <- graph(rbind(c(1, 2), c(2, 3), c(1, 3), c(4, 5), c(5, 6)))
P_irregular <- diag(rowSums(A_irregular)) - A_irregular
P_ring <- diag(rowSums(A_ring)) - A_ring
P_split <- diag(rowSums(A_split)) - A_split
# The same penalty with its regions in another order, and without names.
shuffled <- regions[c(4, 1, 6, 2, 5, 3)]
P_shuffled <- P_irregular[shuffled, shuffled]
P_unnamed <- unname(P_irregular)
# A penalty that is not a graph Laplacian.
P_dense <- P_irregular + 0.5 * crossprod(matrix(sin(1:36), 6, 6) - mean(sin(1:36)))
dimnames(P_dense) <- list(regions, regions)
nb_irregular <- neighbours(A_irregular)
nb_split <- neighbours(A_split)
# A neighbour list in which the first region lists itself.
nb_self <- nb_irregular
nb_self$r1 <- c("r1", nb_self$r1)
A_self <- A_irregular
A_self["r1", "r1"] <- 1

n <- 30
i <- seq_len(n)
train <- data.frame(
  region = factor(regions[(i * 5) %% 6 + 1], levels = regions),
  w = round(cos(i), 6) + 1.5,
  h = factor(c("u", "v")[(i %/% 3) %% 2 + 1])
)
train$y <- sin(i) + as.integer(train$region) / 3
# Training rows without the last region, whose level is kept.
partial <- train[train$region != "r6", ]
fresh <- data.frame(
  region = factor(regions, levels = regions),
  w = c(0.6, 1.1, 1.4, 1.9, 2.2, 0.9),
  h = factor(c("u", "v", "v", "u", "u", "v"))
)
matrix_rows <- function(m) {
  m <- as.matrix(m)
  lapply(seq_len(nrow(m)), function(r) unname(m[r, ]))
}
entry <- function(spec, data = train, ...) {
  standata <- make_standata(as.formula(paste("y ~", spec)), data = data, ...)
  frame <- if (nrow(data) == n) data else data
  sml <- smoothCon(eval(parse(text = spec)), data = frame, knots = NULL,
                   absorb.cons = TRUE, diagonal.penalty = TRUE)
  blocks <- lapply(seq_along(sml), function(j) {
    re <- smooth2random(sml[[j]], names(frame), type = 2)
    new <- brms:::s2rPred(sml[[j]], re, fresh)
    name <- paste0("Zs_", j, "_1")
    if (length(re$rand)) {
      stopifnot(max(abs(standata[[name]] - as.matrix(re$rand[[1]]))) == 0)
    }
    list(
      label = sml[[j]]$label,
      fixed = matrix_rows(re$Xf),
      random = if (length(re$rand)) matrix_rows(re$rand[[1]]) else list(),
      fresh_fixed = matrix_rows(new$Xf),
      fresh_random = if (length(new$rand)) matrix_rows(new$rand[[1]]) else list()
    )
  })
  list(spec = spec, rows = nrow(data), blocks = blocks,
       xs = matrix_rows(standata$Xs))
}
message_of <- function(expr) tryCatch({expr; ""}, error = function(e) conditionMessage(e))
entries <- list(
  entry("s(region, bs = 'mrf', xt = list(penalty = P_irregular))"),
  entry("s(region, bs = 'mrf', xt = list(nb = nb_irregular))"),
  entry("s(region, bs = 'mrf', xt = list(penalty = P_shuffled))"),
  entry("s(region, bs = 'mrf', xt = list(penalty = P_unnamed))"),
  entry("s(region, bs = 'mrf', xt = list(penalty = P_dense))"),
  entry("s(region, bs = 'mrf', xt = list(penalty = P_ring))"),
  entry("s(region, bs = 'mrf', xt = list(penalty = P_split))"),
  entry("s(region, bs = 'mrf', xt = list(nb = nb_split))"),
  entry("s(region, bs = 'mrf', xt = list(nb = nb_self))"),
  entry("s(region, bs = 'mrf', k = 4, xt = list(penalty = P_irregular))"),
  entry("s(region, bs = 'mrf', k = 3, xt = list(penalty = P_dense))"),
  entry("s(region, bs = 'mrf', xt = list(penalty = P_irregular), by = w)"),
  entry("s(region, bs = 'mrf', xt = list(penalty = P_irregular), by = h)"),
  entry("s(region, bs = 'mrf', xt = list(penalty = P_irregular), fx = TRUE)")
)
unobserved <- list(
  entry("s(region, bs = 'mrf', xt = list(penalty = P_irregular))", data = partial,
        drop_unused_levels = FALSE),
  entry("s(region, bs = 'mrf', k = 4, xt = list(penalty = P_irregular))", data = partial,
        drop_unused_levels = FALSE)
)
fit <- suppressWarnings(suppressMessages(brm(
  y ~ s(region, bs = "mrf", xt = list(penalty = P_irregular)), data = train,
  algorithm = "meanfield", output_samples = 3, iter = 2000, refresh = 0, seed = 1
)))
draws <- as.data.frame(as_draws_df(fit))[1:3, ]
unknown <- data.frame(region = factor("r9"), w = 1, h = factor("u"))
out <- list(
  generator = "mgcv_mrf_smooths.R",
  brms = as.character(packageVersion("brms")),
  mgcv = as.character(packageVersion("mgcv")),
  regions = regions,
  train = as.list(transform(train, region = as.character(region), h = as.character(h))),
  fresh = as.list(transform(fresh, region = as.character(region), h = as.character(h))),
  structures = list(
    P_irregular = matrix_rows(P_irregular), P_ring = matrix_rows(P_ring),
    P_split = matrix_rows(P_split), P_dense = matrix_rows(P_dense),
    P_shuffled = matrix_rows(P_shuffled), shuffled = shuffled,
    nb_irregular = matrix_rows(A_irregular), nb_split = matrix_rows(A_split),
    nb_self = matrix_rows(A_self)
  ),
  entries = entries,
  unobserved = unobserved,
  errors = list(
    dropped_levels = message_of(make_standata(
      y ~ s(region, bs = "mrf", xt = list(penalty = P_irregular)), data = partial)),
    too_high = message_of(make_standata(
      y ~ s(region, bs = "mrf", k = 7, xt = list(penalty = P_irregular)), data = train)),
    no_structure = message_of(make_standata(y ~ s(region, bs = "mrf"), data = train)),
    data2 = message_of(make_standata(
      y ~ s(region, bs = "mrf", xt = list(penalty = P_data2)), data = train,
      data2 = list(P_data2 = P_irregular)))
  ),
  fit = list(
    variables = setdiff(variables(fit), c("lprior", "lp__")),
    design = matrix_rows(standata(fit)$Zs_1_1),
    draws = draws[, setdiff(names(draws), c(".chain", ".iteration", ".draw", "lp__", "lprior"))],
    linpred = posterior_linpred(fit, draw_ids = 1:3),
    fresh_linpred = posterior_linpred(fit, newdata = fresh, draw_ids = 1:3),
    new_level_error = message_of(posterior_linpred(fit, newdata = unknown, draw_ids = 1:3)),
    new_level_allowed_error = message_of(posterior_linpred(
      fit, newdata = unknown, draw_ids = 1:3, allow_new_levels = TRUE))
  )
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, dataframe = "columns"),
           "mgcv_mrf_smooths.json")
