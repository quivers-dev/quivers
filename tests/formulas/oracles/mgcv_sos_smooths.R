# Generates mgcv_sos_smooths.json, pinned by tests/formulas/test_mgcv_sos_smooths.py:
# brms's Stan data for mgcv's splines on the sphere s(la, lo, bs = "sos") with
# every penalty choice of m (the second-order spline, the four pseudospline
# orders, and the two penalties of the embedded sphere), mgcv's rounding and
# clamping of m, the default and full-rank basis dimensions, by variables, an
# unpenalized smooth, repeated rows, and user knots. Each entry follows brms's
# data_sm() path as in mgcv_smooths.R: smoothCon with absorb.cons = TRUE and
# diagonal.penalty = TRUE, smooth2random(type = 2) on the training rows,
# brms:::s2rPred() on fresh rows, and a check against make_standata(). The
# file also records mgcv's kernel makeR() at fixed angles, mgcv's raw
# constructor output (knots, design and penalty before smoothCon), the text
# order in which mgcv's uniquecombs() returns the knots, and the failures of
# mgcv on specifications outside its domain.
#
# Run from this directory: Rscript mgcv_sos_smooths.R
suppressMessages({
  library(brms)
  library(mgcv)
  library(jsonlite)
})

n <- 60
i <- seq_len(n)
train <- data.frame(
  la = asin(2 * ((i * 0.618034) %% 1) - 1) * 180 / pi,
  lo = ((i * 0.414214) %% 1) * 360 - 180,
  g = factor(c("a", "b", "c")[(i * 2) %% 3 + 1]),
  w = 0.5 + (i * 0.236068) %% 1
)
train$y <- sin(train$lo * pi / 180) * cos(train$la * pi / 180 - 0.3)
fresh <- data.frame(
  la = c(-90, -45.5, -10, 0, 12.25, 60, 90, train$la[3]),
  lo = c(0, -180, 35, 180, 359, -77.7, 10, train$lo[3]),
  g = factor(c("a", "b", "c", "a", "b", "c", "a", "b"), levels = levels(train$g)),
  w = c(0.6, 1.1, 1.4, 0.7, 0.9, 1.2, 0.55, 1.0)
)
repeated <- rbind(train, train[c(2, 5, 9, 14, 30), ])
user_knots <- list(la = train$la[seq(1, 60, by = 3)], lo = train$lo[seq(1, 60, by = 3)])

matrix_rows <- function(m) {
  m <- as.matrix(m)
  lapply(seq_len(nrow(m)), function(r) unname(m[r, ]))
}

entry <- function(spec, data = "train", knots = NULL) {
  rows <- get(data)
  sm <- smoothCon(eval(parse(text = spec)), data = rows, knots = knots,
                  absorb.cons = TRUE, diagonal.penalty = TRUE)
  blocks <- lapply(seq_along(sm), function(j) {
    one <- sm[[j]]
    re <- smooth2random(one, names(rows), type = 2)
    replay <- brms:::s2rPred(one, re, rows)
    for (b in seq_along(re$rand)) {
      stopifnot(max(abs(replay$rand[[b]] - re$rand[[b]])) < 1e-10)
    }
    new <- brms:::s2rPred(one, re, fresh)
    list(
      fixed = matrix_rows(re$Xf),
      random = unname(lapply(re$rand, matrix_rows)),
      fresh_fixed = matrix_rows(new$Xf),
      fresh_random = unname(lapply(new$rand, matrix_rows))
    )
  })
  formula <- as.formula(paste("y ~", spec))
  standata <- make_standata(formula, data = rows, knots = knots)
  Xs <- do.call(cbind, lapply(sm, function(one) {
    smooth2random(one, names(rows), type = 2)$Xf
  }))
  if (NCOL(Xs)) stopifnot(max(abs(standata$Xs - Xs)) == 0)
  for (j in seq_along(blocks)) {
    for (b in seq_along(blocks[[j]]$random)) {
      brms_z <- standata[[paste0("Zs_", j, "_", b)]]
      ours <- do.call(rbind, blocks[[j]]$random[[b]])
      stopifnot(max(abs(brms_z - ours)) == 0)
    }
  }
  out <- list(spec = spec, data = data, blocks = blocks)
  if (!is.null(knots)) out$knots <- knots
  out
}

specs <- c(
  "s(la, lo, bs = 'sos')",
  "s(la, lo, bs = 'sos', k = 10)",
  "s(la, lo, bs = 'sos', k = 10, m = 1)",
  "s(la, lo, bs = 'sos', k = 10, m = 2)",
  "s(la, lo, bs = 'sos', k = 9, m = 3)",
  "s(la, lo, bs = 'sos', k = 10, m = 4)",
  "s(la, lo, bs = 'sos', k = 10, m = -1)",
  "s(la, lo, bs = 'sos', k = 6, m = -1)",
  "s(la, lo, bs = 'sos', k = 10, m = -2)",
  "s(la, lo, bs = 'sos', k = 3)",
  "s(la, lo, bs = 'sos', k = 10, m = -3)",
  "s(la, lo, bs = 'sos', k = 10, m = 7)",
  "s(la, lo, bs = 'sos', k = 10, m = 1.4)",
  "s(la, lo, bs = 'sos', k = 10, m = 2.5)",
  "s(la, lo, bs = 'sos', k = 10, m = 2, by = g)",
  "s(la, lo, bs = 'sos', k = 10, by = w)",
  "s(la, lo, bs = 'sos', k = 10, m = 1, fx = TRUE)",
  "s(la, lo, bs = 'sos', k = 12, m = -1, by = g)"
)
entries <- lapply(specs, entry)
entries <- c(entries, list(
  entry("s(la, lo, bs = 'sos', k = 60, m = 2)"),
  entry("s(la, lo, bs = 'sos', k = 60, m = -1)"),
  entry("s(la, lo, bs = 'sos', k = 12)", data = "repeated"),
  entry("s(la, lo, bs = 'sos', k = 10, m = 2)", knots = user_knots),
  entry("s(la, lo, bs = 'sos', k = 20, m = -1)", knots = user_knots)
))

# The clamped and rounded orders equal the order mgcv reads them as.
same_order <- list(
  list("s(la, lo, bs = 'sos', k = 10, m = -3)", "s(la, lo, bs = 'sos', k = 10, m = -1)"),
  list("s(la, lo, bs = 'sos', k = 10, m = 7)", "s(la, lo, bs = 'sos', k = 10, m = 4)"),
  list("s(la, lo, bs = 'sos', k = 10, m = 1.4)", "s(la, lo, bs = 'sos', k = 10, m = 1)"),
  list("s(la, lo, bs = 'sos', k = 10, m = 2.5)", "s(la, lo, bs = 'sos', k = 10, m = 2)")
)
for (pair in same_order) {
  a <- make_standata(as.formula(paste("y ~", pair[[1]])), data = train)
  b <- make_standata(as.formula(paste("y ~", pair[[2]])), data = train)
  stopifnot(identical(dim(a$Xs), dim(b$Xs)), max(abs(a$Zs_1_1 - b$Zs_1_1)) == 0)
  if (length(a$Xs)) stopifnot(max(abs(a$Xs - b$Xs)) == 0)
}

# mgcv's kernel between fixed points and knots, for every m.
kernel_points <- data.frame(
  la = c(-90, -60, -12.5, 0, 0, 33, 71, 90, 10, 10),
  lo = c(0, 170, -45, 0, 180, 250, -120, 77, 20, 20.000001)
)
kernel_knots <- data.frame(la = c(10, -35, 0, 90, -89.5), lo = c(20, 300, 180, 0, -170))
kernels <- lapply(-2:4, function(m) {
  R <- mgcv:::makeR(kernel_points$la, kernel_points$lo, kernel_knots$la, kernel_knots$lo, m = m)
  null <- attr(R, "T")
  attributes(R) <- list(dim = dim(R))
  list(m = m, R = matrix_rows(R), null = matrix_rows(null))
})

# mgcv's constructor output before smoothCon, for one low-rank and one
# full-rank basis per null space.
raw <- lapply(
  list(c(10, 0), c(10, 2), c(10, -1), c(10, -2), c(60, 3), c(60, -1)),
  function(km) {
    object <- s(la, lo, bs = "sos", k = km[1], m = km[2])
    object$bs.dim <- km[1]
    object$p.order <- km[2]
    sm <- smooth.construct(object, train, NULL)
    list(
      k = km[1], m = km[2], X = matrix_rows(sm$X), S = matrix_rows(sm$S[[1]]),
      rank = sm$rank, null_dim = sm$null.space.dim, scale = sm$xc.scale,
      knots = matrix_rows(matrix(sm$knt, ncol = 2)),
      fresh = matrix_rows(Predict.matrix(sm, fresh))
    )
  }
)

# mgcv orders the unique rows by the C collation of their pasted text.
text_values <- c(0, 1, -1, 45, 0.1, 1 / 3, -2 / 3, 100000, 123456, 100000.5, 1e-04,
                 1e-05, 0.00012345, 1.5e-05, 123456.7, 359.99999999999994,
                 89.999999999999986, 0.1 + 0.2, 12.3456789012345678, 1e-10,
                 -179.99999999999997, 1234567.25, 0.001, 2e-04, -5e-06, 90, -90, 180)
text_rows <- cbind(
  c(5, 10, -3, -20, 100.5, 9.75, 5, 1e-05, 5 + 1e-15, 0, -3),
  c(2, 1, 7, 7, 0, 30, 2, 4, 2, -0.5, 60)
)
row_text <- list(
  values = text_values,
  text = paste0(text_values),
  rows = matrix_rows(text_rows),
  unique = matrix_rows(unname(uniquecombs(text_rows, TRUE)))
)

# Specifications mgcv or brms fails on.
failure <- function(spec, data = train, knots = NULL) {
  message <- tryCatch({
    make_standata(as.formula(paste("y ~", spec)), data = data, knots = knots)
    NA_character_
  }, error = function(e) conditionMessage(e))
  list(spec = spec, error = message)
}
out_of_range <- train
out_of_range$la[1] <- 120
out_of_range$lo[2] <- 700
failures <- list(
  failure("s(la, lo, bs = 'sos', k = 61)"),
  failure("s(la, lo, bs = 'sos', k = 2)"),
  failure("s(la, lo, bs = 'sos', k = 1)"),
  failure("s(la, lo, bs = 'sos', k = 5, m = -1)"),
  failure("s(la, lo, bs = 'sos', k = 10, m = c(1, 2))"),
  failure("s(la, bs = 'sos')"),
  failure("s(la, lo, w, bs = 'sos')"),
  failure("t2(la, lo, bs = 'sos')")
)
stopifnot(!any(sapply(failures, function(f) is.na(f$error))))
# mgcv does not check the coordinate ranges, and ignores knots given for one
# covariate alone.
accepted <- list(
  out_of_range = is.na(failure("s(la, lo, bs = 'sos', k = 10)", data = out_of_range)$error),
  one_knot_vector = identical(
    make_standata(y ~ s(la, lo, bs = "sos", k = 10), data = train,
                  knots = list(la = user_knots$la))$Zs_1_1,
    make_standata(y ~ s(la, lo, bs = "sos", k = 10), data = train)$Zs_1_1
  )
)
stopifnot(accepted$out_of_range, accepted$one_knot_vector)

frame <- function(d) as.list(transform(d, g = as.character(g)))
out <- list(
  generator = "mgcv_sos_smooths.R",
  brms = as.character(packageVersion("brms")),
  mgcv = as.character(packageVersion("mgcv")),
  train = frame(train),
  fresh = frame(fresh),
  repeated = frame(repeated),
  entries = entries,
  kernel_points = as.list(kernel_points),
  kernel_knots = as.list(kernel_knots),
  kernels = kernels,
  raw = raw,
  row_text = row_text,
  failures = failures,
  accepted = accepted
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), "mgcv_sos_smooths.json")
