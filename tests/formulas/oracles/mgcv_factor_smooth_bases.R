# Generates mgcv_factor_smooth_bases.json, pinned by
# tests/formulas/test_mgcv_factor_smooth_bases.py: brms's design for
# factor-smooth interactions s(x, g, bs = "fs", xt = ...) over bases other
# than the thin plate, and what mgcv and brms do with `by` and `fx`. Each
# entry follows brms's data_sm() path: smoothCon(absorb.cons = TRUE,
# diagonal.penalty = TRUE), smooth2random(type = 2) on the training rows, and
# brms:::s2rPred() on fresh rows, checked against make_standata().
#
# `by`: mgcv's smoothCon multiplies the design X of the smooth by the `by`
# variable, which is what mgcv::gam fits, and smooth2random then reads the
# unmultiplied copy Xb, so brms's Stan data ignore `by`. Both facts are
# recorded as maximal absolute differences.
#
# Run from this directory: Rscript mgcv_factor_smooth_bases.R
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
  w = round(cos(i), 6) + 1.5,
  g = factor(c("a", "b", "c", "d")[(i * 3) %% 4 + 1]),
  h = factor(c("u", "v")[(i %/% 2) %% 2 + 1])
)
train$y <- sin(3 * train$x) + train$z^2
fresh <- data.frame(
  x = c(0.02, 0.05, 0.33, 0.5, 0.77, 0.97),
  z = c(-0.4, -0.2, 0.1, 0.4, 0.9, 1.4),
  w = c(0.6, 1.1, 1.4, 1.9, 2.2, 0.9),
  g = factor(c("a", "b", "c", "d", "a", "b"), levels = c("a", "b", "c", "d")),
  h = factor(c("u", "v", "v", "u", "u", "v"))
)
specs <- c(
  "s(x, g, bs = 'fs', k = 5, xt = list(bs = 'cr'))",
  "s(x, g, bs = 'fs', k = 5, xt = 'cr')",
  "s(x, g, bs = 'fs', k = 6, xt = list(bs = 'ps'))",
  "s(x, g, bs = 'fs', k = 6, m = c(2, 1), xt = list(bs = 'ps'))",
  "s(x, g, bs = 'fs', k = 5, xt = list(bs = 'cc'))",
  "s(x, g, bs = 'fs', k = 6, xt = list(bs = 'bs'))",
  "s(x, g, bs = 'fs', k = 5, xt = list(bs = 'gp'))",
  "s(x, z, g, bs = 'fs', k = 8, xt = list(bs = 'gp'))",
  "s(x, g, bs = 'fs', k = 5, xt = list(bs = 'tp'))"
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
  list(
    spec = spec,
    rank = sm$base$rank,
    null_dim = length(re$rand) - 1,
    levels = sm$flev,
    pen_ind = re$pen.ind,
    random = unname(lapply(re$rand, matrix_rows)),
    fresh_random = unname(lapply(new$rand, matrix_rows))
  )
}
message_of <- function(expr) tryCatch({expr; ""}, error = function(e) conditionMessage(e))
plain <- smoothCon(s(x, g, bs = "fs", k = 5), data = train, absorb.cons = TRUE,
                   diagonal.penalty = TRUE)[[1]]
by_factor <- smoothCon(s(x, g, bs = "fs", k = 5, by = h), data = train, absorb.cons = TRUE,
                       diagonal.penalty = TRUE)
by_numeric <- smoothCon(s(x, g, bs = "fs", k = 5, by = w), data = train, absorb.cons = TRUE,
                        diagonal.penalty = TRUE)[[1]]
data_plain <- make_standata(y ~ s(x, g, bs = "fs", k = 5), data = train)
data_factor <- make_standata(y ~ s(x, g, bs = "fs", k = 5, by = h), data = train)
data_numeric <- make_standata(y ~ s(x, g, bs = "fs", k = 5, by = w), data = train)
blocks <- c("Zs_1_1", "Zs_1_2", "Zs_1_3")
out <- list(
  generator = "mgcv_factor_smooth_bases.R",
  brms = as.character(packageVersion("brms")),
  mgcv = as.character(packageVersion("mgcv")),
  train = as.list(transform(train, g = as.character(g), h = as.character(h))),
  fresh = as.list(transform(fresh, g = as.character(g), h = as.character(h))),
  entries = lapply(specs, entry),
  by = list(
    spec = "s(x, g, bs = 'fs', k = 5)",
    plain = unname(lapply(blocks, function(b) matrix_rows(data_plain[[b]]))),
    # mgcv's own design multiplies by the `by` variable.
    gam_factor_masks = max(sapply(seq_along(by_factor), function(l) {
      max(abs(by_factor[[l]]$X - as.numeric(train$h == levels(train$h)[l]) * plain$X))
    })),
    gam_numeric_multiplies = max(abs(by_numeric$X - train$w * plain$X)),
    # brms's Stan data do not.
    brms_factor_ignores = max(sapply(blocks, function(b) {
      max(abs(data_factor[[b]] - data_plain[[b]]),
          abs(data_factor[[sub("Zs_1", "Zs_2", b)]] - data_plain[[b]]))
    })),
    brms_numeric_ignores = max(sapply(blocks, function(b) {
      max(abs(data_numeric[[b]] - data_plain[[b]]))
    })),
    factor_labels = sapply(by_factor, function(s) s$label),
    numeric_label = by_numeric$label
  ),
  errors = list(
    fx = message_of(make_standata(y ~ s(x, g, bs = "fs", k = 5, fx = TRUE), data = train)),
    cs = message_of(make_standata(y ~ s(x, g, bs = "fs", k = 5, xt = list(bs = "cs")), data = train)),
    ts = message_of(make_standata(y ~ s(x, g, bs = "fs", k = 5, xt = list(bs = "ts")), data = train))
  )
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, dataframe = "columns"),
           "mgcv_factor_smooth_bases.json")
