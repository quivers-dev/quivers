# Generates brms_smooth_names.json, pinned by
# tests/formulas/test_smooth_variable_names.py: brms's variables() of models
# with s() and t2() smooths, factor, ordered-factor and numeric by variables,
# factor-smooth interactions, Gaussian process and random-effect bases,
# unpenalized smooths, two smooths of one covariate, smooths in a
# distributional predictor and in a two-response model, and R's make.unique
# on the names brms repairs with it.
# Run from this directory: Rscript brms_smooth_names.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
set.seed(11)
n <- 40
d <- data.frame(
  y = round(rnorm(n), 3), x = round(runif(n), 3), z = round(runif(n), 3),
  w = round(rnorm(n), 3), g = sample(c("a", "b", "c"), n, TRUE),
  o = sample(c("lo", "mid", "hi"), n, TRUE), stringsAsFactors = FALSE
)
fd <- d
fd$g <- factor(fd$g)
fd$o <- factor(fd$o, levels = c("lo", "mid", "hi"), ordered = TRUE)
forms <- list(
  basic = bf(y ~ s(x) + s(z, k = 5, bs = "cr") + t2(x, z, k = 4)),
  by = bf(y ~ s(x, by = g) + s(z, by = w, k = 5) + t2(x, z, by = g, k = 4) + s(w, by = o, k = 4)),
  bases = bf(y ~ s(x, g, bs = "fs", k = 4, m = 1) + s(z, bs = "gp", k = 5) + s(x, z, k = 6) +
    s(g, bs = "re")),
  duplicate = bf(y ~ s(x, k = 4) + s(x, bs = "cr", k = 5)),
  unpenalized = bf(y ~ s(x, k = 4, fx = TRUE) + s(z, k = 4) + s(w, by = g, k = 4, fx = TRUE)),
  distributional = bf(y ~ s(x, k = 4), sigma ~ s(z, k = 4) + s(x, by = g, k = 4)),
  joint = bf(mvbind(y, w) ~ s(x, k = 4) + s(z, by = g, k = 4)) + set_rescor(FALSE)
)
repaired <- c("s_sx_1[1]", "s_sx_1[1]", "s_sx_1[1]__1", "s_sx_1[1]", "b", "b", "b")
result <- list(
  brms = as.character(packageVersion("brms")), data = d, models = list(),
  make_unique = list(input = repaired, output = make.unique(repaired, "__"))
)
# The names do not depend on the algorithm or the seed. Stan's variational
# start fails for some models and seeds, so a mean-field fit started at zero
# is tried first and full-rank fits with other seeds after it.
attempts <- c(
  list(list(algorithm = "meanfield", iter = 500, init = 0, seed = 1)),
  lapply(1:6, function(seed) list(algorithm = "fullrank", iter = 2000, seed = seed))
)
for (key in names(forms)) {
  fit <- NULL
  for (attempt in attempts) {
    fit <- tryCatch(
      suppressWarnings(suppressMessages(do.call(brm, c(list(forms[[key]]), attempt, list(
        data = fd, refresh = 0, output_samples = 2, silent = 2
      ))))),
      error = function(e) NULL
    )
    if (!is.null(fit)) break
  }
  if (is.null(fit)) stop("no variational fit of model '", key, "' started")
  result$models[[key]] <- setdiff(variables(fit), c("lprior", "lp__"))
}
writeLines(toJSON(result, digits = 15, auto_unbox = TRUE, dataframe = "columns", pretty = TRUE),
  "brms_smooth_names.json")
