# Generates brms_conditional_effects.json, pinned by
# tests/formulas/test_brms_conditional_effects.py.
# brms oracles for conditional_effects and make_conditions: which effects are
# selected by default, the conditions chosen for the remaining variables, the
# prediction grids (resolution, int_conditions, surface, too_far, integer
# variables, trials), the labels (cond__, effect1__, effect2__, cats__), the
# point frames of select_points, and the list names, for several fitted
# models. Only grid and label columns are exported: the estimates depend on
# brms's own draws. Short chains suffice, since no posterior value is used.
# conditional_smooths differs between brms 2.20.4 (installed) and 2.23.0 (the
# reference) for factor by variables and surface = FALSE; the test pins the
# shared grid values and checks the 2.23.0 rule.
# Run from this directory: Rscript brms_conditional_effects.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))

d <- data.frame(
  y = c(0.31, -1.24, 0.85, 1.93, -0.42, 0.13, 1.41, -0.73, 0.62, 2.21,
        -1.52, 0.94, 0.27, -0.66, 1.18, 0.05, -0.91, 1.67, 0.44, -0.18,
        0.72, 1.36, -1.05, 0.58, 0.09, -0.37, 1.02, 0.66, -0.84, 0.21),
  y2 = c(1.1, 0.4, -0.3, 0.9, 1.6, -0.8, 0.2, 0.7, -1.2, 0.5,
         1.3, -0.1, 0.8, 0.3, -0.6, 1.0, 0.1, -0.4, 0.6, 1.2,
         -0.9, 0.0, 0.4, 1.5, -0.2, 0.9, -0.7, 0.3, 1.1, -0.5),
  x = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8,
        0.4, -0.3, 1.1, -0.6, 0.2, 0.8, -1.2, 1.5, -0.1, 0.7,
        -0.9, 1.3, -0.5, 0.1, 0.6, -1.0, 0.9, -0.4, 1.0, -0.7),
  z = c(-0.2, 0.4, 1.1, -0.9, 0.3, -0.5, 0.8, 1.4, -1.3, 0.2,
        0.6, -0.8, -0.1, 1.2, -0.6, 0.5, 0.9, -1.1, 0.0, 0.7,
        -0.4, 1.0, 0.3, -0.7, 1.3, -0.2, 0.1, -1.0, 0.4, 0.6),
  w = c(0.8, 1.9, 0.5, 2.4, 1.2, 0.3, 1.6, 2.2, 0.9, 1.1,
        0.4, 2.0, 1.3, 0.7, 1.8, 0.6, 2.5, 1.0, 0.2, 1.5,
        2.1, 0.8, 1.4, 0.5, 1.7, 2.3, 0.9, 1.2, 0.6, 1.9),
  f = c("q", "p", "r", "p", "q", "r", "r", "p", "q", "p",
        "q", "r", "p", "q", "r", "p", "q", "r", "p", "q",
        "r", "p", "q", "r", "p", "q", "r", "p", "q", "r"),
  g = rep(c("a", "b", "c"), 10),
  k = c(5, 3, 4, 6, 5, 2, 4, 3, 5, 6, 4, 5, 3, 2, 6, 4, 5, 3, 4, 6,
        2, 5, 4, 3, 6, 5, 4, 2, 3, 5),
  m = c(0, 2, 1, 3, 2, 0, 1, 3, 2, 1, 0, 3, 1, 2, 0, 3, 2, 1, 0, 2,
        3, 1, 0, 2, 1, 3, 0, 1, 2, 3),
  stringsAsFactors = FALSE
)
d$s <- c(2, 1, 3, 4, 2, 1, 2, 2, 3, 4, 1, 3, 2, 1, 4, 2, 3, 1, 2, 4,
         1, 3, 2, 1, 4, 3, 2, 1, 2, 3)
d$ys <- pmin(d$k, c(3, 1, 2, 4, 2, 1, 3, 1, 4, 5, 2, 3, 1, 0, 4, 2, 3, 1, 2, 5,
                    1, 3, 2, 1, 5, 4, 2, 1, 2, 3))
d$o <- factor(d$s, levels = 1:4)

quick <- function(...) {
  suppressMessages(suppressWarnings(brm(
    ..., chains = 1, iter = 60, warmup = 30, refresh = 0, seed = 1,
    silent = 2, backend = "cmdstanr"
  )))
}

grid_columns <- function(frame) {
  drop <- c("estimate__", "se__", "lower__", "upper__")
  frame <- frame[, setdiff(names(frame), drop), drop = FALSE]
  for (v in names(frame)) {
    if (is.factor(frame[[v]])) frame[[v]] <- as.character(frame[[v]])
  }
  rownames(frame) <- NULL
  frame
}

export_effects <- function(ce) {
  out <- lapply(ce, function(frame) {
    points <- attr(frame, "points")
    spag <- attr(frame, "spaghetti")
    list(
      grid = grid_columns(frame),
      effects = attr(frame, "effects"),
      response = attr(frame, "response"),
      surface = attr(frame, "surface"),
      points = if (is.null(points)) NULL else grid_columns(points),
      spaghetti_rows = if (is.null(spag)) NULL else nrow(spag),
      spaghetti_samples = if (is.null(spag)) NULL else
        sort(unique(as.character(spag$sample__)))
    )
  })
  list(names = names(ce), effects = unname(out))
}

gauss <- quick(bf(y ~ x * z + f + log(w) + (1 | g), sigma ~ x), data = d)
binom <- quick(ys | trials(k) ~ x + mo(m), data = d, family = binomial())
categ <- quick(o ~ x + f, data = d, family = categorical())
smooth <- quick(y ~ z + s(x, by = f, k = 4), data = d)
mv <- quick(bf(y ~ x + f) + bf(y2 ~ z) + set_rescor(FALSE), data = d)
surface <- quick(y ~ s(x, z, k = 8), data = d)

conds <- make_conditions(d, c("z", "f"))
out <- list(
  data = d[, setdiff(names(d), "o")],
  make_conditions = list(
    z_f = grid_columns(conds),
    w = grid_columns(make_conditions(d, "w"))
  ),
  gauss_default = export_effects(conditional_effects(gauss, resolution = 9)),
  gauss_conditions = export_effects(conditional_effects(
    gauss, effects = c("z:x", "f"), conditions = make_conditions(d, "f"),
    int_conditions = list(z = c(-1, 1)), resolution = 7
  )),
  gauss_named_int = export_effects(conditional_effects(
    gauss, effects = "x:z", int_conditions = list(z = c(low = -1, high = 1)),
    resolution = 5
  )),
  gauss_surface = export_effects(conditional_effects(
    gauss, effects = "x:z", surface = TRUE, resolution = 6, too_far = 0.2
  )),
  gauss_points = export_effects(conditional_effects(
    gauss, effects = "x", conditions = make_conditions(d, c("z", "f")),
    select_points = 0.1, resolution = 4
  )),
  gauss_spaghetti = export_effects(conditional_effects(
    gauss, effects = c("x", "x:f"), spaghetti = TRUE, ndraws = 3,
    resolution = 4
  )),
  gauss_sigma = export_effects(conditional_effects(
    gauss, effects = "x", dpar = "sigma", resolution = 4
  )),
  binom_default = export_effects(conditional_effects(binom, resolution = 5)),
  categorical = export_effects(conditional_effects(
    categ, effects = "x", categorical = TRUE, resolution = 3
  )),
  smooth_default = export_effects(conditional_effects(smooth, resolution = 4)),
  smooths = export_effects(conditional_smooths(smooth, resolution = 4)),
  smooths_flat = export_effects(conditional_smooths(
    smooth, resolution = 4, int_conditions = list(f = c("p", "r"))
  )),
  smooths_surface = export_effects(conditional_smooths(
    surface, resolution = 5, too_far = 0.3
  )),
  smooths_lines = export_effects(conditional_smooths(
    surface, resolution = 5, surface = FALSE
  )),
  mv_default = export_effects(conditional_effects(mv, resolution = 3)),
  mv_resp = export_effects(conditional_effects(mv, resp = "y2", resolution = 3)),
  versions = list(brms = as.character(packageVersion("brms")))
)
writeLines(
  toJSON(out, digits = I(15), auto_unbox = TRUE, dataframe = "columns",
         null = "null", na = "null"),
  "brms_conditional_effects.json"
)
