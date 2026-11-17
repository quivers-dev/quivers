# Generates brms_drop_unused_levels.json, pinned by
# tests/formulas/test_drop_unused_levels.py.
# brms oracle for drop_unused_levels: make_standata() population design X,
# group sizes and indexes under TRUE (default) and FALSE, for factors with a
# level no row uses (f: c; h: c) and a grouping factor with one (g: w).
# Run from this directory: Rscript brms_drop_unused_levels.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(
  y = c(0.31, -1.24, 0.85, 1.93, -0.42, 0.13, 1.41, -0.73, 0.62, 2.21, -1.52, 0.94),
  x = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8, 0.4, -0.3),
  f = factor(rep(c("a", "b"), 6), levels = c("a", "b", "c")),
  h = factor(rep(c("a", "b", "e"), 4), levels = c("a", "b", "c", "e")),
  g = factor(rep(c("u", "v"), each = 6), levels = c("u", "v", "w"))
)
formula <- y ~ x * f + h + (1 | g)
standata <- function(drop) {
  s <- make_standata(formula, data = d, drop_unused_levels = drop)
  list(X_names = colnames(s$X), X = unname(lapply(seq_len(nrow(s$X)),
                                                   function(r) unname(s$X[r, ]))),
       N_1 = s$N_1, J_1 = as.integer(s$J_1))
}
writeLines(
  toJSON(list(brms = as.character(packageVersion("brms")),
              data = transform(d, f = as.character(f), h = as.character(h),
                               g = as.character(g)),
              levels = list(f = levels(d$f), h = levels(d$h), g = levels(d$g)),
              drop = standata(TRUE), keep = standata(FALSE)),
         auto_unbox = TRUE, pretty = TRUE, digits = NA),
  "brms_drop_unused_levels.json"
)
