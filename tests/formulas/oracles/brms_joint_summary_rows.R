# Generates brms_joint_summary_rows.json, pinned by
# tests/formulas/test_postprocess_tables.py::test_joint_fit_tables.
# brms oracle for the row order of summary() of a joint model: the population
# effects (every intercept before the other coefficients), the group-level
# standard deviations, the family parameters and the residual correlation.
# The rows do not depend on the algorithm or the draws, so the cheapest
# variational fit is used.
# Run from this directory: Rscript brms_joint_summary_rows.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(
  y = c(0.31, -1.24, 0.85, 1.93, -0.42, 0.13, 1.41, -0.73, 0.62, 2.21,
        -1.52, 0.94, 0.27, -0.66, 1.18, 0.05, -0.91, 1.67, 0.44, -0.18),
  y2 = c(1.2, 0.4, -0.3, 2.2, 0.9, 1.1, -0.8, 0.2, 1.7, 0.6,
         -0.1, 1.4, 0.8, 0.3, 2.6, -0.5, 1.0, 0.7, 1.9, 0.0),
  x = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8,
        0.4, -0.3, 1.1, -0.6, 0.2, 0.8, -1.2, 1.5, -0.1, 0.7),
  g = rep(c("a", "b", "c", "d"), 5)
)
formula <- "bf(y ~ x + (1 | g)) + bf(y2 ~ x) + set_rescor(TRUE)"
fit <- brm(eval(parse(text = formula)), data = d, algorithm = "meanfield",
           iter = 200, init = 0, refresh = 0, seed = 1, output_samples = 10)
s <- suppressWarnings(summary(fit))
writeLines(
  toJSON(list(
    brms = as.character(packageVersion("brms")), formula = formula, data = d,
    fixed = rownames(s$fixed), spec_pars = rownames(s$spec_pars),
    rescor_pars = rownames(s$rescor_pars),
    random = lapply(s$random, rownames)
  ), auto_unbox = FALSE, pretty = TRUE, digits = NA),
  "brms_joint_summary_rows.json"
)
