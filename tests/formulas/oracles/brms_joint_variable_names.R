# Generates brms_joint_variable_names.json, pinned by
# tests/formulas/test_joint_variable_names.py.
# brms oracle for the parameter names of a joint model whose design labels
# need brms's renaming: factor levels (f[q] -> fq), interactions (x:f[q] ->
# x:fq), I() and poly() columns, and a factor in a varying slope.
# Run from this directory: Rscript brms_joint_variable_names.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(
  y = c(0.31, -1.24, 0.85, 1.93, -0.42, 0.13, 1.41, -0.73, 0.62, 2.21,
        -1.52, 0.94, 0.27, -0.66, 1.18, 0.05, -0.91, 1.67, 0.44, -0.18),
  y2 = c(1.2, 0.4, -0.3, 2.2, 0.9, 1.1, -0.8, 0.2, 1.7, 0.6,
         -0.1, 1.4, 0.8, 0.3, 2.6, -0.5, 1.0, 0.7, 1.9, 0.0),
  x = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8,
        0.4, -0.3, 1.1, -0.6, 0.2, 0.8, -1.2, 1.5, -0.1, 0.7),
  f = factor(rep(c("p", "q"), 10)),
  h = factor(rep(c("u", "v", "w", "u", "v"), 4)),
  g = rep(c("a", "b", "c", "d"), 5)
)
formula <- bf(y ~ x * f + (1 + f | g)) +
  bf(y2 ~ I(x^2) + h, sigma ~ f) + set_rescor(FALSE)
fit <- brm(formula, data = d, algorithm = "meanfield", iter = 200,
           refresh = 0, seed = 1, output_samples = 10)
writeLines(
  toJSON(list(brms = as.character(packageVersion("brms")), data = d,
              variables = setdiff(variables(fit), c("lprior", "lp__"))),
         auto_unbox = TRUE, pretty = TRUE, digits = NA),
  "brms_joint_variable_names.json"
)
