# Generates loo_psis_pathfinder.json, pinned by
# tests/formulas/test_pathfinder.py.
# loo oracle for the Pareto-smoothed importance weights of multi-path
# Pathfinder's resampling step: loo::psis() with r_eff = 1 on fixed log
# importance ratios (heavy-tailed, light-tailed, and too few for a tail fit).
# Run from this directory: Rscript loo_psis_pathfinder.R
suppressMessages(library(loo))
suppressMessages(library(jsonlite))
set.seed(20221)
cases <- list(
  heavy = log(abs(rt(400, df = 2))) + rnorm(400, sd = 0.5),
  light = rnorm(1000, sd = 0.3),
  small = c(-0.4, 0.1, 0.3, -1.2, 0.8, 0.05, -0.3, 0.2, 0.6, -0.9,
            0.15, -0.05, 0.4, -0.7, 0.25, 0.35, -0.15, 0.45, -0.25, 0.55)
)
out <- lapply(cases, function(ratios) {
  fit <- suppressWarnings(psis(ratios, r_eff = 1))
  lw <- weights(fit, log = TRUE, normalize = TRUE)
  list(log_ratios = ratios, log_weights = as.numeric(lw),
       pareto_k = as.numeric(pareto_k_values(fit)))
})
out$loo_version <- as.character(packageVersion("loo"))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE),
           "loo_psis_pathfinder.json")
