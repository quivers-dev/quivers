# Generates brms_missing_slopes.json, pinned by
# tests/formulas/test_missing_varying_slopes.py: brms's prior rows and Stan
# predictor statements for mi() terms used as varying slopes.
# Run from this directory: Rscript brms_missing_slopes.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
# Dyadic values, exact in single precision.
d <- data.frame(
  y = c(0.125, 0.375, -0.25, 0.25, 0.5, 0.0, 0.75, -0.125, 0.25, 0.625, -0.375, 0.25),
  x = c(1.0, NA, 0.5, NA, 2.0, 1.5, NA, 0.25, 0.875, 1.25, 0.375, 1.75),
  z = c(0.25, 0.125, -0.25, 0.375, 0.0, 0.25, -0.25, 0.625, 0.5, -0.375, 0.125, -0.125),
  w = c(0.5, -0.25, 0.25, 0.75, -0.625, 0.125, 0.375, -0.25, 0.75, 0.25, -0.5, 0.625),
  g = rep(c("a", "b", "c"), 4)
)
describe <- function(formula) {
  prior <- as.data.frame(get_prior(formula, d))
  prior <- prior[prior$class %in% c("sd", "b") & nzchar(prior$coef),
                 c("class", "coef", "group", "resp")]
  lines <- strsplit(make_stancode(formula, d), "\n")[[1]]
  list(prior = prior, predictor = trimws(grep("mu_y\\[n\\] \\+=", lines, value = TRUE)))
}
out <- list(
  data = d,
  slope = describe(bf(y ~ mi(x) + (mi(x) | g)) + bf(x | mi() ~ z) + set_rescor(FALSE)),
  slope_only = describe(bf(y ~ mi(x) + (0 + mi(x) | g)) + bf(x | mi() ~ z) + set_rescor(FALSE)),
  interaction = describe(bf(y ~ mi(x) * w + (1 + mi(x):w | g)) + bf(x | mi() ~ z) + set_rescor(FALSE)),
  orphan = describe(bf(y ~ z + (mi(x) | g)) + bf(x | mi() ~ z) + set_rescor(FALSE))
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, dataframe = "columns", na = "null"),
           "brms_missing_slopes.json")
