# Generates brms_prior_table_options.json, pinned by
# tests/formulas/test_prior_table_options.py: the rows of brms::get_prior under
# drop_unused_levels and knots, and where brms declares an equated mixture
# parameter.
# Run from this directory: Rscript brms_prior_table_options.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
rows <- 0:23
d <- data.frame(
  x = (rows * 0.618034) %% 1,
  f = factor(c("a", "b")[rows %% 2 + 1], levels = c("a", "b", "c")),
  g = factor(c("u", "v")[rows %/% 12 + 1], levels = c("u", "v", "w"))
)
d$y <- sin(3 * d$x) + 0.5 * (d$f == "b") + 0.1 * cos(7 * rows)
table_of <- function(...) {
  p <- as.data.frame(get_prior(...))
  p[, c("class", "coef", "group")]
}
knots <- list(x = c(0, 0.2, 0.5, 0.8, 1))
out <- list(
  data = d,
  dropped = table_of(y ~ x + f + (1 | g), data = d),
  kept = table_of(y ~ x + f + (1 | g), data = d, drop_unused_levels = FALSE),
  smooth = table_of(y ~ s(x, bs = "cr", k = 5) + f, data = d),
  smooth_knots = table_of(y ~ s(x, bs = "cr", k = 5) + f, data = d, knots = knots),
  kept_levels = make_standata(y ~ x + f + (1 | g), data = d, drop_unused_levels = FALSE)$N_1,
  dropped_levels = make_standata(y ~ x + f + (1 | g), data = d)$N_1
)
code <- make_stancode(bf(y ~ x, sigma2 = "sigma1"), data = d,
                      family = mixture(gaussian, gaussian))
lines <- strsplit(code, "\n")[[1]]
blocks <- grep("^[a-z ]+[{]", lines)
declared <- grep("real sigma2;", lines, fixed = TRUE)
out$equated_block <- trimws(sub("[{]", "", lines[max(blocks[blocks < declared])]))
out$equated_assignment <- trimws(grep("sigma2 = sigma1", lines, value = TRUE))
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, dataframe = "columns"),
           "brms_prior_table_options.json")
