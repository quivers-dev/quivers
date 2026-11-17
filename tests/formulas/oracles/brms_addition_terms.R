# Generates brms_addition_terms.json, pinned by tests/formulas/test_addition_terms.py:
# brms Stan code of rate() for every family that admits it, threshold counts
# of thres()/cat(), subset() with 0/1 values, and update_adterms() results.
# Run from this directory: Rscript brms_addition_terms.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(y = c(1, 2, 2, 3, 1, 2), x = c(0.1, -0.3, 0.5, 1.2, -0.8, 0.4),
                t = c(1.5, 2, 3, 1, 2.5, 4), keep = c(1, 0, 1, 1, 0, 1))
pick <- function(code) grep("_lpmf|denom", strsplit(code, "\n")[[1]], value = TRUE)
rate <- list(
  poisson_log = pick(make_stancode(y | rate(t) ~ x, d, family = poisson())),
  poisson_identity = pick(make_stancode(y | rate(t) ~ x, d, family = poisson("identity"))),
  negbinomial = pick(make_stancode(y | rate(t) ~ x, d, family = negbinomial())),
  geometric = pick(make_stancode(y | rate(t) ~ x, d, family = geometric()))
)
nthres <- list(
  thres = make_standata(y | thres(4) ~ x, d, family = cumulative())$nthres,
  cat = suppressWarnings(make_standata(y | cat(5) ~ x, d, family = cumulative())$nthres)
)
subset_n <- make_standata(y | subset(keep) ~ x, d, family = gaussian())$N
updates <- list(
  update = deparse1(update_adterms(y | weights(w) + trials(n) ~ x, ~ trials(10) + cens(c))),
  replace = deparse1(update_adterms(y | weights(w) ~ x, ~ cens(c), action = "replace")),
  add = deparse1(update_adterms(y ~ x, ~ rate(t)))
)
writeLines(toJSON(list(rate = rate, nthres = nthres, subset_n = subset_n,
                       updates = updates), auto_unbox = TRUE),
           "brms_addition_terms.json")
