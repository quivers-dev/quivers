# Generates brms_vignette_spelling.json, pinned by
# tests/formulas/test_brms_spelling.py: the family names brms accepts and how
# it resolves their spellings; brms's coding of integer and ordered-factor
# ordinal responses (make_standata Y and nthres, with thres() and thres(gr));
# R's model.matrix for ordered-factor predictors (contr.poly) with brms's
# coefficient names; and brms's hypothesis() table and density ratio on fixed
# draws.
#
# Run from this directory: Rscript brms_vignette_spelling.R
suppressMessages({
  library(brms)
  library(jsonlite)
})

out <- list(brms_version = as.character(packageVersion("brms")))

# -- family names -------------------------------------------------------------
ns <- asNamespace("brms")
registered <- sub("^\\.family_", "", ls(ns, all.names = TRUE, pattern = "^\\.family_"))
out$families <- setdiff(registered, c("info", "custom"))
spellings <- c(
  "normal", "gaussian", "student", "Gamma", "gamma", "Beta", "beta",
  "inverse.gaussian", "negbinomial", "NegBinomial", "zi_poisson",
  "zi_negbinomial", "hu_gamma", "hu_lognormal", "bernoulli", "Cumulative"
)
out$resolved <- lapply(
  setNames(spellings, spellings),
  function(name) brms:::validate_family(name)$family
)
stats_families <- list(
  gaussian = gaussian(), binomial = binomial(), poisson = poisson(),
  Gamma = Gamma(), inverse.gaussian = inverse.gaussian()
)
out$stats_families <- lapply(stats_families, function(f) brms:::validate_family(f)$family)

# -- ordinal response coding --------------------------------------------------
d <- data.frame(x = seq(-1, 1, length.out = 12), g = rep(c("a", "b"), 6))
d$y14 <- rep(1:4, 3)
d$y03 <- rep(0:3, 3)
d$y245 <- rep(c(2, 4, 5), 4)
d$yg <- c(1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1, 4)
d$yo <- factor(rep(c("lo", "mid", "hi", "top"), 3),
  levels = c("lo", "mid", "hi", "top"), ordered = TRUE
)
standata <- function(formula, family) {
  tryCatch(
    {
      s <- make_standata(formula, d, family = family)
      list(Y = as.integer(s$Y), nthres = as.integer(s$nthres), ncat = s$ncat)
    },
    error = function(e) list(error = conditionMessage(e))
  )
}
out$ordinal <- list(
  y14_cumulative = standata(y14 ~ x, cumulative()),
  y03_cumulative = standata(y03 ~ x, cumulative()),
  y245_cumulative = standata(y245 ~ x, cumulative()),
  yo_cumulative = standata(yo ~ x, cumulative()),
  y14_sratio = standata(y14 ~ x, sratio()),
  y14_cratio = standata(y14 ~ x, cratio()),
  y14_acat = standata(y14 ~ x, acat()),
  y14_thres_gr = standata(y14 | thres(gr = g) ~ x, cumulative()),
  yg_thres_gr = standata(yg | thres(gr = g) ~ x, cumulative()),
  y14_thres3 = standata(y14 | thres(3) ~ x, cumulative()),
  y14_thres5 = standata(y14 | thres(5) ~ x, cumulative()),
  y14_thres2 = standata(y14 | thres(2) ~ x, cumulative()),
  y03_hurdle = standata(y03 ~ x, hurdle_cumulative()),
  y14_hurdle = standata(y14 ~ x, hurdle_cumulative()),
  y14_categorical = standata(y14 ~ x, categorical()),
  y245_categorical = standata(y245 ~ x, categorical())
)

# -- ordered-factor predictors ------------------------------------------------
lev <- c("lo", "mid", "hi", "top", "max")
p <- data.frame(
  o = factor(lev[c(1, 2, 3, 4, 5, 2, 3, 1, 5, 4)], levels = lev, ordered = TRUE),
  z = c(0.5, -1, 2, 0.25, 1.5, -0.5, 3, 1, -2, 0.75),
  f = factor(rep(c("a", "b"), 5))
)
p$y <- p$z + 1
design <- function(formula) {
  m <- model.matrix(formula, p)
  list(names = colnames(m), matrix = unname(m))
}
out$poly <- list(
  main = design(~o),
  no_intercept = design(~ 0 + o),
  interaction = design(~ o * z),
  contr = lapply(
    setNames(c(2, 3, 4, 6), c("2", "3", "4", "6")),
    function(n) list(names = colnames(contr.poly(n)), matrix = unname(contr.poly(n)))
  ),
  brms_names = colnames(make_standata(y ~ o * z, p)$X),
  brms_group_names = get_prior(y ~ o + (o | f), p)$coef[
    get_prior(y ~ o + (o | f), p)$class == "sd"
  ]
)

# -- hypothesis ---------------------------------------------------------------
i <- 1:400
draws <- data.frame(
  b_a = sin(i * 1.3) + 0.2 * cos(i * 0.7) + 0.3,
  b_b = cos(i * 0.9) * 0.5 - 0.1 + 0.001 * i,
  sigma = exp(0.3 * sin(i * 0.37))
)
prior <- 1.5 * sin(i * 0.51) + 0.4 * cos(i * 1.9)
table <- function(h) {
  h <- h$hypothesis
  h$Evid.Ratio[is.na(h$Evid.Ratio)] <- NA
  as.list(h)
}
strings <- c(
  "b_a > 0", "b_a + b_b = 0", "b_a - b_b < 0.2", "exp(b_a) > sigma^2",
  "b_a^2 / (b_a^2 + sigma^2) = 0.5"
)
out$hypothesis <- list(
  strings = strings,
  alpha = 0.1,
  table = table(brms:::hypothesis.default(draws, strings, alpha = 0.1)),
  robust = table(brms:::hypothesis.default(draws, c(first = "b_a * 2 = b_b"), robust = TRUE)),
  density_ratio = brms:::density_ratio(draws$b_a + draws$b_b, prior, point = 0),
  density = list(
    a = brms:::density_ratio(draws$b_a, point = 0),
    prior = brms:::density_ratio(prior, point = 0),
    below = brms:::density_ratio(draws$sigma, point = 0.6),
    above = brms:::density_ratio(draws$sigma, point = 1.45)
  )
)

write(
  toJSON(out, digits = NA, auto_unbox = TRUE, na = "null", pretty = TRUE),
  "brms_vignette_spelling.json"
)
