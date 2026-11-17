# Generates brms_vector_constants.json, pinned by
# tests/formulas/test_vector_constant_priors.py. brms fixes a parameter vector
# to a vector with constant(v, broadcast = FALSE): stan_constant_prior() of
# brms 2.23.0 emits `<par> = <v>` for the whole vector (b, b_<dpar>, sd_<id>),
# so entry k of v is the constant of the k-th coefficient of that vector. This
# script records brms's coefficient order of each vector: the columns of the
# population design X without the intercept, and the sd rows of each group.
# Run from this directory: Rscript brms_vector_constants.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
r <- 0:23
d <- data.frame(
  x = sin(r * 0.7), z = cos(r * 1.3), w = 1 + (r %% 3) * 0.5,
  f = c("u", "v", "t")[r %% 3 + 1], g = c("a", "b", "c", "d")[r %% 4 + 1],
  h = c("p", "q")[(r %/% 4) %% 2 + 1]
)
d$y <- 1 + 0.5 * d$x - 0.3 * d$z + 0.1 * cos(r * 2.1)
cases <- list(
  list(formula = "y ~ x + z"),
  list(formula = "y ~ x + f"),
  list(formula = "y ~ x * z + w"),
  list(formula = "y ~ x + (1 + x | g)"),
  list(formula = "y ~ x + (1 + x + z | g) + (1 | h)"),
  list(formula = "y ~ x", sigma = "sigma ~ x + z")
)
result <- lapply(cases, function(case) {
  formula <- if (is.null(case$sigma)) bf(as.formula(case$formula)) else
    bf(as.formula(case$formula), as.formula(case$sigma))
  standata <- make_standata(formula, data = d)
  prior <- get_prior(formula, data = d)
  sd <- prior[prior$class == "sd" & nzchar(prior$coef), ]
  out <- list(formula = case$formula, b = setdiff(colnames(standata$X), "Intercept"),
              sd = lapply(split(sd$coef, factor(sd$group, unique(sd$group))), as.list))
  if (!is.null(case$sigma)) {
    out$sigma <- sub("sigma ", "", case$sigma)
    out$b_sigma <- setdiff(colnames(standata$X_sigma), "Intercept")
  }
  out
})
writeLines(
  toJSON(list(brms = as.character(packageVersion("brms")), data = d, cases = result),
         auto_unbox = TRUE, pretty = TRUE, digits = NA, dataframe = "columns"),
  "brms_vector_constants.json"
)
