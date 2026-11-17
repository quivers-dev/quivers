# Generates brms_multivariate.json, pinned by tests/formulas/test_multivariate_public.py.
# brms oracle for multivariate naming, log_lik units, and the residual-correlation
# parameterization. Run from this directory: Rscript brms_multivariate.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
set.seed(1)
d <- data.frame(
  y_1 = rnorm(12), y.2 = rnorm(12), k = rbinom(12, 1, 0.5),
  x = rnorm(12), z = rnorm(12), g = rep(c("a", "b", "c"), 4)
)
quick <- function(formula) {
  brm(formula, data = d, algorithm = "meanfield", iter = 200, refresh = 0,
      seed = 1, output_samples = 10)
}
lines <- function(code, pattern) grep(pattern, strsplit(code, "\n")[[1]], value = TRUE)
rescor_formula <- bf(mvbind(y_1, y.2) ~ x + (1 | p | g)) + set_rescor(TRUE)
mixed_formula <- bf(y_1 ~ x + (1 | p | g), sigma ~ z + (1 | p | g)) +
  bf(k ~ x + (1 | p | g), family = bernoulli()) + set_rescor(FALSE)
student_formula <- bf(mvbind(y_1, y.2) ~ x, family = student()) + set_rescor(TRUE)
rescor_fit <- quick(rescor_formula)
mixed_fit <- quick(mixed_formula)
result <- list(
  rescor_variables = setdiff(variables(rescor_fit), c("lprior", "lp__")),
  mixed_variables = setdiff(variables(mixed_fit), c("lprior", "lp__")),
  rescor_log_lik_dim = dim(log_lik(rescor_fit)),
  rescor_log_lik_resp_dim = dim(log_lik(rescor_fit, resp = "y1")),
  mixed_log_lik_dim = dim(log_lik(mixed_fit)),
  mixed_log_lik_resp_dim = dim(log_lik(mixed_fit, resp = "k")),
  rescor_predict_dim = dim(posterior_predict(rescor_fit)),
  rescor_stan = lines(make_stancode(rescor_formula, data = d),
                      "Lrescor|LSigma|multi_normal"),
  student_stan = lines(make_stancode(student_formula, data = d),
                       "Lrescor|Sigma|multi_student|real<lower=1> nu"),
  default_rescor_gaussian = suppressWarnings(suppressMessages(
    brms:::validate_formula(mvbf(bf(y_1 ~ x, family = gaussian()), bf(y.2 ~ x, family = gaussian())), data = d)$rescor)),
  default_rescor_student = suppressWarnings(suppressMessages(
    brms:::validate_formula(mvbf(bf(y_1 ~ x, family = student()),
                                 bf(y.2 ~ x, family = student())), data = d)$rescor)),
  default_rescor_mixed = suppressMessages(
    brms:::validate_formula(mvbf(bf(y_1 ~ x, family = gaussian()), bf(k ~ x, family = bernoulli())),
                            data = d)$rescor),
  responses = names(make_standata(rescor_formula, data = d))[c(1, 3)]
)
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), "brms_multivariate.json")
