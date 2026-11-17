# brms's own truncated expectations (brms:::posterior_epred_trunc_<family>)
# at fixed parameters, pinned in tests/continuous/test_truncated_mean.py.
# Each function reads prep$dpars and the bounds as draws-by-observations
# matrices; one draw and one observation per case is enough.
suppressMessages(library(brms))
m <- function(x) matrix(x, nrow = 1)
show <- function(name, value) cat(name, " ", format(as.numeric(value), digits = 17), "\n", sep = "")
cases <- list(c(0.5, 2), c(0.5, Inf), c(-Inf, 0.8), c(3, Inf))
for (b in cases) {
  lb <- m(b[1]); ub <- m(b[2]); tag <- paste0("[", b[1], ",", b[2], "]")
  show(paste0("gaussian", tag), brms:::posterior_epred_trunc_gaussian(
    list(dpars = list(mu = m(0.3), sigma = m(1.2))), lb, ub))
  show(paste0("student", tag), brms:::posterior_epred_trunc_student(
    list(dpars = list(mu = m(0.3), sigma = m(1.2), nu = m(3))), lb, ub))
  show(paste0("lognormal", tag), brms:::posterior_epred_trunc_lognormal(
    list(dpars = list(mu = m(0.2), sigma = m(0.7))), lb, ub))
  show(paste0("gamma", tag), brms:::posterior_epred_trunc_gamma(
    list(dpars = list(mu = m(1.4), shape = m(2.5))), lb, ub))
  show(paste0("exponential", tag), brms:::posterior_epred_trunc_exponential(
    list(dpars = list(mu = m(1.4))), lb, ub))
  show(paste0("weibull", tag), brms:::posterior_epred_trunc_weibull(
    list(dpars = list(mu = m(1.4), shape = m(1.7))), lb, ub))
}
# Counts: brms sums x * pmf(x) over lb < x <= ub. Its array code needs at
# least two draws and two observations; all four cells are the same case.
w <- function(x) matrix(x, nrow = 2, ncol = 2)
first <- function(x) x[1, 1]
lb <- w(1); ub <- w(6)
show("poisson(1,6]", first(suppressMessages(brms:::posterior_epred_trunc_discrete(
  "pois", list(lambda = w(3.2)), lb, ub))))
show("negbinomial(1,6]", first(suppressMessages(brms:::posterior_epred_trunc_discrete(
  "nbinom", list(mu = w(3.2), size = w(2.5)), lb, ub))))
show("geometric(1,6]", first(suppressMessages(brms:::posterior_epred_trunc_discrete(
  "nbinom", list(mu = w(3.2), size = w(1)), lb, ub))))
show("binomial(1,6]", first(suppressMessages(brms:::posterior_epred_trunc_discrete(
  "binom", list(size = w(9), prob = w(0.35)), lb, ub))))
