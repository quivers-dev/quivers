# Generates bridge_numerics.json, pinned by tests/formulas/test_bridge_evidence.py.
# Oracles for the deterministic parts of the bridgesampling estimator that brms
# calls: coda spectrum0.ar and effectiveSize, Matrix nearPD, the fixed-point
# scheme .run.iterative.scheme (normal and warp3), the restart after
# non-convergence, error_measures of a normal bridge object, and post_prob.
# Inputs are drawn here and exported with 17 significant digits, so the Python
# tests recompute every value from the same numbers.
# Run from this directory: Rscript bridge_numerics.R
suppressMessages(library(bridgesampling))
suppressMessages(library(coda))
suppressMessages(library(Matrix))
suppressMessages(library(jsonlite))

set.seed(20261116)
iid <- rnorm(200)
ar1 <- as.numeric(arima.sim(list(ar = 0.8), n = 500))
short <- cumsum(rnorm(30)) / 5
spectra <- list(
  iid = list(x = iid, spec = spectrum0.ar(iid)$spec, order = spectrum0.ar(iid)$order),
  ar1 = list(x = ar1, spec = spectrum0.ar(ar1)$spec, order = spectrum0.ar(ar1)$order),
  short = list(x = short, spec = spectrum0.ar(short)$spec, order = spectrum0.ar(short)$order)
)

chain1 <- cbind(rnorm(150), as.numeric(arima.sim(list(ar = 0.5), n = 150)), runif(150))
chain2 <- cbind(rnorm(150), as.numeric(arima.sim(list(ar = 0.5), n = 150)), runif(150))
ess <- effectiveSize(mcmc.list(mcmc(chain1), mcmc(chain2)))

not_pd <- matrix(c(1, 0.9, 0.7, 0.9, 1, 0.3, 0.7, 0.3, 1), 3) +
  diag(c(0, 0, -0.5))
pd <- matrix(c(2, 0.3, 0.1, 0.3, 1, 0.2, 0.1, 0.2, 0.5), 3)
near <- list(
  not_pd = list(x = not_pd, mat = as.matrix(nearPD(not_pd)$mat)),
  pd = list(x = pd, mat = as.matrix(nearPD(pd)$mat))
)

# Fixed log densities of a two-dimensional problem with a known logml.
n1 <- 400; n2 <- 400
mu <- c(0.3, -0.2); s <- c(0.5, 0.8)
post <- cbind(rnorm(n1, mu[1], s[1]), rnorm(n1, mu[2], s[2]))
logml_true <- -3.7
log_q <- function(th) logml_true + dnorm(th[, 1], mu[1], s[1], log = TRUE) +
  dnorm(th[, 2], mu[2], s[2], log = TRUE)
m <- c(0.28, -0.15); V <- diag(c(0.3, 0.7))
prop <- cbind(rnorm(n2, m[1], sqrt(V[1, 1])), rnorm(n2, m[2], sqrt(V[2, 2])))
dn <- function(th) dnorm(th[, 1], m[1], sqrt(V[1, 1]), log = TRUE) +
  dnorm(th[, 2], m[2], sqrt(V[2, 2]), log = TRUE)
q11 <- log_q(post); q12 <- dn(post); q21 <- log_q(prop); q22 <- dn(prop)
normal <- bridgesampling:::.run.iterative.scheme(
  q11 = q11, q12 = q12, q21 = q21, q22 = q22, r0 = 0.5, tol = 1e-10,
  L = NULL, method = "normal", maxiter = 1000, silent = TRUE,
  criterion = "r", neff = 350
)
L <- t(chol(V))
warp <- bridgesampling:::.run.iterative.scheme(
  q11 = q11, q12 = q12, q21 = q21, q22 = q22, r0 = 0.5, tol = 1e-10,
  L = L, method = "warp3", maxiter = 1000, silent = TRUE,
  criterion = "r", neff = 350
)
# Too few iterations to converge: the scheme reports the iterates of r.
stalled <- bridgesampling:::.run.iterative.scheme(
  q11 = q11, q12 = q12, q21 = q21, q22 = q22, r0 = 0.5, tol = 1e-10,
  L = NULL, method = "normal", maxiter = 3, silent = TRUE,
  criterion = "r", neff = 350
)
restart <- bridgesampling:::.run.iterative.scheme(
  q11 = q11, q12 = q12, q21 = q21, q22 = q22,
  r0 = sqrt(as.numeric(stalled$r_vals[[length(stalled$r_vals) - 1]]) *
    as.numeric(stalled$r_vals[[length(stalled$r_vals)]])),
  tol = 1e-4, L = NULL, method = "normal", maxiter = 3, silent = TRUE,
  criterion = "logml", neff = 350
)
bridge_object <- structure(
  list(logml = as.numeric(normal$logml), niter = normal$niter, method = "normal",
       q11 = q11, q12 = q12, q21 = q21, q22 = q22),
  class = "bridge"
)
errors <- error_measures(bridge_object)

logml_values <- c(-10.2, -11.7, -9.95)
probs_uniform <- bridgesampling:::.post_prob_calc(
  logml_values, c("a", "b", "c"), NULL
)
probs_prior <- bridgesampling:::.post_prob_calc(
  logml_values, c("a", "b", "c"), c(0.5, 0.3, 0.2)
)

out <- list(
  spectra = spectra,
  ess = list(chain1 = chain1, chain2 = chain2, ess = as.numeric(ess)),
  near_pd = near,
  scheme = list(
    q11 = q11, q12 = q12, q21 = q21, q22 = q22, neff = 350, L = L,
    normal_logml = as.numeric(normal$logml), normal_niter = normal$niter,
    warp_logml = as.numeric(warp$logml), warp_niter = warp$niter,
    stalled_r = sapply(stalled$r_vals, as.numeric), restart_logml = as.numeric(restart$logml),
    restart_niter = restart$niter
  ),
  errors = list(re2 = as.numeric(errors$re2), cv = as.numeric(errors$cv)),
  post_prob = list(
    logml = logml_values, uniform = as.numeric(probs_uniform),
    prior = c(0.5, 0.3, 0.2), with_prior = as.numeric(probs_prior)
  ),
  versions = list(
    bridgesampling = as.character(packageVersion("bridgesampling")),
    coda = as.character(packageVersion("coda")),
    Matrix = as.character(packageVersion("Matrix"))
  )
)
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE, matrix = "rowmajor"),
           "bridge_numerics.json")
