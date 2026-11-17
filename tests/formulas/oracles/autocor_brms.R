# Generates the brms values pinned in tests/formulas/test_autocor.py.
# Run with: Rscript tests/formulas/oracles/autocor_brms.R
suppressMessages(library(brms))
options(digits = 17)
# Unsorted grouped series: brms sorts by group, then time.
d <- data.frame(
  y = c(0.3, -1.2, 0.8, 2.1, -0.4, 1.5, 0.9),
  x = c(1.0, 0.5, -0.2, 0.7, 1.3, -0.9, 0.1),
  t = c(3, 1, 2, 2, 1, 4, 3),
  g = c("b", "a", "b", "a", "b", "a", "a")
)
sd_arma <- make_standata(y ~ x + arma(time = t, gr = g, p = 2, q = 1), data = d)
cat("sorted_y:", sd_arma$Y, "\n")
cat("J_lag:", sd_arma$J_lag, "\n")
sd_cov <- make_standata(y ~ x + arma(time = t, gr = g, p = 1, q = 1, cov = TRUE), data = d)
cat("begin_tg:", sd_cov$begin_tg, "nobs_tg:", sd_cov$nobs_tg, "\n")
sd_unstr <- make_standata(y ~ x + unstr(t, g), data = d)
cat("Jtime_tg:", t(sd_unstr$Jtime_tg), "n_unique_t:", sd_unstr$n_unique_t, "\n")
cat("arma1:", brms:::get_cor_matrix_arma1(0.4, -0.3, 3), "\n")
cat("ar1:", brms:::get_cor_matrix_ar1(0.4, 3), "\n")
cat("ma1:", brms:::get_cor_matrix_ma1(-0.3, 3), "\n")
cat("cosy:", brms:::get_cor_matrix_cosy(0.3, 3), "\n")
# SAR eigenvalue bounds of a row-standardized path graph.
W <- matrix(0, 5, 5)
for (i in 1:4) { W[i, i + 1] <- 1; W[i + 1, i] <- 1 }
W <- W / rowSums(W)
ds <- data.frame(y = c(0.2, -0.5, 1.1, 0.4, -0.3), x = c(0.1, 0.4, -0.6, 1.2, 0.3))
sd_sar <- make_standata(y ~ x + sar(W, type = "lag"), data = ds, data2 = list(W = W))
cat("sar_eigen_min:", min(sd_sar$eigenMsar), "max:", max(sd_sar$eigenMsar), "\n")
pr <- function(f, ...) {
  p <- get_prior(f, ...)
  p <- p[p$class %in% c("ar", "ma", "cosy", "cortime", "lagsar", "errorsar"), ]
  print(as.data.frame(p)[, c("prior", "class", "lb", "ub")])
}
pr(y ~ x + arma(time = t, gr = g, p = 1, q = 1), data = d)
pr(y ~ x + arma(time = t, gr = g, p = 1, q = 1, cov = TRUE), data = d)
pr(y ~ x + cosy(time = t, gr = g), data = d)
pr(y ~ x + unstr(t, g), data = d)
pr(y ~ x + sar(W, type = "error"), data = ds, data2 = list(W = W))
# BYM2 scaling factor of a 4-location path and a 5-cycle (brms .car_scale).
for (K in c(4, 5)) {
  A <- matrix(0, K, K)
  for (i in 1:(K - 1)) { A[i, i + 1] <- 1; A[i + 1, i] <- 1 }
  if (K == 5) { A[1, K] <- 1; A[K, 1] <- 1 }
  rownames(A) <- seq_len(K)
  dc <- data.frame(y = seq_len(K), g = seq_len(K))
  s <- make_standata(y ~ 1 + car(A, gr = g, type = "bym2"), data = dc,
                     data2 = list(A = A), family = poisson())
  cat("car_scale", K, ":", s$car_scale, "\n")
}
