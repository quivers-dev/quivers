# brms Gaussian-process oracle for tests/formulas/test_gp.py and test_gp_exact.py.
# Rscript tests/formulas/oracles/gp_brms.R > tests/formulas/oracles/gp_brms.json
# Generated with brms 2.20.4; the standata and default priors of approximate
# gp() terms are unchanged at brms commit b124949c (2.23.0), whose kernels
# beyond exp_quad change only the spectral density. brms 2.20.4 has no
# default for c, so the cases state the 2.23.0 default c = 5/4 explicitly.
suppressMessages({library(brms); library(jsonlite)})
n <- 24
i <- seq_len(n)
d <- data.frame(
  x = round(((i * 0.618034) %% 1) * 4 - 1, 3),
  z = round(((i * 0.414214) %% 1) * 2, 3),
  w = round(0.5 + ((i * 0.7548777) %% 1), 3),
  g = factor(c("a", "b", "c")[(i %% 3) + 1])
)
d$x[c(5, 17)] <- d$x[3]
d$z[c(5, 17)] <- d$z[3]
d$y <- round(3 * sin(2 * d$x) + d$z + ((i * 0.3) %% 1), 3)
d$y2 <- 10 * d$y
specs <- c(
  "y ~ gp(x, k = 5, c = 5/4)",
  "y ~ gp(x, k = 6, c = 1.5, gr = FALSE)",
  "y ~ gp(x, z, k = 4, c = c(1.5, 2))",
  "y ~ gp(x, z, k = 3, c = 5/4, iso = FALSE)",
  "y ~ gp(x, by = g, k = 5, c = 5/4)",
  "y ~ gp(x, by = g, k = 4, c = 5/4, cmc = FALSE)",
  "y ~ gp(x, by = w, k = 5, c = 5/4)",
  "y ~ gp(x, k = 5, c = 5/4, scale = FALSE)",
  "y2 ~ gp(x, k = 5, c = 5/4)"
)
mat <- function(v) if (length(dim(v)) != 2) as.numeric(v) else unname(lapply(seq_len(nrow(v)), function(r) as.numeric(v[r, ])))
out <- list(data = as.list(d[, c("x", "z", "w", "y", "y2")]), g = as.character(d$g), cases = list())
for (f in specs) {
  sd <- make_standata(as.formula(f), data = d)
  keep <- grep("gp|slambda", names(sd), value = TRUE)
  stan <- lapply(sd[keep], mat)
  pr <- get_prior(as.formula(f), data = d)
  pr <- pr[pr$class %in% c("sdgp", "lscale"), c("prior", "class", "coef")]
  out$cases[[length(out$cases) + 1]] <- list(
    formula = f, standata = stan,
    priors = unname(lapply(seq_len(nrow(pr)), function(r) as.list(pr[r, ])))
  )
}
# Exact GPs (no k): brms's distinct scaled inputs Xgp, row map Jgp, counts
# Nsubgp, and the default priors, used by tests/formulas/test_gp_exact.py.
# The exact standata and priors are unchanged at commit b124949c (2.23.0),
# whose data_gp and frame_gp read k = NA as in 2.20.4. The Stan text records
# brms's 1e-12 diagonal and the non-centered cholesky_decompose(cov) * zgp.
exact_specs <- c(
  "y ~ gp(x)",
  "y ~ gp(x, gr = FALSE)",
  "y ~ gp(x, z)",
  "y ~ gp(x, z, iso = FALSE)",
  "y ~ gp(x, by = g)",
  "y ~ gp(x, by = g, cmc = FALSE)",
  "y ~ gp(x, by = w)",
  "y ~ gp(x, scale = FALSE)",
  "y ~ gp(x, by = g, gr = FALSE)"
)
out$exact_cases <- list()
for (f in exact_specs) {
  sd <- make_standata(as.formula(f), data = d)
  keep <- grep("gp", names(sd), value = TRUE)
  stan <- lapply(sd[keep], mat)
  pr <- get_prior(as.formula(f), data = d)
  pr <- pr[pr$class %in% c("sdgp", "lscale"), c("prior", "class", "coef")]
  out$exact_cases[[length(out$exact_cases) + 1]] <- list(
    formula = f, standata = stan,
    priors = unname(lapply(seq_len(nrow(pr)), function(r) as.list(pr[r, ])))
  )
}
code <- make_stancode(y ~ gp(x), data = d)
out$exact_stan <- list(
  jitter = grepl("cov[n, n] += 1e-12;", code, fixed = TRUE),
  noncentered = grepl("return cholesky_decompose(cov) * zgp;", code, fixed = TRUE),
  zgp = grepl("vector[Nsubgp_1] zgp_1;", code, fixed = TRUE),
  # brms 2.20.4 names the exp_quad function gp(); 2.23.0 names it gp_exp_quad().
  call = grepl("gp_pred_1 = gp(Xgp_1, sdgp_1[1], lscale_1[1], zgp_1);", code, fixed = TRUE)
)
cat(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null"))
