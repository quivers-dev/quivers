# Generates brms_postprocess.json, pinned by tests/formulas/test_postprocess_tables.py
# and tests/formulas/test_postprocess_loo.py.
# brms and loo oracles for post-processing: parameter names of two fitted
# models, brms's summary/fixef/ranef/coef/VarCorr/vcov/posterior_summary/
# posterior_interval/predictive_interval/posterior_table tables computed on
# the exported draws, rhat and neff_ratio, and loo/waic/loo_compare/E_loo/
# loo_subsample/loo_model_weights results on the exported pointwise
# log-likelihoods. Every value is computed by brms or loo from the exported
# draws, so the Python tests recompute it from the same numbers.
# Run from this directory: Rscript brms_postprocess.R
suppressMessages(library(brms))
suppressMessages(library(loo))
suppressMessages(library(posterior))
suppressMessages(library(jsonlite))

d <- data.frame(
  y = c(0.31, -1.24, 0.85, 1.93, -0.42, 0.13, 1.41, -0.73, 0.62, 2.21,
        -1.52, 0.94, 0.27, -0.66, 1.18, 0.05, -0.91, 1.67, 0.44, -0.18,
        0.72, 1.36, -1.05, 0.58, 0.09, -0.37, 1.02, 0.66, -0.84, 0.21),
  x = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8,
        0.4, -0.3, 1.1, -0.6, 0.2, 0.8, -1.2, 1.5, -0.1, 0.7,
        -0.9, 1.3, -0.5, 0.1, 0.6, -1.0, 0.9, -0.4, 1.0, -0.7),
  z = c(-0.2, 0.4, 1.1, -0.9, 0.3, -0.5, 0.8, 1.4, -1.3, 0.2,
        0.6, -0.8, -0.1, 1.2, -0.6, 0.5, 0.9, -1.1, 0.0, 0.7,
        -0.4, 1.0, 0.3, -0.7, 1.3, -0.2, 0.1, -1.0, 0.4, 0.6),
  g = rep(c("a", "b", "c"), 10),
  f = factor(rep(c("p", "q"), 15))
)
d$r <- factor(c(1, 0, 2, 3, 1, 1, 2, 0, 2, 3, 0, 2, 1, 0, 2, 1, 0, 3, 1, 1,
                2, 3, 0, 2, 1, 1, 2, 2, 0, 1), levels = 0:3, ordered = TRUE)

fit_model <- function(formula, family) {
  # brms 2.20.4 emits pre-2.33 Stan array syntax for some families; the
  # generated program is rewritten to current syntax, compiled with cmdstanr,
  # and its draws are attached to an empty brmsfit, as brms itself does.
  empty <- brm(formula, data = d, family = family, empty = TRUE)
  code <- gsub("int\\[\\] ([A-Za-z_]+)", "array[] int \\1", stancode(empty))
  model <- cmdstanr::cmdstan_model(cmdstanr::write_stan_file(code))
  draws <- model$sample(
    data = unclass(standata(empty)), chains = 2, iter_warmup = 150,
    iter_sampling = 150, seed = 11, refresh = 0, adapt_delta = 0.95,
    show_messages = FALSE
  )
  empty$fit <- brms:::read_csv_as_stanfit(draws$output_files())
  empty$backend <- "cmdstanr"
  rename_pars(empty)
}

exact <- function(x) {
  # jsonlite prints at most 15 significant digits; draws are exported as
  # 17-digit strings so that every double, and every tie, round-trips.
  out <- array(sprintf("%.17g", x), dim = dim(x))
  out
}

array_list <- function(x) {
  # a 3-way array as rows x columns x components, keeping all dimnames
  list(dim = dim(x), dimnames = dimnames(x), values = as.vector(x))
}

table_list <- function(x) {
  list(rows = rownames(x), columns = colnames(x), values = unname(as.matrix(x)))
}

models <- list(
  gaussian = list(formula = bf(y ~ x + f + (1 + x | g), sigma ~ z), family = gaussian()),
  cumulative = list(formula = bf(r ~ x + (1 | g)), family = cumulative())
)

result <- lapply(models, function(spec) {
  fit <- fit_model(spec$formula, spec$family)
  draws <- as_draws_array(fit)
  variables <- variables(fit)
  keep <- setdiff(variables, c("lprior", "lp__"))
  s <- summary(fit, prob = 0.9)
  re <- ranef(fit, probs = c(0.1, 0.9))
  cf <- coef(fit, robust = TRUE)
  vc <- VarCorr(fit)
  out <- list(
    variables = variables,
    draws = lapply(setNames(keep, keep), function(v) exact(unname(as.matrix(extract_variable_matrix(draws, v))))),
    summary = list(
      fixed = table_list(s$fixed),
      spec_pars = table_list(s$spec_pars),
      random = lapply(s$random, table_list)
    ),
    fixef = table_list(fixef(fit)),
    fixef_robust = table_list(fixef(fit, robust = TRUE, probs = c(0.05, 0.5, 0.95))),
    ranef = lapply(re, array_list),
    coef = lapply(cf, array_list),
    varcorr = lapply(vc, function(block) lapply(block, function(x) {
      if (length(dim(x)) == 3) array_list(x) else table_list(x)
    })),
    vcov = table_list(vcov(fit)),
    vcov_cor = table_list(vcov(fit, correlation = TRUE)),
    posterior_summary = table_list(posterior_summary(fit, variable = keep)),
    posterior_interval = table_list(posterior_interval(fit, variable = keep, prob = 0.8)),
    rhat = as.list(rhat(fit)[keep]),
    neff_ratio = as.list(neff_ratio(fit)[keep]),
    nobs = nobs(fit),
    ngrps = ngrps(fit),
    log_lik = exact(unname(log_lik(fit)))
  )
  if (identical(fit$family$family, "cumulative")) {
    pp <- posterior_predict(fit)[, 1:6]
    out$predict_draws = exact(unname(pp))
    out$posterior_table = table_list(posterior_table(pp, levels = 1:4))
  } else {
    pp <- posterior_predict(fit)
    out$predictive_interval = table_list(predictive_interval(pp, prob = 0.8))
    out$predict_draws = exact(unname(pp[, 1:5]))
    out$predict_summary = table_list(posterior_summary(pp[, 1:5], probs = c(0.1, 0.9)))
  }
  out
})

# loo-package oracles on the exported pointwise log-likelihoods. r_eff = 1
# isolates the estimators from relative-efficiency estimation.
ll1 <- matrix(as.numeric(result$gaussian$log_lik), nrow = nrow(result$gaussian$log_lik))
ll2 <- sweep(ll1, 2, 0.3 * sin(seq_len(ncol(ll1))), "+")
l1 <- loo(ll1, r_eff = rep(1, ncol(ll1)))
l2 <- loo(ll2, r_eff = rep(1, ncol(ll2)))
w1 <- waic(ll1)
cmp <- loo_compare(list(m1 = l1, m2 = l2))
psis1 <- psis(-ll1, r_eff = rep(1, ncol(ll1)))
yrep <- matrix(as.numeric(result$gaussian$predict_draws), nrow = nrow(result$gaussian$predict_draws))
e_mean <- E_loo(yrep, psis_object = psis(-ll1[, 1:5], r_eff = rep(1, 5)), type = "mean", log_ratios = -ll1[, 1:5])
e_var <- E_loo(yrep, psis_object = psis(-ll1[, 1:5], r_eff = rep(1, 5)), type = "var", log_ratios = -ll1[, 1:5])
e_q <- E_loo(yrep, psis_object = psis(-ll1[, 1:5], r_eff = rep(1, 5)), type = "quantile", probs = c(0.1, 0.9), log_ratios = -ll1[, 1:5])
weights_stacking <- loo_model_weights(list(ll1, ll2), method = "stacking", r_eff_list = list(rep(1, 30), rep(1, 30)))
weights_pbma <- loo_model_weights(list(ll1, ll2), method = "pseudobma", BB = FALSE, r_eff_list = list(rep(1, 30), rep(1, 30)))

# loo_subsample through the function interface: an intercept-only Normal
# model with exported draws of (mu, sigma).
sub_draws <- cbind(
  mu = 0.25 + 0.1 * sin(seq_len(400) * 0.37),
  sigma = 1.0 + 0.08 * cos(seq_len(400) * 0.53)
)
sub_y <- data.frame(y = d$y)
llfun <- function(data_i, draws) dnorm(data_i$y, draws[, "mu"], draws[, "sigma"], log = TRUE)
sub_obs <- c(2L, 5L, 9L, 14L, 17L, 21L, 26L, 29L)
sub_plpd <- loo_subsample(llfun, data = sub_y, draws = sub_draws, observations = sub_obs,
                          loo_approximation = "plpd", r_eff = rep(1, 30))
sub_lpd <- loo_subsample(llfun, data = sub_y, draws = sub_draws, observations = sub_obs,
                         loo_approximation = "lpd", r_eff = rep(1, 30))
sub_full <- loo(llfun, data = sub_y, draws = sub_draws, r_eff = rep(1, 30))

result$loo <- list(
  loo = list(estimates = table_list(l1$estimates), pointwise = table_list(l1$pointwise),
             pareto_k = l1$diagnostics$pareto_k),
  loo2 = list(estimates = table_list(l2$estimates), pointwise = table_list(l2$pointwise)),
  waic = list(estimates = table_list(w1$estimates), pointwise = table_list(w1$pointwise)),
  compare = table_list(cmp),
  psis_log_weights_col1 = as.vector(weights(psis1, log = TRUE, normalize = TRUE)[, 1]),
  psis_log_weights_5 = exact(weights(psis(-ll1[, 1:5], r_eff = rep(1, 5)), log = TRUE, normalize = TRUE)),
  psis_pareto_k_5 = psis(-ll1[, 1:5], r_eff = rep(1, 5))$diagnostics$pareto_k,
  e_loo = list(mean = e_mean$value, mean_k = e_mean$pareto_k,
               var = e_var$value, var_k = e_var$pareto_k,
               quantile = unname(e_q$value), quantile_k = e_q$pareto_k),
  weights = list(stacking = as.vector(weights_stacking), pseudobma = as.vector(weights_pbma)),
  subsample = list(
    draws = unname(sub_draws), y = d$y, observations = sub_obs,
    plpd = list(estimates = table_list(sub_plpd$estimates)),
    lpd = list(estimates = table_list(sub_lpd$estimates)),
    full = list(estimates = table_list(sub_full$estimates))
  )
)

result$data <- d
result$versions <- list(
  brms = as.character(packageVersion("brms")),
  loo = as.character(packageVersion("loo")),
  posterior = as.character(packageVersion("posterior"))
)
write_json(result, "brms_postprocess.json", digits = NA, auto_unbox = TRUE, pretty = FALSE)
