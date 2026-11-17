# Generates brms_membership_covariance.json, pinned by
# tests/formulas/test_membership_covariance.py: brms's mm(..., cov = K) terms.
# For each formula it records the Stan data (member codes J, weights W, member
# covariates Z, and the Cholesky factor Lcov of the known covariance), the
# generated Stan statements for the group-level effects and the predictor, the
# sd prior rows, and posterior_linpred and log_lik on training and fresh rows
# at three draws of a meanfield fit, which serve only as evaluation points.
# Run from this directory: Rscript brms_membership_covariance.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(
  y = c(0.3, -1.2, 0.8, 1.9, -0.4, 0.1, 1.4, -0.7, 0.6, 2.2, -1.5, 0.9),
  x1 = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8, 0.4, -0.3),
  x2 = c(1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2) / 2,
  g1 = c("a", "b", "c", "a", "b", "c", "d", "a", "b", "c", "d", "a"),
  g2 = c("b", "c", "d", "d", "a", "a", "b", "c", "e", "e", "a", "b"),
  w1 = c(1, 2, 1, 3, 1, 1, 2, 1, 1, 2, 1, 1),
  w2 = c(1, 1, 2, 1, 3, 1, 1, 2, 1, 1, 2, 1)
)
fresh <- data.frame(
  y = c(0.2, -0.5, 1.1, 0.4),
  x1 = c(0.1, -0.6, 1.3, 0.8),
  x2 = c(0.5, 1, 0, 0.5),
  g1 = c("e", "a", "c", "b"),
  g2 = c("a", "d", "c", "e"),
  w1 = c(1, 1, 2, 1),
  w2 = c(2, 1, 1, 3)
)
levels <- c("a", "b", "c", "d", "e")
# A covariance whose Cholesky factor has negative entries, and one whose
# factor is entrywise positive: brms's scale_r_cor_cov skips entries of Lcov
# that are not above 1e-10, so only the second gives correlated coefficients
# the Kronecker covariance in brms.
K <- matrix(c(1.0, 0.5, -0.3, 0.1, 0.2,
              0.5, 1.2, 0.2, -0.4, 0.1,
              -0.3, 0.2, 0.9, 0.3, -0.2,
              0.1, -0.4, 0.3, 1.1, 0.25,
              0.2, 0.1, -0.2, 0.25, 0.8), 5, 5, dimnames = list(levels, levels))
P <- 0.3 + diag(c(0.7, 0.9, 0.6, 1.0, 0.8))
dimnames(P) <- list(levels, levels)
stopifnot(any(t(chol(K)) < -1e-3), all(t(chol(P))[lower.tri(P, diag = TRUE)] > 1e-10))
formulas <- c(
  "y ~ x1 + (1 | mm(g1, g2, cov = K))",
  "y ~ x1 + (1 + x1 | mm(g1, g2, weights = cbind(w1, w2), cov = P))",
  "y ~ x1 + (1 + mmc(x1, x2) || mm(g1, g2, cov = K))"
)
data2 <- list(K = K, P = P)
attempt <- function(expr) tryCatch(expr, error = function(e) list(error = conditionMessage(e)))
result <- lapply(formulas, function(text) {
  formula <- as.formula(text)
  standata <- unclass(make_standata(formula, data = d, data2 = data2))
  keep <- grepl("^(N_[0-9]+|M_[0-9]+|J_[0-9]+_[0-9]+|W_[0-9]+_[0-9]+|Z_[0-9]+_[0-9]+_[0-9]+)$",
                names(standata))
  code <- strsplit(make_stancode(formula, data = d, data2 = data2), "\n")[[1]]
  prior <- get_prior(formula, data = d, data2 = data2)
  fit <- suppressMessages(brm(formula, data = d, data2 = data2, algorithm = "meanfield",
                              output_samples = 3, iter = 2000, refresh = 0, seed = 1, init = 0))
  draws <- as.data.frame(as_draws_df(fit))[1:3, ]
  draws <- draws[, !grepl("^(lp__|lprior|\\.chain|\\.iteration|\\.draw)$", names(draws))]
  list(
    formula = text,
    data = lapply(standata[keep], function(value) as.numeric(value)),
    Lcov = unname(standata$Lcov_1),
    predictor = trimws(grep("mu\\[n\\] \\+=", code, value = TRUE)),
    effects = trimws(grep("^ *r_1(_[0-9]+)? = ", code, value = TRUE)),
    sd = prior$coef[prior$class == "sd" & nzchar(prior$coef)],
    group = unique(prior$group[prior$class == "sd" & nzchar(prior$group)]),
    variables = names(draws),
    draws = draws,
    linpred = posterior_linpred(fit, draw_ids = 1:3),
    log_lik = log_lik(fit, draw_ids = 1:3),
    new_linpred = posterior_linpred(fit, newdata = fresh, draw_ids = 1:3),
    new_log_lik = log_lik(fit, newdata = fresh, draw_ids = 1:3),
    new_level = attempt(posterior_linpred(fit, newdata = transform(fresh, g1 = "z"),
                                          draw_ids = 1:3, allow_new_levels = TRUE))
  )
})
writeLines(
  toJSON(list(brms = as.character(packageVersion("brms")), data = d, fresh = fresh,
              levels = levels, K = K, P = P, cases = result),
         auto_unbox = TRUE, pretty = TRUE, digits = NA, dataframe = "columns"),
  "brms_membership_covariance.json"
)
