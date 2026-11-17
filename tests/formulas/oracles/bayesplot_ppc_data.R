# Generates bayesplot_ppc_data.json, pinned by tests/formulas/test_pp_check.py.
# Oracles for the data behind the bayesplot ppc_* plots that brms pp_check
# forwards to: ppc_data/ppd_data, ppc_stat_data (one and two statistics,
# grouped), ppc_intervals_data (with x and group), ppc_error_data,
# ppc_error_binnned_data, ppc_bars_data (counts and proportions, grouped), the
# rootogram table of ppc_rootogram, ppc_scatter_data, ppc_scatter_avg_data,
# rstantools::loo_pit, the E_loo quantiles behind ppc_loo_intervals, the
# Kaplan-Meier curves behind ppc_km_overlay, and the PIT values and ECDF
# behind ppc_pit_ecdf. Inputs are drawn here and exported.
# Run from this directory: Rscript bayesplot_ppc_data.R
suppressMessages(library(bayesplot))
suppressMessages(library(loo))
suppressMessages(library(survival))
suppressMessages(library(jsonlite))

set.seed(11)
N <- 12; S <- 6
y <- round(rnorm(N), 3)
yrep <- matrix(round(rnorm(S * N), 3), S, N)
group <- rep(c("b", "a", "c"), 4)
x <- round(runif(N, -1, 1), 3)
ycount <- rpois(N, 2)
yrepcount <- matrix(rpois(S * N, 2), S, N)
ybin <- rbinom(N, 1, 0.4)
epred <- matrix(round(runif(S * N, 0.05, 0.95), 3), S, N)
log_ratios <- matrix(rnorm(400 * N, 0, 0.6), 400, N)
yrep_loo <- matrix(round(rnorm(400 * N), 3), 400, N)
psis_object <- suppressWarnings(psis(log_ratios, r_eff = NA))
lw <- weights(psis_object, log = TRUE)
status <- c(1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0)
ypos <- round(rexp(N), 3)
yreppos <- matrix(round(rexp(S * N), 3), S, N)

frame <- function(d) {
  d <- as.data.frame(d)
  for (v in names(d)) if (is.factor(d[[v]])) d[[v]] <- as.character(d[[v]])
  d
}

rootogram <- function(y, yrep, prob = 0.9) {
  alpha <- (1 - prob) / 2
  probs <- c(alpha, 1 - alpha)
  ymax <- max(y, yrep)
  xpos <- 0L:ymax
  tyrep <- as.list(rep(NA, nrow(yrep)))
  for (i in seq_along(tyrep)) {
    tyrep[[i]] <- table(yrep[i, ])
    matches <- match(xpos, rownames(tyrep[[i]]))
    tyrep[[i]] <- as.numeric(tyrep[[i]][matches])
  }
  tyrep <- do.call(rbind, tyrep)
  tyrep[is.na(tyrep)] <- 0
  tyexp <- sqrt(colMeans(tyrep))
  tyquantile <- sqrt(t(apply(tyrep, 2, quantile, probs = probs)))
  ty <- table(y)
  ty <- sqrt(as.numeric(ty[match(xpos, rownames(ty))]))
  ty[is.na(ty)] <- 0
  data.frame(xpos = xpos, ypos = ty / 2, ty = ty, tyexp = tyexp,
             tylower = tyquantile[, 1], tyupper = tyquantile[, 2])
}

km <- function(y, yrep, status) {
  data <- ppc_data(y, yrep, group = status)
  data$status <- ifelse(data$is_y, as.numeric(as.character(data$group)), 1)
  fit <- survfit(Surv(value, status) ~ rep_label, data = data)
  strata <- rep(names(fit$strata), fit$strata)
  data.frame(strata = strata, time = fit$time, n_risk = fit$n.risk,
             n_event = fit$n.event, n_censor = fit$n.censor, surv = fit$surv)
}

pit <- ppc_data(y, yrep) |>
  dplyr::group_by(y_id) |>
  dplyr::group_map(~ mean(.x$value[.x$is_y] >= .x$value[!.x$is_y])) |>
  unlist()
K <- min(S + 1, N)

out <- list(
  inputs = list(
    y = y, yrep = yrep, group = group, x = x, ycount = ycount,
    yrepcount = yrepcount, ybin = ybin, epred = epred,
    log_ratios = log_ratios, yrep_loo = yrep_loo, status = status,
    ypos = ypos, yreppos = yreppos, lw = lw
  ),
  ppc_data = frame(ppc_data(y, yrep, group = group)),
  ppd_data = frame(ppd_data(yrep)),
  stat = frame(ppc_stat_data(y, yrep, stat = "mean")),
  stat_grouped = frame(ppc_stat_data(y, yrep, group = group, stat = "median")),
  stat_2d = frame(ppc_stat_data(y, yrep, stat = c("mean", "sd"))),
  ppd_stat = frame(ppd_stat_data(yrep, stat = "sd")),
  intervals = frame(ppc_intervals_data(y, yrep, x = x, group = group)),
  intervals_plain = frame(ppc_intervals_data(y, yrep, prob = 0.6, prob_outer = 0.95)),
  error = frame(ppc_error_data(y, yrep, group = group)),
  error_binned = frame(bayesplot:::ppc_error_binnned_data(ybin, epred)),
  bars = frame(ppc_bars_data(ycount, yrepcount)),
  bars_grouped = frame(ppc_bars_data(ycount, yrepcount, group = group, freq = FALSE)),
  rootogram = rootogram(ycount, yrepcount),
  scatter = frame(ppc_scatter_data(y, yrep)),
  scatter_avg = frame(ppc_scatter_avg_data(y, yrep, group = group)),
  loo_pit = rstantools::loo_pit(yrep_loo, y = y, lw = lw),
  loo_quantiles = t(E_loo(yrep_loo, psis_object, type = "quantile",
                          probs = c(0.05, 0.25, 0.5, 0.75, 0.95))$value),
  km = km(ypos, yreppos, status),
  pit_ecdf = list(pit = pit, K = K,
                  ecdf = ecdf(pit)(seq(0, 1, length.out = K))),
  versions = list(
    bayesplot = as.character(packageVersion("bayesplot")),
    loo = as.character(packageVersion("loo")),
    survival = as.character(packageVersion("survival"))
  )
)
writeLines(toJSON(out, digits = I(15), auto_unbox = TRUE, dataframe = "columns",
                  matrix = "rowmajor", na = "null"),
           "bayesplot_ppc_data.json")
