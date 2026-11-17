# Generates brms_threshold_group_models.json, pinned by
# tests/formulas/test_threshold_group_models.py: brms's thres(n, gr) in a
# multivariate model, with hurdle_cumulative, and in mixtures of ordinal
# families. It records standata, Stan statements, and messages, and variables,
# log_lik, and posterior_epred at fixed posterior draws on the fitted and on
# fresh rows.
# Run from this directory: Rscript brms_threshold_group_models.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
options(digits = 17)
# Group a has four categories, b three, c two. R codes categories from one.
d <- data.frame(
  y = c(1, 2, 3, 4, 2, 3, 1, 4, 2, 3, 4, 1, 2, 3,
        1, 2, 3, 2, 1, 3, 2, 3, 1, 2, 3, 1,
        1, 2, 2, 1, 2, 1, 1, 2, 2, 1),
  x = c(0.25, -0.5, 1.0, 1.5, -0.25, 0.75, -1.0, 2.0, 0.0, 0.5, 1.25, -1.5, 0.125, 0.875,
        -0.75, 0.25, 1.5, 0.5, -1.25, 1.0, 0.0, 0.75, -0.5, 0.375, 1.75, -1.0,
        -0.5, 0.75, 1.25, -1.0, 0.5, 0.0, -0.25, 1.5, 0.25, -0.75),
  g = c(rep("a", 14), rep("b", 12), rep("c", 10))
)
d$r <- rep(c(1, 3, 2, 4, 2, 3, 1, 2, 4), 4)
d$w <- 0.5 * d$x + rep(c(-0.3, 0.4, 0.1, -0.6), 9)
# Code zero is the hurdle category.
d$z <- d$y
d$z[c(3, 9, 16, 22, 28, 33)] <- 0
d$n <- c(a = 3, b = 2, c = 1)[d$g]
fresh <- d[c(1, 4, 15, 17, 27, 28), ]
fresh$x <- fresh$x + 0.5

structure_of <- function(formula, family, pattern) {
  tryCatch({
    sdata <- make_standata(formula, data = d, family = family)
    lines <- trimws(strsplit(make_stancode(formula, data = d, family = family), "\n")[[1]])
    list(status = "ok", nthres = as.integer(sdata$nthres),
         lines = lines[grepl(pattern, lines) & !grepl("lprior \\+=|^//|^\\*|^/\\*|^real |^return", lines)])
  }, error = function(e) list(status = "refused", message = conditionMessage(e)))
}
keep <- "Intercept_[a-z0-9_]*[0-9];|merged_lpmf|target \\+="
cases <- list(
  hurdle = structure_of(z | thres(gr = g) ~ x, hurdle_cumulative(), keep),
  hurdle_column = structure_of(z | thres(n, gr = g) ~ x, hurdle_cumulative(), keep),
  hurdle_scalar = structure_of(z | thres(3, gr = g) ~ x, hurdle_cumulative(), keep),
  hurdle_plain = structure_of(z | thres(3) ~ x, hurdle_cumulative(), keep),
  hurdle_too_few = structure_of(z | thres(2) ~ x, hurdle_cumulative(), keep),
  mixture = structure_of(y | thres(gr = g) ~ x, mixture(cumulative(), acat()), keep),
  mixture_hurdle = structure_of(z | thres(gr = g) ~ x,
                                mixture(hurdle_cumulative(), hurdle_cumulative()), keep),
  mixture_mixed = structure_of(z | thres(gr = g) ~ x,
                               mixture(hurdle_cumulative(), cumulative()), keep)
)

# Meanfield output draws suffice: they are only evaluation points.
points <- function(formula, family = NULL, resp = NULL) {
  fit <- NULL
  message <- NULL
  for (seed in 1:5) {
    fit <- tryCatch(suppressMessages(brm(formula, data = d, family = family,
      algorithm = "meanfield", output_samples = 3, iter = 2000,
      refresh = 0, seed = seed, init = 0)),
      error = function(e) { message <<- conditionMessage(e); NULL })
    if (!is.null(fit)) break
  }
  if (is.null(fit)) return(list(error = message))
  draws <- as.data.frame(as_draws_df(fit))[1:3, ]
  draws <- draws[, !grepl("^(lp__|lprior|\\.chain|\\.iteration|\\.draw)$", names(draws))]
  attempt <- function(expr) tryCatch(expr, error = function(e) list(error = conditionMessage(e)))
  out <- list(variables = names(draws), draws = draws)
  if (is.null(resp)) {
    out$log_lik <- attempt(log_lik(fit, draw_ids = 1:3))
    out$epred <- attempt(posterior_epred(fit, draw_ids = 1:3))
    out$new_log_lik <- attempt(log_lik(fit, newdata = fresh, draw_ids = 1:3))
    out$new_epred <- attempt(posterior_epred(fit, newdata = fresh, draw_ids = 1:3))
  } else {
    for (name in resp) {
      out[[paste0("log_lik_", name)]] <- attempt(log_lik(fit, resp = name, draw_ids = 1:3))
      out[[paste0("epred_", name)]] <- attempt(posterior_epred(fit, resp = name, draw_ids = 1:3))
    }
  }
  out
}
values <- list(
  joint = points(bf(y | thres(gr = g) ~ x, family = cumulative()) +
                   bf(r ~ x, family = cumulative()) + bf(w ~ x, family = gaussian()) +
                   set_rescor(FALSE), resp = c("y", "r")),
  hurdle = points(z | thres(gr = g) ~ x, hurdle_cumulative()),
  mixture = points(y | thres(gr = g) ~ x, mixture(cumulative(), acat()))
)
writeLines(toJSON(list(data = d, fresh = fresh, cases = cases, values = values),
                  auto_unbox = TRUE, digits = NA, dataframe = "columns", na = "null"),
           "brms_threshold_group_models.json")
