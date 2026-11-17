# Generates brms_threshold_groups.json, pinned by
# tests/formulas/test_threshold_groups.py: brms's standata, Stan statements,
# prior rows, and messages for thres(n, gr), and variables, log_lik,
# posterior_epred, and posterior_linpred at fixed posterior draws.
# Run from this directory: Rscript brms_threshold_groups.R
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
d$n <- c(a = 3, b = 2, c = 1)[d$g]
d$wide <- c(a = 4, b = 2, c = 3)[d$g]
fresh <- d[c(1, 4, 15, 17, 27, 28), ]
fresh$x <- fresh$x + 0.5

lines_of <- function(formula, family) {
  trimws(strsplit(make_stancode(formula, data = d, family = family), "\n")[[1]])
}
structure_of <- function(formula, family = cumulative()) {
  tryCatch({
    sdata <- make_standata(formula, data = d, family = family)
    lines <- lines_of(formula, family)
    list(status = "ok", nthres = as.integer(sdata$nthres),
         Jthres = unname(sdata$Jthres), ngrthres = sdata$ngrthres,
         lines = lines[grepl("Intercept_[0-9]|merged_Intercept|^target \\+=", lines) &
                         !grepl("lprior \\+=|^real |^//", lines)])
  }, error = function(e) list(status = "refused", message = conditionMessage(e)))
}
cases <- list(
  inferred = structure_of(y | thres(gr = g) ~ x),
  column = structure_of(y | thres(n, gr = g) ~ x),
  wide = structure_of(y | thres(wide, gr = g) ~ x),
  scalar = structure_of(y | thres(5, gr = g) ~ x),
  sratio = structure_of(y | thres(gr = g) ~ x, sratio()),
  cratio = structure_of(y | thres(gr = g) ~ x, cratio()),
  acat = structure_of(y | thres(gr = g) ~ x, acat()),
  category_specific = structure_of(y | thres(gr = g) ~ x + cs(x), acat()),
  too_few = structure_of(y | thres(1, gr = g) ~ x)
)
prior <- as.data.frame(get_prior(y | thres(gr = g) ~ x, d, family = cumulative()))
prior <- prior[prior$class == "Intercept", c("prior", "class", "coef", "group")]

# Meanfield output draws suffice: they are only evaluation points.
points <- function(formula, family) {
  fit <- NULL
  message <- NULL
  for (seed in 1:5) {
    fit <- tryCatch(suppressMessages(brm(formula, data = d, family = family,
      algorithm = "meanfield", output_samples = 3, iter = 2000,
      refresh = 0, seed = seed, init = 0)),
      error = function(e) { message <<- conditionMessage(e); NULL })
    if (!is.null(fit)) break
  }
  # A model brms cannot fit is recorded by its message.
  if (is.null(fit)) return(list(error = message))
  draws <- as.data.frame(as_draws_df(fit))[1:3, ]
  draws <- draws[, !grepl("^(lp__|lprior|\\.chain|\\.iteration|\\.draw)$", names(draws))]
  attempt <- function(expr) tryCatch(expr, error = function(e) list(error = conditionMessage(e)))
  list(variables = names(draws), draws = draws,
       log_lik = attempt(log_lik(fit, draw_ids = 1:3)),
       epred = attempt(posterior_epred(fit, draw_ids = 1:3)),
       linpred = attempt(posterior_linpred(fit, draw_ids = 1:3)),
       new_log_lik = attempt(log_lik(fit, newdata = fresh, draw_ids = 1:3)),
       new_epred = attempt(posterior_epred(fit, newdata = fresh, draw_ids = 1:3)))
}
values <- list(
  cumulative = points(y | thres(gr = g) ~ x, cumulative()),
  cumulative_probit = points(y | thres(gr = g) ~ x, cumulative("probit")),
  sratio = points(y | thres(gr = g) ~ x, sratio()),
  cratio = points(y | thres(gr = g) ~ x, cratio("cloglog")),
  acat = points(y | thres(gr = g) ~ x, acat())
)
writeLines(toJSON(list(data = d, fresh = fresh, cases = cases, prior = prior,
                       values = values),
                  auto_unbox = TRUE, digits = NA, dataframe = "columns", na = "null"),
           "brms_threshold_groups.json")
