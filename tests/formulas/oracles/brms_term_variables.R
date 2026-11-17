# Generates brms_term_variables.json, pinned by
# tests/formulas/test_term_postprocessing_matrix.py.
# brms oracle for the parameter names (variables()) of one small model per
# term type. Per-level and per-row variables (r_*, z_*, s_*, zs_*, zgp_*,
# rcar, err, Ymi, Xme, zme, udf_*, hs_local, R2D2_phi), the centered
# Intercept, and the full by-level correlation matrices Cor_* that repeat
# the cor_* pairs are dropped, and so are lprior and lp__.
# Run from this directory: Rscript brms_term_variables.R [key ...]
# Without keys every model is fitted. Each fit's names are cached as
# <key>.json in the directory named by BRMS_TERM_VARIABLES_CACHE (a session
# temporary directory by default), so a rerun fits only the missing keys.
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
n <- 24
r <- 0:(n - 1)
set.seed(1)
d <- data.frame(
  x = sin(r * 0.7), z = cos(r * 1.3), w = 1 + (r %% 3) * 0.5,
  g = c("a", "b", "c", "d")[r %% 4 + 1],
  g2 = c("a", "b", "c", "d")[(r + 1) %% 4 + 1],
  h = c("p", "q")[(r %/% 4) %% 2 + 1],
  f = c("u", "v", "t")[r %% 3 + 1],
  time = r %% 6, sub = r %/% 6, se = 0.2 + 0.05 * (r %% 4),
  k = r %% 4, n = 10
)
d$by2 <- ifelse(d$g %in% c("a", "b"), "L", "R")
d$y <- 1 + 0.5 * d$x - 0.3 * d$z + 0.3 * rnorm(n)
d$y2 <- 0.5 - 0.2 * d$x + 0.3 * rnorm(n)
d$ymis <- ifelse(r %% 5 == 0, NA, d$y)
d$pos <- exp(0.3 * d$y)
d$cnt <- rpois(n, 3)
d$ordy <- factor(c("lo", "mid", "hi", "top")[(r * 7) %% 4 + 1],
                 levels = c("lo", "mid", "hi", "top"), ordered = TRUE)
d$caty <- c("A", "B", "C")[(r * 5) %% 3 + 1]
unit <- 1 / (1 + exp(-d$y + 1))
d$P <- cbind(p1 = unit * 0.5, p2 = (1 - unit) * 0.5, p3 = 0.5)
W <- matrix(0, n, n)
for (i in 1:(n - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
W <- W / rowSums(W)
V <- diag(n) + 0.1
M <- matrix(c(0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0), 4, 4,
            dimnames = list(letters[1:4], letters[1:4]))
models <- list(
  plain = list(y ~ x + z),
  intercept_column = list(y ~ 0 + Intercept + x),
  bf_single = list(bf(y ~ x, sigma ~ z)),
  group = list(y ~ x + (1 + x | g)),
  gr_by = list(y ~ x + (1 + x | gr(g, by = by2))),
  gr_student = list(y ~ x + (1 | gr(g, dist = "student"))),
  mm = list(y ~ x + (1 | mm(g, g2))),
  mo = list(y ~ mo(k) + (1 | g)),
  me = list(y ~ me(x, se) + z),
  mi_response = list(bf(ymis | mi() ~ x)),
  s = list(y ~ s(x) + (1 | g)),
  t2 = list(y ~ t2(x, z)),
  gp_k = list(y ~ gp(x, k = 5, c = 5 / 4)),
  gp = list(y ~ gp(x)),
  ar = list(y ~ x + ar(time = time, gr = sub)),
  ma = list(y ~ x + ma(time = time, gr = sub)),
  arma = list(y ~ x + arma(time = time, gr = sub, p = 1, q = 1)),
  ar_latent = list(cnt ~ x + ar(time = time, gr = sub), family = poisson()),
  cosy = list(y ~ x + cosy(time = time, gr = sub)),
  unstr = list(y ~ x + unstr(time = time, gr = sub)),
  sar = list(y ~ x + sar(W), data2 = list(W = W)),
  car = list(cnt ~ x + car(M, gr = g), family = poisson(), data2 = list(M = M)),
  car_bym2 = list(cnt ~ x + car(M, gr = g, type = "bym2"), family = poisson(),
                  data2 = list(M = M)),
  fcor = list(y ~ x + fcor(V), data2 = list(V = V)),
  cumulative = list(ordy ~ x + (1 | g), family = cumulative()),
  cs = list(ordy ~ x + cs(z), family = acat()),
  categorical = list(bf(caty ~ x, muC ~ z + (1 | g)), family = categorical()),
  logistic_normal = list(P ~ x, family = logistic_normal()),
  mixture = list(y ~ x, family = mixture(gaussian, gaussian)),
  distributional = list(bf(y ~ x, sigma ~ z + (1 | g))),
  nonlinear = list(
    bf(pos ~ b1 * exp(b2 * x), b1 ~ 1, b2 ~ 1 + (1 | g), nl = TRUE),
    prior = prior(normal(0, 1), nlpar = "b1") + prior(normal(0, 1), nlpar = "b2")
  ),
  horseshoe = list(y ~ x + z, prior = set_prior(horseshoe(1))),
  r2d2 = list(y ~ x + z, prior = set_prior(R2D2())),
  joint_rescor = list(bf(y ~ x) + bf(y2 ~ x + (1 | g)) + set_rescor(TRUE)),
  joint_nuts = list(bf(y ~ x) + bf(y2 ~ x) + set_rescor(TRUE))
)
dropped <- paste0(
  "^(r_|z_|s_|zs_|zgp_|rcar|err\\[|Ymi|Xme|zme|udf_|hs_local|R2D2_phi|Cor_|",
  "lprior$|lp__$|Intercept$)"
)
keys <- commandArgs(trailingOnly = TRUE)
if (length(keys) == 0) keys <- names(models)
cache <- Sys.getenv("BRMS_TERM_VARIABLES_CACHE", tempdir())
dir.create(cache, showWarnings = FALSE, recursive = TRUE)
for (key in keys) {
  path <- file.path(cache, paste0(key, ".json"))
  if (file.exists(path)) next
  arguments <- models[[key]]
  names(arguments)[1] <- "formula"
  # A variational fit is the cheapest way to obtain the names, which do not
  # depend on the algorithm or the seed. Stan's variational start fails for
  # some models and seeds, so a mean-field fit started at zero is tried
  # first and full-rank fits with other seeds after it.
  attempts <- c(
    list(list(algorithm = "meanfield", iter = 200, init = 0, seed = 1)),
    lapply(1:6, function(seed) {
      list(algorithm = "fullrank", iter = 2000, seed = seed)
    })
  )
  fit <- NULL
  for (attempt in attempts) {
    fit <- tryCatch(
      suppressWarnings(do.call(brm, c(arguments, attempt, list(
        data = d, refresh = 0, output_samples = 10
      )))),
      error = function(e) NULL
    )
    if (!is.null(fit)) break
  }
  if (is.null(fit)) stop("no variational fit of model '", key, "' started")
  kept <- grep(dropped, variables(fit), value = TRUE, invert = TRUE)
  stopifnot(length(kept) > 0)
  writeLines(toJSON(kept), path)
}
result <- list()
for (key in names(models)) {
  path <- file.path(cache, paste0(key, ".json"))
  if (file.exists(path)) result[[key]] <- fromJSON(path)
}
writeLines(toJSON(result, pretty = TRUE), "brms_term_variables.json")
