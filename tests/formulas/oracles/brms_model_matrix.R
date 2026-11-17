# Generates brms_model_matrix.json, pinned by tests/formulas/test_formula_operators.py.
# R's model.matrix (which brms's make_standata uses for the population design)
# for the R formula operators brms users write: I() with ^, poly() orthogonal
# and raw, splines::ns() and splines::bs() (with rows outside the boundary
# knots and quantile knots on them), scale(), log(), interactions with
# :, *, ^ and %in%, and term removal with -, plus predict() of the stateful
# transforms on new rows, some outside the training range; brms's cmc = FALSE
# design, the intercept class under sparse = TRUE, and unused = ~ u rows.
# Run from this directory: Rscript brms_model_matrix.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
suppressMessages(library(splines))
d <- data.frame(
  y = c(0.31, -1.24, 0.85, 1.93, -0.42, 0.13, 1.41, -0.73, 0.62, 2.21, -1.52, 0.94),
  x = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8, 0.4, -0.3),
  z = c(0.4, 1.9, 2.6, 0.2, 3.1, 1.4, 0.7, 2.2, 1.0, 0.3, 2.8, 1.6),
  f = factor(rep(c("a", "b", "c"), 4)),
  h = factor(strsplit("uvuvuvvuvuuv", "")[[1]])
)
formulas <- c(
  "y ~ I(x^2)", "y ~ poly(x, 2)", "y ~ poly(x, 3, raw = TRUE)",
  "y ~ ns(x, df = 3)", "y ~ ns(x, df = 4, intercept = TRUE)",
  "y ~ ns(x, knots = c(-0.5, 0.6))", "y ~ bs(x, df = 4)",
  "y ~ bs(x, df = 5, degree = 2)", "y ~ bs(x, knots = c(-0.5, 0.6), intercept = TRUE)",
  "y ~ scale(x)",
  "y ~ log(z)", "y ~ x:z", "y ~ x * z", "y ~ (x + z + f)^2", "y ~ x %in% f",
  "y ~ x + z - z", "y ~ f * x", "y ~ 0 + f", "y ~ x - 1"
)
matrices <- lapply(formulas, function(text) {
  m <- model.matrix(as.formula(text), d)
  list(formula = text, columns = colnames(m), values = unname(m))
})
newx <- c(-2.0, -1.1, 0.0, 0.75, 1.7, 2.5)
ns_fit <- ns(d$x, df = 3)
ns_knots <- ns(d$x, knots = c(-0.5, 0.6))
poly_fit <- poly(d$x, 2)
bs_fit <- bs(d$x, df = 4)
# Rows outside the boundary knots, at training and at prediction.
bs_bounded <- suppressWarnings(bs(d$x, df = 5, Boundary.knots = c(-1.2, 1.5)))
# Quantile knots that fall on both boundary knots and are moved inside.
tied <- c(0, 0, 0, 0, 1, 2, 3, 3, 3, 3)
bs_tied <- suppressWarnings(bs(tied, df = 5))
scaled <- scale(d$x)
standata <- make_standata(bf(y ~ ns(x, df = 3) + scale(x)), data = d)
cmc_false <- make_standata(bf(y ~ 0 + f, cmc = FALSE), data = d)
sparse_prior <- get_prior(bf(y ~ x, sparse = TRUE), data = d)
with_missing <- d
with_missing$u <- c(rep(1, 11), NA)
unused_rows <- make_standata(bf(y ~ x, unused = ~ u), data = with_missing)$N
writeLines(
  toJSON(list(
    brms = as.character(packageVersion("brms")), data = d, matrices = matrices,
    newx = newx,
    ns_new = unname(predict(ns_fit, newx)),
    ns_knots_new = unname(predict(ns_knots, newx)),
    poly_new = unname(predict(poly_fit, newx)),
    bs_new = unname(predict(bs_fit, newx[newx >= min(d$x) & newx <= max(d$x)])),
    bs_inside = newx >= min(d$x) & newx <= max(d$x),
    bs_all_new = unname(suppressWarnings(predict(bs_fit, newx))),
    bs_bounded = unname(unclass(bs_bounded)[, 1:5]),
    bs_bounded_new = unname(suppressWarnings(predict(bs_bounded, newx))),
    bs_tied_x = tied, bs_tied_knots = unname(attr(bs_tied, "knots")),
    bs_tied = unname(unclass(bs_tied)[, 1:5]),
    scale_center = attr(scaled, "scaled:center"),
    scale_scale = attr(scaled, "scaled:scale"),
    brms_X = unname(standata$X), brms_X_columns = colnames(standata$X),
    cmc_false_X = unname(cmc_false$X), cmc_false_columns = colnames(cmc_false$X),
    sparse_prior = data.frame(class = sparse_prior$class, coef = sparse_prior$coef),
    unused_rows = unused_rows
  ), auto_unbox = TRUE, pretty = TRUE, digits = NA),
  "brms_model_matrix.json"
)
