# Generates brms_population_coordinates.json, pinned by
# tests/formulas/test_population_coordinates.py.
# brms oracle for centering and QR coordinates: the population design X of
# make_standata(), its column means, the scaled thin QR factors brms builds in
# transformed data (Stan's qr_thin_Q/qr_thin_R, whose R has a positive
# diagonal), and the Stan lines that sample and back-transform them.
# Run from this directory: Rscript brms_population_coordinates.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(
  y = c(0.31, -1.24, 0.85, 1.93, -0.42, 0.13, 1.41, -0.73, 0.62, 2.21,
        -1.52, 0.94, 0.27, -0.66, 1.18, 0.05, -0.91, 1.67, 0.44, -0.18),
  x = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8,
        0.4, -0.3, 1.1, -0.6, 0.2, 0.8, -1.2, 1.5, -0.1, 0.7),
  z = c(-0.2, 0.4, 1.1, -0.9, 0.3, -0.5, 0.8, 1.4, -1.3, 0.2,
        0.6, -0.8, -0.1, 1.2, -0.6, 0.5, 0.9, -1.1, 0.0, 0.7)
)
d$w <- d$x + 0.3 * d$z + c(0.05, -0.02, 0.01, 0.03, -0.04, 0.02, 0.0, -0.01,
                           0.02, -0.03, 0.01, 0.04, -0.02, 0.0, 0.03, -0.01,
                           0.02, -0.04, 0.01, 0.0)
qr_factors <- function(X) {
  decomposition <- qr(X)
  Q <- qr.Q(decomposition)
  R <- qr.R(decomposition)
  signs <- sign(diag(R))
  Q <- Q %*% diag(signs, nrow = length(signs))
  R <- diag(signs, nrow = length(signs)) %*% R
  N <- nrow(X)
  list(XQ = Q * sqrt(N - 1), XR = R / sqrt(N - 1),
       XR_inv = solve(R / sqrt(N - 1)))
}
lines_of <- function(code, pattern) {
  trimws(grep(pattern, strsplit(code, "\n")[[1]], value = TRUE))
}
standata <- make_standata(bf(y ~ x + z + w, decomp = "QR"), data = d)
X <- standata$X
centered <- sweep(X[, -1, drop = FALSE], 2, colMeans(X[, -1, drop = FALSE]))
code_qr <- make_stancode(bf(y ~ x + z + w, decomp = "QR"), data = d)
code_center <- make_stancode(bf(y ~ x + z + w), data = d)
uncentered <- make_standata(bf(y ~ x + z + w, center = FALSE, decomp = "QR"), data = d)
code_uncentered <- make_stancode(bf(y ~ x + z + w, center = FALSE, decomp = "QR"), data = d)
writeLines(
  toJSON(list(
    brms = as.character(packageVersion("brms")), data = d,
    X = unname(X), colnames = colnames(X),
    means = unname(colMeans(X[, -1, drop = FALSE])),
    centered = qr_factors(centered),
    uncentered = qr_factors(uncentered$X),
    uncentered_colnames = colnames(uncentered$X),
    stan_qr = lines_of(code_qr, "qr_thin|XR_inv|bQ|b_Intercept|Intercept -"),
    stan_center = lines_of(code_center, "means_X|Xc\\[|b_Intercept"),
    stan_uncentered = lines_of(code_uncentered, "qr_thin|XQ \\* bQ|b = XR_inv")
  ), auto_unbox = TRUE, pretty = TRUE, digits = NA),
  "brms_population_coordinates.json"
)
