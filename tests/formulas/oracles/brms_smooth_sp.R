# Generates brms_smooth_sp.json, pinned by tests/formulas/test_smooth_sp.py:
# brms builds the same Stan program and data for s() and t2() terms with and
# without mgcv's sp argument, which enters only the term label, and mgcv's
# smoothCon and smooth2random outputs do not depend on it.
# Run from this directory: Rscript brms_smooth_sp.R
suppressMessages(library(brms))
suppressMessages(library(mgcv))
suppressMessages(library(jsonlite))
rows <- 0:29
d <- data.frame(x = (rows * 0.618034) %% 1, z = cos(rows))
d$y <- sin(3 * d$x) + 0.1 * d$z
same_data <- function(a, b) {
  all(vapply(names(a), function(n) isTRUE(all.equal(unname(a[[n]]), unname(b[[n]]),
                                                     check.attributes = FALSE)), TRUE))
}
labels <- function(f) {
  p <- as.data.frame(get_prior(f, d))
  p$coef[p$class == "sds" & nzchar(p$coef)]
}
s1 <- smoothCon(s(x, sp = 0.5), d, absorb.cons = TRUE)[[1]]
s0 <- smoothCon(s(x), d, absorb.cons = TRUE)[[1]]
r1 <- smooth2random(s1, names(d), type = 2)
r0 <- smooth2random(s0, names(d), type = 2)
out <- list(
  s_code = identical(make_stancode(y ~ s(x, sp = 0.5), d), make_stancode(y ~ s(x), d)),
  s_data = same_data(make_standata(y ~ s(x, sp = 0.5), d), make_standata(y ~ s(x), d)),
  t2_code = identical(make_stancode(y ~ t2(x, z, sp = c(1, 2, 3)), d),
                      make_stancode(y ~ t2(x, z), d)),
  t2_data = same_data(make_standata(y ~ t2(x, z, sp = c(1, 2, 3)), d),
                      make_standata(y ~ t2(x, z), d)),
  mgcv_design = isTRUE(all.equal(s1$X, s0$X)),
  mgcv_random = isTRUE(all.equal(r1$rand, r0$rand, check.attributes = FALSE)) &&
    isTRUE(all.equal(r1$Xf, r0$Xf, check.attributes = FALSE)),
  s_label = labels(y ~ s(x, sp = 0.5)),
  t2_label = labels(y ~ t2(x, z, sp = c(1, 2, 3)))
)
writeLines(toJSON(out, auto_unbox = TRUE), "brms_smooth_sp.json")
