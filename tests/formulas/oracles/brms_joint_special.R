# Generates brms_joint_special.json, pinned by tests/formulas/test_joint_special_terms.py.
# brms oracle for mo() and mm() inside a multivariate model: the response-
# suffixed data (Xmo_<resp>_k, Jmo_<resp>, J_<id>_<resp>_k, W_<id>_<resp>_k) and the
# generated per-response linear predictors.
# Run from this directory: Rscript brms_joint_special.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(
  y1 = c(0.3, -1.2, 0.8, 1.9, -0.4, 0.1, 1.4, -0.7, 0.6, 2.2, -1.5, 0.9),
  y2 = c(1.1, 0.2, -0.5, 0.7, 1.6, -0.9, 0.4, 0.0, -1.3, 0.8, 0.5, -0.2),
  x = c(1, 2, 3, 4, 2, 3, 1, 4, 3, 2, 4, 1),
  z = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8, 0.4, -0.3),
  g1 = c("a", "b", "c", "a", "b", "c", "d", "a", "b", "c", "d", "a"),
  g2 = c("b", "c", "d", "d", "a", "a", "b", "c", "e", "e", "a", "b"),
  w1 = c(1, 2, 1, 3, 1, 1, 2, 1, 1, 2, 1, 1),
  w2 = c(1, 1, 2, 1, 3, 1, 1, 2, 1, 1, 2, 1)
)
formula <- bf(y1 ~ mo(x) * z) +
  bf(y2 ~ z + (1 | mm(g1, g2, weights = cbind(w1, w2)))) + set_rescor(FALSE)
standata <- unclass(make_standata(formula, data = d))
keep <- grepl("^(Xmo|Jmo|Csp|J_|W_|Z_|N_|Ksp)", names(standata))
code <- strsplit(make_stancode(formula, data = d), "\n")[[1]]
writeLines(
  toJSON(list(
    brms = as.character(packageVersion("brms")),
    data = d,
    standata = lapply(standata[keep], as.numeric),
    predictor = trimws(grep("mu_y[12]\\[n\\] \\+=", code, value = TRUE)),
    simo = get_prior(formula, data = d)$coef[get_prior(formula, data = d)$class == "simo"]
  ), auto_unbox = TRUE, pretty = TRUE, digits = NA),
  "brms_joint_special.json"
)
