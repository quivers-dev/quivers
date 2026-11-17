# Generates brms_monotonic.json, pinned by tests/formulas/test_monotonic.py.
# brms oracle for mo(): the coding of integer and ordered-factor predictors
# (Xmo, Jmo), the special-effects covariates (Csp), the simplex and coefficient
# labels, and the generated Stan linear predictor of each special term.
# Run from this directory: Rscript brms_monotonic.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(
  y = c(0.3, -1.2, 0.8, 1.9, -0.4, 0.1, 1.4, -0.7, 0.6, 2.2, -1.5, 0.9),
  x = c(1, 2, 3, 4, 2, 3, 1, 4, 3, 2, 4, 1),
  x2 = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2),
  z = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8, 0.4, -0.3),
  f = factor(c("a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c")),
  g = c("u", "v", "w", "u", "v", "w", "u", "v", "w", "u", "v", "w")
)
d$xo <- factor(c("lo", "mid", "hi", "top", "mid", "hi", "lo", "top", "hi", "mid", "top", "lo"),
               levels = c("lo", "mid", "hi", "top", "max"), ordered = TRUE)
# An unused interior level: brms drops unused levels before coding.
d$xs <- factor(c("a", "c", "d", "a", "c", "d", "a", "c", "d", "a", "c", "d"),
               levels = c("a", "b", "c", "d"), ordered = TRUE)
formulas <- c(
  "y ~ mo(x)",
  "y ~ mo(xo)",
  "y ~ mo(xs)",
  "y ~ mo(x) * z",
  "y ~ mo(x) * mo(x2)",
  "y ~ mo(x):f",
  "y ~ mo(x, id = 'i') * z",
  "y ~ mo(x) + (mo(x) | g)"
)
result <- lapply(formulas, function(text) {
  formula <- as.formula(text)
  standata <- unclass(make_standata(formula, data = d))
  keep <- grepl("^(Ksp|Imo|Jmo|Xmo_[0-9]+|Csp_[0-9]+|con_simo_[0-9]+)$", names(standata))
  data <- lapply(standata[keep], function(value) as.numeric(value))
  code <- strsplit(make_stancode(formula, data = d), "\n")[[1]]
  predictor <- trimws(grep("mu\\[n\\] \\+=", code, value = TRUE))
  shared <- trimws(grep("simplex\\[Jmo\\[[0-9]+\\]\\] simo_[0-9]+ = ", code, value = TRUE))
  prior <- get_prior(formula, data = d)
  list(
    formula = text,
    data = data,
    predictor = predictor,
    shared = shared,
    b = prior$coef[prior$class == "b" & nzchar(prior$coef)],
    simo = prior$coef[prior$class == "simo"],
    simo_prior = unique(prior$prior[prior$class == "simo"])
  )
})
frame <- d
frame$xo <- as.character(frame$xo)
frame$f <- as.character(frame$f)
frame$xs <- as.character(frame$xs)
writeLines(
  toJSON(list(brms = as.character(packageVersion("brms")), data = frame, cases = result),
         auto_unbox = TRUE, pretty = TRUE, digits = NA),
  "brms_monotonic.json"
)
