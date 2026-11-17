# Generates brms_membership.json, pinned by tests/formulas/test_membership.py.
# brms oracle for mm() and mmc(): the member codes (J), weights (W), member
# covariates (Z), group levels, labels, and the generated Stan linear predictor.
# Run from this directory: Rscript brms_membership.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(
  y = c(0.3, -1.2, 0.8, 1.9, -0.4, 0.1, 1.4, -0.7, 0.6, 2.2, -1.5, 0.9),
  x1 = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8, 0.4, -0.3),
  x2 = c(1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2) / 2,
  g1 = c("a", "b", "c", "a", "b", "c", "d", "a", "b", "c", "d", "a"),
  g2 = c("b", "c", "d", "d", "a", "a", "b", "c", "e", "e", "a", "b"),
  g3 = c("c", "a", "e", "b", "d", "b", "a", "e", "c", "b", "d", "e"),
  w1 = c(1, 2, 1, 3, 1, 1, 2, 1, 1, 2, 1, 1),
  w2 = c(1, 1, 2, 1, 3, 1, 1, 2, 1, 1, 2, 1),
  w3 = c(2, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 1)
)
formulas <- c(
  "y ~ (1 | mm(g1, g2))",
  "y ~ (1 | mm(g1, g2, weights = cbind(w1, w2)))",
  "y ~ (1 | mm(g1, g2, weights = cbind(w1, w2), scale = FALSE))",
  "y ~ (1 | mm(g1, g2, g3, weights = cbind(w1, w2, w3)))",
  "y ~ (1 + mmc(x1, x2) || mm(g1, g2))",
  "y ~ (1 + x1 || mm(g1, g2))",
  "y ~ (1 + mmc(x1, x2) | mm(g1, g2))"
)
result <- lapply(formulas, function(text) {
  formula <- as.formula(text)
  standata <- unclass(make_standata(formula, data = d))
  keep <- grepl("^(N_[0-9]+|M_[0-9]+|J_[0-9]+_[0-9]+|W_[0-9]+_[0-9]+|Z_[0-9]+_[0-9]+_[0-9]+)$",
                names(standata))
  data <- lapply(standata[keep], function(value) as.numeric(value))
  code <- strsplit(make_stancode(formula, data = d), "\n")[[1]]
  prior <- get_prior(formula, data = d)
  list(
    formula = text,
    data = data,
    predictor = trimws(grep("mu\\[n\\] \\+=", code, value = TRUE)),
    effects = trimws(grep("^ *r_1_[0-9]+ = ", code, value = TRUE)),
    sd = prior$coef[prior$class == "sd" & nzchar(prior$coef)],
    group = unique(prior$group[prior$class == "sd" & nzchar(prior$group)])
  )
})
writeLines(
  toJSON(list(brms = as.character(packageVersion("brms")), data = d, cases = result),
         auto_unbox = TRUE, pretty = TRUE, digits = NA),
  "brms_membership.json"
)
