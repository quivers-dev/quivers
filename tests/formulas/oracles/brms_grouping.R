# Generates brms_grouping.json, pinned by tests/formulas/test_grouping_syntax.py.
# brms oracle for grouping-term syntax: interaction (g1:g2) and nested (g1/g2)
# factor coding and labels, gr() options, the known-covariance factor Lcov of
# gr(cov = ), and the refusal of coefficients repeated across blocks.
# Run from this directory: Rscript brms_grouping.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(
  y = c(0.3, -1.2, 0.8, 1.9, -0.4, 0.1, 1.4, -0.7, 0.6, 2.2, -1.5, 0.9),
  x = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8, 0.4, -0.3),
  g1 = rep(c("a", "b"), each = 6),
  g2 = rep(c("u", "v", "w"), 4)
)
A <- matrix(c(1, 0.5, 0.2, 0.5, 2, 0.3, 0.2, 0.3, 1.5), 3,
            dimnames = list(c("u", "v", "w"), c("u", "v", "w")))
codes <- function(text) {
  standata <- unclass(make_standata(as.formula(text), data = d, data2 = list(A = A)))
  prior <- get_prior(as.formula(text), data = d, data2 = list(A = A))
  sd <- prior[prior$class == "sd" & nzchar(prior$coef), ]
  list(
    formula = text,
    J = lapply(standata[grepl("^J_[0-9]+$", names(standata))], as.numeric),
    N = lapply(standata[grepl("^N_[0-9]+$", names(standata))], as.numeric),
    Lcov = if ("Lcov_1" %in% names(standata)) unname(standata$Lcov_1) else NULL,
    sd = data.frame(coef = sd$coef, group = sd$group),
    cor = unique(prior$group[prior$class == "cor" & nzchar(prior$group)]),
    stan = trimws(grep("mu\\[n\\] \\+=|r_[0-9]+_[0-9]+ = ", strsplit(
      make_stancode(as.formula(text), data = d, data2 = list(A = A)), "\n")[[1]], value = TRUE))
  )
}
formulas <- c(
  "y ~ (1 | g1:g2)",
  "y ~ (1 | g1/g2)",
  "y ~ (1 + x | gr(g2, cor = FALSE))",
  "y ~ (1 + x || g2)",
  "y ~ (1 | gr(g2, id = 'i')) + (0 + x | gr(g2, id = 'i'))",
  "y ~ (1 | gr(g2, cov = A))"
)
duplicate <- tryCatch(make_standata(y ~ (1 | g1) + (x | g1), data = d),
                      error = function(e) conditionMessage(e))
writeLines(
  toJSON(list(brms = as.character(packageVersion("brms")), data = d,
              cases = lapply(formulas, codes), duplicate = duplicate),
         auto_unbox = TRUE, pretty = TRUE, digits = NA),
  "brms_grouping.json"
)
