# Generates brms_missing.json, pinned by tests/formulas/test_joint_missing.py.
# brms oracle for mi() responses and predictors, mi(sdy), me(), set_mecor().
# Run from this directory: Rscript brms_missing.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(
  y = c(0.1, 0.4, -0.2, 0.3, 0.5, 0.0, 0.7, -0.1),
  x = c(1.0, NA, 0.5, NA, 2.0, 1.5, NA, 0.2),
  z = c(0.2, 0.1, -0.3, 0.4, 0.0, 0.3, -0.2, 0.6),
  s = c(0.1, 0.2, 0.3, 0.1, 0.2, 0.2, 0.4, 0.3)
)
lines <- function(code, pattern) grep(pattern, strsplit(code, "\n")[[1]], value = TRUE)
mi <- bf(y ~ mi(x) + mi(x):z) + bf(x | mi() ~ z) + set_rescor(FALSE)
mi_data <- make_standata(mi, data = d)
mi_code <- make_stancode(mi, data = d)
me_response <- make_stancode(bf(x | mi(s) ~ z), data = d)
complete <- transform(d, x = ifelse(is.na(x), 0.1, x))
me_formula <- bf(y ~ me(x, s) + me(z, s)) + set_mecor(TRUE)
me_code <- make_stancode(me_formula, data = complete)
fit_mi <- brm(mi, data = d, algorithm = "meanfield", iter = 300, refresh = 0,
              seed = 1, output_samples = 5)
fit_me <- brm(me_formula, data = complete, algorithm = "meanfield", iter = 300,
              refresh = 0, seed = 1, output_samples = 5)
result <- list(
  mi = list(
    Nmi_x = mi_data$Nmi_x,
    Jmi_x = as.integer(mi_data$Jmi_x),
    stan = lines(mi_code, "Ymi_x|Yl_x|bsp_y"),
    variables = setdiff(variables(fit_mi), c("lprior", "lp__")),
    log_lik_x = as.numeric(is.na(log_lik(fit_mi, resp = "x")[1, ]))
  ),
  me_response = list(
    Yl = trimws(lines(me_response, "vector\\[N\\] Yl;")),
    stan = lines(me_response, "Jme|noise")
  ),
  me = list(
    stan = lines(me_code, "Xme|zme|meanme|sdme|Lme|corme|Xn"),
    variables = grep("^Xme", setdiff(variables(fit_me), c("lprior", "lp__")),
                     value = TRUE, invert = TRUE),
    prior = get_prior(me_formula, data = complete)[, c("prior", "class", "coef")]
  )
)
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), "brms_missing.json")
