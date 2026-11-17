# Generates brms_group_options.json, pinned by tests/formulas/test_group_options.py.
# brms oracle for gr(by = ) and gr(dist = "student"): level and by codes (J,
# Jby), the generated Stan lines that build the group-level effects, and the
# prior catalogue (sd by level, one cor per by level, df with its default).
# gr(pw = ) postdates brms 2.20.4 and is pinned from the 2.23 source instead.
# Run from this directory: Rscript brms_group_options.R
suppressMessages(library(brms))
suppressMessages(library(jsonlite))
d <- data.frame(
  y = c(0.3, -1.2, 0.8, 1.9, -0.4, 0.1, 1.4, -0.7, 0.6, 2.2, -1.5, 0.9),
  x = c(0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8, 0.4, -0.3),
  g = c("a", "b", "c", "d", "a", "b", "c", "d", "a", "b", "c", "d"),
  f = c("p", "p", "q", "q", "p", "p", "q", "q", "p", "p", "q", "q")
)
formulas <- c(
  "y ~ x + (1 + x | gr(g, by = f))",
  "y ~ x + (1 + x || gr(g, by = f))",
  "y ~ x + (1 + x | gr(g, dist = 'student'))",
  "y ~ x + (1 | gr(g, dist = 'student'))"
)
result <- lapply(formulas, function(text) {
  formula <- as.formula(text)
  standata <- unclass(make_standata(formula, data = d))
  keep <- grepl("^(J_[0-9]+|Jby_[0-9]+|Nby_[0-9]+|N_[0-9]+|M_[0-9]+)$", names(standata))
  code <- strsplit(make_stancode(formula, data = d), "\n")[[1]]
  prior <- get_prior(formula, data = d)
  rows <- prior[prior$class %in% c("sd", "cor", "L", "df"), ]
  list(
    formula = text,
    data = lapply(standata[keep], as.numeric),
    stan = trimws(grep("r_1|dfm|udf|df_1|scale_r_cor|sd_1|L_1", code, value = TRUE)),
    prior = data.frame(prior = rows$prior, class = rows$class, coef = rows$coef,
                       group = rows$group, lb = rows$lb)
  )
})
writeLines(
  toJSON(list(brms = as.character(packageVersion("brms")), data = d, cases = result),
         auto_unbox = TRUE, pretty = TRUE, digits = NA),
  "brms_group_options.json"
)
