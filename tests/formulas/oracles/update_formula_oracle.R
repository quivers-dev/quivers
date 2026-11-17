# R's update.formula on the cases pinned in tests/formulas/test_update_formula.py.
cases <- list(
  c("y ~ x + z", ". ~ . + w"),
  c("y ~ x + z", ". ~ . - z"),
  c("y ~ x", "log(.) ~ ."),
  c("y ~ x", "y2 ~ z"),
  c("y ~ x + (1 | g)", ". ~ . + w"),
  c("y ~ x + (1 | g) + (1 | h)", ". ~ . + w"),
  c("y ~ x + (1 | g)", "~ . + w"),
  c("y ~ x + (1 | g)", ". ~ . - (1 | g)"),
  c("y ~ x + (1 | g)", ". ~ . - x"),
  c("y ~ x + (1 + x | g)", ". ~ . - (1 + x | g) + (1 | g)"),
  c("y ~ x + z", ". ~ . + x"),
  c("y ~ x", ". ~ . - x"),
  c("y ~ x + z", ". ~ . - 1"),
  c("y ~ x + z", ". ~ . + 0"),
  c("y ~ 0 + x", ". ~ . + 1"),
  c("y ~ x + z", ". ~ w + ."),
  c("y ~ x + s(z)", ". ~ . + w"),
  c("y ~ x + z", ". ~ . - (x + z) + w"),
  c("y ~ x + (1 | g)", ". ~ . + (1 | h)"),
  c("y ~ x + z", "sqrt(.) ~ . - z + w"),
  c("y ~ x - 1", ". ~ . + w"),
  c("y ~ x * z", ". ~ . + w"),
  c("y ~ x + I(z - 1)", ". ~ . - I(z - 1)")
)
for (case in cases) {
  out <- update.formula(as.formula(case[1]), as.formula(case[2]))
  cat(case[1], " || ", case[2], " || ", paste(deparse(out, width.cutoff = 500L), collapse = ""), "\n", sep = "")
}
