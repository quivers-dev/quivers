# `quivers.formulas.formula`

[Parsing a formula](../../guides/analysis-data-and-formulas.md#formulas-quiversformulas) separates the
ordinary predictor columns (`FixedColumn`) from the group terms (`RandomTerm`), and the `Formula`
record carries the design information from which compilation builds parameter sites and indexed
lookups. `formula_from_data` builds it over `formulae.design_matrices`, so its
[design matrices](../../tutorials/python/08-analysis-pipelines.md#a-first-fit-gaussian-regression-in-one-line)
depend on the data and the category encoding as well as on the formula.

`Formula`, `FormulaData`, and `FixedColumn` are immutable values that
compare by every field. Two formulas are equal only when their formula
strings, response names, fixed columns (labels and data), random
terms, response values, group levels, and group indices all agree, and
hashing agrees with equality. The per-row arrays are stored as
read-only copies made at construction: writing into
`formula.response_values` or a column's `data` raises `ValueError`, and
changing the source data frame afterwards does not change the formula.
Copy a field (`formula.response_values.copy()`) to get a writable array.

::: quivers.formulas.formula
