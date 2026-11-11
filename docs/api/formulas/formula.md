# `quivers.formulas.formula`

[Parsing a formula](../../guides/analysis-data-and-formulas.md#formulas-quiversformulas) separates ordinary predictor columns from group-specific terms. The
resulting record carries design information that later compilation uses to create
parameter sites and indexed lookups. A formula describes model structure; its [design
matrices](../../tutorials/python/08-analysis-pipelines.md#a-first-fit-gaussian-regression-in-one-line) also depend on the data and category encoding.

Parsed-formula IR: `Formula`, `FixedColumn`, `RandomTerm`, plus the
`formula_from_data` adapter over `formulae.design_matrices`.

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
