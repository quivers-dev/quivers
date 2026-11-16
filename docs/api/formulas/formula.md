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
terms, offsets, response modifiers, response values, group levels, and group indices all agree, and
hashing agrees with equality. The arrays with one entry per row are stored as
read-only copies made at construction: writing into
`formula.response_values` or a column's `data` raises `ValueError`, and
changing the source data frame afterwards does not change the formula.
Copy a field (`formula.response_values.copy()`) to get a writable array.

`Formula.response_layout` retains response carrier and event shape, named
vector components, and scalar factor levels with their declared order.
`cbind(a, b, c)` contributes one vector event per row, whereas a factor
response contributes one integer code per row. `FormulaData` retains this
layout through the lens complement; neither serialization nor prediction
flattens the event into observation rows.

`Formula.offsets` retains coefficient-free `offset(expr)` and `exposure(expr)`
columns as read-only `FixedColumn` values. Exposure contributes the logarithm of
positive finite values. The [offset guide](../../guides/analysis-data-and-formulas.md#offsets-and-observation-exposure)
shows how these terms affect fitting and posterior predictors.

`formula_from_data(..., weights="column")` retains a finite nonnegative
numeric column as `Formula.modifiers.weights`. `Formula.modifiers.decision`
retains a normalized binary decision column for a joint response model.
Both belong to `ResponseModifiers`; `FormulaData.modifiers` preserves them
in the lens complement. The
[likelihood-weight guide](../../guides/analysis-data-and-formulas.md#likelihood-weights)
explains the objective, row alignment, and prediction contract.

`cs(expr)` marks a transition-specific fixed or group column through
`FixedColumn.category_specific`. The formula lens retains those flags and
the authored fixed-column order in `FormulaData.category_specific_columns`
and `fixed_column_order`; serialization through the value IR does not
silently turn a category-specific coefficient into an ordinary scalar.
`cs(1)` uses an internally prepared ones column, with an explicit refusal
if the caller already uses its reserved name. The
[ordinal guide](../../guides/analysis-data-and-formulas.md#ordinal-families-thresholds-and-category-specific-effects)
states supported constructions and prediction shapes.

::: quivers.formulas.formula
