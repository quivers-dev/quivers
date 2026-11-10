# `quivers.formulas.formula`

[Parsing a formula](../../guides/analysis-data-and-formulas.md#formulas-quiversformulas) separates ordinary predictor columns from group-specific terms. The
resulting record carries design information that later compilation uses to create
parameter sites and indexed lookups. A formula describes model structure; its [design
matrices](../../tutorials/python/08-analysis-pipelines.md#a-first-fit-gaussian-regression-in-one-line) also depend on the data and category encoding.

Parsed-formula IR: `Formula`, `FixedColumn`, `RandomTerm`, plus the
`formula_from_data` adapter over `formulae.design_matrices`.

::: quivers.formulas.formula
