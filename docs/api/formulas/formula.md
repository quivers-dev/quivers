# `quivers.formulas.formula`

Parsing a formula separates ordinary predictor columns from group-specific terms. The
resulting record carries design information that later compilation uses to create
parameter sites and indexed lookups. A formula describes model structure; its design
matrices also depend on the data and category encoding.

See [formula terms and generated models](../../guides/analysis-data-and-formulas.md#formulas-quiversformulas) and
[a Gaussian regression fit](../../tutorials/python/08-analysis-pipelines.md#a-first-fit-gaussian-regression-in-one-line).

For regression and partial pooling, see [@gelman2013bayesian].

Parsed-formula IR: `Formula`, `FixedColumn`, `RandomTerm`, plus the
`formula_from_data` adapter over `formulae.design_matrices`.

::: quivers.formulas.formula
