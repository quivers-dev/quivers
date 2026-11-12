# `quivers.data.schema`

A `DatasetSchema` records the meaning and organization of encoded columns, including their roles
as model inputs and observations ([data frames](../../guides/analysis-data-and-formulas.md#dataframes-quiversdata)).
It is the shared description from which QVR declarations are built and data are packed, and the
`compose` helper keeps the model's finite index sets aligned with the
[category codes of the table](../../guides/inference-foundations.md#host-data-per-row-covariates-and-index-arrays).

::: quivers.data.schema
