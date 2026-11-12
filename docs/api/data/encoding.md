# `quivers.data.encoding`

[Column encoding](../../guides/analysis-data-and-formulas.md#dataframes-quiversdata) turns a column of a
data frame into runtime values: a participant name becomes an integer index, and a numeric response
a tensor. The `ColumnRole` and `MissingPolicy` enums record a column's role and its policy for
missing values, which together with the category mapping fix what an encoded value means when a
model is fit and its predictions inspected
([data inputs](../../guides/inference-foundations.md#host-data-per-row-covariates-and-index-arrays)).

::: quivers.data.encoding
