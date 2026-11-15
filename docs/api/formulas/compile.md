# `quivers.formulas.compile`

The `FormulaToQVRModule` lens turns a `Formula` into a `quivers.dsl.ast_nodes.Module`, expanding
formula terms into explicit QVR declarations and program steps
([inspect the generated QVR](../../guides/analysis-data-and-formulas.md#inspect-or-dump-the-generated-qvr)).
A group intercept becomes a set of group effects and a lookup of each observation's group. The
generated module shows exactly which priors, axes, and likelihood the formula produced.
Its optional `distributional` mapping takes parsed `Formula` values for auxiliary
parameters, giving each predictor its own coefficient namespace while sharing group
coding. The lens complement still reconstructs the primary location formula;
auxiliary formulas are retained in the configured lens. Offset and exposure
columns enter native predictor bindings without coefficient draws; their arrays
are retained by `FormulaData.offsets` and returned by `fixed_column_observations`.

::: quivers.formulas.compile
