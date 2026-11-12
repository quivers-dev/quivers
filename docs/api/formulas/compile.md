# `quivers.formulas.compile`

The `FormulaToQVRModule` lens turns a `Formula` into a `quivers.dsl.ast_nodes.Module`, expanding
formula terms into explicit QVR declarations and program steps
([inspect the generated QVR](../../guides/analysis-data-and-formulas.md#inspect-or-dump-the-generated-qvr)).
A group intercept becomes a set of group effects and a lookup of each observation's group. The
generated module shows exactly which priors, axes, and likelihood the formula produced.

::: quivers.formulas.compile
