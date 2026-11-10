# `quivers.formulas.compile`

This conversion [expands formula terms](../../guides/analysis-data-and-formulas.md#inspect-or-dump-the-generated-qvr) into explicit QVR declarations and program steps.
A group intercept becomes a collection of latent effects plus a lookup for each
observation’s group. Inspect the generated module when you need to see exactly which
priors, axes, and likelihood the formula frontend has constructed.

The `FormulaToQVRModule` lens: `Formula` → `quivers.dsl.ast_nodes.Module`.

::: quivers.formulas.compile
