# `quivers.formulas.compile`

This conversion expands formula terms into explicit QVR declarations and program steps.
A group intercept becomes a collection of latent effects plus a lookup for each
observation’s group. Inspect the generated module when you need to see exactly which
priors, axes, and likelihood the formula frontend has constructed.

See [inspecting the generated QVR](../../guides/analysis-data-and-formulas.md#inspect-or-dump-the-generated-qvr) and
[a Gaussian regression fit](../../tutorials/python/08-analysis-pipelines.md#a-first-fit-gaussian-regression-in-one-line).

For regression and partial pooling, see [@gelman2013bayesian].

The `FormulaToQVRModule` lens: `Formula` → `quivers.dsl.ast_nodes.Module`.

::: quivers.formulas.compile
