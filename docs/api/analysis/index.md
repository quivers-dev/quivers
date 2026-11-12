# `quivers.analysis`

The analysis tools inspect a compiled QVR program without rewriting it. They derive chain shapes,
initialization recipes, [saturation diagnostics](../../guides/analysis-fitting-and-diagnostics.md#algebra-guided-training-tooling),
lexical scope paths, and plate graphs, which locate a mismatched axis, a parameter shared by
accident, or a declaration outside the scope of its use. The package exports the training
diagnostics, and the plate and scope modules also serve the
[REPL](../../guides/repl-and-lsp.md#program-exploration).

::: quivers.analysis
