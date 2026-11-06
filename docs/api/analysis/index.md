# `quivers.analysis`

Program analysis reports which sites repeat, how tensor dimensions change, and where
names are visible. Use these reports to locate a mismatched axis, an unexpectedly shared
parameter, or a declaration that falls outside the scope of its use.

See [shape and initialization diagnostics](../../guides/analysis-fitting-and-diagnostics.md#algebra-guided-training-tooling) and [interactive program exploration](../../guides/repl-and-lsp.md#program-exploration).

Static analysis and inspection utilities derive chain shapes, initialization
recipes, saturation diagnostics, scope paths, and plate graphs without
rewriting a compiled model. The package-level exports cover the training
diagnostics; the plate and scope modules support the REPL and can also be used
by downstream tooling.

::: quivers.analysis
