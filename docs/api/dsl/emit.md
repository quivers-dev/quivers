# `quivers.dsl.emit`

Emission prints a syntax tree as QVR text with a consistent layout. Use it to inspect
generated programs or test whether parsing and printing preserve the represented source
structure. Translations into other programming languages use the transpilation API.

See [printing a module as QVR](../../guides/analysis-fitting-and-diagnostics.md#module-to-source-quiversdslemit) and
[inspecting a formula-generated model](../../guides/analysis-data-and-formulas.md#inspect-or-dump-the-generated-qvr).

Emission from an AST to canonical `.qvr` source. The emitter walks a
`Module`; reparsing its output with `quivers.dsl.loads` produces the
same compiled program.

::: quivers.dsl.emit
