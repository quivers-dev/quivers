# `quivers.dsl.emit`

`module_to_source` prints an AST `Module` as `.qvr` source in a fixed layout
([module to source](../../guides/analysis-fitting-and-diagnostics.md#module-to-source-quiversdslemit)).
Reparsing its output with `quivers.dsl.loads` gives the same compiled program, which makes it
suited to [inspecting generated programs](../../guides/analysis-data-and-formulas.md#inspect-or-dump-the-generated-qvr)
and to testing that parsing and printing agree. Translation into other languages is
[transpilation](../transpile/index.md).

::: quivers.dsl.emit
