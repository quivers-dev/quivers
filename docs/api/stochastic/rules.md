# Rule Systems

A rule specifies how recognized constituents combine into another constituent. Rule
indices record the allowed combinations, and optional weights score their applications.
Merging systems makes their rules jointly available. A probabilistic grammar
additionally requires the appropriate normalization of its rule weights.

See [declaring combinatory rules](../../semantics/schemas.md#5-rule-declarations) and
[a parser built from schemas](../../examples/schema-chart-parser.md).

Composable structural rule systems for chart parsing. A `RuleSystem`
stores binary and unary inference rules as index tensors. The `+`
operator merges rule systems for hybrid grammars.

A rule system may carry **learnable weights**, one log weight per binary
rule and one per unary rule. Parsing adds these weights to the chart
combination scores. When passed to a `ChartParser`, the weights
initialize its `nn.Parameter` tensors, or fixed buffers when
`learnable_rule_weights=False`. Merging with `+` preserves each rule's
weight; duplicate rules keep the weight from the left operand.

The convenience functions `ccg_rules` and `lambek_rules` instantiate
[rule schema](biclosed.md) presets over a given category system.
For custom grammars, compose the schema primitives directly via `|`.

::: quivers.stochastic._rule_system

::: quivers.stochastic.rules
    options:
      filters: ["!^_", "!^RuleSystem$"]
