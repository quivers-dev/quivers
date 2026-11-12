# Rule Systems

A [rule](../../semantics/schemas.md#5-rule-declarations) says which recognized constituents combine
into another constituent. A `RuleSystem` stores binary and unary inference rules as index tensors,
and the `+` operator merges two rule systems into one grammar. A
[probabilistic grammar](../../examples/schema-chart-parser.md) also needs its rule weights
normalized.

A rule system may carry **learnable weights**, one log weight per binary
rule and one per unary rule. Parsing adds these weights to the chart
combination scores. When passed to a `ChartParser`, the weights
initialize its `nn.Parameter` tensors, or fixed buffers when
`learnable_rule_weights=False`. Merging with `+` preserves each rule's
weight, and a duplicate rule keeps the weight from the left operand.

The convenience functions `ccg_rules` and `lambek_rules` instantiate
[rule schema](biclosed.md) presets over a given category system.
A custom grammar composes the schema primitives directly with `|`.

::: quivers.stochastic._rule_system

::: quivers.stochastic.rules
    options:
      filters: ["!^_", "!^RuleSystem$"]
