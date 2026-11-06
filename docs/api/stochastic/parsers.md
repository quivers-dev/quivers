# Chart Parsers

A chart records results for spans of an input sentence so the parser can reuse a partial
analysis in several larger analyses. This avoids separately rebuilding every parse tree.
The selected rules, weights, and semiring determine whether the answer is recognition, a
best score, or an aggregate weight.

See [rules, charts, and schedules](../../guides/deduction.md#the-framework) and
[comparing inside scores for a parser](../../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores).

For sharing a parsing algorithm across weighting schemes, see [@goodman1999semiring].

Differentiable CKY chart parsers implemented as weighted deductive
systems. `ChartParser` extends `DeductiveSystem` by composing
`LexicalAxiom`, `BinarySpanDeduction`, `UnarySpanDeduction`, `SpanGoal`,
and `CKYSchedule` in one `nn.Module`. It supports learnable rule weights
and semiring-parameterized scoring for log probability, Viterbi,
Boolean recognition, and counting.

`ChartParser.from_schema(schema, category_system, ...)` constructs a
parser from composable `RuleSchema` objects.
`ChartParser.from_category_system(...)` instead accepts an explicit
`RuleSystem`.

Includes concrete `CCGParser` and `LambekParser` convenience subclasses.

::: quivers.stochastic.parsers

::: quivers.stochastic.ccg

::: quivers.stochastic.lambek
