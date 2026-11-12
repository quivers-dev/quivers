# Chart Parsers

A [chart parser](https://en.wikipedia.org/wiki/Chart_parser) records results for spans of an input
sentence, so that one partial analysis serves many larger ones
([the framework](../../guides/deduction.md#the-framework)). `ChartParser` is a differentiable CKY
parser built as a `DeductiveSystem` from `LexicalAxiom`, `BinarySpanDeduction`,
`UnarySpanDeduction`, `SpanGoal`, and `CKYSchedule` in one `nn.Module`. It supports learnable rule
weights and scoring in any chart semiring: log probability,
[inside weight](../../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores), Viterbi,
Boolean recognition, or counting [@goodman1999semiring].
`ChartParser.from_schema(schema, category_system, ...)` builds a parser from rule schemas, and
`ChartParser.from_category_system(...)` takes an explicit `RuleSystem`. `CCGParser` and
`LambekParser` are convenience subclasses.

::: quivers.stochastic.parsers

::: quivers.stochastic.ccg

::: quivers.stochastic.lambek
