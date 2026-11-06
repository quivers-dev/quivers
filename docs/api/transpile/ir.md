# Structural IR

An intermediate representation (IR) records program structure without committing to one
target language’s syntax. A distribution call can retain its parameters and support
while renderers decide how that target spells it. This separation helps distinguish a
modeling operation from a formatting choice.

See [the shared intermediate representation](../../semantics/transpile-architecture.md#2-the-ir) and
[the score-preservation criterion for a translation](../../semantics/transpile-correctness/index.md#2-correctness-criterion).

The structural intermediate representation renderers consume.

::: quivers.transpile.ir
