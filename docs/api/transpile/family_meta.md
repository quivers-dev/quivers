# Family metadata

`FAMILY_META` records, for each registered distribution family, the
[facts that only transpilation needs](../../semantics/transpile-architecture.md#3-familymeta-the-registry-for-transpile-only-facts),
such as each target's parameter conventions. A target may expect a precision where QVR gives a
standard deviation: for a Normal with standard deviation 2, the renderer passes precision 1/4 so
that the [distribution is preserved](../../semantics/transpile-correctness/index.md#2-correctness-criterion).

::: quivers.transpile.family_meta
