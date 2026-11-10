# Family metadata

[Distribution metadata](../../semantics/transpile-architecture.md#3-familymeta-the-registry-for-transpile-only-facts) records parameter conventions needed by each renderer. A target
may expect precision where QVR supplies standard deviation. For a Normal with standard
deviation 2, the renderer must pass precision 1/4 to [preserve the distribution](../../semantics/transpile-correctness/index.md#2-correctness-criterion).

Transpilation metadata for registered distribution families.

::: quivers.transpile.family_meta
