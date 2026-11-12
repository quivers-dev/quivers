# `quivers.transpile`

[Transpilation](../../semantics/transpile-architecture.md#1-the-shared-pipeline) translates a
checked QVR module into the source of another probabilistic programming language. The operations a
target supports determine which models it accepts, and validation checks the generated syntax and
[compares log densities at fixed latent values](../../semantics/transpile-correctness/index.md#2-correctness-criterion).
Every public name of the package is importable from `quivers.transpile`. The constants and type
aliases its leaf modules define are listed below under the package path.

::: quivers.transpile
    options:
      members:
        - transpile
        - available_targets
        - UnsupportedConstruct
        - RefusedDeclaration
        - unsupported_for
        - Backend
        - STAN_LIKE
        - PYTHON_DEEP
        - CHURCH_LIKE
        - CATEGORICAL_METADATA_IGNORABLE
        - QIEC_SURFACE
        - EmitPretty
        - parser_registry
        - target_protocol
        - FAMILY_META
        - MarginalizeAtomSet
        - DYNAMIC_TARGET_LANGUAGES
        - TARGET_NAME_SEPARATOR
        - QiecFeature
        - EmitFn
