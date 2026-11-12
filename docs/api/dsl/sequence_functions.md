# Sequence functions

These compiler helpers elaborate
[learned functions over whole sequences](../../reference/qvr/probabilistic-programs.md#sequence-functions),
a `[role=function]` morphism with `param_source=self_attention` or `param_source=positions(N)`,
to comprehensions of the checked core over a sequence's open extent. Their runtime form is
[`SequenceFunction`][quivers.continuous.deterministic.SequenceFunction].

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.sequence_functions
