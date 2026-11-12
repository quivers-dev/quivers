# Checked programs

A `CheckedProgram` is a program morphism that runs its checked [Quivers Indexed Effect Core](../qiec/index.md) (QIEC) computation on the
[reference machine](../../developer/qiec.md#reference-evaluator), the library's interpreter for QIEC
([running and scoring](../../reference/qvr/probabilistic-programs.md#running-and-scoring)). The
compiler produces one for a program whose calls it cannot turn into compiled steps, so that callers
that expect a morphism can still sample and score it
([runtime providers](../../reference/qvr/execution-and-tooling.md#runtime-providers)).

::: quivers.effects.checked_program
