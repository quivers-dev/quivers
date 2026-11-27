# Checked programs

A `CheckedProgram` is a program morphism that runs its checked [Quivers Indexed Effect Core](../qiec/index.md) (QIEC) computation on the
[reference machine](../../developer/qiec.md#reference-evaluator), the library's interpreter for QIEC
([running and scoring](../../reference/qvr/probabilistic-programs.md#running-and-scoring)). The
compiler produces one for a program whose calls it cannot turn into compiled steps. It runs
under the effect handler stack with tensor values. Inference methods such as MCMC and SVI
read the latents from a trace and use the same log joint and its gradients
([runtime providers](../../reference/qvr/execution-and-tooling.md#runtime-providers)).

::: quivers.effects.checked_program
