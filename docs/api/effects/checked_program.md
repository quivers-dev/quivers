# Checked programs

This wrapper exposes a core computation that has passed type and effect checking through
the [runtime morphism interface](../../reference/qvr/execution-and-tooling.md#runtime-providers). It [executes that computation on the reference machine](../../reference/qvr/probabilistic-programs.md#running-and-scoring),
connecting the checked program to callers that expect a morphism.

A program morphism that runs its checked computation on the reference machine.

::: quivers.effects.checked_program
