# Checked programs

This wrapper exposes a core computation that has passed type and effect checking through
the runtime morphism interface. It executes that computation on the reference machine,
connecting the checked program to callers that expect a morphism.

See [running a checked program](../../reference/qvr/probabilistic-programs.md#running-and-scoring) and
[the runtime providers needed for execution](../../reference/qvr/execution-and-tooling.md#runtime-providers).

For algebraic effects and handler interpretation, see [@plotkin2009handlers; @bauer2015programming].

A program morphism that runs its checked computation on the reference machine.

::: quivers.effects.checked_program
