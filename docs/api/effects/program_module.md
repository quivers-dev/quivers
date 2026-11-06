# Program encoding

This module connects the Python program API to checked core execution. Use it when a
caller needs to invoke a program through the runtime module interface and collect its
results and scores.

See [returning program values and scores](../../reference/qvr/probabilistic-programs.md#running-and-scoring) and
[the runtime providers needed for execution](../../reference/qvr/execution-and-tooling.md#runtime-providers).

For algebraic effects and handler interpretation, see [@plotkin2009handlers; @bauer2015programming].

The kernel encoding of a `MonadicProgram`.

::: quivers.effects.program_module
