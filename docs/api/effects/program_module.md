# Program encoding

This module encodes a `MonadicProgram` as a [Quivers Indexed Effect Core](../qiec/index.md) (QIEC) computation. Each step's Python part is a request
on the prelude's `Compute` interface over the values it reads, and each draw is a `Random.sample`
request on the sampleable that part returns. Running the encoding on the reference machine
collects the program's [results and scores](../../reference/qvr/probabilistic-programs.md#running-and-scoring)
([runtime providers](../../reference/qvr/execution-and-tooling.md#runtime-providers)).

::: quivers.effects.program_module
