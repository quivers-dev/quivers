# Program runtime

`sample_program` and `run_program` [run a program's entry point](../../reference/qvr/execution-and-tooling.md#run-an-entry)
on the [reference machine](../../developer/qiec.md#reference-evaluator) and return its
[value and log joint density](../../reference/qvr/probabilistic-programs.md#running-and-scoring).
`sample_program` draws the sites it is not given, and `run_program` replays a value for every
site.

::: quivers.qiec.program_runtime
