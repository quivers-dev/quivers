# Interpreter

`run_program` runs a `MonadicProgram` on the
[reference machine](../../developer/qiec.md#reference-evaluator) under the handler stack
[@plotkin2009handlers; @bauer2015programming]. It encodes the program as a QIEC computation,
installs the stacked handlers as lexically scoped handlers of the program's `random` and `score`
instances, and runs the program's Python steps through `Compute` handlers over PyTorch. A draw
receives its value from its handler, and an
[observation adds its score](../../guides/inference-foundations.md#conditioning-on-observations) as
the handler interprets it. A caller that wants an
[execution trace](../../tutorials/python/05-variational-inference.md#tracing-the-model) stacks a
`TraceHandler` first.

::: quivers.effects.interpreter
