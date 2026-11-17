# Terms

The core [separates values from computations](../../reference/qvr/computations.md#pure-and-effectful-binding)
[@moggi1991notions]. A value can be passed as it is, while a computation may
[request effects](../../semantics/typing.md#6-computations-and-rows) and must be sequenced to give
its result. The term classes make these distinctions explicit, including the continuation that
receives the answer to an operation.

`LogMass(sampleable, origin, batch=())` is the structural total-mass
query, with scalar `LogWeight` result by default. A nonempty `batch`
retains the construction's declared batch axes and returns one log
weight per batch entry. `LogDensity` evaluates the same raw sampleable
value at an outcome. Both terms preserve their source origins through
substitution and serialization; neither introduces a scoring effect.

::: quivers.qiec.terms
