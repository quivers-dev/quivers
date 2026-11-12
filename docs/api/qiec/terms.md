# Terms

The core [separates values from computations](../../reference/qvr/computations.md#pure-and-effectful-binding)
[@moggi1991notions]. A value can be passed as it is, while a computation may
[request effects](../../semantics/typing.md#6-computations-and-rows) and must be sequenced to give
its result. The term classes make these distinctions explicit, including the continuation that
receives the answer to an operation.

::: quivers.qiec.terms
