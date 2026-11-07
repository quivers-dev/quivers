# Terms

The core [separates values from computations](../../reference/qvr/computations.md#pure-and-effectful-binding). A value can be passed directly; a
computation may [request effects](../../semantics/typing.md#6-computations-and-rows) and must be sequenced to obtain its result [@moggi1991notions]. Terms make
those distinctions explicit, including the continuation that receives an effect
operation’s answer.

Stable value and computation terms.

::: quivers.qiec.terms
