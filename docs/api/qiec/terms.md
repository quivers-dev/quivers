# Terms

The core separates values from computations. A value can be passed directly; a
computation may request effects and must be sequenced to obtain its result. Terms make
those distinctions explicit, including the continuation that receives an effect
operation’s answer.

See [values, computations, and sequencing](../../reference/qvr/computations.md#pure-and-effectful-binding) and
[the typing of computations and their effects](../../semantics/typing.md#6-computations-and-rows).

For the value/computation distinction motivating monadic semantics, see [@moggi1991notions].

Stable value and computation terms.

::: quivers.qiec.terms
