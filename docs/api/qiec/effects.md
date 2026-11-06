# Effects

An effect interface declares what operations can be requested and the types of their
arguments and results. A lexical instance identifies a particular use of that interface;
an effect row records the requests permitted at a boundary. Two state instances can
share an interface without sharing the same state.

See [effect instances and rows](../../reference/qvr/effects-and-handlers.md#effect-rows) and
[writing and running a handler](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler).

For the general semantics of effect handlers, see [@plotkin2009handlers; @bauer2015programming].

Typed effect interfaces, lexical rows, and handler signatures.

::: quivers.qiec.effects
