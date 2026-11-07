# Effects

An effect interface declares what operations can be requested and the types of their
arguments and results [@plotkin2009handlers; @bauer2015programming]. A [lexical instance](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler) identifies a particular use of that interface;
an [effect row](../../reference/qvr/effects-and-handlers.md#effect-rows) records the requests permitted at a boundary. Two state instances can
share an interface without sharing the same state.

Typed effect interfaces, lexical rows, and handler signatures.

::: quivers.qiec.effects
