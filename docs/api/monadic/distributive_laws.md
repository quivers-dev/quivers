# Distributive Laws

A distributive law specifies how two layers of monadic structure interact. Moving from a
list of optional values to an optional list, for instance, requires deciding how a
missing element affects the whole result. The compatibility equations ensure that the
chosen map respects the two monads’ units and multiplications.

See [compatibility of two monads](../../guides/monadic.md#distributive-laws) and
[an introduction to sequencing computations](../../getting-started/concepts.md#6-monads-and-handlers-describe-how-computations-continue).

For the semantics of monadic computation, see [@moggi1991notions; @wadler1992essence].

Distributive laws between monads and comonads.

::: quivers.monadic.distributive_laws
