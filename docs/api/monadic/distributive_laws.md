# Distributive Laws

A distributive law specifies how [two layers of monadic structure interact](../../guides/monadic.md#distributive-laws). Moving from a
list of optional values to an optional list, for instance, requires deciding how a
missing element affects the whole result. The compatibility equations ensure that the
chosen map respects the two monads’ units and multiplications [@moggi1991notions; @wadler1992essence].

Distributive laws between monads and comonads.

::: quivers.monadic.distributive_laws
