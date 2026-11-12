# Stdlib Effect Instances

Each instance carries a different [kind of information](../../guides/effects.md#stdlib-effect-instances)
along with a computation's result: possible failure (`Maybe`), alternatives (`Alternative_`), a
shared input (`Reader`), state passed explicitly (`State`), accumulated output (`Writer`), a
continuation (`Continuation`), or nothing (`Identity`). The
[monad](https://en.wikipedia.org/wiki/Monad_(functional_programming)) interface sequences each one
[@moggi1991notions; @wadler1992essence]. Every instance is registered with its type class abstract
base classes.

::: quivers.monadic.instances
