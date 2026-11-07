# Stdlib Effect Instances

Choose an instance by [what a computation carries along with its result](../../guides/effects.md#stdlib-effect-instances): possible failure
(`Maybe`), alternatives (`ListMonad`), shared input (`Reader`), mutable state passed
explicitly (`State`), or accumulated output (`Writer`). The same sequencing interface
then handles that information consistently [@moggi1991notions; @wadler1992essence]. The classes below specify the concrete
representations and supported operations.

Effect instances registered with their typeclass ABCs: `Identity`,
`Maybe`, `Alternative_`, `Continuation`, `State`, `Reader`, `Writer`,
and `ListMonad`.

::: quivers.monadic.instances
