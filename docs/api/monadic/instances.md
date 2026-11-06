# Stdlib Effect Instances

Choose an instance by what a computation carries along with its result: possible failure
(`Maybe`), alternatives (`ListMonad`), shared input (`Reader`), mutable state passed
explicitly (`State`), or accumulated output (`Writer`). The same sequencing interface
then handles that information consistently. The classes below specify the concrete
representations and supported operations.

See [the standard effect instances](../../guides/effects.md#stdlib-effect-instances) and
[the mathematical interpretation of the interfaces](../../semantics/effects.md#2-typeclass-denotations).

For the semantics of monadic computation, see [@moggi1991notions; @wadler1992essence].

Effect instances registered with their typeclass ABCs: `Identity`,
`Maybe`, `Alternative_`, `Continuation`, `State`, `Reader`, `Writer`,
and `ListMonad`.

::: quivers.monadic.instances
