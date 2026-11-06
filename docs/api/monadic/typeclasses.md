# Typeclass Hierarchy

The typeclasses group operations for mapping, combining, and sequencing computations. An
abstract base class identifies the methods an instance supplies. Its accompanying laws
specify how those methods must agree, such as whether two successive mappings equal one
mapping of the composite function.

For the relationship between mapping and sequencing, see Milewski's [functors](https://bartoszmilewski.com/2015/01/20/functors/) and [monads](https://bartoszmilewski.com/2016/11/21/monads-programmers-definition/) chapters [@milewski2015functors; @milewski2016monads].

See [mapping, application, and sequencing](../../guides/effects.md#typeclass-hierarchy) and
[the mathematical interpretation of the interfaces](../../semantics/effects.md#2-typeclass-denotations).

For the semantics of monadic computation, see [@moggi1991notions; @wadler1992essence].

ABCs for `Functor`, `Applicative`, `Monad`, `Alternative`, `MonadPlus`,
`Foldable`, `Traversable`, and `MonadTrans`.

For applicative functors and their laws, see [@mcbride2008applicative].

Sources for the constructions below: [@hughes2000arrows; @bauer2015programming; @bumford2026effect].

::: quivers.monadic.typeclasses

## Package constants

The typeclass theories of `quivers.monadic`, each a `TypeclassTheory`.

::: quivers.monadic
    options:
      members:
        - ThFunctor
        - ThApplicative
        - ThMonad
        - ThAlternative
        - ThMonadPlus
        - ThMonadTrans
        - ThFoldable
        - ThTraversable
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
