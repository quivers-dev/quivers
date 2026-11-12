# Typeclass Hierarchy

The [type classes](https://en.wikipedia.org/wiki/Type_class) `Functor`, `Applicative`, `Monad`,
`Alternative`, `MonadPlus`, `Foldable`, `Traversable`, and `MonadTrans` group the operations for
[mapping, combining, and sequencing computations](../../guides/effects.md#typeclass-hierarchy), each
as an abstract base class with laws [@moggi1991notions; @wadler1992essence]. A functor law, for
instance, requires two successive maps to equal one map of the composite function. An applicative
combines computations whose structure is fixed before their results are known
[@mcbride2008applicative], while monadic bind can choose the next computation from an earlier
result.

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
