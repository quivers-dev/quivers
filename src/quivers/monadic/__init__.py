"""Monadic structures: typeclass hierarchy, instances,
algebraic effects, comonads, algebras, and distributive laws.

The `Monad` ABC lives in `typeclasses`. Concrete monad
implementations (``FuzzyPowersetMonad``,
``GiryMonad``, etc.) subclass it directly and provide the
``fmap_obj`` / ``fmap`` / ``pure`` / ``apply`` / ``join`` operations.
``unit`` and ``multiply`` are exposed as aliases on the concrete
classes for the Eilenberg–Moore vocabulary.
"""

from quivers.monadic.typeclasses import (
    Functor,
    Applicative,
    Monad,
    Alternative,
    MonadPlus,
    Foldable,
    Traversable,
    MonadTrans,
)
from quivers.monadic.monads import (
    FuzzyPowersetMonad,
    KleisliCategory,
)
from quivers.monadic.comonads import (
    Comonad,
    CoKleisliCategory,
    CofreeComonad,
)
from quivers.monadic.algebras import (
    Algebra,
    FreeAlgebra,
    ObservedAlgebra,
    Coalgebra,
    CofreeCoalgebra,
    ObservedCoalgebra,
    EilenbergMooreCategory,
)
from quivers.monadic.distributive_laws import (
    DistributiveLaw,
)
from quivers.monadic.instances import (
    Identity,
    Maybe,
    Alternative_,
    Continuation,
    State,
    Reader,
    Writer,
)
from quivers.monadic.algebraic import (
    Operation,
    EffectSignature,
    Handler,
    FreeMonad,
)
from quivers.monadic.bridges import (
    Kleisli,
    ArrowMonad,
    CoKleisli,
    kleisli,
    arrow_monad,
    cokleisli,
)
from quivers.monadic.laws import (
    check_functor_laws,
    check_monad_laws,
)
from quivers.monadic.theories import (
    TypeclassTheory,
    ThFunctor,
    ThApplicative,
    ThMonad,
    ThAlternative,
    ThMonadPlus,
    ThMonadTrans,
    ThFoldable,
    ThTraversable,
)

__all__ = [
    # typeclasses
    "Functor",
    "Applicative",
    "Monad",
    "Alternative",
    "MonadPlus",
    "Foldable",
    "Traversable",
    "MonadTrans",
    # monads
    "FuzzyPowersetMonad",
    "KleisliCategory",
    # comonads
    "Comonad",
    "CoKleisliCategory",
    "CofreeComonad",
    # algebras
    "Algebra",
    "FreeAlgebra",
    "ObservedAlgebra",
    "Coalgebra",
    "CofreeCoalgebra",
    "ObservedCoalgebra",
    "EilenbergMooreCategory",
    # distributive_laws
    "DistributiveLaw",
    # instances
    "Identity",
    "Maybe",
    "Alternative_",
    "Continuation",
    "State",
    "Reader",
    "Writer",
    # algebraic
    "Operation",
    "EffectSignature",
    "Handler",
    "FreeMonad",
    # bridges
    "Kleisli",
    "ArrowMonad",
    "CoKleisli",
    "kleisli",
    "arrow_monad",
    "cokleisli",
    # laws
    "check_functor_laws",
    "check_monad_laws",
    # theories
    "TypeclassTheory",
    "ThFunctor",
    "ThApplicative",
    "ThMonad",
    "ThAlternative",
    "ThMonadPlus",
    "ThMonadTrans",
    "ThFoldable",
    "ThTraversable",
]
