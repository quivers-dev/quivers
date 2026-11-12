"""Carrier maps, algebra homomorphisms, and change of base for morphisms.

A morphism in [`quivers.core.morphisms`][quivers.core.morphisms] has
entries in the carrier of its algebra ``V``. Two morphisms over
different algebras do not compose (`Morphism.__rshift__` raises
``incompatible algebras``); ``f.change_base(φ)`` moves a morphism to
another algebra by applying a map ``φ`` between the carriers to every
entry.

* `CarrierMap` is the abstract base: a source algebra, a target
  algebra, and an entrywise map ``apply``. It promises no law, so
  change of base along it is an entrywise transformation and need not
  preserve composition.
* `AlgebraHomomorphism` is a carrier map that preserves the algebra's
  operations: ``φ(a ⊗ b) = φ(a) ⊗ φ(b)``, ``φ(a ⊕ b) = φ(a) ⊕ φ(b)``,
  ``φ(1) = 1`` and ``φ(0) = 0``. Between semirings this is a semiring
  homomorphism [Golan (1999)], and change of base along it is a functor
  between the categories of matrices: ``(f >> g).change_base(φ) =
  f.change_base(φ) >> g.change_base(φ)``.

The shipped homomorphisms are `IdentityHom`, `Embedding` (of a
sub-algebra such as ``BOOLEAN ↪ PRODUCT_FUZZY``) and `CountingToReal`.
Every other shipped map (`Expectation`, `LogProb`, `MaxPlus`,
`Threshold`, `MaterialImplication`, `ProbabilityClamp`,
`CountingFromReal`, `ProbabilityToReal`) is a `CarrierMap` that fails
at least one of those equations; the test suite records a
counterexample for each.

A registry `CARRIER_MAP_REGISTRY` keyed by ``(source.name,
target.name)`` lets the compiler and user code look up the shipped map
between two algebras.

References
----------
Golan, J. S. (1999). *Semirings and their Applications*. Springer.
[doi:10.1007/978-94-015-9333-5](https://doi.org/10.1007/978-94-015-9333-5)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.core.algebras import (
    COUNTING,
    GODEL,
    LOG_PROB,
    MARKOV,
    MAX_PLUS,
    PROBABILITY,
    REAL,
)
from quivers.core.algebras import (
    BOOLEAN,
    PRODUCT_FUZZY,
    Algebra,
)


class CarrierMap(ABC):
    """An entrywise map from one algebra's carrier to another's.

    A carrier map promises no law: change of base along it applies the
    map to every entry and need not preserve composition or identities.
    `AlgebraHomomorphism` is the subclass whose maps preserve the
    algebra operations.
    """

    @property
    @abstractmethod
    def source(self) -> Algebra:
        """The source algebra ``V``."""

    @property
    @abstractmethod
    def target(self) -> Algebra:
        """The target algebra ``W``."""

    @abstractmethod
    def apply(self, t: torch.Tensor) -> torch.Tensor:
        """Map a tensor with entries in ``source.L`` to a tensor
        with entries in ``target.L``.

        The shape is preserved; only the per-entry lattice changes.
        """

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.source.name} -> {self.target.name})"


class AlgebraHomomorphism(CarrierMap):
    """A carrier map that preserves the algebra operations.

    ``φ(a ⊗ b) = φ(a) ⊗ φ(b)``, ``φ(⋁ a_i) = ⋁ φ(a_i)``, ``φ(I) = I``
    and ``φ(⊥) = ⊥`` on the source carrier. Between semirings this is a
    semiring homomorphism [Golan (1999)], so change of base along it
    commutes with composition and sends identities to identities.
    Membership is a claim that the equations hold; the test suite checks
    every shipped instance.
    """


class IdentityHom(AlgebraHomomorphism):
    """The identity homomorphism ``id_V : V → V``.

    Useful as the unit of composition; ``f.change_base(IdentityHom(V))``
    leaves the morphism in V unchanged.
    """

    def __init__(self, q: Algebra) -> None:
        self._q = q

    @property
    def source(self) -> Algebra:
        return self._q

    @property
    def target(self) -> Algebra:
        return self._q

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t


class Embedding(AlgebraHomomorphism):
    """The inclusion of a sub-algebra into a super-algebra.

    The canonical example is ``Boolean ↪ ProductFuzzyAlgebra``: a
    boolean tensor with entries in ``{0, 1}`` embeds as a
    product-fuzzy tensor with entries in ``[0, 1]`` whose support
    is the boolean's truth assignment.

    Embeddings preserve every algebra operation up to the
    embedding inclusion; they are *strict* homomorphisms.
    """

    def __init__(self, source: Algebra, target: Algebra) -> None:
        self._source = source
        self._target = target

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        # Cast to the target's natural dtype (float for fuzzy /
        # markov, bool stays bool for boolean) — no value change.
        return t.to(dtype=torch.float32) if t.dtype == torch.bool else t


class Expectation(CarrierMap):
    """Markov → ProductFuzzyAlgebra, reading probabilities as memberships.

    A Markov kernel ``k : A → B`` has rows that sum to 1 (a
    probability distribution over B per a∈A). Reading the
    per-entry probability as a *fuzzy membership* of (a, b)
    in the relation gives a ``ProductFuzzyAlgebra`` morphism.

    The map clamps each entry to ``[0, 1]``. It does not preserve the
    join (a sum of probabilities is not their noisy-OR), so change of
    base along it does not commute with composition.
    """

    def __init__(self) -> None:
        self._source = MARKOV
        self._target = PRODUCT_FUZZY

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0.0, max=1.0)


class LogProb(CarrierMap):
    """ProductFuzzyAlgebra → LogProb via ``log``.

    Maps entries in (0, 1] to (-∞, 0] for
    log-space computation. Entries that are exactly 0 in the
    source go to ``-inf``. In practice the input is clamped to
    ``[ε, 1]`` first to avoid numerical blow-up.
    """

    def __init__(self) -> None:
        self._source = PRODUCT_FUZZY
        self._target = LOG_PROB

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return torch.log(t.clamp(min=1e-30))


class MaxPlus(CarrierMap):
    """ProductFuzzyAlgebra → MaxPlus via ``log``.

    The max-plus tropical algebra on the reals has tensor =
    addition and join = max. A ProductFuzzyAlgebra morphism transports
    via ``log`` so the product t-norm becomes addition in
    log-space; subsequent joins (which were noisy-OR in
    ProductFuzzyAlgebra) get replaced by max in Viterbi-style
    Viterbi-MAP aggregations.

    `LogProb` and `MaxPlus` share the per-entry map ``log`` and differ
    only in the target algebra. Neither preserves the join: ``log`` of a
    noisy-OR is neither the logsumexp nor the maximum of the logs.
    """

    def __init__(self) -> None:
        self._source = PRODUCT_FUZZY
        self._target = MAX_PLUS

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return torch.log(t.clamp(min=1e-30))


class Threshold(CarrierMap):
    """ProductFuzzyAlgebra → Boolean via thresholding.

    Entries above ``tau`` map to True; entries at or below
    ``tau`` map to False. It preserves order (a ≤ b ⟹ threshold(a) ≤ threshold(b)) but
    not the tensor: two entries just above ``tau`` threshold to True
    while their product may not.

    Parameters
    ----------
    tau : float
        Threshold value in [0, 1].
    """

    def __init__(self, tau: float = 0.5) -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"Threshold: tau must be in [0, 1]; got {tau}")
        self._tau = tau
        self._source = PRODUCT_FUZZY
        self._target = BOOLEAN

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return (t > self._tau).to(dtype=torch.float32)


class MaterialImplication(CarrierMap):
    """ProductFuzzyAlgebra → Godel, keeping each entry's value.

    The map is the identity on ``[0, 1]`` (clamped), so subsequent
    ``>>`` composes with min and max rather than product and noisy-OR.
    It preserves neither operation (``0.5 · 0.5 = 0.25`` but
    ``min(0.5, 0.5) = 0.5``), so it is a carrier map and not a
    homomorphism.
    """

    def __init__(self) -> None:
        self._source = PRODUCT_FUZZY
        self._target = GODEL

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0.0, max=1.0)


# Shipped carrier-map instances, exposed as module-level
# singletons. Users compose with ``f.change_base(EXPECTATION)`` to
# perform the change of base.
EXPECTATION = Expectation()
"""The `Expectation` carrier-map singleton, Markov to ProductFuzzy."""
LOG_PROB_MAP = LogProb()
"""The `LogProb` carrier-map singleton, ProductFuzzy to LogProb."""
MAX_PLUS_MAP = MaxPlus()
"""The `MaxPlus` carrier-map singleton, ProductFuzzy to MaxPlus."""
MATERIAL_IMPLICATION = MaterialImplication()
"""The `MaterialImplication` carrier-map singleton, ProductFuzzy to Godel."""


def threshold(tau: float = 0.5) -> Threshold:
    """Build a `Threshold` carrier map at the given threshold value.

    Parameters
    ----------
    tau : float
        The threshold; entries strictly above it map to true.

    Returns
    -------
    Threshold
        The ProductFuzzy-to-Boolean carrier map at ``tau``.
    """
    return Threshold(tau)


def embedding(source: Algebra, target: Algebra) -> Embedding:
    """Build an `Embedding` homomorphism, a sub-algebra inclusion.

    Parameters
    ----------
    source : Algebra
        The sub-algebra.
    target : Algebra
        The super-algebra that ``source`` includes into.

    Returns
    -------
    Embedding
        The inclusion ``source → target``.
    """
    return Embedding(source, target)


class ProbabilityClamp(CarrierMap):
    """Real → Probability via clamping to the unit interval.

    Coerces a real-valued tensor into a probability tensor by
    clamping entries to ``[0, 1]``. It preserves order but not the
    operations: ``clamp(-1 + 0.5) = 0`` while the bounded sum of the
    clamped entries is ``0.5``.
    """

    def __init__(self) -> None:
        self._source = REAL
        self._target = PROBABILITY

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0.0, max=1.0)


class CountingFromReal(CarrierMap):
    """Real → Counting via floor-and-clamp-to-non-negative.

    Coerces a real-valued tensor into a non-negative integer
    counting tensor by flooring and clamping at zero.
    It does not preserve the operations (``floor(0.5 + 0.5) = 1`` while
    ``floor(0.5) + floor(0.5) = 0``); `CountingToReal` is a one-sided
    inverse on counting tensors.
    """

    def __init__(self) -> None:
        self._source = REAL
        self._target = COUNTING

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0.0).floor()


class ProbabilityToReal(CarrierMap):
    """Bounded-sum product algebra → Real, the inclusion of ``[0, 1]``.

    The entries already lie in ``[0, 1] ⊂ ℝ``. The inclusion preserves
    the product but not the join: the source sums with a cap at 1
    (``0.5 ⊕ 0.75 = 1``) while the real sum is ``1.25``.
    """

    def __init__(self) -> None:
        self._source = PROBABILITY
        self._target = REAL

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t


class CountingToReal(AlgebraHomomorphism):
    """Counting → Real (sub-algebra inclusion).

    Non-negative integers embed canonically in the reals; this
    is the strict inclusion homomorphism.
    """

    def __init__(self) -> None:
        self._source = COUNTING
        self._target = REAL

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype=torch.float32)


PROBABILITY_CLAMP = ProbabilityClamp()
"""The `ProbabilityClamp` carrier-map singleton, Real to Probability."""
COUNTING_FROM_REAL = CountingFromReal()
"""The `CountingFromReal` carrier-map singleton, Real to Counting."""
PROBABILITY_TO_REAL = ProbabilityToReal()
"""The `ProbabilityToReal` carrier-map singleton, Probability to Real."""
COUNTING_TO_REAL = CountingToReal()
"""The `CountingToReal` homomorphism singleton, Counting to Real."""


CARRIER_MAP_REGISTRY: dict[tuple[str, str], CarrierMap] = {
    ("Markov", "ProductFuzzy"): EXPECTATION,
    ("ProductFuzzy", "LogProb"): LOG_PROB_MAP,
    ("ProductFuzzy", "MaxPlus"): MAX_PLUS_MAP,
    ("ProductFuzzy", "Boolean"): Threshold(0.5),
    ("ProductFuzzy", "Godel"): MATERIAL_IMPLICATION,
    ("Boolean", "ProductFuzzy"): Embedding(BOOLEAN, PRODUCT_FUZZY),
    ("Real", "Probability"): PROBABILITY_CLAMP,
    ("Real", "Counting"): COUNTING_FROM_REAL,
    ("Probability", "Real"): PROBABILITY_TO_REAL,
    ("Counting", "Real"): COUNTING_TO_REAL,
}
"""The shipped carrier maps keyed by ``(source.name, target.name)``,
so compiler and user code can look up the standard bridge between two
algebras rather than construct one by hand."""


def lookup_carrier_map(source: Algebra, target: Algebra) -> CarrierMap | None:
    """Look up the shipped carrier map ``source → target``.

    Identity is always available: algebras of one type yield
    ``IdentityHom(source)``.

    Parameters
    ----------
    source : Algebra
        The algebra to map from.
    target : Algebra
        The algebra to map to.

    Returns
    -------
    CarrierMap or None
        The registered bridge from `CARRIER_MAP_REGISTRY`, or
        ``None`` when no canonical bridge is known.
    """
    if type(source) is type(target):
        return IdentityHom(source)
    return CARRIER_MAP_REGISTRY.get((source.name, target.name))


__all__ = [
    "CarrierMap",
    "AlgebraHomomorphism",
    "IdentityHom",
    "Embedding",
    "Expectation",
    "LogProb",
    "MaxPlus",
    "Threshold",
    "MaterialImplication",
    "ProbabilityClamp",
    "CountingFromReal",
    "ProbabilityToReal",
    "CountingToReal",
    "EXPECTATION",
    "LOG_PROB_MAP",
    "MAX_PLUS_MAP",
    "MATERIAL_IMPLICATION",
    "PROBABILITY_CLAMP",
    "COUNTING_FROM_REAL",
    "PROBABILITY_TO_REAL",
    "COUNTING_TO_REAL",
    "threshold",
    "embedding",
    "CARRIER_MAP_REGISTRY",
    "lookup_carrier_map",
]
