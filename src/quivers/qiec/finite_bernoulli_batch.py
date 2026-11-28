"""Exact independent row axes for finite Bernoulli source computations."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from random import Random

from quivers.qiec._finite_choices import _choose_position, _selector_probabilities
from quivers.qiec._random_site_graph import (
    _Address,
    _check_first_order_graph,
    _random_computation,
)
from quivers.qiec.bernoulli_estimators import _check_request
from quivers.qiec.contracts import SiteEstimator
from quivers.qiec.differentiation import (
    ExactGradient,
    ExactJvp,
    JvpLimit,
    JvpUnsupported,
    _bound,
    _fraction,
)
from quivers.qiec.finite_marginalization import (
    _checked_finite_unit,
    _History,
    _prior_histories,
)
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.laws import _validate_limit
from quivers.qiec.module import QiecModule

type _AxisAddress = tuple[str, int, _Address]


@dataclass(frozen=True, slots=True)
class BernoulliBatchCotangent:
    """Adjoints for the shared prefix and the row-local input vectors.

    Parameters
    ----------
    shared : tuple[Fraction, ...]
        The sum of all rows' adjoints for the same shared inputs.
    rows : tuple[tuple[Fraction, ...], ...]
        The independent row-local adjoints in axis order.
    """

    shared: tuple[Fraction, ...]
    rows: tuple[tuple[Fraction, ...], ...]


@dataclass(frozen=True, slots=True)
class BernoulliBatchLinearization:
    """The exact ordered row outputs and their source-parameter gradients.

    Parameters
    ----------
    shared_arity : int
        The length of the deterministic parameter prefix shared by all rows.
    rows : tuple[ExactGradient, ...]
        The row values and gradients in original source parameter order.
    """

    shared_arity: int
    rows: tuple[ExactGradient, ...]

    @property
    def values(self) -> tuple[Fraction, ...]:
        """Give the row values in axis order.

        Returns
        -------
        tuple[Fraction, ...]
            The output vector.
        """
        return tuple(row.value for row in self.rows)

    def jvp(
        self,
        shared: tuple[int | Fraction, ...],
        rows: tuple[tuple[int | Fraction, ...], ...],
    ) -> tuple[Fraction, ...]:
        """Apply the exact batch Jacobian to shared and row-local tangents.

        Parameters
        ----------
        shared : tuple[int or Fraction, ...]
            The one common tangent for the shared source inputs.
        rows : tuple[tuple[int or Fraction, ...], ...]
            A local tangent for each row in axis order.

        Returns
        -------
        tuple[Fraction, ...]
            The output tangent for each row.

        Raises
        ------
        ValueError
            If any of the input tangent axes have the wrong extent.
        """
        if len(shared) != self.shared_arity or len(rows) != len(self.rows):
            raise ValueError(
                "batch tangents must match the shared prefix and row count"
            )
        shared_seed = tuple(map(_fraction, shared))
        results: list[Fraction] = []
        for row, local in zip(self.rows, rows, strict=True):
            if len(local) != len(row.gradient) - self.shared_arity:
                raise ValueError("each local tangent must match its row's input arity")
            seed = (*shared_seed, *map(_fraction, local))
            _bound(*seed)
            result = sum(
                (
                    slope * value
                    for slope, value in zip(row.gradient, seed, strict=True)
                ),
                Fraction(0),
            )
            _bound(result)
            results.append(result)
        return tuple(results)

    def vjp(self, cotangents: tuple[int | Fraction, ...]) -> BernoulliBatchCotangent:
        """Apply the transpose Jacobian in row order.

        Parameters
        ----------
        cotangents : tuple[int or Fraction, ...]
            One output cotangent for each row.

        Returns
        -------
        BernoulliBatchCotangent
            The shared summed adjoints and the separate row-local adjoints.

        Raises
        ------
        ValueError
            If the output cotangent has the wrong row extent.
        """
        if len(cotangents) != len(self.rows):
            raise ValueError("the batch cotangent must match the row count")
        shared = [Fraction(0)] * self.shared_arity
        locals_: list[tuple[Fraction, ...]] = []
        for row, cotangent in zip(self.rows, cotangents, strict=True):
            seed = _fraction(cotangent)
            _bound(seed)
            projected = tuple(seed * slope for slope in row.gradient)
            _bound(*projected)
            for position, contribution in enumerate(projected[: self.shared_arity]):
                shared[position] += contribution
                _bound(shared[position])
            locals_.append(projected[self.shared_arity :])
        return BernoulliBatchCotangent(tuple(shared), tuple(locals_))


@dataclass(frozen=True, slots=True)
class BernoulliRowReplay:
    """A replay tape for one original row and its canonical sample scopes.

    Parameters
    ----------
    axis : str
        The finite batch axis label.
    ordinal : int
        The original row ordinal.
    choices : tuple
        The canonical local addresses and replayed Boolean outcomes in order.
    """

    axis: str
    ordinal: int
    choices: _History

    @property
    def addresses(self) -> tuple[_AxisAddress, ...]:
        """Give the original request addresses with their row coordinates.

        Returns
        -------
        tuple
            An axis label, row ordinal and original address per occurrence.
        """
        return tuple((self.axis, self.ordinal, address) for address, _ in self.choices)


@dataclass(frozen=True, slots=True)
class BernoulliBatchPath:
    """One checked conditional path in an independent row.

    Parameters
    ----------
    tape : BernoulliRowReplay
        The complete addressed replay choices.
    probability : Fraction
        The source probability of those choices in their original scope.
    action : ExactGradient
        The source result and derivatives with the replay choices fixed.
    """

    tape: BernoulliRowReplay
    probability: Fraction
    action: ExactGradient


@dataclass(frozen=True, slots=True)
class BernoulliBatchReplay:
    """One complete jointly addressed replay in the independent product law.

    Parameters
    ----------
    tapes : tuple[BernoulliRowReplay, ...]
        The original tapes in row order.
    probability : Fraction
        Their joint probability in the independent source law.
    linearization : BernoulliBatchLinearization
        The row outputs and derivatives with these choices fixed.
    """

    tapes: tuple[BernoulliRowReplay, ...]
    probability: Fraction
    linearization: BernoulliBatchLinearization


@dataclass(frozen=True, slots=True)
class BernoulliBatchPlan:
    """A checked independent finite row axis over the same scalar program.

    Parameters
    ----------
    axis : str
        The axis label for the row namespaces.
    computation : ComputationId
        The named scalar source run independently in every row.
    shared_point, row_points : tuple
        The exact deterministic shared parameters and row-local parameters.
    shape : tuple
        The same ordered sequence of local addresses on every history.
    expected : BernoulliBatchLinearization
        The exact expected row values and their derivatives, including the
        parameter dependence of each row's probability.
    paths : tuple[tuple[BernoulliBatchPath, ...], ...]
        Each row's original independent support, in ordinal order.
    source_steps : int
        The total measured scalar reference work over the row and basis runs.
    selectors : tuple[tuple[Fraction, ...], ...]
        The exact categorical sampler for each independent row support.
    """

    axis: str
    computation: ComputationId
    shared_point: tuple[Fraction, ...]
    row_points: tuple[tuple[Fraction, ...], ...]
    shape: tuple[_Address, ...]
    expected: BernoulliBatchLinearization
    paths: tuple[tuple[BernoulliBatchPath, ...], ...]
    source_steps: int
    selectors: tuple[tuple[Fraction, ...], ...] = field(repr=False)

    def _replayed(self, paths: tuple[BernoulliBatchPath, ...]) -> BernoulliBatchReplay:
        """Assemble a joint history under the independent row law.

        Parameters
        ----------
        paths : tuple[BernoulliBatchPath, ...]
            One already checked path per row, in ordinal order.

        Returns
        -------
        BernoulliBatchReplay
            The exact product mass and fixed-replay output map.
        """
        probability = Fraction(1)
        for path in paths:
            probability *= path.probability
        return BernoulliBatchReplay(
            tuple(path.tape for path in paths),
            probability,
            BernoulliBatchLinearization(
                len(self.shared_point), tuple(path.action for path in paths)
            ),
        )

    def replay(self, tapes: tuple[BernoulliRowReplay, ...]) -> BernoulliBatchReplay:
        """Replay complete row-local choices at their original addresses.

        Parameters
        ----------
        tapes : tuple[BernoulliRowReplay, ...]
            One tape per row in axis order, with each choice fixed.

        Returns
        -------
        BernoulliBatchReplay
            The exact fixed-replay value map and its derivative.

        Raises
        ------
        ValueError
            If any axis, ordinal, address, Boolean choice or tape count differs
            from the checked row shape.
        """
        if len(tapes) != len(self.paths):
            raise ValueError("a batch replay needs one addressed tape per row")
        paths: list[BernoulliBatchPath] = []
        for ordinal, (tape, support) in enumerate(zip(tapes, self.paths, strict=True)):
            if (
                tape.axis != self.axis
                or type(tape.ordinal) is not int
                or tape.ordinal != ordinal
            ):
                raise ValueError("batch replay tapes must match the axis and row order")
            if tuple(address for address, _ in tape.choices) != self.shape or any(
                type(choice) is not bool for _, choice in tape.choices
            ):
                raise ValueError(
                    "batch replay choices must match the checked original addresses"
                )
            path = next(item for item in support if item.tape == tape)
            paths.append(path)
        return self._replayed(tuple(paths))

    def sample(self, *, seed: int | None = None) -> BernoulliBatchReplay:
        """Draw separately from each row's complete source law.

        Parameters
        ----------
        seed : int or None
            An optional seed for exact integer draws in row order.

        Returns
        -------
        BernoulliBatchReplay
            One joint ordered trace and its fixed-replay linearization.
        """
        random = Random(seed)
        selected = tuple(
            support[_choose_position(selectors, random)]
            for support, selectors in zip(self.paths, self.selectors, strict=True)
        )
        return self._replayed(selected)


def plan_bernoulli_batch(
    module: QiecModule,
    computation: ComputationId,
    axis: str,
    shared: tuple[int | Fraction, ...],
    rows: tuple[tuple[int | Fraction, ...], ...],
    policies: tuple[SiteEstimator, ...],
    *,
    max_rows: int = 256,
    max_paths: int = 256,
    max_work: int = 1_000_000,
    fuel: int = 50_000,
) -> BernoulliBatchPlan:
    """Validate a finite independent row axis and its exact linearizations.

    Source inputs start with the shared deterministic parameters, followed
    by that row's local inputs. All reached requests must be direct
    Bernoulli draws on one nominal Random row. The call graph is acyclic.
    Every complete trace of every row must have the same ordered local
    addresses. Row random choices are separate even when all row inputs
    and local address sequences coincide.

    Each scalar basis direction gives the fixed-path derivatives and
    the derivative of the row expectation. The replay maps hold choices
    fixed. The expectation maps include their changing probabilities.

    Parameters
    ----------
    module : QiecModule
        The actual checked source.
    computation : ComputationId
        The scalar entry to run on each row.
    axis : str
        The nonempty axis label used to namespace requests.
    shared : tuple[int or Fraction, ...]
        The exact shared deterministic input prefix.
    rows : tuple[tuple[int or Fraction, ...], ...]
        The exact local inputs in ordinal row order.
    policies : tuple[SiteEstimator, ...]
        The explicit enumeration policies for the common static sites.
    max_rows, max_paths, max_work, fuel : int
        Bounds on row count, complete paths per scalar run, cumulative
        reference steps accepted, and steps per source replay.

    Returns
    -------
    BernoulliBatchPlan
        The checked row support, replay values and exact batch derivatives.

    Raises
    ------
    ValueError
        If an axis, input table, policy, or bound is malformed.
    JvpUnsupported
        If the row graph, operation, probability or trace shape is outside
        the admitted finite independent fragment.
    JvpBoundaryError
        If an active source decision is at an input equality boundary.
    JvpLimit
        If a row, path, arithmetic or reference-work limit is exceeded.
    """
    for name, value in (
        ("max_rows", max_rows),
        ("max_paths", max_paths),
        ("max_work", max_work),
        ("fuel", fuel),
    ):
        _validate_limit(name, value)
    if not isinstance(axis, str) or not axis.strip() or not rows:
        raise ValueError("a Bernoulli batch needs an axis label and at least one row")
    if len(rows) > max_rows:
        raise JvpLimit(f"Bernoulli batch exceeds {max_rows} rows")
    _, owner = _random_computation(module, computation, "Bernoulli batch")
    _check_first_order_graph(
        module, owner, "Bernoulli batch", _check_request, require_acyclic=True
    )
    arity = len(owner.parameters)
    if len(shared) > arity or any(len(shared) + len(row) != arity for row in rows):
        raise ValueError("shared and row-local inputs must match the source arity")
    shared_point = tuple(map(_fraction, shared))
    row_points = tuple(tuple(map(_fraction, row)) for row in rows)
    shape: tuple[_Address, ...] | None = None
    batches: list[tuple[BernoulliBatchPath, ...]] = []
    expectations: list[ExactGradient] = []
    selectors: list[tuple[Fraction, ...]] = []
    source_steps = 0
    for ordinal, row in enumerate(row_points):
        point = (*shared_point, *row)
        results: list[ExactJvp] = []
        supports: list[tuple[tuple[_History, ExactJvp, ExactJvp], ...]] = []
        for index in range(max(1, arity)):
            seed = tuple(int(position == index) for position in range(arity))
            finite = _checked_finite_unit(
                module, computation, point, seed, policies, max_paths, fuel
            )
            source_steps += finite.source_steps
            if source_steps > max_work:
                raise JvpLimit(f"Bernoulli batch exceeds {max_work} reference steps")
            results.append(finite.exact)
            support = tuple(_prior_histories(finite.tree))
            for history, _, _ in support:
                addresses = tuple(address for address, _ in history)
                if shape is None:
                    shape = addresses
                if addresses != shape:
                    raise JvpUnsupported(
                        "Bernoulli batch needs the same ordered draw addresses on every history"
                    )
            supports.append(support)
        expected = ExactGradient(
            results[0].value, tuple(result.tangent for result in results[:arity])
        )
        expectations.append(expected)
        paths: list[BernoulliBatchPath] = []
        for across_bases in zip(*supports, strict=True):
            history, prior, value = across_bases[0]
            gradient = tuple(result.tangent for _, _, result in across_bases[:arity])
            assert all(
                bits == history
                and mass.value == prior.value
                and pair.value == value.value
                for bits, mass, pair in across_bases
            )
            paths.append(
                BernoulliBatchPath(
                    BernoulliRowReplay(axis, ordinal, history),
                    prior.value,
                    ExactGradient(value.value, gradient),
                )
            )
        batches.append(tuple(paths))
        selectors.append(
            _selector_probabilities(tuple(path.probability for path in paths))
        )
    assert shape is not None
    return BernoulliBatchPlan(
        axis,
        computation,
        shared_point,
        row_points,
        shape,
        BernoulliBatchLinearization(len(shared_point), tuple(expectations)),
        tuple(batches),
        source_steps,
        tuple(selectors),
    )


__all__ = [
    "BernoulliBatchCotangent",
    "BernoulliBatchLinearization",
    "BernoulliBatchPath",
    "BernoulliBatchPlan",
    "BernoulliBatchReplay",
    "BernoulliRowReplay",
    "plan_bernoulli_batch",
]
