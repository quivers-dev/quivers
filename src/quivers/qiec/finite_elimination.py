"""Exact elimination of finite factor tables with checked QIEC operators."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from math import prod

from quivers.core.elimination import elimination_order
from quivers.qiec.effects import EMPTY_ROW
from quivers.qiec.execution import ExecutionFailure, run_named
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.laws import (
    FiniteLawCounterexample,
    FiniteLawLimit,
    FiniteLawUnsupported,
    FiniteValue,
    _Shape,
    _check_body,
    _check_reachable,
    _observed,
    _shape,
    _size,
    _validate_limit,
    _values,
)
from quivers.qiec.module import (
    NamedComputation,
    QiecModule,
    _children,
    called_computations,
    validate_module,
)
from quivers.qiec.numbers import host_value
from quivers.qiec.serialization import dumps
from quivers.qiec.terms import Handle, NewInstance, Perform
from quivers.qiec.types import TypeExpr

type _OperationTable = tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class FiniteAlgebra:
    """The declared binary operations and identities of a finite algebra.

    Parameters
    ----------
    addition : ComputationId
        The closed named binary sum.
    multiplication : ComputationId
        The closed named binary product.
    zero : FiniteValue
        The declared additive identity and product annihilator.
    one : FiniteValue
        The declared multiplicative identity.
    """

    addition: ComputationId
    multiplication: ComputationId
    zero: FiniteValue
    one: FiniteValue


class FiniteAlgebraLawError(ValueError):
    """A required law fails on a concrete tuple of carrier values.

    Parameters
    ----------
    law : str
        The name of the failed law.
    counterexample : FiniteLawCounterexample
        The arguments and the two exact unequal results.
    """

    def __init__(self, law: str, counterexample: FiniteLawCounterexample) -> None:
        super().__init__(f"{law} fails at {counterexample.arguments!r}")
        self.law = law
        self.counterexample = counterexample


@dataclass(frozen=True, slots=True)
class FiniteAlgebraProof:
    """Evidence from exhaustive tables for the actual checked operators.

    Parameters
    ----------
    algebra : FiniteAlgebra
        The nominal subject and its checked identities.
    carrier : TypeExpr
        The common finite QIEC type at every argument and result.
    fingerprint : str
        A binding to the checked source and exact named algebra.
    values : int
        The size of the complete finite carrier.
    checks : int
        The number of operator table entries and equality checks.
    noncommutative : FiniteLawCounterexample or None
        An exact witness if the product cannot be permuted.
    """

    algebra: FiniteAlgebra
    carrier: TypeExpr
    fingerprint: str
    values: int
    checks: int
    noncommutative: FiniteLawCounterexample | None


@dataclass(frozen=True, slots=True)
class FiniteBaseChange:
    """A named QIEC map from the input algebra to another finite algebra.

    Parameters
    ----------
    target : FiniteAlgebra
        The algebra in which to contract the mapped factors.
    mapping : ComputationId
        A closed unary computation from the source to the target carrier.
    """

    target: FiniteAlgebra
    mapping: ComputationId


@dataclass(frozen=True, slots=True)
class FiniteBaseChangeProof:
    """Evidence that a checked map preserves both operations and units.

    Parameters
    ----------
    source : FiniteAlgebraProof
        The input algebra and its binding to the checked module.
    target : FiniteAlgebraProof
        The algebra used for elimination in that same module.
    mapping : ComputationId
        The checked nominal unary computation in the module.
    checks : int
        The combined number of table entries and equality checks.
    """

    source: FiniteAlgebraProof
    target: FiniteAlgebraProof
    mapping: ComputationId
    checks: int


@dataclass(frozen=True, slots=True)
class FiniteFactor:
    """A finite table in its written axis order.

    Parameters
    ----------
    axes : tuple[str, ...]
        The table's distinct axes, with the last axis varying fastest.
    cells : tuple[FiniteValue, ...]
        The table entries in that order.
    """

    axes: tuple[str, ...]
    cells: tuple[FiniteValue, ...]


@dataclass(frozen=True, slots=True)
class FiniteSourceFactor:
    """One finite factor supplied by a checked named computation.

    Parameters
    ----------
    axes : tuple[str, ...]
        One distinct axis per parameter, in the computation's parameter order.
    computation : ComputationId
        The source computation evaluated for each assignment.
    """

    axes: tuple[str, ...]
    computation: ComputationId


@dataclass(frozen=True, slots=True)
class FiniteEliminationStep:
    """One exact sum of an ordered product of factor tables.

    Parameters
    ----------
    variable : str
        The variable summed out at the step.
    positions : tuple[int, ...]
        Zero-based positions in the current ordered list of factors.
    bag : tuple[str, ...]
        The full axis set of the joined factors, in table order.
    cells : int
        The number of assignments in that bag.
    """

    variable: str
    positions: tuple[int, ...]
    bag: tuple[str, ...]
    cells: int


@dataclass(frozen=True, slots=True)
class FiniteEliminationWork:
    """The executed operations on cells of the checked answer algebra.

    Parameters
    ----------
    additions : int
        Sum-table lookups during contraction.
    multiplications : int
        Product-table lookups during contraction.
    """

    additions: int
    multiplications: int


@dataclass(frozen=True, slots=True)
class FiniteEliminationResult:
    """The contracted factor, its exact algebra, and the chosen elimination.

    Parameters
    ----------
    factor : FiniteFactor
        The value indexed in the requested output order.
    proof : FiniteAlgebraProof
        The fresh proof used to permit elimination for the actual operators.
    steps : tuple[FiniteEliminationStep, ...]
        The finite sums and the product positions they cover.
    largest : int
        The largest number of assignments in an input, intermediate bag or factor,
        or output.
    base_change : FiniteBaseChangeProof or None
        The proved map when the input was contracted in a different algebra.
    work : FiniteEliminationWork
        The counted table lookups performed by the contractions.
    """

    factor: FiniteFactor
    proof: FiniteAlgebraProof
    steps: tuple[FiniteEliminationStep, ...]
    largest: int
    base_change: FiniteBaseChangeProof | None = None
    work: FiniteEliminationWork = FiniteEliminationWork(0, 0)


@dataclass(frozen=True, slots=True)
class FiniteSourceEliminationResult:
    """A contraction with its independently counted source evaluation work.

    Parameters
    ----------
    result : FiniteEliminationResult
        The finite contraction and the counts of its operator table lookups.
    source_evaluations : int
        The number of top-level factor evaluations, one per factor cell.
        Operator proof evaluations and calls within each factor are separate.
    """

    result: FiniteEliminationResult
    source_evaluations: int


@dataclass(frozen=True, slots=True)
class _Tables:
    """The finite operation tables and the proof over their entries.

    Parameters
    ----------
    proof : FiniteAlgebraProof
        The exact evidence assembled from these operators.
    values : tuple[FiniteValue, ...]
        The carrier in deterministic lexical order.
    zero : int
        The table index of the additive identity.
    one : int
        The table index of the multiplicative identity.
    add : _OperationTable
        The sum indexed by left and right operands.
    multiply : _OperationTable
        The product indexed by left and right operands.
    """

    proof: FiniteAlgebraProof
    values: tuple[FiniteValue, ...]
    zero: int
    one: int
    add: _OperationTable
    multiply: _OperationTable


def _law_witness(
    values: tuple[FiniteValue, ...],
    arguments: tuple[int, ...],
    left: int,
    right: int,
) -> FiniteLawCounterexample:
    """Decode a table inequality in its exact carrier values.

    Parameters
    ----------
    values : tuple[FiniteValue, ...]
        The ordered finite carrier.
    arguments : tuple[int, ...]
        Table indices of the witnesses.
    left : int
        Index of the law's left answer.
    right : int
        Index of the law's right answer.

    Returns
    -------
    FiniteLawCounterexample
        The failing assignment and its two answers.
    """
    return FiniteLawCounterexample(
        tuple(values[index] for index in arguments), values[left], values[right]
    )


def _prove(
    values: tuple[FiniteValue, ...],
    zero: int,
    one: int,
    add: _OperationTable,
    multiply: _OperationTable,
) -> FiniteLawCounterexample | None:
    """Check the two monoids, two distributive laws, and zero exactly.

    Parameters
    ----------
    values : tuple[FiniteValue, ...]
        The full finite domain.
    zero, one : int
        The proposed identity indices.
    add, multiply : _OperationTable
        The two closed operator tables.

    Returns
    -------
    FiniteLawCounterexample or None
        The first failed product swap, if one exists.

    Raises
    ------
    FiniteAlgebraLawError
        If a law required by an ordered elimination fails.
    """
    domain = range(len(values))
    noncommutative = None

    def assert_law(law: str, arguments: tuple[int, ...], left: int, right: int) -> None:
        """Reject a failed equality with its decoded witness.

        Parameters
        ----------
        law : str
            The equality being checked.
        arguments : tuple[int, ...]
            The indices of the law variables.
        left, right : int
            The result indices of its sides.

        Raises
        ------
        FiniteAlgebraLawError
            If the sides are unequal in the finite carrier.
        """
        if left != right:
            raise FiniteAlgebraLawError(
                law, _law_witness(values, arguments, left, right)
            )

    for a in domain:
        assert_law("left additive identity", (a,), add[zero][a], a)
        assert_law("right additive identity", (a,), add[a][zero], a)
        assert_law("left multiplicative identity", (a,), multiply[one][a], a)
        assert_law("right multiplicative identity", (a,), multiply[a][one], a)
        assert_law("left zero annihilation", (a,), multiply[zero][a], zero)
        assert_law("right zero annihilation", (a,), multiply[a][zero], zero)
    for a, b in product(domain, repeat=2):
        assert_law("additive commutativity", (a, b), add[a][b], add[b][a])
        differs = multiply[a][b] != multiply[b][a]
        if differs and noncommutative is None:
            noncommutative = _law_witness(
                values, (a, b), multiply[a][b], multiply[b][a]
            )
    for a, b, c in product(domain, repeat=3):
        assignment = a, b, c
        assert_law(
            "additive associativity",
            assignment,
            add[add[a][b]][c],
            add[a][add[b][c]],
        )
        assert_law(
            "multiplicative associativity",
            assignment,
            multiply[multiply[a][b]][c],
            multiply[a][multiply[b][c]],
        )
        assert_law(
            "left distributivity",
            assignment,
            multiply[a][add[b][c]],
            add[multiply[a][b]][multiply[a][c]],
        )
        assert_law(
            "right distributivity",
            assignment,
            multiply[add[a][b]][c],
            add[multiply[a][c]][multiply[b][c]],
        )
    return noncommutative


def _tables(
    module: QiecModule, algebra: FiniteAlgebra, max_checks: int, fuel: int
) -> _Tables:
    """Read every operator entry from the bound checked implementation.

    Parameters
    ----------
    module : QiecModule
        The module declaring the exact operators to call.
    algebra : FiniteAlgebra
        Their identities and the two individual units.
    max_checks : int
        The upper bound on the complete finite proof.
    fuel : int
        The evaluation step limit for each operator table entry.

    Returns
    -------
    _Tables
        The actual finite operations and their exhaustive law evidence.

    Raises
    ------
    FiniteLawUnsupported
        If an operator or unit is outside the closed finite domain.
    FiniteLawLimit
        If the proof or an operator evaluation exceeds its bound.
    """
    _validate_limit("max_checks", max_checks)
    _validate_limit("fuel", fuel)
    validate_module(module)
    owners = {item.id: item for item in module.computations}
    if algebra.addition not in owners or algebra.multiplication not in owners:
        raise FiniteLawUnsupported("unknown finite algebra operator")
    add, multiply = owners[algebra.addition], owners[algebra.multiplication]
    carrier = add.type.result
    for operator in (add, multiply):
        if (
            operator.type.effects != EMPTY_ROW
            or operator.type.result != carrier
            or tuple(parameter.type for parameter in operator.parameters)
            != (carrier, carrier)
        ):
            raise FiniteLawUnsupported(
                "finite algebra operators need the same closed binary type"
            )
    shape = _shape(carrier)
    count = _size(shape)
    checks = 4 * count**3 + 4 * count**2 + 6 * count
    if checks > max_checks:
        raise FiniteLawLimit(
            f"finite algebra needs {checks} checks, exceeding {max_checks}"
        )
    _check_reachable(module, add, multiply)
    values = _values(shape)
    indices = {value: position for position, value in enumerate(values)}
    zero = indices[_observed(shape, algebra.zero)]
    one = indices[_observed(shape, algebra.one)]

    def operation(name: str) -> _OperationTable:
        """Evaluate one closed operator on all ordered input pairs.

        Parameters
        ----------
        name : str
            The checked computation's name.

        Returns
        -------
        _OperationTable
            The two-dimensional table of exact result indices.
        """

        def at(left: FiniteValue, right: FiniteValue) -> int:
            """Read a single operator result in the finite carrier.

            Parameters
            ----------
            left, right : FiniteValue
                The operands in written order.

            Returns
            -------
            int
                The exact result's index.

            Raises
            ------
            FiniteLawLimit
                If the operator does not finish within the fuel bound.
            """
            return indices[_evaluate_finite(module, name, (left, right), shape, fuel)]

        return tuple(tuple(at(left, right) for right in values) for left in values)

    additions = operation(add.name)
    multiplications = operation(multiply.name)
    noncommutative = _prove(values, zero, one, additions, multiplications)
    fingerprint = sha256(
        dumps(
            (
                "finite-elimination-v1",
                module,
                algebra.addition,
                algebra.multiplication,
                algebra.zero,
                algebra.one,
            )
        ).encode("utf-8")
    ).hexdigest()
    proof = FiniteAlgebraProof(
        algebra, carrier, fingerprint, count, checks, noncommutative
    )
    return _Tables(proof, values, zero, one, additions, multiplications)


def _evaluate_finite(
    module: QiecModule,
    name: str,
    arguments: tuple[FiniteValue, ...],
    shape: _Shape,
    fuel: int,
) -> FiniteValue:
    """Call one checked finite operation and check its exact result carrier.

    Parameters
    ----------
    module : QiecModule
        The checked operator module.
    name : str
        Name of the computation to evaluate.
    arguments : tuple[FiniteValue, ...]
        The exact arguments in source order.
    shape : _Shape
        The shape of the expected finite result.
    fuel : int
        Step limit for this call.

    Returns
    -------
    FiniteValue
        The checked host representation of the result.

    Raises
    ------
    FiniteLawLimit
        If the computation exhausts its fuel.
    """
    try:
        answer = run_named(module, name, arguments, fuel=fuel, record_trace=False)
    except ExecutionFailure as error:
        if error.diagnostic.code == "qiec-run-fuel":
            raise FiniteLawLimit(
                f"fuel exhausted for {name!r} at {arguments!r}"
            ) from error
        raise
    return _observed(shape, host_value(answer.value))


def _prove_base_change(
    module: QiecModule,
    source: _Tables,
    target: _Tables,
    mapping_id: ComputationId,
    fuel: int,
) -> tuple[int, ...]:
    """Read a checked map and prove both binary laws and both identities.

    Parameters
    ----------
    module : QiecModule
        The checked module supplying all three sets of operations.
    source, target : _Tables
        The already proved input and output algebras.
    mapping_id : ComputationId
        The nominal unary computation to check.
    fuel : int
        The step limit for evaluating a map entry.

    Returns
    -------
    tuple[int, ...]
        Target indices in the source carrier's order.

    Raises
    ------
    FiniteLawUnsupported
        If the mapping does not have a supported signature or body.
    FiniteAlgebraLawError
        If a homomorphism law fails, with its exact values.
    """
    mapping = next(
        (item for item in module.computations if item.id == mapping_id), None
    )
    if (
        mapping is None
        or mapping.type.effects != EMPTY_ROW
        or mapping.type.result != target.proof.carrier
        or tuple(parameter.type for parameter in mapping.parameters)
        != (source.proof.carrier,)
    ):
        raise FiniteLawUnsupported("finite base change needs a closed unary mapping")
    _check_reachable(module, mapping, mapping)
    shape = _shape(target.proof.carrier)
    indices = {value: position for position, value in enumerate(target.values)}
    image = tuple(
        indices[_evaluate_finite(module, mapping.name, (value,), shape, fuel)]
        for value in source.values
    )

    def require(
        law: str, arguments: tuple[FiniteValue, ...], left: int, right: int
    ) -> None:
        """Compare the target values of a base-change equation.

        Parameters
        ----------
        law : str
            The identity that is being checked.
        arguments : tuple[FiniteValue, ...]
            Its source-carrier arguments, in written order.
        left, right : int
            The target indices of its two sides.

        Raises
        ------
        FiniteAlgebraLawError
            If the two target values differ.
        """
        if left != right:
            raise FiniteAlgebraLawError(
                law,
                FiniteLawCounterexample(
                    arguments, target.values[left], target.values[right]
                ),
            )

    require("base change preserves zero", (), image[source.zero], target.zero)
    require("base change preserves one", (), image[source.one], target.one)
    for i, j in product(range(len(source.values)), repeat=2):
        arguments = source.values[i], source.values[j]
        require(
            "base change preserves addition",
            arguments,
            image[source.add[i][j]],
            target.add[image[i]][image[j]],
        )
        require(
            "base change preserves multiplication",
            arguments,
            image[source.multiply[i][j]],
            target.multiply[image[i]][image[j]],
        )
    return image


def check_finite_algebra(
    module: QiecModule,
    algebra: FiniteAlgebra,
    *,
    max_checks: int = 65_536,
    fuel: int = 65_536,
) -> FiniteAlgebraProof:
    """Prove the complete finite operator tables before choosing a product order.

    Addition must form a commutative monoid and multiplication a monoid
    whose zero annihilates. The product distributes over addition on
    both sides. Product commutativity is checked separately.

    Parameters
    ----------
    module : QiecModule
        The checked source containing both binary computations.
    algebra : FiniteAlgebra
        The nominal operations and two identities.
    max_checks : int
        The bound on the required operator entries and law comparisons.
    fuel : int
        The evaluation step limit for each entry in either operation table.

    Returns
    -------
    FiniteAlgebraProof
        The exact evidence and any witness preventing a product swap.

    Raises
    ------
    FiniteAlgebraLawError
        If any required law fails, with a concrete equation witness.
    FiniteLawUnsupported
        If the signature, operator bodies, or units leave the finite fragment.
    FiniteLawLimit
        If the complete proof or an operator runs past its bound.
    """
    return _tables(module, algebra, max_checks, fuel).proof


@dataclass(frozen=True, slots=True)
class _IndexedFactor:
    """A factor whose values are encoded as indices into the checked carrier.

    Parameters
    ----------
    axes : tuple[str, ...]
        The factor's ordered axes.
    cells : tuple[int, ...]
        The carrier-value indices in row-major order.
    """

    axes: tuple[str, ...]
    cells: tuple[int, ...]

    def at(self, coordinates: Mapping[str, int], extents: Mapping[str, int]) -> int:
        """Read one ordered factor entry.

        Parameters
        ----------
        coordinates : Mapping[str, int]
            An assignment to every axis used by the factor.
        extents : Mapping[str, int]
            The lengths of those axes.

        Returns
        -------
        int
            The indexed carrier value at the assignment.
        """
        offset = 0
        for axis in self.axes:
            offset = offset * extents[axis] + coordinates[axis]
        return self.cells[offset]


def _elimination_steps(
    scopes: tuple[tuple[str, ...], ...],
    output: tuple[str, ...],
    axes: tuple[str, ...],
    extents: Mapping[str, int],
    permute: bool,
) -> tuple[FiniteEliminationStep, ...]:
    """Choose finite sums without moving an unproved product.

    Parameters
    ----------
    scopes : tuple[tuple[str, ...], ...]
        The input scopes in written factor order.
    output : tuple[str, ...]
        The axes retained in the result.
    axes : tuple[str, ...]
        All axes in order of first mention.
    extents : Mapping[str, int]
        The number of indices on each axis.
    permute : bool
        Whether exact multiplication commutativity was established.

    Returns
    -------
    tuple[FiniteEliminationStep, ...]
        An ordered plan, using noncontiguous buckets only when permitted.
    """
    pending = tuple(axis for axis in axes if axis not in output)
    active = list(scopes)
    planned: list[FiniteEliminationStep] = []

    def step(variable: str) -> FiniteEliminationStep:
        """Read the smallest permitted bucket for one variable.

        Parameters
        ----------
        variable : str
            Candidate sum in this round.

        Returns
        -------
        FiniteEliminationStep
            Its containing factors, including the interval gaps if needed.
        """
        positions = tuple(
            index for index, scope in enumerate(active) if variable in scope
        )
        if not permute:
            positions = tuple(range(positions[0], positions[-1] + 1))
        bag = tuple(
            axis for axis in axes if any(axis in active[index] for index in positions)
        )
        return FiniteEliminationStep(
            variable, positions, bag, prod(extents[axis] for axis in bag)
        )

    if permute:
        pending = elimination_order(
            axes, scopes, output, cost=lambda bag: prod(extents[axis] for axis in bag)
        )
    while pending:
        chosen = (
            step(pending[0])
            if permute
            else min(map(step, pending), key=lambda item: item.cells)
        )
        remaining = tuple(axis for axis in chosen.bag if axis != chosen.variable)
        _replace_factors(active, chosen.positions, remaining)
        planned.append(chosen)
        pending = tuple(axis for axis in pending if axis != chosen.variable)
    return tuple(planned)


def _replace_factors[Factor](
    active: list[Factor], positions: tuple[int, ...], replacement: Factor
) -> None:
    """Replace selected factors in their original positions.

    Parameters
    ----------
    active : list[Factor]
        The current factors or their scopes in written order.
    positions : tuple[int, ...]
        The positions selected, in increasing order.
    replacement : Factor
        Their aggregate, written where the first selected factor appeared.
    """
    for index in reversed(positions[1:]):
        del active[index]
    active[positions[0]] = replacement


def _contract(
    tables: _Tables,
    factors: tuple[_IndexedFactor, ...],
    output: tuple[str, ...],
    summed: tuple[str, ...],
    extents: Mapping[str, int],
) -> tuple[_IndexedFactor, FiniteEliminationWork]:
    """Evaluate one ordered factor product under a finite sum.

    Parameters
    ----------
    tables : _Tables
        The actual checked operator tables.
    factors : tuple[_IndexedFactor, ...]
        The product factors in their required order.
    output : tuple[str, ...]
        The indices kept by the step.
    summed : tuple[str, ...]
        The indices removed by the step.
    extents : Mapping[str, int]
        The sizes of all factor axes.

    Returns
    -------
    tuple of _IndexedFactor and FiniteEliminationWork
        The exact output table in output order and the table lookups performed.
    """
    if any(extents[axis] == 0 for axis in output):
        return _IndexedFactor(output, ()), FiniteEliminationWork(0, 0)
    cells = []
    additions = multiplications = 0
    for selected in product(*(range(extents[axis]) for axis in output)):
        coordinates = dict(zip(output, selected, strict=True))
        total = tables.zero
        for assignment in product(*(range(extents[axis]) for axis in summed)):
            coordinates.update(zip(summed, assignment, strict=True))
            term = tables.one
            for factor in factors:
                term = tables.multiply[term][factor.at(coordinates, extents)]
                multiplications += 1
            total = tables.add[total][term]
            additions += 1
        cells.append(total)
    return _IndexedFactor(output, tuple(cells)), FiniteEliminationWork(
        additions, multiplications
    )


def _factor_geometry(
    scopes: tuple[tuple[str, ...], ...],
    output: tuple[str, ...],
    extents: Mapping[str, int],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Check the geometry of factor tables without observing their entries.

    Parameters
    ----------
    scopes : tuple[tuple[str, ...], ...]
        The axes of all factors in written order.
    output : tuple[str, ...]
        The axes retained after elimination.
    extents : Mapping[str, int]
        The cardinality of every input axis.

    Returns
    -------
    tuple[tuple[str, ...], tuple[int, ...]]
        The axes in first occurrence order and the area of every factor.

    Raises
    ------
    ValueError
        If the factor or output axes and cardinalities are invalid.
    """
    if not scopes:
        raise ValueError("finite elimination needs at least one factor")
    if any(len(set(scope)) != len(scope) for scope in scopes):
        raise ValueError("a finite factor cannot repeat an axis")
    axes = tuple(dict.fromkeys(axis for scope in scopes for axis in scope))
    if any(not isinstance(axis, str) or not axis for axis in axes):
        raise ValueError("finite axes need nonempty names")
    if set(axes) != set(extents) or any(
        type(size) is not int or size < 0 for size in extents.values()
    ):
        raise ValueError("each finite axis needs one nonnegative integer extent")
    if len(set(output)) != len(output) or not set(output) <= set(axes):
        raise ValueError("output axes must be distinct input axes")
    sizes = tuple(prod(extents[axis] for axis in scope) for scope in scopes)
    return axes, sizes


def _eliminate_factors(
    module: QiecModule,
    algebra: FiniteAlgebra,
    scopes: tuple[tuple[str, ...], ...],
    axes: tuple[str, ...],
    sizes: tuple[int, ...],
    output: tuple[str, ...],
    extents: Mapping[str, int],
    load_factors: Callable[[], tuple[FiniteFactor, ...]],
    *,
    base_change: FiniteBaseChange | None,
    max_checks: int,
    max_cells: int,
    fuel: int,
) -> FiniteEliminationResult:
    """Plan and bound the exact contraction before obtaining its cells.

    Parameters
    ----------
    module : QiecModule
        The checked source supplying the algebra.
    algebra : FiniteAlgebra
        The named algebra of the input factors.
    scopes : tuple[tuple[str, ...], ...]
        The validated axes of the factors in written order.
    axes : tuple[str, ...]
        The distinct axes in order of first occurrence.
    sizes : tuple[int, ...]
        The validated input factor areas.
    output : tuple[str, ...]
        The output axes in requested order.
    extents : Mapping[str, int]
        The exact cardinalities.
    load_factors : Callable[[], tuple[FiniteFactor, ...]]
        The already checked shape's cells, materialized after the area check.
    base_change : FiniteBaseChange or None
        The optional proved change of answer algebra.
    max_checks, max_cells, fuel : int
        The proof, area and individual evaluation limits.

    Returns
    -------
    FiniteEliminationResult
        The exact result and its contraction plan and work counts.

    Raises
    ------
    FiniteAlgebraLawError
        If a required equation fails.
    FiniteLawLimit
        If a proof, evaluation or table exceeds its checked limit.
    FiniteLawUnsupported
        If the checked algebra or factor cells fall outside the finite fragment.
    """
    initial = max((*sizes, prod(extents[axis] for axis in output)))
    if initial > max_cells:
        raise FiniteLawLimit(
            f"finite table needs {initial} cells, exceeding {max_cells}"
        )

    source = _tables(module, algebra, max_checks, fuel)
    tables = source
    image = tuple(range(len(source.values)))
    proof = None
    if base_change is not None:
        count = len(source.values)
        mapping_checks = count + 2 * count**2 + 2
        remaining = max_checks - source.proof.checks - mapping_checks
        if remaining < 1:
            raise FiniteLawLimit(
                f"finite base-change proof exceeds {max_checks} checks"
            )
        tables = _tables(module, base_change.target, remaining, fuel)
        image = _prove_base_change(module, source, tables, base_change.mapping, fuel)
        proof = FiniteBaseChangeProof(
            source.proof,
            tables.proof,
            base_change.mapping,
            source.proof.checks + tables.proof.checks + mapping_checks,
        )
    shape = _shape(source.proof.carrier)
    steps = _elimination_steps(
        scopes, output, axes, extents, tables.proof.noncommutative is None
    )
    largest = max(
        (
            initial,
            *(
                max(
                    step.cells,
                    prod(extents[axis] for axis in step.bag if axis != step.variable),
                )
                for step in steps
            ),
        )
    )
    if largest > max_cells:
        raise FiniteLawLimit(f"finite bag needs {largest} cells, exceeding {max_cells}")
    indices = {value: index for value, index in zip(source.values, image, strict=True)}
    active = [
        _IndexedFactor(
            factor.axes,
            tuple(indices[_observed(shape, value)] for value in factor.cells),
        )
        for factor in load_factors()
    ]
    additions = multiplications = 0
    for step in steps:
        selected = tuple(active[index] for index in step.positions)
        kept = tuple(axis for axis in step.bag if axis != step.variable)
        value, work = _contract(tables, selected, kept, (step.variable,), extents)
        additions += work.additions
        multiplications += work.multiplications
        _replace_factors(active, step.positions, value)
    result, work = _contract(tables, tuple(active), output, (), extents)
    factor = FiniteFactor(output, tuple(tables.values[index] for index in result.cells))
    total = FiniteEliminationWork(
        additions + work.additions, multiplications + work.multiplications
    )
    return FiniteEliminationResult(factor, tables.proof, steps, largest, proof, total)


def eliminate_finite(
    module: QiecModule,
    algebra: FiniteAlgebra,
    factors: tuple[FiniteFactor, ...],
    output: tuple[str, ...],
    extents: Mapping[str, int],
    *,
    base_change: FiniteBaseChange | None = None,
    max_checks: int = 65_536,
    max_cells: int = 65_536,
    fuel: int = 65_536,
) -> FiniteEliminationResult:
    """Eliminate named axes of explicit finite factors using checked operators.

    Every run proves the required laws against the actual closed named
    operations. With a noncommutative product, a step contracts the interval
    from the first to the last participating factor, preserving their order
    and including any factors in between. With commutativity, a step may
    select a smaller bucket. Empty sums yield the checked zero.

    If a map preserves both binary operations and both units, mapping
    every input factor before elimination has the same result as mapping
    the contracted value. The target's proved multiplication determines
    the target plan's order.

    Parameters
    ----------
    module : QiecModule
        The checked source supplying the actual operations.
    algebra : FiniteAlgebra
        The two nominal binary operations and their identities.
    factors : tuple[FiniteFactor, ...]
        At least one factor, in written product order.
    output : tuple[str, ...]
        The retained axes in output order. An empty tuple returns a scalar.
    extents : Mapping[str, int]
        Exactly one nonnegative cardinality for each input axis.
    base_change : FiniteBaseChange or None
        An optional named homomorphism to a different finite answer algebra.
    max_checks : int
        Maximum operator entries and comparisons in all required law proofs.
    max_cells : int
        Maximum cardinality of an input, intermediate bag or factor, or output.
    fuel : int
        The evaluation step limit for an individual operator call.

    Returns
    -------
    FiniteEliminationResult
        Exact factor entries and the proved, bounded elimination plan.

    Raises
    ------
    ValueError
        If factor axes, cardinalities, shapes, output, or limits are invalid.
    FiniteLawUnsupported
        If the factor entries or operators are outside the finite carrier.
    FiniteAlgebraLawError
        If a required law fails, with its exact counterexample.
    FiniteLawLimit
        If the law proof, operator evaluation, or a table exceeds a bound.
    """
    _validate_limit("max_cells", max_cells)
    scopes = tuple(factor.axes for factor in factors)
    axes, sizes = _factor_geometry(scopes, output, extents)
    if any(
        len(factor.cells) != size for factor, size in zip(factors, sizes, strict=True)
    ):
        raise ValueError("finite factor cells do not match their axis extents")
    return _eliminate_factors(
        module,
        algebra,
        scopes,
        axes,
        sizes,
        output,
        extents,
        lambda: factors,
        base_change=base_change,
        max_checks=max_checks,
        max_cells=max_cells,
        fuel=fuel,
    )


def _require_pure_factors(
    roots: tuple[NamedComputation, ...],
    owners: Mapping[ComputationId, NamedComputation],
) -> None:
    """Reject effects in any factor body or transitive callee before evaluation.

    Parameters
    ----------
    roots : tuple[NamedComputation, ...]
        The named factor computations.
    owners : Mapping[ComputationId, NamedComputation]
        The checked definitions supplying their callees.

    Raises
    ------
    FiniteLawUnsupported
        If a reachable body has effects, static parameters or nonfinite terms.
    """
    pending = list(roots)
    checked: set[ComputationId] = set()
    while pending:
        owner = pending.pop()
        if owner.id in checked:
            continue
        checked.add(owner.id)
        if owner.telescope or owner.type.effects != EMPTY_ROW:
            raise FiniteLawUnsupported(
                f"source factor reaches `{owner.name}` with effects or static parameters"
            )
        nodes: list[object] = [owner.body]
        seen: set[int] = set()
        while nodes:
            node = nodes.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, (Perform, Handle, NewInstance)):
                raise FiniteLawUnsupported(
                    f"source factor reaches `{owner.name}` with an effect or handler"
                )
            nodes.extend(_children(node))
        for parameter in owner.parameters:
            _shape(parameter.type)
        _shape(owner.type.result)
        _check_body(owner.body)
        pending.extend(owners[callee] for callee in called_computations(owner.body))


def eliminate_finite_source(
    module: QiecModule,
    algebra: FiniteAlgebra,
    factors: tuple[FiniteSourceFactor, ...],
    output: tuple[str, ...],
    *,
    base_change: FiniteBaseChange | None = None,
    max_checks: int = 65_536,
    max_cells: int = 65_536,
    max_source_cells: int = 65_536,
    fuel: int = 65_536,
) -> FiniteSourceEliminationResult:
    """Eliminate finite factors defined by pure named QIEC computations.

    The axis at each position has that parameter's exact finite type. A
    repeated axis name denotes the same type and value order in every
    factor. The compiler bounds the total source cells and every bag of
    the proved elimination plan before asking the reference evaluator for
    a factor cell. The last parameter varies fastest. Reachable requests,
    including locally handled draws in computations with an empty row,
    are refused before tabulation.

    Parameters
    ----------
    module : QiecModule
        The checked source defining the factors and algebra operations.
    algebra : FiniteAlgebra
        The nominal binary operations and identities of the input algebra.
    factors : tuple[FiniteSourceFactor, ...]
        The computations and their positional axis names, in product order.
    output : tuple[str, ...]
        The retained axes in output order. An empty tuple returns a scalar.
    base_change : FiniteBaseChange or None
        An optional checked map to a different finite answer algebra.
    max_checks : int
        Maximum operator entries and comparisons in the algebra proofs.
    max_cells : int
        Maximum area of any input, intermediate bag or factor, or output.
    max_source_cells : int
        Maximum total number of top-level source factor evaluations.
    fuel : int
        The reference step limit for each source or operator evaluation.

    Returns
    -------
    FiniteSourceEliminationResult
        The exact contraction and its distinctly counted source work.

    Raises
    ------
    ValueError
        If an axis, extent, output, bound, or parameter count is invalid.
    FiniteLawUnsupported
        If a computation has an unknown identity, an inconsistent axis
        type, an inadmissible value, or reaches an effect.
    FiniteLawLimit
        If a proof, table, or evaluation exceeds a checked bound.
    FiniteAlgebraLawError
        If a required algebra law fails.
    """
    _validate_limit("max_cells", max_cells)
    _validate_limit("max_source_cells", max_source_cells)
    validate_module(module)
    owners = {owner.id: owner for owner in module.computations}
    if algebra.addition not in owners or any(
        factor.computation not in owners for factor in factors
    ):
        raise FiniteLawUnsupported("unknown named factor or finite algebra operator")
    roots = tuple(owners[factor.computation] for factor in factors)
    carrier = owners[algebra.addition].type.result
    shape = _shape(carrier)
    axis_types: dict[str, TypeExpr] = {}
    scopes = tuple(factor.axes for factor in factors)
    for scope, owner in zip(scopes, roots, strict=True):
        if owner.type.result != carrier:
            raise FiniteLawUnsupported(
                f"source factor `{owner.name}` has a different carrier"
            )
        if len(scope) != len(owner.parameters):
            raise ValueError(
                f"source factor `{owner.name}` needs one axis per parameter"
            )
        for axis, parameter in zip(scope, owner.parameters, strict=True):
            if not isinstance(axis, str) or not axis:
                raise ValueError("finite axes need nonempty names")
            if axis_types.setdefault(axis, parameter.type) != parameter.type:
                raise FiniteLawUnsupported(
                    f"source factor axis {axis!r} has inconsistent types"
                )
    shapes = {axis: _shape(type_) for axis, type_ in axis_types.items()}
    extents = {axis: _size(domain) for axis, domain in shapes.items()}
    axes, sizes = _factor_geometry(scopes, output, extents)
    needed = sum(sizes)
    if needed > max_source_cells:
        raise FiniteLawLimit(
            f"source factors need {needed} cells, exceeding {max_source_cells}"
        )
    _require_pure_factors(roots, owners)

    def load_factors() -> tuple[FiniteFactor, ...]:
        """Evaluate each checked factor once per cell in positional order.

        Returns
        -------
        tuple[FiniteFactor, ...]
            The actual observed factor values in written product order.

        Raises
        ------
        FiniteLawLimit
            If a source computation exhausts its evaluation fuel.
        """
        domains = {axis: _values(domain) for axis, domain in shapes.items()}
        return tuple(
            FiniteFactor(
                scope,
                tuple(
                    _evaluate_finite(module, owner.name, point, shape, fuel)
                    for point in product(*(domains[axis] for axis in scope))
                ),
            )
            for scope, owner in zip(scopes, roots, strict=True)
        )

    result = _eliminate_factors(
        module,
        algebra,
        scopes,
        axes,
        sizes,
        output,
        extents,
        load_factors,
        base_change=base_change,
        max_checks=max_checks,
        max_cells=max_cells,
        fuel=fuel,
    )
    return FiniteSourceEliminationResult(result, needed)


__all__ = [
    "FiniteAlgebra",
    "FiniteAlgebraLawError",
    "FiniteAlgebraProof",
    "FiniteBaseChange",
    "FiniteBaseChangeProof",
    "FiniteEliminationResult",
    "FiniteEliminationStep",
    "FiniteEliminationWork",
    "FiniteFactor",
    "FiniteSourceEliminationResult",
    "FiniteSourceFactor",
    "check_finite_algebra",
    "eliminate_finite",
    "eliminate_finite_source",
]
