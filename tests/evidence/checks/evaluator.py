"""Checks of the reference machine's values and gradients on tensors.

The machine holds every number as a tensor, so the gradient of a value
it computes is the gradient torch takes through the machine's steps. The
references here are independent of torch: every primitive is computed by
the standard library in double precision, its derivative by extrapolated
central differences of that function (oracle O7), and the derivative of a
deduction's weight by the expected rule counts of its enumerated
derivations (oracle O1). Points where differences do not apply, ties of a
selection and empty or impossible sums, have derivatives worked out by
hand.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from fractions import Fraction

import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.qiec.canonical import tensor_type
from quivers.qiec.evaluator import Evaluator, RuntimeAttachments
from quivers.qiec.identifiers import SourceOrigin
from quivers.qiec.kinds import NAT
from quivers.qiec.numbers import NumberDomain
from quivers.qiec.primitives import primitive
from quivers.qiec.program_runtime import deduction_item, run_deduction, run_program
from quivers.qiec.terms import (
    CholeskyFactor,
    Comprehension,
    Gather,
    Local,
    PrimitiveApplication,
    Reduction,
    Rowwise,
    TensorValue,
    Value,
    Var,
)
from quivers.qiec.types import INT, REAL, IndexLiteral, TypeExpr
from tests.evidence.checks._registry import add
from tests.evidence.checks.machine import (
    AXIOM_WEIGHTS,
    JOIN,
    LIFT,
    NONLEXICAL_WEIGHTS,
    SENTENCES,
    module_of,
    nonlexical_trees,
    parses,
)
from tests.evidence.compare import (
    ELEMENTARY,
    Accuracy,
    Mismatch,
    compare_number,
    differentiate,
)
from tests.evidence.oracles import derivations
from tests.evidence.oracles.gradients import richardson
from tests.evidence.problems import logarithm

ORIGIN = SourceOrigin(
    module="evidence",
    structural_path=("evaluator",),
    role="primitive",
    source_protocol="test",
)
"""The origin of the terms built here."""

DOUBLE = NumberDomain(real="float64")
"""The machine's numbers in double precision."""

STEP = 1e-3
"""The larger step of the central differences."""

type Reference = Callable[..., float]
"""A function of host numbers computed by the standard library."""


def _digamma(value: float) -> float:
    """The logarithmic derivative of the gamma function.

    Parameters
    ----------
    value : float
        A positive number.

    Returns
    -------
    float
        ``psi(value)``, by the recurrence up to thirty-two and the
        asymptotic series there, whose first omitted term is below the
        double-precision rounding of the value.
    """
    result = 0.0
    while value < 32.0:
        result -= 1.0 / value
        value += 1.0
    inverse = 1.0 / value
    square = inverse * inverse
    return (
        result
        + math.log(value)
        - 0.5 * inverse
        - square
        * (
            1.0 / 12.0
            - square
            * (
                1.0 / 120.0
                - square * (1.0 / 252.0 - square * (1.0 / 240.0 - square / 132.0))
            )
        )
    )


def _erfinv(value: float) -> float:
    """The inverse error function, by bisection on ``math.erf``.

    Parameters
    ----------
    value : float
        A number in the open interval from minus one to one.

    Returns
    -------
    float
        The ``x`` with ``erf(x) == value`` to the last bit bisection
        reaches.
    """
    low, high = -6.0, 6.0
    for _ in range(200):
        middle = (low + high) / 2
        if math.erf(middle) < value:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _sigmoid(value: float) -> float:
    """The logistic function.

    Parameters
    ----------
    value : float
        The argument.

    Returns
    -------
    float
        ``1 / (1 + exp(-value))``.
    """
    return 1.0 / (1.0 + math.exp(-value))


def _softplus(value: float) -> float:
    """``log(1 + exp(value))``.

    Parameters
    ----------
    value : float
        The argument.

    Returns
    -------
    float
        The softplus.
    """
    return math.log1p(math.exp(value))


SELU_ALPHA = 1.6732632423543772848170429916717
"""The negative-side scale of the scaled exponential linear unit."""

SELU_SCALE = 1.0507009873554804934193349852946
"""The overall scale of the scaled exponential linear unit."""

REFERENCES: dict[str, tuple[Reference, tuple[tuple[float, ...], ...]]] = {
    "add_real": (lambda a, b: a + b, ((0.3, -1.2),)),
    "sub_real": (lambda a, b: a - b, ((0.3, -1.2),)),
    "mul_real": (lambda a, b: a * b, ((0.3, -1.2),)),
    "div_real": (lambda a, b: a / b, ((0.3, -1.2),)),
    "neg_real": (lambda a: -a, ((0.3,),)),
    "abs_real": (abs, ((-0.7,), (0.7,))),
    "min_real": (min, ((0.3, -1.2), (-1.2, 0.3))),
    "max_real": (max, ((0.3, -1.2), (-1.2, 0.3))),
    "pow_real": (math.pow, ((1.3, 2.5), (0.6, -0.4))),
    "exp": (math.exp, ((0.7,), (-1.9,))),
    "log": (math.log, ((1.7,), (0.2,))),
    "sqrt": (math.sqrt, ((1.7,),)),
    "expm1": (math.expm1, ((0.7,), (-0.01,))),
    "log1p": (math.log1p, ((0.7,), (-0.5,))),
    "log2": (math.log2, ((1.7,),)),
    "log10": (math.log10, ((1.7,),)),
    "rsqrt": (lambda a: 1.0 / math.sqrt(a), ((1.7,),)),
    "square": (lambda a: a * a, ((-0.7,),)),
    "reciprocal": (lambda a: 1.0 / a, ((-0.7,),)),
    "sin": (math.sin, ((0.7,),)),
    "cos": (math.cos, ((0.7,),)),
    "tan": (math.tan, ((0.7,),)),
    "asin": (math.asin, ((0.4,),)),
    "acos": (math.acos, ((0.4,),)),
    "atan": (math.atan, ((0.4,),)),
    "sinh": (math.sinh, ((0.7,),)),
    "cosh": (math.cosh, ((0.7,),)),
    "tanh": (math.tanh, ((0.7,),)),
    "asinh": (math.asinh, ((0.7,),)),
    "acosh": (math.acosh, ((1.6,),)),
    "atanh": (math.atanh, ((0.4,),)),
    "erf": (math.erf, ((0.4,),)),
    "erfc": (math.erfc, ((0.4,),)),
    "erfinv": (_erfinv, ((0.3,),)),
    "lgamma": (math.lgamma, ((2.7,), (0.3,))),
    "digamma": (_digamma, ((2.7,),)),
    "sigmoid": (_sigmoid, ((0.7,), (-2.0,))),
    "relu": (lambda a: max(a, 0.0), ((0.7,), (-0.7,))),
    "relu6": (lambda a: min(max(a, 0.0), 6.0), ((0.7,), (6.5,))),
    "elu": (lambda a: a if a > 0 else math.expm1(a), ((0.7,), (-0.7,))),
    "selu": (
        lambda a: SELU_SCALE * (a if a > 0 else SELU_ALPHA * math.expm1(a)),
        ((0.7,), (-0.7,)),
    ),
    "gelu": (lambda a: 0.5 * a * (1.0 + math.erf(a / math.sqrt(2.0))), ((0.7,),)),
    "silu": (lambda a: a * _sigmoid(a), ((0.7,),)),
    "mish": (lambda a: a * math.tanh(_softplus(a)), ((0.7,),)),
    "softplus": (_softplus, ((0.7,),)),
    "logsigmoid": (lambda a: -_softplus(-a), ((0.7,),)),
    "softsign": (lambda a: a / (1.0 + abs(a)), ((0.7,), (-0.7,))),
    "log_probability": (math.log, ((0.3,),)),
    "log1mexp": (lambda a: math.log(-math.expm1(a)), ((-0.3,), (-2.0,))),
    "log_ndtr": (
        lambda a: math.log(0.5 * math.erfc(-a / math.sqrt(2.0))),
        ((-0.4,), (1.3,)),
    ),
    "positive_parameter": (lambda a: a, ((0.7,),)),
    "observation_weight": (lambda a: a, ((0.7,),)),
    "as_weight": (lambda a: a, ((-0.7,),)),
    "weight_value": (lambda a: a, ((-0.7,),)),
    "add_weight": (lambda a, b: a + b, ((-0.7, -0.2),)),
    "scale_weight": (lambda a, b: a * b, ((0.6, -1.3),)),
}
"""Every real primitive, its standard-library value, and interior points."""


def evaluator() -> Evaluator:
    """An evaluator over the machine's numbers in double precision.

    Returns
    -------
    Evaluator
        The evaluator.
    """
    return Evaluator(RuntimeAttachments(numbers=DOUBLE))


def _application(name: str, arguments: tuple[Value, ...]) -> PrimitiveApplication:
    """A primitive applied to values.

    Parameters
    ----------
    name : str
        The primitive.
    arguments : tuple of Value
        Its arguments.

    Returns
    -------
    PrimitiveApplication
        The application.
    """
    signature = primitive(name)
    return PrimitiveApplication(
        primitive=signature.id,
        name=name,
        arguments=arguments,
        result_type=signature.result,
        origin=ORIGIN,
    )


def _locals(types: tuple[TypeExpr, ...]) -> tuple[Local, ...]:
    """One local per argument.

    Parameters
    ----------
    types : tuple of TypeExpr
        The arguments' types.

    Returns
    -------
    tuple of Local
        The locals ``a0``, ``a1``, and so on.
    """
    return tuple(Local(name=f"a{i}", type=type_) for i, type_ in enumerate(types))


def _leaves(point: tuple[float, ...]) -> list[torch.Tensor]:
    """Double-precision leaves at a point.

    Parameters
    ----------
    point : tuple of float
        The point.

    Returns
    -------
    list of torch.Tensor
        One scalar leaf per coordinate.
    """
    return [
        torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in point
    ]


def _check_derivative(
    found: torch.Tensor,
    reference: Reference,
    point: tuple[float, ...],
    position: int,
    what: str,
) -> None:
    """Compare one derivative with extrapolated differences of the reference.

    Parameters
    ----------
    found : torch.Tensor
        The machine's derivative.
    reference : Reference
        The standard-library function.
    point : tuple of float
        The point, interior to the domain.
    position : int
        The argument differentiated by.
    what : str
        A description for the report.

    Raises
    ------
    Mismatch
        If the derivative differs from the differences by more than their
        error estimate and the rounding of an elementary function.
    """
    names = [f"a{i}" for i in range(len(point))]

    def value(at: Mapping[str, float]) -> float:
        """The reference at named coordinates.

        Parameters
        ----------
        at : Mapping of str to float
            The coordinates.

        Returns
        -------
        float
            The value.
        """
        return reference(*(at[name] for name in names))

    difference = richardson(
        value, dict(zip(names, point, strict=True)), names[position], step=STEP
    )
    derivative = float(found)
    allowed = difference.error + Accuracy(
        domain="linear", operations=4 * ELEMENTARY
    ).tolerance(difference.value)
    deviation = abs(derivative - difference.value)
    if not deviation <= allowed:
        raise Mismatch(
            "value",
            deviation,
            f"{what}: derivative {derivative!r} for {difference.value!r}",
        )


def check_primitive_gradients() -> None:
    """Every real primitive's value and derivatives at interior points.

    Raises
    ------
    Mismatch
        If a value differs from the standard library, a result is not a
        double-precision scalar, or a derivative is missing or differs from
        the differences.
    """
    for name, (reference, points) in REFERENCES.items():
        signature = primitive(name)
        locals_ = _locals(signature.parameters)
        term = _application(name, tuple(Var(local=local) for local in locals_))
        for point in points:
            leaves = _leaves(point)
            value = evaluator().evaluate_value(
                term, dict(zip(locals_, leaves, strict=True))
            )
            what = f"{name}{point}"
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
                raise Mismatch("dtype", math.inf, f"{what}: {value!r}")
            compare_number(
                float(value),
                reference(*point),
                Accuracy(domain="linear", operations=4 * ELEMENTARY),
                what,
            )
            gradients = differentiate(value, leaves, what)
            for position, gradient in enumerate(gradients):
                _check_derivative(gradient, reference, point, position, what)


def _vector(local: Local) -> Var:
    """A reference to a vector local.

    Parameters
    ----------
    local : Local
        The local.

    Returns
    -------
    Var
        The reference.
    """
    return Var(local=local)


def _vector_type(size: int) -> TypeExpr:
    """``Tensor[Real]([size])``.

    Parameters
    ----------
    size : int
        The extent.

    Returns
    -------
    TypeExpr
        The type.
    """
    return tensor_type(REAL, (IndexLiteral(value=size, sort=NAT),))


def _structures(xs: Local, index: Local) -> tuple[Value, ...]:
    """Terms over a vector that reduce, gather, map, and factor it.

    Parameters
    ----------
    xs : Local
        A vector of three reals.
    index : Local
        An Int selecting an entry.

    Returns
    -------
    tuple of Value
        A scalar term per structure.
    """
    vector = _vector(xs)
    binder = Local(name="i", type=INT)
    squares = Comprehension(
        binder=binder,
        extent=IndexLiteral(value=3, sort=NAT),
        body=_application(
            "mul_real",
            (
                Gather(value=vector, index=Var(local=binder), result_type=REAL),
                Gather(value=vector, index=Var(local=binder), result_type=REAL),
            ),
        ),
        result_type=_vector_type(3),
    )
    matrix = TensorValue(
        items=(
            TensorValue(
                items=(
                    _application(
                        "add_real",
                        (
                            Gather(
                                value=vector,
                                index=Var(local=index),
                                result_type=REAL,
                            ),
                            _application(
                                "exp",
                                (
                                    Gather(
                                        value=vector,
                                        index=Var(local=index),
                                        result_type=REAL,
                                    ),
                                ),
                            ),
                        ),
                    ),
                    Reduction(operator="sum", value=vector, result_type=REAL),
                ),
                result_type=_vector_type(2),
            ),
            TensorValue(
                items=(
                    Reduction(operator="sum", value=vector, result_type=REAL),
                    Reduction(operator="sum", value=squares, result_type=REAL),
                ),
                result_type=_vector_type(2),
            ),
        ),
        result_type=tensor_type(
            REAL, (IndexLiteral(value=2, sort=NAT), IndexLiteral(value=2, sort=NAT))
        ),
    )
    factor = CholeskyFactor(value=matrix, result_type=matrix.result_type)
    return (
        Reduction(operator="sum", value=squares, result_type=REAL),
        Reduction(operator="logsumexp", value=vector, result_type=REAL),
        Reduction(operator="prod", value=vector, result_type=REAL),
        Reduction(operator="max", value=vector, result_type=REAL),
        Reduction(operator="mean", value=vector, result_type=REAL),
        Gather(
            value=Rowwise(
                operator="log_softmax", value=vector, result_type=_vector_type(3)
            ),
            index=Var(local=index),
            result_type=REAL,
        ),
        Reduction(operator="sum", value=factor, result_type=REAL),
    )


def vector_type() -> TypeExpr:
    """``Tensor[Real]([3])``.

    Returns
    -------
    TypeExpr
        The type.
    """
    return _vector_type(3)


def _structure_references(index: int) -> tuple[Reference, ...]:
    """The standard-library values of `_structures`.

    Parameters
    ----------
    index : int
        The selected entry.

    Returns
    -------
    tuple of Reference
        One function of the three entries per term.
    """

    def factor_sum(a: float, b: float, c: float) -> float:
        """The sum of the entries of a two-by-two Cholesky factor.

        Parameters
        ----------
        a, b, c : float
            The vector.

        Returns
        -------
        float
            The sum.
        """
        entries = (a, b, c)
        top = entries[index] + math.exp(entries[index])
        off = a + b + c
        bottom = a * a + b * b + c * c
        first = math.sqrt(top)
        second = off / first
        return first + second + math.sqrt(bottom - second * second)

    def log_softmax(a: float, b: float, c: float) -> float:
        """The selected entry of the log-softmax.

        Parameters
        ----------
        a, b, c : float
            The vector.

        Returns
        -------
        float
            The entry.
        """
        entries = (a, b, c)
        peak = max(entries)
        total = math.fsum(math.exp(item - peak) for item in entries)
        return entries[index] - peak - math.log(total)

    return (
        lambda a, b, c: a * a + b * b + c * c,
        lambda a, b, c: math.log(math.exp(a) + math.exp(b) + math.exp(c)),
        lambda a, b, c: a * b * c,
        lambda a, b, c: max(a, b, c),
        lambda a, b, c: (a + b + c) / 3.0,
        log_softmax,
        factor_sum,
    )


STRUCTURE_POINT = (0.4, -1.1, 1.7)
"""A vector with distinct entries."""


def check_structure_gradients() -> None:
    """Reductions, gathers, comprehensions, tensors and a factorization.

    Raises
    ------
    Mismatch
        If a value or a derivative differs from the standard library.
    """
    xs = Local(name="xs", type=vector_type())
    index = Local(name="k", type=INT)
    for selected in (0, 2):
        references = _structure_references(selected)
        for term, reference in zip(_structures(xs, index), references, strict=True):
            leaf = torch.tensor(
                STRUCTURE_POINT, dtype=torch.float64, requires_grad=True
            )
            value = evaluator().evaluate_value(
                term, {xs: leaf, index: torch.tensor(selected)}
            )
            what = f"{type(term).__name__} at {selected}"
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
                raise Mismatch("dtype", math.inf, f"{what}: {value!r}")
            compare_number(
                float(value),
                reference(*STRUCTURE_POINT),
                Accuracy(domain="linear", operations=16 * ELEMENTARY),
                what,
            )
            (gradient,) = differentiate(value, [leaf], what)
            for position in range(3):
                _check_derivative(
                    gradient[position], reference, STRUCTURE_POINT, position, what
                )


def check_boundary_gradients() -> None:
    """Derivatives at ties, kinks and impossible or empty sums, by hand.

    A selection gives the whole derivative to the first of tied operands;
    a kink of a piecewise function takes the derivative of the piece that
    holds the point; an empty or all-impossible sum has derivative zero,
    never NaN.

    Raises
    ------
    Mismatch
        If a value or a derivative differs from the hand-derived one.
    """
    cases: list[
        tuple[
            str, Value, tuple[Local, ...], tuple[float, ...], float, tuple[float, ...]
        ]
    ] = []
    a, b = Local(name="a", type=REAL), Local(name="b", type=REAL)
    for name in ("max_real", "min_real"):
        cases.append(
            (
                f"{name} at a tie",
                _application(name, (Var(local=a), Var(local=b))),
                (a, b),
                (0.5, 0.5),
                0.5,
                (1.0, 0.0),
            )
        )
    cases.append(
        (
            "relu at zero",
            _application("relu", (Var(local=a),)),
            (a,),
            (0.0,),
            0.0,
            (0.0,),
        )
    )
    cases.append(
        (
            "abs at zero",
            _application("abs_real", (Var(local=a),)),
            (a,),
            (0.0,),
            0.0,
            (0.0,),
        )
    )
    xs = Local(name="xs", type=vector_type())
    cases.append(
        (
            "maximum of tied entries",
            Reduction(operator="max", value=Var(local=xs), result_type=REAL),
            (xs,),
            (1.0, 1.0, 0.0),
            1.0,
            (1.0, 0.0, 0.0),
        )
    )
    cases.append(
        (
            "log-sum-exp of impossible entries",
            Reduction(operator="logsumexp", value=Var(local=xs), result_type=REAL),
            (xs,),
            (-math.inf, -math.inf, -math.inf),
            -math.inf,
            (0.0, 0.0, 0.0),
        )
    )
    for what, term, locals_, point, wanted, gradient in cases:
        vector = locals_[0].type == vector_type()
        leaves = (
            [torch.tensor(point, dtype=torch.float64, requires_grad=True)]
            if vector
            else _leaves(point)
        )
        value = evaluator().evaluate_value(
            term, dict(zip(locals_, leaves, strict=True))
        )
        assert isinstance(value, torch.Tensor)
        compare_number(
            float(value), wanted, Accuracy(domain="linear", operations=1), what
        )
        found = differentiate(value, leaves, what)
        flat = torch.cat([item.reshape(-1) for item in found])
        for position, (number, expected) in enumerate(
            zip(flat.tolist(), gradient, strict=True)
        ):
            compare_number(
                number,
                expected,
                Accuracy(domain="linear", operations=1),
                f"{what} d/d{position}",
            )


def _structural_gradients(
    value: torch.Tensor,
    leaves: Mapping[str, torch.Tensor],
    used: set[str],
    what: str,
) -> list[torch.Tensor]:
    """Gradients of a weight, zero exactly for weights no derivation uses.

    A weight that no derivation uses is never read by the run, so the
    value does not depend on it at any point and its derivative is zero by
    structure. Every other weight must have a gradient.

    Parameters
    ----------
    value : torch.Tensor
        The weight.
    leaves : Mapping of str to torch.Tensor
        The weights differentiated by, by name.
    used : set of str
        The names some derivation uses.
    what : str
        A description for the report.

    Returns
    -------
    list of torch.Tensor
        One gradient per leaf, zero for an unused weight.

    Raises
    ------
    Mismatch
        If the weight has no gradient path while some leaf is used, or a
        used leaf has no gradient.
    """
    if not used & set(leaves):
        return [torch.zeros_like(leaf) for leaf in leaves.values()]
    if not value.requires_grad:
        raise Mismatch("missing", math.inf, f"{what}: the value has no gradient path")
    found = torch.autograd.grad(value, list(leaves.values()), allow_unused=True)
    gradients: list[torch.Tensor] = []
    for (name, leaf), gradient in zip(leaves.items(), found, strict=True):
        if gradient is None:
            if name in used:
                raise Mismatch("missing", math.inf, f"{what}: no gradient for {name}")
            gradients.append(torch.zeros_like(leaf))
            continue
        gradients.append(gradient)
    return gradients


def check_lexical_gradient() -> None:
    """The derivative of a sentence's log weight in the rule log weights.

    It is the expected number of uses of the rule under the normalized
    weights of the derivations (oracle O1), here exact in fractions.

    Raises
    ------
    Mismatch
        If a derivative is missing or differs from the expected count.
    """
    module = module_of("lexical", SEMIRING="LogProb")
    weights = {"join": JOIN, "lift": LIFT}
    for sentence in SENTENCES:
        trees = parses(sentence)
        weight_of = {
            **weights,
            **{f"{i}:{c}": Fraction(1) for i in range(len(sentence)) for c in "SV"},
        }
        leaves = {
            name: torch.tensor(
                logarithm(value), dtype=torch.float64, requires_grad=True
            )
            for name, value in weights.items()
        }
        found = run_deduction(
            module,
            "G",
            tokens=sentence,
            parameters={f"G.rule.{name}:": leaf for name, leaf in leaves.items()},
        ).weight
        what = f"machine gradient {sentence}"
        if not isinstance(found, torch.Tensor):
            raise Mismatch("dtype", math.inf, f"{what}: {found!r}")
        used = {name for tree in trees for name in derivations.factors(tree)}
        for name, gradient in zip(
            leaves, _structural_gradients(found, leaves, used, what), strict=True
        ):
            compare_number(
                float(gradient),
                derivations.logarithmic_gradient(trees, weight_of, name),
                Accuracy(domain="log", operations=8 * len(trees) + 8, magnitude=8.0),
                f"{what} d/d{name}",
            )


def check_nonlexical_gradient() -> None:
    """The derivative of a chart's log weight in rule and axiom weights.

    Each of two equal axioms is an occurrence of its own, so each gets
    the expected number of uses of that occurrence.

    Raises
    ------
    Mismatch
        If a derivative is missing or differs from the expected count.
    """
    module = module_of("nonlexical", SEMIRING="LogProb")
    trees = nonlexical_trees(2)
    weight_of = {
        **NONLEXICAL_WEIGHTS,
        **{f"axiom{i}": weight for i, weight in enumerate(AXIOM_WEIGHTS)},
    }
    rules = {
        name: torch.tensor(logarithm(value), dtype=torch.float64, requires_grad=True)
        for name, value in NONLEXICAL_WEIGHTS.items()
    }
    axioms = [
        torch.tensor(logarithm(value), dtype=torch.float64, requires_grad=True)
        for value in AXIOM_WEIGHTS
    ]
    found = run_deduction(
        module,
        "W",
        axioms=[deduction_item(module, "W", "A")] * 2,
        axiom_weights=axioms,
        parameters={f"W.rule.{name}:": leaf for name, leaf in rules.items()},
        fuel=200_000,
    ).weight
    what = "machine chart gradient"
    if not isinstance(found, torch.Tensor):
        raise Mismatch("dtype", math.inf, f"{what}: {found!r}")
    names = [*rules, *(f"axiom{i}" for i in range(len(axioms)))]
    gradients = differentiate(found, [*rules.values(), *axioms], what)
    for name, gradient in zip(names, gradients, strict=True):
        compare_number(
            float(gradient),
            derivations.logarithmic_gradient(trees, weight_of, name),
            Accuracy(domain="log", operations=8 * len(trees) + 8, magnitude=8.0),
            f"{what} d/d{name}",
        )


def recursive_joint(t: float, y: float) -> float:
    """The log joint of the recursive program, from its definition.

    ``t ~ Normal(0, 1)`` and ``y ~ Normal(t + 3, 1)``, since the
    recursive computation adds one three times.

    Parameters
    ----------
    t : float
        The draw.
    y : float
        The observation.

    Returns
    -------
    float
        The log joint.
    """
    half_log_two_pi = 0.5 * math.log(2.0 * math.pi)
    return -0.5 * t * t - half_log_two_pi - 0.5 * (y - t - 3.0) ** 2 - half_log_two_pi


def check_program_gradient() -> None:
    """A program's log joint and its derivatives in a site and a datum.

    The program calls a recursive computation, so only the machine runs
    it; its densities go through the tensor backend.

    Raises
    ------
    Mismatch
        If the joint or a derivative differs from the definition's.
    """
    module = module_of("recursive")
    point = (0.3, 0.5)
    t, y = _leaves(point)
    run = run_program(
        module,
        "recursive",
        data={"y": y},
        sites={"t": t},
        distributions=TensorBackend(),
    )
    what = "machine program joint"
    compare_number(
        float(run.log_joint),
        recursive_joint(*point),
        Accuracy(domain="linear", operations=8 * ELEMENTARY),
        what,
    )
    gradients = differentiate(run.log_joint, [t, y], what)
    for position, gradient in enumerate(gradients):
        _check_derivative(gradient, recursive_joint, point, position, what)


def _register() -> None:
    """Register the checks."""
    add(
        "evaluator/gradient/primitives",
        "machine-program",
        "Q1g",
        check_primitive_gradients,
    )
    add(
        "evaluator/gradient/structures",
        "machine-program",
        "Q1g",
        check_structure_gradients,
    )
    add(
        "evaluator/gradient/boundaries",
        "machine-program",
        "Q1g",
        check_boundary_gradients,
    )
    add(
        "evaluator/gradient/program",
        "machine-program",
        "Q1g",
        check_program_gradient,
    )
    add(
        "machine/gradient/lexical/logprob",
        "machine-lexical",
        "Q1g",
        check_lexical_gradient,
    )
    add(
        "machine/gradient/nonlexical/logprob",
        "machine-nonlexical",
        "Q1g",
        check_nonlexical_gradient,
    )


_register()

__all__ = ["REFERENCES", "evaluator", "recursive_joint"]
