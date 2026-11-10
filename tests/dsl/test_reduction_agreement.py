"""Every let reduction means one thing on every execution path.

A reduction builtin (``sum``, ``mean``, ``prod``, ``max``, ``min``,
``logsumexp``) collapses the last axis of its argument, and a rowwise one
(``softmax``, ``log_softmax``, ``cumsum``, ``sort``, ``normalize``) acts
along it. The tables below apply each builtin to a vector over a plate, to
a plate of event vectors, and to a matrix a factor builds over two plates,
and compare the compiled runtime, the reference machine, and the rendered
NumPyro and Pyro bodies at one point. The rendered bodies are evaluated
statement by statement on the same arrays: NumPyro's ``jnp`` calls through
NumPy, whose reductions take the same ``axis`` keyword, and Pyro's through
PyTorch itself. A builtin a target has no rendering for must be refused
there, not rendered with a different meaning.

The host-data contract is pinned here too: a datum whose shape or element
type contradicts the checked type of the free name it fills is refused by
the compiled runtime, naming the datum and both shapes.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.effects.interpreter import check_host_data
from quivers.qiec.canonical import tensor_type
from quivers.qiec.kinds import NAT
from quivers.qiec.types import REAL, IndexVariable
from quivers.transpile import UnsupportedConstruct, transpile

#: The reductions, each collapsing its argument's last axis.
REDUCTIONS: tuple[str, ...] = ("sum", "mean", "prod", "max", "min", "logsumexp")

#: The rowwise operations, each acting along its argument's last axis.
ROWWISE: tuple[str, ...] = ("softmax", "log_softmax", "cumsum", "sort", "normalize")

#: The builtins the Python targets render; every other one is refused.
RENDERED: frozenset[str] = frozenset({"sum", "mean", "prod", "max", "min", "softmax"})

_HEADER = """
object Row : FinSet 3
object Unit : FinSet 2
object Dim : FinSet 2
"""

#: Each shape: the program steps building the argument ``v``, the sites
#: they draw, and the values those sites take. The values are positive
#: so ``normalize`` and ``prod`` stay well conditioned.
SHAPES: dict[str, tuple[str, dict[str, list[float] | list[list[float]]]]] = {
    "scalar-plate": (
        "    sample v : Row <- Normal(loc=0.0, scale=1.0)\n",
        {"v": [0.5, 2.0, 1.25]},
    ),
    "vector-event": (
        "    sample v : Row <- Normal(loc=0.0, scale=1.0) [over=Dim, iid_over=Row]\n",
        {"v": [[0.5, 2.0], [1.5, 0.25], [3.0, 1.0]]},
    ),
    "matrix": (
        "    sample x : Row <- Normal(loc=0.0, scale=1.0)\n"
        "    sample w : Unit <- Normal(loc=0.0, scale=1.0)\n"
        "    let v = factor r : Row, u : Unit in x[r] * w[u]\n",
        {"x": [0.5, 2.0, 1.25], "w": [1.5, 0.75]},
    ),
}

#: The expected value of each builtin along a tensor's last axis.
EXPECTED: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "sum": lambda t: t.sum(-1),
    "mean": lambda t: t.mean(-1),
    "prod": lambda t: t.prod(-1),
    "max": lambda t: t.amax(-1),
    "min": lambda t: t.amin(-1),
    "logsumexp": lambda t: t.logsumexp(-1),
    "softmax": lambda t: t.softmax(-1),
    "log_softmax": lambda t: t.log_softmax(-1),
    "cumsum": lambda t: t.cumsum(-1),
    "sort": lambda t: t.sort(-1).values,
    "normalize": lambda t: t / t.sum(-1, keepdim=True),
}


def _source(builtin: str, shape: str) -> str:
    """A program applying one builtin to one argument shape.

    Parameters
    ----------
    builtin : str
        The builtin.
    shape : str
        The key of the argument's shape in `SHAPES`.

    Returns
    -------
    str
        QVR source returning the builtin's value.
    """
    steps, _ = SHAPES[shape]
    return (
        f"{_HEADER}\nprogram net : Row -> Row\n{steps}"
        f"    let r = {builtin}(v)\n    return r\n\nexport net\n"
    )


def _argument(shape: str) -> torch.Tensor:
    """The builtin's argument at the shape's site values.

    Parameters
    ----------
    shape : str
        The key of the shape in `SHAPES`.

    Returns
    -------
    torch.Tensor
        The vector, the plate of event vectors, or the outer product.
    """
    _, sites = SHAPES[shape]
    if shape == "matrix":
        return torch.outer(torch.tensor(sites["x"]), torch.tensor(sites["w"]))
    return torch.tensor(sites["v"])


def _nested(value: list[float] | list[list[float]]) -> tuple[object, ...]:
    """A site value as the reference machine's nested tuples.

    Parameters
    ----------
    value : list[float] | list[list[float]]
        The value.

    Returns
    -------
    tuple[object, ...]
        The same entries, nested as tuples.
    """
    return tuple(_nested(item) if isinstance(item, list) else item for item in value)


def _flat(value: object) -> list[float]:
    """The entries of a nested host value, outermost first.

    Parameters
    ----------
    value : object
        A number, a tensor, an array, or nested tuples of numbers.

    Returns
    -------
    list[float]
        Its entries.
    """
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, tuple | list):
        return [entry for item in value for entry in _flat(item)]
    assert isinstance(value, int | float)
    return [float(value)]


def _assert_values(actual: object, expected: torch.Tensor, path: str) -> None:
    """Require a value to equal the expected tensor, shape and entries.

    Parameters
    ----------
    actual : object
        The value a path computed.
    expected : torch.Tensor
        The value along the last axis.
    path : str
        The path's name, for the failure message.
    """
    shape = tuple(np.shape(actual)) if not isinstance(actual, tuple) else None
    if shape is not None:
        assert shape == tuple(expected.shape), (path, shape, tuple(expected.shape))
    entries = _flat(actual)
    wanted = _flat(expected)
    assert len(entries) == len(wanted), (path, entries, wanted)
    for got, want in zip(entries, wanted, strict=True):
        assert math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-6), (
            path,
            entries,
            wanted,
        )


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """NumPy's spelling of ``jax.nn.softmax``.

    Parameters
    ----------
    values : np.ndarray
        The logits.
    axis : int
        The axis normalized over.

    Returns
    -------
    np.ndarray
        The probabilities along ``axis``.
    """
    shifted = np.exp(values - values.max(axis=axis, keepdims=True))
    return shifted / shifted.sum(axis=axis, keepdims=True)


#: The names a rendered NumPyro body reads, bound to the NumPy functions
#: that take the same arguments and the same ``axis`` keyword.
_NUMPYRO_NAMESPACE: dict[str, object] = {
    "jnp": SimpleNamespace(
        sum=np.sum,
        mean=np.mean,
        prod=np.prod,
        max=np.max,
        min=np.min,
        stack=np.stack,
    ),
    "jnn": SimpleNamespace(softmax=_softmax),
}


def _rendered_lets(source: str, target: str, names: frozenset[str]) -> list[str]:
    """The rendered model's deterministic assignments to the given names.

    A site's assignment draws through the library's ``sample`` and is
    skipped; the site's value is bound by the caller instead.

    Parameters
    ----------
    source : str
        QVR source.
    target : str
        ``"numpyro"`` or ``"pyro"``.
    names : frozenset[str]
        The let names.

    Returns
    -------
    list[str]
        Each assignment's source, in body order.
    """
    tree = ast.parse(transpile(parse(source), target=target).decode())
    model = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "model"
    )
    return [
        ast.unparse(node)
        for node in ast.walk(model)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
        and not _draws(node.value)
    ]


def _draws(value: ast.expr) -> bool:
    """Whether an assigned expression is a call to a ``sample`` function.

    Parameters
    ----------
    value : ast.expr
        The right-hand side.

    Returns
    -------
    bool
        True for ``<library>.sample(...)``.
    """
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "sample"
    )


def _evaluate_rendered(
    source: str,
    target: str,
    sites: Mapping[str, list[float] | list[list[float]]],
) -> object:
    """Run the rendered let statements on the site values.

    Parameters
    ----------
    source : str
        QVR source whose result is the let ``r``.
    target : str
        ``"numpyro"`` or ``"pyro"``.
    sites : Mapping[str, list[float] | list[list[float]]]
        The site values.

    Returns
    -------
    object
        The value the rendered body binds to ``r``.
    """
    lets = _rendered_lets(source, target, frozenset({"v", "r"}))
    namespace: dict[str, object]
    if target == "numpyro":
        namespace = {
            **_NUMPYRO_NAMESPACE,
            **{name: np.asarray(value) for name, value in sites.items()},
        }
    else:
        namespace = {
            "torch": torch,
            **{name: torch.tensor(value) for name, value in sites.items()},
        }
    for statement in lets:
        exec(statement, namespace)
    return namespace["r"]


@pytest.mark.parametrize("shape", tuple(SHAPES))
@pytest.mark.parametrize("builtin", REDUCTIONS + ROWWISE)
def test_a_builtin_acts_along_the_last_axis_on_every_path(
    builtin: str, shape: str
) -> None:
    source = _source(builtin, shape)
    _, sites = SHAPES[shape]
    expected = EXPECTED[builtin](_argument(shape))
    program = loads(source)
    assert not program.qiec.gap, program.qiec.gap
    compiled = program.morphism.rsample(
        torch.zeros(1, 1),
        observations={name: torch.tensor(value) for name, value in sites.items()},
    )
    _assert_values(compiled, expected, "compiled runtime")
    reference = program.run(
        "net", sites={name: _nested(value) for name, value in sites.items()}
    )
    _assert_values(reference.value, expected, "reference machine")
    for target in ("numpyro", "pyro"):
        if builtin in RENDERED:
            _assert_values(_evaluate_rendered(source, target, sites), expected, target)
        else:
            with pytest.raises(UnsupportedConstruct):
                transpile(parse(source), target=target)


def test_the_checked_contract_types_a_reduction_by_its_last_axis() -> None:
    program = loads(_source("sum", "matrix"))
    (entry,) = program.entry_points()
    assert entry.result == "Tensor[Real]([3])"


_GATHER = """
object Row : FinSet 3
object Unit : FinSet 2
object Target : Real 1

program net : Row -> Target
    sample w : Unit <- Normal(loc=0.0, scale=1.0)
    let pre = x[row_idx] * w[unit_idx]
    let mu = sum(pre)
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return y

export net
"""


def _gather_point(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    """A point for `_GATHER` whose data inhabit their checked types.

    Parameters
    ----------
    **overrides : torch.Tensor
        Values replacing the defaults, by name.

    Returns
    -------
    dict[str, torch.Tensor]
        The data, the site, and the observation.
    """
    point = {
        "x": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "row_idx": torch.tensor([0, 1, 3]),
        "unit_idx": torch.tensor([0, 1, 1]),
        "w": torch.tensor([0.5, -1.0]),
        "y": torch.tensor([0.0, 0.0, 0.0]),
    }
    point.update(overrides)
    return point


def test_a_gathered_table_takes_open_extents_and_its_index_the_plate() -> None:
    program = loads(_GATHER)
    (entry,) = program.entry_points()
    types = {parameter.name: parameter.type for parameter in entry.parameters}
    assert types["x"] == "Tensor[Real]([x_extent])"
    assert types["row_idx"] == "Tensor[Int]([3])"
    assert types["unit_idx"] == "Tensor[Int]([3])"


def test_data_that_inhabit_the_checked_types_score_the_checked_model() -> None:
    program = loads(_GATHER)
    point = _gather_point()
    joint = program.morphism.log_joint(torch.zeros(1, 1), point)
    normal = torch.distributions.Normal
    mu = (point["x"][point["row_idx"]] * point["w"][point["unit_idx"]]).sum()
    expected = (
        normal(0.0, 1.0).log_prob(point["w"]).sum()
        + normal(mu, 1.0).log_prob(point["y"]).sum()
    )
    assert float(joint.reshape(-1)[0]) == pytest.approx(float(expected), rel=1e-6)


def test_a_two_dimensional_index_for_a_plated_name_is_refused() -> None:
    program = loads(_GATHER)
    point = _gather_point(
        row_idx=torch.tensor([[0, 0], [1, 1], [2, 2]]),
        unit_idx=torch.tensor([[0, 1], [0, 1], [0, 1]]),
    )
    message = (
        r"host datum 'row_idx' has shape \(3, 2\), but the checked program "
        r"reads it at type Tensor\[Int\]\(\[3\]\), of shape \(3,\)"
    )
    with pytest.raises(ValueError, match=message):
        program.morphism.log_joint(torch.zeros(1, 1), point)
    with pytest.raises(ValueError, match=message):
        program.morphism.rsample(torch.zeros(1, 1), observations=point)


def test_a_real_index_is_refused_by_its_element_type() -> None:
    program = loads(_GATHER)
    point = _gather_point(row_idx=torch.tensor([0.0, 1.0, 2.0]))
    with pytest.raises(TypeError, match=r"host datum 'row_idx' has dtype torch\.float"):
        program.morphism.log_joint(torch.zeros(1, 1), point)


def test_an_integer_tensor_for_real_data_is_refused() -> None:
    program = loads(_GATHER)
    point = _gather_point(x=torch.tensor([1, 2, 3, 4]))
    with pytest.raises(TypeError, match=r"host datum 'x' has dtype torch\.int64"):
        program.morphism.log_joint(torch.zeros(1, 1), point)


def test_a_name_filling_a_plated_draws_parameter_is_one_value_per_row() -> None:
    program = loads(
        """
object Resp : FinSet 4
program m : Resp -> Resp
    observe y : Resp <- Normal(loc=mu, scale=1.0)
    return y
export m
"""
    )
    (entry,) = program.entry_points()
    types = {parameter.name: parameter.type for parameter in entry.parameters}
    assert types["mu"] == "Tensor[Real]([4])"
    mu = torch.tensor([0.5, -1.0, 2.0, 0.0])
    joint = program.morphism.log_joint(
        torch.zeros(1, 1), {"mu": mu, "y": torch.zeros(4)}
    )
    expected = torch.distributions.Normal(mu, 1.0).log_prob(torch.zeros(4)).sum()
    assert float(joint.reshape(-1)[0]) == pytest.approx(float(expected), rel=1e-6)
    with pytest.raises(ValueError, match=r"host datum 'mu' has shape \(\)"):
        program.morphism.log_joint(
            torch.zeros(1, 1), {"mu": torch.tensor(0.5), "y": torch.zeros(4)}
        )


def test_a_datum_may_lead_with_the_input_batch_axis() -> None:
    program = loads(
        """
object Resp : FinSet 4
program m : Resp -> Resp
    observe y : Resp <- Normal(loc=mu, scale=1.0)
    return y
export m
"""
    )
    # The datum's shape inhabits the checked type once per run; how a run
    # reads its row of the datum is the input batch's own contract.
    mu = torch.tensor([[0.5, -1.0, 2.0, 0.0]])
    program.morphism.log_joint(torch.zeros(1, 1), {"mu": mu, "y": torch.zeros(4)})
    with pytest.raises(ValueError, match=r"host datum 'mu' has shape \(2, 4\)"):
        program.morphism.log_joint(
            torch.zeros(1, 1), {"mu": mu.expand(2, 4), "y": torch.zeros(4)}
        )


def test_an_open_extent_must_agree_across_the_data_it_shapes() -> None:
    program = loads(_GATHER)
    monadic = program.morphism
    extent = IndexVariable("n", NAT)
    monadic.host_data_types = {
        "a": tensor_type(REAL, (extent,)),
        "b": tensor_type(REAL, (extent,)),
    }
    check_host_data(monadic, {"a": torch.zeros(4), "b": torch.ones(4)})
    with pytest.raises(ValueError, match=r"'b' fixes extent 'n' at 5, but 'a'"):
        check_host_data(monadic, {"a": torch.zeros(4), "b": torch.ones(5)})
