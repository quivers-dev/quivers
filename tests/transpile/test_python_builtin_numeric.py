"""Every math builtin a Python target maps, applied to plain numbers and scored.

A let that applies a math builtin to a constant, or to the Python
number an unrolled ``factor`` binder becomes, hands the target's array
library a number rather than an array. Each Python-family target's
table of builtins is read here, every builtin it maps is applied both
to a constant and to the binder, and the rendered program is scored
in the target's image against the reference machine, which is itself
pinned to an independent ``torch`` computation. A surface builtin the
table leaves out must be refused when the program is rendered.
"""

from __future__ import annotations

from collections.abc import Callable
import pathlib

import pytest
import torch
import torch.distributions as D
import torch.nn.functional as F

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.dsl.pure_builtins import PRIMITIVE_BUILTINS
from quivers.transpile import transpile
from quivers.transpile.renderers._python_helpers import (
    _BINARY_POINTWISE_SYMBOLS,
    _LET_CALL_SYMBOLS,
)
from quivers.transpile import UnsupportedConstruct
from tests.transpile import _docker, _equivalence
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

#: The Python-family targets, each with its own builtin table.
_TARGETS: tuple[str, ...] = ("edward2", "numpyro", "pymc", "pyro")

#: Each one-argument surface builtin's independent ``torch`` meaning.
_UNARY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "abs": torch.abs,
    "acos": torch.acos,
    "acosh": torch.acosh,
    "asin": torch.asin,
    "asinh": torch.asinh,
    "atan": torch.atan,
    "atanh": torch.atanh,
    "ceil": torch.ceil,
    "cos": torch.cos,
    "cosh": torch.cosh,
    "digamma": torch.digamma,
    "elu": F.elu,
    "erf": torch.erf,
    "erfc": torch.erfc,
    "erfinv": torch.erfinv,
    "exp": torch.exp,
    "expm1": torch.expm1,
    "floor": torch.floor,
    "gelu": F.gelu,
    "lgamma": torch.lgamma,
    "log": torch.log,
    "log10": torch.log10,
    "log1p": torch.log1p,
    "log2": torch.log2,
    "logsigmoid": F.logsigmoid,
    "mish": F.mish,
    "reciprocal": torch.reciprocal,
    "relu": F.relu,
    "relu6": F.relu6,
    "round": torch.round,
    "rsqrt": torch.rsqrt,
    "selu": F.selu,
    "sigmoid": torch.sigmoid,
    "sign": torch.sign,
    "silu": F.silu,
    "sin": torch.sin,
    "sinh": torch.sinh,
    "softplus": F.softplus,
    "softsign": F.softsign,
    "sqrt": torch.sqrt,
    "square": torch.square,
    "tan": torch.tan,
    "tanh": torch.tanh,
    "trunc": torch.trunc,
}

#: Each two-argument pointwise builtin's independent ``torch`` meaning.
_BINARY: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "max": torch.maximum,
    "min": torch.minimum,
    "pow": torch.pow,
}

#: The constant each builtin is applied to, inside its domain; the
#: binder adds a tenth of the step index (at most 0.2) to it.
_ARGUMENT: dict[str, float] = {"acosh": 1.5}
_DEFAULT_ARGUMENT = 0.4

_STEPS = 3
_DATA: list[float] = [0.5, -0.2, 0.7]
_POINTS: list[float] = [-0.3, 0.2, 0.8]


def _argument(name: str) -> float:
    """The constant a builtin is applied to.

    Parameters
    ----------
    name : str
        The builtin.

    Returns
    -------
    float
        A value inside the builtin's domain, with room for the binder's
        offset.
    """
    return _ARGUMENT.get(name, _DEFAULT_ARGUMENT)


def _mapped(target: str) -> tuple[list[str], list[str]]:
    """The surface builtins a target's tables map.

    Parameters
    ----------
    target : str
        A Python-family target.

    Returns
    -------
    tuple[list[str], list[str]]
        The one-argument builtins, then the two-argument ones.
    """
    unary = sorted(name for name in _UNARY if name in _LET_CALL_SYMBOLS[target])
    binary = sorted(
        name for name in _BINARY if name in _BINARY_POINTWISE_SYMBOLS[target]
    )
    return unary, binary


def _source(unary: list[str], binary: list[str]) -> str:
    """A program whose lets apply each builtin to a constant and a binder.

    Parameters
    ----------
    unary : list[str]
        One-argument builtins.
    binary : list[str]
        Two-argument builtins.

    Returns
    -------
    str
        QVR source scoring ``mu`` times each builtin's applications, one
        constant, one ``factor`` and one score per builtin.
    """
    calls = [
        (f"{name}({_argument(name)})", f"{name}(real(t) / 10.0 + {_argument(name)})")
        for name in unary
    ] + [(f"{name}(0.6, 1.5)", f"{name}(real(t) / 10.0 + 0.6, 1.5)") for name in binary]
    steps: list[str] = []
    for index, (constant, stepped) in enumerate(calls):
        steps += [
            f"    let c{index} = {constant}",
            f"    let k{index} = factor t : Step in {stepped}",
            f"    score s{index} = mu * (c{index} + sum(k{index}))",
        ]
    return "\n".join(
        [
            f"object Step : FinSet {_STEPS}",
            "object Val : Real 1",
            "",
            "program p : Step -> Val",
            "    sample mu <- Normal(loc=0.0, scale=1.0)",
            "    observe y : Step <- Normal(loc=mu, scale=1.0)",
            *steps,
            "    return mu",
            "",
            "export p",
            "",
        ]
    )


def _oracle(unary: list[str], binary: list[str], mu: float) -> float:
    """The joint at ``mu``, computed in ``torch`` without the DSL.

    Parameters
    ----------
    unary : list[str]
        One-argument builtins.
    binary : list[str]
        Two-argument builtins.
    mu : float
        The clamped latent.

    Returns
    -------
    float
        The log joint.
    """
    total = torch.tensor(0.0, dtype=torch.float64)
    for step in range(_STEPS + 1):
        offset = 0.0 if step == 0 else (step - 1) / 10.0
        for name in unary:
            argument = torch.tensor(_argument(name) + offset, dtype=torch.float64)
            total = total + _UNARY[name](argument)
        for name in binary:
            left = torch.tensor(0.6 + offset, dtype=torch.float64)
            total = total + _BINARY[name](left, torch.tensor(1.5, dtype=torch.float64))
    latent = torch.tensor(mu, dtype=torch.float64)
    data = torch.tensor(_DATA, dtype=torch.float64)
    return float(
        D.Normal(0.0, 1.0).log_prob(latent)
        + D.Normal(latent, 1.0).log_prob(data).sum()
        + latent * total
    )


def _reference(source: str, unary: list[str], binary: list[str]) -> list[float]:
    """The torch runtime's joint at each point, held to the oracle.

    Parameters
    ----------
    source : str
        The program.
    unary : list[str]
        One-argument builtins.
    binary : list[str]
        Two-argument builtins.

    Returns
    -------
    list[float]
        The log joint at each point.
    """
    model = loads(source).morphism
    densities: list[float] = []
    for mu in _POINTS:
        sites = {"mu": torch.tensor([mu]), "y": torch.tensor(_DATA)}
        joint = float(model.log_joint(torch.zeros(1, 1), sites).sum())
        assert joint == pytest.approx(_oracle(unary, binary, mu), rel=1e-5)
        densities.append(joint)
    return densities


@pytest.mark.parametrize("target", _TARGETS)
def test_target_maps_or_refuses_every_surface_builtin(target: str) -> None:
    surface = set(PRIMITIVE_BUILTINS.values())
    assert set(_UNARY) | set(_BINARY) <= surface
    unary, binary = _mapped(target)
    for name in sorted(set(_UNARY) - set(unary)):
        with pytest.raises(UnsupportedConstruct, match=repr(name)):
            transpile(parse(_source([name], [])), target=target)
    for name in sorted(set(_BINARY) - set(binary)):
        with pytest.raises(UnsupportedConstruct, match=repr(name)):
            transpile(parse(_source([], [name])), target=target)


@pytest.mark.parametrize("target", _TARGETS)
def test_target_scores_every_builtin_on_plain_numbers(
    target: str, scratch: pathlib.Path
) -> None:
    unary, binary = _mapped(target)
    source = _source(unary, binary)
    image, ext, script_name = _BACKENDS_WITH_IMAGES[target]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    raw = _docker.run_probe(
        image=image,
        script=pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name,
        source=transpile(parse(source), target=target),
        source_ext=ext,
        points=[{"params": {"mu": mu}, "data": {"y": _DATA}} for mu in _POINTS],
        scratch=scratch / target,
        shapes={"mu": [], "y": [_STEPS]},
        dtypes={"mu": "float", "y": "float"},
    )
    _equivalence.assert_log_density_match(
        _reference(source, unary, binary),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=_STEPS),
        context=f"{target}@builtins",
    )
