"""Every target's spelling of every math builtin, scored against the reference.

A let applies a math builtin by its QVR name, and each target settles
each builtin in
[`BUILTIN_SPELLINGS`][quivers.transpile._builtin_spellings.BUILTIN_SPELLINGS]:
a function of its own, a definition in terms of other builtins, code of
its renderer's own, or a refusal. Every target's table is checked to
settle every surface builtin; every builtin a target spells is applied
to a constant, to an unrolled ``factor`` binder, and to a plated
observation, and the rendered program is scored in the target's image
against the reference machine, which is itself pinned to an independent
``torch`` computation; every builtin a target refuses must be refused by
name when the program is rendered.
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
from quivers.dsl.pure_builtins import BUILTIN_REGISTRY, PRIMITIVE_BUILTINS
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from quivers.transpile._builtin_spellings import (
    BUILTIN_SPELLINGS,
    ELABORATED_BUILTINS,
    MATH_BUILTINS,
    Elaborated,
    Refusal,
)
from tests.transpile import _docker, _equivalence
from tests.transpile.probes import _protocol
from tests.transpile.probes.church import ChurchProbe
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

#: Each one-argument builtin's independent ``torch`` meaning.
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

#: Each three-argument pointwise builtin's independent ``torch`` meaning,
#: applied as ``f(x > _THRESHOLD, x, 1.0 - x)``.
_TERNARY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "select": lambda x: torch.where(x > _THRESHOLD, x, 1.0 - x),
}

#: The condition ``select`` reads: the constant falls below it, and the
#: binder's and the plated operand's entries fall on both sides.
_THRESHOLD = 0.43

#: The constant each builtin is applied to, inside its domain. The
#: binder adds a tenth of the step index (at most 0.2), and the plated
#: operand a tenth of an observation (within 0.07), so every argument
#: stays inside the domain. ``round`` sits on a tie, which a target
#: rounding half up or away from zero gets wrong.
_ARGUMENT: dict[str, float] = {"acosh": 1.5, "round": 2.5, "pow": 0.6}
_DEFAULT_ARGUMENT = 0.4
_EXPONENT = 1.5

_STEPS = 3
_DATA: list[float] = [0.5, -0.2, 0.7]
_POINTS: list[float] = [-0.3, 0.2, 0.8]

_TARGETS: tuple[str, ...] = tuple(sorted(available_targets()))


def _argument(name: str) -> float:
    """The constant a builtin is applied to.

    Parameters
    ----------
    name : str
        The builtin.

    Returns
    -------
    float
        A value inside the builtin's domain.
    """
    return _ARGUMENT.get(name, _DEFAULT_ARGUMENT)


def _spelled(target: str) -> list[str]:
    """The builtins a target spells rather than refuses.

    Parameters
    ----------
    target : str
        A transpile target.

    Returns
    -------
    list[str]
        The builtins, sorted.
    """
    table = BUILTIN_SPELLINGS[target]
    return sorted(
        name for name in MATH_BUILTINS if not isinstance(table[name], Refusal)
    )


def _application(name: str, shift: str) -> str:
    """A builtin applied to its constant plus ``shift``.

    Parameters
    ----------
    name : str
        The builtin.
    shift : str
        QVR source added to the constant, or ``""``.

    Returns
    -------
    str
        The call.
    """
    argument = f"{shift}{_argument(name)}"
    if name in _TERNARY:
        return f"{name}({argument} > {_THRESHOLD}, {argument}, 1.0 - ({argument}))"
    return (
        f"{name}({argument}, {_EXPONENT})" if name in _BINARY else f"{name}({argument})"
    )


def _source(names: list[str]) -> str:
    """A program applying each builtin to a constant, a binder, and a plate.

    One let per builtin and shape keeps each expression small, and one
    score per builtin weighs its three applications by ``mu``.

    Parameters
    ----------
    names : list[str]
        The builtins.

    Returns
    -------
    str
        QVR source.
    """
    steps: list[str] = []
    for index, name in enumerate(names):
        steps += [
            f"    let c{index} = {_application(name, '')}",
            f"    let k{index} = factor t : Step in "
            f"{_application(name, 'real(t) / 10.0 + ')}",
            f"    let v{index} = {_application(name, 'y / 10.0 + ')}",
            f"    score s{index} = mu * (c{index} + sum(k{index}) + sum(v{index}))",
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


def _apply(name: str, argument: torch.Tensor) -> torch.Tensor:
    """A builtin's ``torch`` meaning at ``argument``.

    Parameters
    ----------
    name : str
        The builtin.
    argument : torch.Tensor
        Its first argument.

    Returns
    -------
    torch.Tensor
        The value.
    """
    if name in _BINARY:
        return _BINARY[name](argument, torch.tensor(_EXPONENT, dtype=torch.float64))
    if name in _TERNARY:
        return _TERNARY[name](argument)
    return _UNARY[name](argument)


def _oracle(names: list[str], mu: float) -> float:
    """The joint at ``mu``, computed in ``torch`` without the DSL.

    Parameters
    ----------
    names : list[str]
        The builtins.
    mu : float
        The clamped latent.

    Returns
    -------
    float
        The log joint.
    """
    data = torch.tensor(_DATA, dtype=torch.float64)
    steps = torch.arange(_STEPS, dtype=torch.float64) / 10.0
    total = torch.tensor(0.0, dtype=torch.float64)
    for name in names:
        base = torch.tensor(_argument(name), dtype=torch.float64)
        total = (
            total
            + _apply(name, base)
            + _apply(name, steps + base).sum()
            + _apply(name, data / 10.0 + base).sum()
        )
    latent = torch.tensor(mu, dtype=torch.float64)
    return float(
        D.Normal(0.0, 1.0).log_prob(latent)
        + D.Normal(latent, 1.0).log_prob(data).sum()
        + latent * total
    )


def _reference(source: str, names: list[str]) -> list[float]:
    """The torch runtime's joint at each point, held to the oracle.

    Parameters
    ----------
    source : str
        The program.
    names : list[str]
        The builtins it applies.

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
        assert joint == pytest.approx(_oracle(names, mu), rel=1e-5)
        densities.append(joint)
    return densities


def test_every_target_settles_every_surface_builtin() -> None:
    surface = {
        builtin
        for builtin in PRIMITIVE_BUILTINS.values()
        if builtin not in {"int", "real", "weight", "weight_value"}
    }
    assert set(MATH_BUILTINS) == surface
    assert set(_UNARY) | set(_BINARY) | set(_TERNARY) == surface
    assert set(BUILTIN_SPELLINGS) == set(_TARGETS)
    elaborated = {
        name for name, spec in BUILTIN_REGISTRY.items() if spec.qiec_form == "shape"
    }
    assert ELABORATED_BUILTINS == elaborated
    for target in _TARGETS:
        assert set(BUILTIN_SPELLINGS[target]) == surface | elaborated, target
        for name in elaborated:
            assert isinstance(BUILTIN_SPELLINGS[target][name], Elaborated), name


@pytest.mark.parametrize("target", _TARGETS)
def test_target_refuses_by_name_every_builtin_it_cannot_express(target: str) -> None:
    refused = sorted(set(MATH_BUILTINS) - set(_spelled(target)))
    for name in refused:
        with pytest.raises(
            UnsupportedConstruct, match=f"{target} cannot express {name!r}"
        ):
            transpile(parse(_source([name])), target=target)


@pytest.mark.parametrize("target", sorted(_BACKENDS_WITH_IMAGES))
def test_target_scores_every_builtin_it_spells(
    target: str, scratch: pathlib.Path
) -> None:
    names = _spelled(target)
    source = _source(names)
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
        timeout=600.0,
        shapes={"mu": [], "y": [_STEPS]},
        dtypes={"mu": "float", "y": "float"},
    )
    _equivalence.assert_log_density_match(
        _reference(source, names),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=_STEPS),
        context=f"{target}@builtins",
    )


@pytest.mark.requires_probe("church")
def test_church_scores_every_builtin_it_spells(scratch: pathlib.Path) -> None:
    names = _spelled("church")
    source = _source(names)
    result = ChurchProbe().evaluate(
        transpile(parse(source), target="church"),
        "builtins",
        [_protocol.Point(params={"mu": mu}, data={"y": list(_DATA)}) for mu in _POINTS],
        scratch=scratch / "church",
    )
    _equivalence.assert_log_density_match(
        _reference(source, names),
        result.log_densities,
        atol=_equivalence.adaptive_atol(n_obs=_STEPS),
        context="church@builtins",
    )


def _computation_source(names: list[str]) -> str:
    """A program calling one named computation per builtin.

    Each computation branches, so the planner keeps it a function of
    its own rather than inlining its body: Stan renders it in the
    `functions` block, whose builtins are spelled by the same table.

    Parameters
    ----------
    names : list[str]
        The builtins.

    Returns
    -------
    str
        QVR source scoring ``mu`` times each computation's value at the
        builtin's constant.
    """
    lines = [f"object Step : FinSet {_STEPS}", "object Val : Real 1", ""]
    for index, name in enumerate(names):
        body = (
            f"{name}(x, {_EXPONENT})"
            if name in _BINARY
            else f"{name}(x > {_THRESHOLD}, x, 1.0 - x)"
            if name in _TERNARY
            else f"{name}(x)"
        )
        lines += [
            f"define f{index}(x : Real) : Real !{{}} =",
            "    if x < -100.0 then",
            "        return 0.0",
            "    else",
            f"        return {body}",
            "",
        ]
    lines += [
        "program p : Step -> Val",
        "    sample mu <- Normal(loc=0.0, scale=1.0)",
        "    observe y : Step <- Normal(loc=mu, scale=1.0)",
    ]
    for index, name in enumerate(names):
        lines += [
            f"    let c{index} <- f{index}({_argument(name)})",
            f"    score s{index} = mu * c{index}",
        ]
    return "\n".join([*lines, "    return mu", "", "export p", ""])


def _computation_oracle(names: list[str], mu: float) -> float:
    """The joint of `_computation_source` at ``mu``, computed in ``torch``.

    Parameters
    ----------
    names : list[str]
        The builtins.
    mu : float
        The clamped latent.

    Returns
    -------
    float
        The log joint.
    """
    total = sum(
        float(_apply(name, torch.tensor(_argument(name), dtype=torch.float64)))
        for name in names
    )
    latent = torch.tensor(mu, dtype=torch.float64)
    data = torch.tensor(_DATA, dtype=torch.float64)
    return float(
        D.Normal(0.0, 1.0).log_prob(latent)
        + D.Normal(latent, 1.0).log_prob(data).sum()
        + latent * total
    )


def test_stan_scores_every_builtin_a_named_computation_applies(
    scratch: pathlib.Path,
) -> None:
    names = _spelled("stan")
    source = _computation_source(names)
    rendered = transpile(parse(source), target="stan")
    assert b"functions {" in rendered and b"round(" not in rendered
    model = loads(source).morphism
    reference: list[float] = []
    for mu in _POINTS:
        sites = {"mu": torch.tensor([mu]), "y": torch.tensor(_DATA)}
        joint = float(model.log_joint(torch.zeros(1, 1), sites).sum())
        assert joint == pytest.approx(_computation_oracle(names, mu), rel=1e-5)
        reference.append(joint)
    image, ext, script_name = _BACKENDS_WITH_IMAGES["stan"]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    raw = _docker.run_probe(
        image=image,
        script=pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name,
        source=rendered,
        source_ext=ext,
        points=[{"params": {"mu": mu}, "data": {"y": _DATA}} for mu in _POINTS],
        scratch=scratch / "stan-computations",
        timeout=600.0,
        shapes={"mu": [], "y": [_STEPS]},
        dtypes={"mu": "float", "y": "float"},
    )
    _equivalence.assert_log_density_match(
        reference,
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=_STEPS),
        context="stan@computation-builtins",
    )
