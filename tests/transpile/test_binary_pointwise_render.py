"""The pointwise two-argument builtins render as each Python target's elementwise call.

QVR's ``max(a, b)``, ``min(a, b)``, and ``pow(a, b)`` are pointwise. The
one-argument ``max`` and ``min`` reduce an axis, and each Python target's
reduction reads a second positional argument as that axis, so the binary
form must take the target's elementwise function. An ``Int`` operand, such
as a ``factor`` binder in index arithmetic, must stay an integer, or the
subscript it builds is a float that the target rejects, and an axis
reduction over integers would silently read the wrong entry
(``max(-1, 0)`` read as a reduction is ``-1``, the last entry).

Each rendered binding is evaluated on concrete arrays and compared with
the reference value: torch runs Pyro's lines directly, and NumPy stands in
for ``jax.numpy``, ``pymc.math``, and ``tf.math``, whose elementwise and
stacking calls share NumPy's signatures. The Docker probe tier runs the
real libraries on the same constructs in ``test_construct_let_numeric``.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.transpile import transpile

TEMPLATE = """\
object Step : FinSet 4
object Val : Real 1

program p : Step -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    let ys = [1.0, 2.0, 3.0, 4.0]
    let zs = [2.5, 0.5, 3.5, 1.0]
    let out = {expression}
    score s = mu
    return mu

export p
"""

_YS = (1.0, 2.0, 3.0, 4.0)
_ZS = (2.5, 0.5, 3.5, 1.0)

#: ``(QVR expression, value of mu, reference value, integer-valued)``.
_CASES: dict[str, tuple[str, float, tuple[float, ...] | float, bool]] = {
    "min-real": ("min(mu, 1.0)", 0.5, 0.5, False),
    "max-real": ("max(mu, 0.0)", -0.5, 0.0, False),
    "max-vector": (
        "max(ys, zs)",
        0.0,
        tuple(max(a, b) for a, b in zip(_YS, _ZS, strict=True)),
        False,
    ),
    "min-vector": (
        "min(ys, zs)",
        0.0,
        tuple(min(a, b) for a, b in zip(_YS, _ZS, strict=True)),
        False,
    ),
    "max-index": (
        "factor t : Step in ys[max(t - 1, 0)]",
        0.0,
        tuple(_YS[max(t - 1, 0)] for t in range(4)),
        False,
    ),
    "min-index": (
        "factor t : Step in ys[min(t + 1, 3)]",
        0.0,
        tuple(_YS[min(t + 1, 3)] for t in range(4)),
        False,
    ),
    "max-int": (
        "factor t : Step in max(t - 2, 0)",
        0.0,
        tuple(float(max(t - 2, 0)) for t in range(4)),
        True,
    ),
    "min-int": (
        "factor t : Step in min(t, 2)",
        0.0,
        tuple(float(min(t, 2)) for t in range(4)),
        True,
    ),
    "pow-real": ("pow(mu, 2.0)", 1.5, 2.25, False),
}

#: The elementwise symbol each target spells ``max`` / ``min`` / ``pow`` as.
_ELEMENTWISE: dict[str, dict[str, str]] = {
    "numpyro": {"max": "jnp.maximum(", "min": "jnp.minimum(", "pow": "jnp.power("},
    "pyro": {"max": "torch.maximum(", "min": "torch.minimum(", "pow": "torch.pow("},
    "pymc": {
        "max": "pymc.math.maximum(",
        "min": "pymc.math.minimum(",
        "pow": "pymc.math.pow(",
    },
    "edward2": {
        "max": "tf.math.maximum(",
        "min": "tf.math.minimum(",
        "pow": "tf.math.pow(",
    },
}

#: Each target's reductions, none of which a binary call may render as.
_REDUCTIONS: dict[str, tuple[str, ...]] = {
    "numpyro": ("jnp.max(", "jnp.min("),
    "pyro": ("torch.amax(", "torch.amin(", "torch.max(", "torch.min("),
    "pymc": ("pymc.math.max(", "pymc.math.min("),
    "edward2": ("tf.reduce_max(", "tf.reduce_min("),
}


def _numpy_namespace(target: str) -> dict[str, object]:
    """The NumPy stand-in for a target's array library.

    Parameters
    ----------
    target : str
        ``numpyro``, ``pymc``, or ``edward2``.

    Returns
    -------
    dict[str, object]
        The names the rendered lines reference.
    """
    math = types.SimpleNamespace(
        maximum=np.maximum,
        minimum=np.minimum,
        pow=np.power,
        stack=np.stack,
        sum=np.sum,
    )
    if target == "numpyro":
        return {
            "jnp": types.SimpleNamespace(
                **vars(math), power=np.power, asarray=np.asarray
            )
        }
    if target == "pymc":
        tensors = types.SimpleNamespace(
            pt=types.SimpleNamespace(as_tensor_variable=np.asarray)
        )
        return {"pymc": types.SimpleNamespace(math=math, pytensorf=tensors)}
    return {"tf": types.SimpleNamespace(math=math, stack=np.stack)}


def _evaluate(target: str, expression: str, mu: float) -> object:
    """Render a case and evaluate its bindings on concrete values.

    Parameters
    ----------
    target : str
        The Python target.
    expression : str
        The QVR expression bound to ``out``.
    mu : float
        The site's value.

    Returns
    -------
    object
        The value the rendered ``out`` binding computes.
    """
    source = transpile(parse(TEMPLATE.format(expression=expression)), target=target)
    lines = [line.strip() for line in source.decode().splitlines()]
    bindings = [
        line
        for line in lines
        if line.split("=", 1)[0].strip() in {"ys", "zs", "out"} and "=" in line
    ]
    assert [line.split("=", 1)[0].strip() for line in bindings] == ["ys", "zs", "out"]
    scope: dict[str, object]
    if target == "pyro":
        scope = {"torch": torch, "mu": torch.tensor(mu)}
    else:
        scope = {**_numpy_namespace(target), "mu": np.float32(mu)}
    for line in bindings:
        name, value = line.split("=", 1)
        scope[name.strip()] = eval(value, scope)
    if target != "pyro":
        for name in ("ys", "zs"):
            scope[name] = np.asarray(scope[name])
    return eval(bindings[-1].split("=", 1)[1], scope)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_reference_runtime_computes_each_case(case: str) -> None:
    expression, _mu, _expected, _integral = _CASES[case]
    assert loads(TEMPLATE.format(expression=expression)) is not None


@pytest.mark.parametrize("target", sorted(_ELEMENTWISE))
@pytest.mark.parametrize("case", sorted(_CASES))
def test_binary_builtins_render_pointwise_and_keep_integers(
    target: str, case: str
) -> None:
    expression, mu, expected, integral = _CASES[case]
    source = transpile(
        parse(TEMPLATE.format(expression=expression)), target=target
    ).decode()
    out = next(line for line in source.splitlines() if line.strip().startswith("out ="))
    builtin = case.split("-")[0]
    assert _ELEMENTWISE[target][builtin] in out
    assert not any(reduction in out for reduction in _REDUCTIONS[target])
    value = _evaluate(target, expression, mu)
    array = (
        value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    )
    assert array.tolist() == pytest.approx(expected)
    if integral:
        assert np.issubdtype(array.dtype, np.integer), array.dtype


#: ``(target, QVR expression, text the rendered program must contain)``
#: for the targets outside the Python family, whose numeric agreement
#: the probe tier checks.
_SPELLINGS: tuple[tuple[str, str, str], ...] = (
    ("stan", "min(mu, 1.0)", "fmin(mu,1.0)"),
    ("stan", "max(ys, zs)", "fmax(ys[m_Step],zs[m_Step])"),
    ("stan", "factor t : Step in ys[max(t - 1, 0)]", "ys[max(0-1,0)+1]"),
    ("turing", "max(ys, zs)", "max.(ys, zs)"),
    ("turing", "min(mu, 1.0)", "min.(mu, 1"),
    ("jags", "factor t : Step in ys[max(t - 1, 0)]", "ys[1]"),
    ("jags", "factor t : Step in ys[min(t + 1, 3)]", "ys[4]"),
    ("webppl", "max(ys, zs)", "var max = function(a, b)"),
)


@pytest.mark.parametrize(("target", "expression", "spelling"), _SPELLINGS)
def test_other_targets_spell_the_pointwise_form(
    target: str, expression: str, spelling: str
) -> None:
    source = transpile(
        parse(TEMPLATE.format(expression=expression)), target=target
    ).decode()
    assert spelling in source.replace(" ", "") or spelling in source
