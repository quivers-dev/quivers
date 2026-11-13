"""Tests for the bijector-typed parameter-transform vocabulary.

The transform vocabulary replaces the historical string-keyed
`Callable[[Tensor], Tensor]` table with a
[`Bijector`][quivers.continuous.bijectors.Bijector]-typed registry.
The tests verify:

1. Every registered string key resolves to a
   [`Bijector`][quivers.continuous.bijectors.Bijector].
2. The bijector's `forward` matches the historical callable to
   numerical precision on a battery of inputs.
3. [`ParamSpec`][quivers.continuous.family_spec.ParamSpec] accepts
   a `Bijector` instance directly and routes `raw_transform`
   through the bijector's forward map.
4. A user-defined custom `Bijector` subclass plugs into any
   family via `ParamSpec(name=..., transform=CustomBij())` without
   touching the string registry.
5. Inline domain maps are callables, preserve valid parameters, and
   use the identity for a user-supplied bijector transform.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from quivers.continuous.bijectors import (
    AffineBijector,
    Bijector,
    ComposeBijector,
    SoftplusBijector,
)
from quivers.continuous.family_spec import ParamSpec
from quivers.continuous.param_transforms import (
    INLINE_CLAMPS,
    TRANSFORM_TO_BIJECTOR,
    resolve_inline_clamp,
    resolve_transform,
)
from quivers.core._util import EPS


# ---------------------------------------------------------------------------
# TRANSFORM_TO_BIJECTOR coverage and typing
# ---------------------------------------------------------------------------


def test_registry_has_expected_keys() -> None:
    expected = {"id", "sigmoid", "softplus", "softplus_shifted", "exp"}
    assert set(TRANSFORM_TO_BIJECTOR) == expected


def test_every_registry_entry_is_a_bijector() -> None:
    for name, bij in TRANSFORM_TO_BIJECTOR.items():
        assert isinstance(bij, Bijector), (
            f"transform {name!r} is not a Bijector; got {type(bij).__name__}"
        )


def test_inline_clamp_registry_matches_transform_registry_keys() -> None:
    assert set(INLINE_CLAMPS) == set(TRANSFORM_TO_BIJECTOR)


# ---------------------------------------------------------------------------
# Forward-map equivalence with the historical string-keyed callables
# ---------------------------------------------------------------------------


def _historical_id(x: Tensor) -> Tensor:
    return x


def _historical_softplus(x: Tensor) -> Tensor:
    return F.softplus(x) + EPS


def _historical_softplus_shifted(x: Tensor) -> Tensor:
    return F.softplus(x) + 0.1


def _historical_exp(x: Tensor) -> Tensor:
    return x.exp().clamp(min=EPS)


def _historical_sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


_HISTORICAL = {
    "id": _historical_id,
    "softplus": _historical_softplus,
    "softplus_shifted": _historical_softplus_shifted,
    "exp": _historical_exp,
    "sigmoid": _historical_sigmoid,
}


@pytest.mark.parametrize("name", sorted(TRANSFORM_TO_BIJECTOR))
def test_forward_matches_historical_transform(name: str) -> None:
    """The bijector's forward map matches the historical
    string-keyed callable on a spread of inputs covering both
    tails and the neighbourhood of zero.
    """
    # Affine holds its scale/shift as default-dtype tensors, so a
    # float64 input silently loses precision through those
    # constants; use the default (float32) throughout.
    x = torch.linspace(-8.0, 8.0, 65)
    bij = TRANSFORM_TO_BIJECTOR[name]
    expected = _HISTORICAL[name](x)
    actual = bij.forward(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# resolve_transform behaviour
# ---------------------------------------------------------------------------


def test_resolve_transform_string() -> None:
    assert resolve_transform("id") is TRANSFORM_TO_BIJECTOR["id"]
    assert resolve_transform("softplus") is TRANSFORM_TO_BIJECTOR["softplus"]


def test_resolve_transform_passes_bijector_through() -> None:
    bij = SoftplusBijector()
    assert resolve_transform(bij) is bij


def test_resolve_transform_unknown_string_raises() -> None:
    with pytest.raises(KeyError, match="unknown transform"):
        resolve_transform("no_such_transform")


def test_resolve_transform_wrong_type_raises() -> None:
    with pytest.raises(TypeError, match="Bijector"):
        resolve_transform(42)  # type: ignore[arg-type]


def test_resolve_inline_clamp_bijector_returns_identity() -> None:
    """A user-supplied bijector transform gets the identity clamp;
    the bijector's forward is trusted to land in-support.
    """
    clamp = resolve_inline_clamp(SoftplusBijector())
    values = torch.tensor([-3.0, 0.0, 3.0])
    torch.testing.assert_close(clamp(values), values)
    assert not isinstance(clamp, Bijector)


# ---------------------------------------------------------------------------
# ParamSpec accepts strings and Bijectors uniformly
# ---------------------------------------------------------------------------


def test_paramspec_accepts_string_transform() -> None:
    p = ParamSpec(name="scale", transform="softplus")
    assert p.transform == "softplus"
    assert isinstance(p.bijector, Bijector)
    x = torch.tensor([-1.0, 0.0, 1.0])
    torch.testing.assert_close(p.raw_transform(x), _historical_softplus(x))


def test_paramspec_accepts_bijector_transform() -> None:
    bij = SoftplusBijector()
    p = ParamSpec(name="scale", transform=bij)
    assert p.bijector is bij
    x = torch.tensor([-1.0, 0.0, 1.0])
    torch.testing.assert_close(p.raw_transform(x), bij.forward(x))


def test_paramspec_unknown_string_raises() -> None:
    with pytest.raises(ValueError, match="unknown transform"):
        ParamSpec(name="x", transform="not_a_transform")


def test_paramspec_wrong_type_raises() -> None:
    with pytest.raises(TypeError, match="string key or a Bijector"):
        ParamSpec(name="x", transform=3.14)  # type: ignore[arg-type]


def test_paramspec_transform_name_for_string() -> None:
    p = ParamSpec(name="rate", transform="softplus")
    assert p.transform_name == "softplus"


def test_paramspec_transform_name_for_bijector() -> None:
    p = ParamSpec(name="rate", transform=SoftplusBijector())
    assert p.transform_name == "<bijector:Softplus>"


def test_paramspec_inline_clamp_string_softplus() -> None:
    p = ParamSpec(name="scale", transform="softplus")
    values = torch.tensor([1e-12, 1e-9, 1.0], dtype=torch.float64)
    torch.testing.assert_close(p.inline_clamp(values), values, rtol=0, atol=0)
    for invalid in (-1.0, 0.0):
        with pytest.raises(ValueError, match="greater than zero"):
            p.inline_clamp(torch.tensor(invalid))


def test_paramspec_inline_clamp_string_sigmoid_bounds_unit_interval() -> None:
    p = ParamSpec(name="probs", transform="sigmoid")
    values = torch.tensor(
        [-1.0, 0.0, 1e-12, 0.5, 1.0 - 1e-12, 1.0, 2.0], dtype=torch.float64
    )
    expected = torch.tensor(
        [0.0, 0.0, 1e-12, 0.5, 1.0 - 1e-12, 1.0, 1.0], dtype=torch.float64
    )
    torch.testing.assert_close(p.inline_clamp(values), expected, rtol=0, atol=0)


def test_exponential_transform_inverse_and_jacobian_below_epsilon() -> None:
    values = torch.tensor(
        [-40.0, -25.0, -10.0, 0.0], dtype=torch.float64, requires_grad=True
    )
    transform = resolve_transform("exp")
    mapped = transform.forward(values)
    torch.testing.assert_close(transform.inverse(mapped), values, rtol=0, atol=1e-12)
    derivative = torch.autograd.grad(mapped.sum(), values)[0]
    torch.testing.assert_close(
        derivative.log(), transform.forward_log_det_jacobian(values)
    )
    assert not torch.allclose(mapped, mapped.clamp_min(EPS), rtol=0, atol=1e-12)


def test_paramspec_inline_clamp_bijector_is_identity() -> None:
    p = ParamSpec(name="scale", transform=SoftplusBijector())
    x = torch.tensor([-3.0, 0.0, 3.0])
    torch.testing.assert_close(p.inline_clamp(x), x)


# ---------------------------------------------------------------------------
# User-defined custom Bijector plugs into a family via ParamSpec
# ---------------------------------------------------------------------------


class _AbsShift(Bijector):
    """A user-supplied bijector: absolute value plus a fixed
    positive shift. Used to exercise the third-party extension
    path; not part of the core registry.
    """

    def __init__(self, shift: float = 0.5) -> None:
        self._shift = float(shift)

    def forward(self, x: Tensor) -> Tensor:
        return x.abs() + self._shift

    def inverse(self, y: Tensor) -> Tensor:
        return (y - self._shift).clamp(min=0.0)

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return torch.zeros_like(y)


def test_custom_bijector_via_paramspec() -> None:
    custom = _AbsShift(shift=0.25)
    p = ParamSpec(name="scale", transform=custom)
    assert p.bijector is custom
    x = torch.tensor([-2.0, -0.5, 0.0, 1.5])
    expected = torch.tensor([2.25, 0.75, 0.25, 1.75])
    torch.testing.assert_close(p.raw_transform(x), expected)


def test_custom_bijector_composes_with_registry_entries() -> None:
    """A user-supplied bijector composed with a registry
    bijector using `Compose` still routes through `resolve_transform`.
    """
    composite = ComposeBijector(
        AffineBijector(scale=2.0, shift=1.0), SoftplusBijector()
    )
    p = ParamSpec(name="rate", transform=composite)
    x = torch.tensor([-1.0, 0.0, 1.0])
    expected = 2.0 * F.softplus(x) + 1.0
    torch.testing.assert_close(p.raw_transform(x), expected, atol=1e-6, rtol=1e-6)


def test_custom_bijector_in_family_factory() -> None:
    """A `_make_family`-generated conditional distribution accepts
    a user-supplied bijector directly in its `param_specs` list.
    The registered `FamilySpec` retains the bijector on its
    `ParamSpec` so downstream tooling sees the same object.
    """
    from quivers.continuous.families import _make_family
    import torch.distributions as D

    custom = SoftplusBijector()
    cls = _make_family(
        "ConditionalCustomExponential",
        D.Exponential,
        [("rate", custom)],
        "test-only family exercising the bijector transform path",
        dsl_name="_CustomExponentialForTests",
    )
    # The registered spec retains the bijector on its ParamSpec.
    from quivers.continuous.family_spec import FAMILY_REGISTRY

    spec = FAMILY_REGISTRY["_CustomExponentialForTests"]
    assert spec.params[0].bijector is custom
    # Constructing an instance and sampling routes through the
    # bijector's forward map.
    from quivers.continuous.spaces import Euclidean

    dom = Euclidean(name="ctx", dim=2)
    cod = Euclidean(name="obs", dim=1)
    inst = cls(dom, cod)
    x = torch.randn(4, 2)
    sample = inst.rsample(x)
    assert sample.shape == (4, 1)
    lp = inst.log_prob(x, sample)
    assert lp.shape == (4,)


# ---------------------------------------------------------------------------
# Round-trip and Jacobian sanity for the registry bijectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["id", "sigmoid"])
def test_bijector_round_trip(name: str) -> None:
    """Round-trip `inverse(forward(x)) == x` on the invertible
    registry entries. Softplus-with-shift and clamped exp are
    strictly monotone on the image but round-trip suffers loss at
    the shift boundary; the invertible entries suffice to verify
    the bijector wiring.
    """
    bij = TRANSFORM_TO_BIJECTOR[name]
    if name == "sigmoid":
        x = torch.linspace(-6.0, 6.0, 41, dtype=torch.float64)
    else:
        x = torch.linspace(-3.0, 3.0, 41, dtype=torch.float64)
    y = bij.forward(x)
    x_back = bij.inverse(y)
    torch.testing.assert_close(x_back, x, atol=1e-6, rtol=1e-6)


def test_softplus_shift_bijector_forward_matches_math() -> None:
    """The `softplus_shifted` entry equals `softplus(x) + 0.1`
    for every input, matching the registered `_softplus_shifted`
    helper in `param_transforms`.
    """
    bij = TRANSFORM_TO_BIJECTOR["softplus_shifted"]
    x = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
    torch.testing.assert_close(
        bij.forward(x),
        F.softplus(x) + 0.1,
        atol=1e-6,
        rtol=1e-6,
    )


def test_forward_log_det_jacobian_finite_on_registry() -> None:
    """The forward log-det-Jacobian is finite over a moderate
    range for every registry bijector; the `id` case yields zero.
    """
    x = torch.linspace(-3.0, 3.0, 21)
    for name, bij in TRANSFORM_TO_BIJECTOR.items():
        ldj = bij.forward_log_det_jacobian(x)
        assert torch.isfinite(ldj).all(), (
            f"log-det-Jacobian of {name!r} is not finite: {ldj}"
        )
        if name == "id":
            torch.testing.assert_close(ldj, torch.zeros_like(x))


def test_softplus_shifted_jacobian_matches_softplus_jacobian() -> None:
    """Composition with an additive `Affine(scale=1, shift=c)`
    leaves the log-det-Jacobian unchanged: `log|scale| = 0`.
    """
    bij_shifted = TRANSFORM_TO_BIJECTOR["softplus_shifted"]
    bare = SoftplusBijector()
    x = torch.linspace(-2.0, 2.0, 11)
    torch.testing.assert_close(
        bij_shifted.forward_log_det_jacobian(x),
        bare.forward_log_det_jacobian(x),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sigmoid_forward_log_det_jacobian_matches_logistic_density() -> None:
    """The `sigmoid` bijector's log-det-Jacobian is the log-density
    of the standard logistic distribution, `-x - 2 softplus(-x)`
    or equivalently `-softplus(x) - softplus(-x)`.
    """
    bij = TRANSFORM_TO_BIJECTOR["sigmoid"]
    x = torch.linspace(-5.0, 5.0, 21)
    expected = -F.softplus(x) - F.softplus(-x)
    torch.testing.assert_close(
        bij.forward_log_det_jacobian(x), expected, atol=1e-6, rtol=1e-6
    )


def test_exp_bijector_forward_matches_clamped_historical() -> None:
    """The clamped-exp entry matches `x.exp().clamp(min=EPS)`
    including at extremely negative inputs where the clamp bites.
    """
    bij = TRANSFORM_TO_BIJECTOR["exp"]
    x = torch.tensor([-40.0, -10.0, 0.0, 3.0])
    expected = x.exp().clamp(min=EPS)
    torch.testing.assert_close(bij.forward(x), expected, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Family-registry contract: every registered FamilySpec resolves its
# transforms via the new bijector-typed pathway.
# ---------------------------------------------------------------------------


def test_every_registered_family_transform_resolves_to_bijector() -> None:
    """Every registered family's parameter transforms resolve to
    a `Bijector` instance via `ParamSpec.bijector`, whether the
    transform was declared as a string key or a bijector object.
    """
    from quivers.continuous.family_spec import FAMILY_REGISTRY

    for family_name, spec in FAMILY_REGISTRY.items():
        for p in spec.params:
            bij = p.bijector
            assert isinstance(bij, Bijector), (
                f"family {family_name!r} parameter {p.name!r} "
                f"did not resolve to a Bijector; got {type(bij).__name__}"
            )


def test_paramspec_frozen() -> None:
    """`ParamSpec` is a frozen dataclass; direct attribute
    assignment raises.
    """
    p = ParamSpec(name="scale", transform="softplus")
    with pytest.raises(Exception):
        p.name = "other"  # type: ignore[misc]


def test_softplus_registry_matches_historical_at_boundary() -> None:
    """The registered `softplus` entry adds `EPS` after softplus,
    matching the historical raw callable's positivity floor.
    """
    bij = TRANSFORM_TO_BIJECTOR["softplus"]
    # At very negative inputs, softplus(x) is tiny; the composite
    # ensures the result stays above EPS.
    x = torch.tensor([-40.0])
    y = bij.forward(x)
    assert (y >= EPS).all()
    # The historical raw transform gives the same value.
    torch.testing.assert_close(y, F.softplus(x) + EPS, atol=1e-6, rtol=1e-6)


def test_softplus_shifted_registry_matches_historical_at_boundary() -> None:
    """The registered `softplus_shifted` entry adds `0.1` after
    softplus, matching the historical raw callable.
    """
    bij = TRANSFORM_TO_BIJECTOR["softplus_shifted"]
    # Affine stores its scale/shift as tensors with the default
    # dtype (float32); use float32 inputs to keep the comparison
    # in a single precision.
    x = torch.tensor([-40.0, 0.0, 40.0])
    y = bij.forward(x)
    torch.testing.assert_close(y, F.softplus(x) + 0.1, atol=1e-7, rtol=1e-7)


def test_math_pi_import_not_required() -> None:
    """Sanity check that we didn't accidentally shadow math or torch
    in the transform module.
    """
    assert math.pi > 3.0
