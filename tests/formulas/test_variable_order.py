"""``brms_order`` lists variables in the order of brms's ``variables()``.

Oracles: the ``variables()`` lists of brms 2.20.4 fits recorded by
``oracles/brms_term_variables.R``, ``oracles/brms_smooth_names.R`` and
``oracles/brms_joint_variable_names.R``, in brms's order. brms orders
parameters in ``reorder_pars`` by class, then intercepts first, keeping the
order of its Stan program otherwise. Each test scrambles the classes of a
recorded list while keeping the order inside a class, and the function must
return the recorded list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from quivers.formulas._variable_catalogue import brms_order

ORACLES = Path(__file__).parent / "oracles"
TERMS = json.loads((ORACLES / "brms_term_variables.json").read_text())
SMOOTHS = json.loads((ORACLES / "brms_smooth_names.json").read_text())["models"]
JOINT = json.loads((ORACLES / "brms_joint_variable_names.json").read_text())

#: The family's distributional parameters of each recorded model, as brms's
#: ``valid_dpars`` lists them.
CASES: dict[str, tuple[list[str], tuple[str, ...]]] = {
    "plain": (TERMS["plain"], ("mu", "sigma")),
    "group": (TERMS["group"], ("mu", "sigma")),
    "gr_by": (TERMS["gr_by"], ("mu", "sigma")),
    "mo": (TERMS["mo"], ("mu", "sigma")),
    "me": (TERMS["me"], ("mu", "sigma")),
    "unstr": (TERMS["unstr"], ("mu", "sigma")),
    "car_bym2": (TERMS["car_bym2"], ("mu",)),
    "cs": (TERMS["cs"], ("mu", "disc")),
    "categorical": (TERMS["categorical"], ("muB", "muC")),
    "logistic_normal": (
        TERMS["logistic_normal"],
        ("mup2", "mup3", "sigmap2", "sigmap3"),
    ),
    "mixture": (
        TERMS["mixture"],
        ("mu1", "sigma1", "mu2", "sigma2", "theta1", "theta2"),
    ),
    "distributional": (TERMS["distributional"], ("mu", "sigma")),
    "horseshoe": (TERMS["horseshoe"], ("mu", "sigma")),
    "r2d2": (TERMS["r2d2"], ("mu", "sigma")),
    "joint_rescor": (TERMS["joint_rescor"], ("mu", "sigma")),
    "smooth_by": (SMOOTHS["by"], ("mu", "sigma")),
    "smooth_bases": (SMOOTHS["bases"], ("mu", "sigma")),
    "smooth_joint": (SMOOTHS["joint"], ("mu", "sigma")),
    "joint_names": (JOINT["variables"], ("mu", "sigma")),
}


def _class(name: str) -> str:
    return name.split("[", 1)[0].split("_", 1)[0]


def _scrambled(names: list[str]) -> list[str]:
    """Reverse the class blocks and put intercepts last inside each class."""
    classes = list(dict.fromkeys(_class(name) for name in names))
    scrambled: list[str] = []
    for label in reversed(classes):
        members = [name for name in names if _class(name) == label]
        intercepts = [
            name for name in members if name.split("[", 1)[0].endswith("_Intercept")
        ]
        scrambled += [name for name in members if name not in intercepts] + intercepts
    return scrambled


@pytest.mark.parametrize("key", sorted(CASES))
def test_scrambled_classes_return_to_brms_order(key: str) -> None:
    names, dpars = CASES[key]
    scrambled = _scrambled(names)
    values = {name: torch.zeros(1, 1) for name in scrambled}
    assert list(brms_order(values, dpars)) == names
    if len({_class(name) for name in names}) > 1:
        assert scrambled != names


def test_values_follow_their_names() -> None:
    values = {"sigma": torch.ones(1, 2), "b_x": torch.zeros(1, 2)}
    ordered = brms_order(values, ("mu", "sigma"))
    assert list(ordered) == ["b_x", "sigma"]
    assert ordered["sigma"] is values["sigma"] and ordered["b_x"] is values["b_x"]


def test_intercepts_lead_their_class_only() -> None:
    names = ["sd_g__x", "b_x", "sd_g__Intercept", "b_sigma_Intercept", "b_Intercept_2"]
    values = {name: torch.zeros(1, 1) for name in names}
    assert list(brms_order(values, ("mu", "sigma"))) == [
        "b_sigma_Intercept",
        "b_Intercept_2",
        "b_x",
        "sd_g__Intercept",
        "sd_g__x",
    ]


def test_family_parameters_sit_between_lscale_and_hs() -> None:
    names = [
        "hs_global",
        "nu",
        "lscale_gpx",
        "sigma",
        "r_g[a,Intercept]",
        "rescor__a__b",
    ]
    values = {name: torch.zeros(1, 1) for name in names}
    assert list(brms_order(values, ("mu", "sigma", "nu"))) == [
        "lscale_gpx",
        "sigma",
        "nu",
        "hs_global",
        "rescor__a__b",
        "r_g[a,Intercept]",
    ]
    # A parameter that is not one of the family's is not placed among them.
    assert list(brms_order(values, ("mu", "sigma")))[-1] == "nu"


def test_names_without_a_brms_class_come_last_in_their_given_order() -> None:
    names = ["mixture_intercept_spacing[1]", "lp__", "cutpoint_offset[1]", "b_x"]
    values = {name: torch.zeros(1, 1) for name in names}
    assert list(brms_order(values, ("mu",))) == [
        "b_x",
        "lp__",
        "mixture_intercept_spacing[1]",
        "cutpoint_offset[1]",
    ]


def test_a_wrong_class_order_does_not_reproduce_brms() -> None:
    # Family parameters before the group standard deviations is not brms's order.
    names, _ = CASES["group"]
    wrong = sorted(
        names,
        key=lambda name: ["b", "sigma", "sd", "cor"].index(_class(name)),
    )
    assert wrong != names
    values = {name: torch.zeros(1, 1) for name in wrong}
    assert list(brms_order(values, ("mu", "sigma"))) == names
