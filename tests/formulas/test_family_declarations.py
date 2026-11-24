"""Families opt in to formula features by declaration.

Each declared property is checked twice. A family registered nowhere, with
its own name and parameter names, gets the behaviour once it declares the
property: the emitted program is that of the built-in family up to the
renaming. Dropping the declaration is refused with an error naming the
missing property, or selects the documented alternative.
"""

from __future__ import annotations

import ast
from pathlib import Path

import didactic.api as dx
import numpy as np
import pandas as pd
import pytest
import torch
from torch.distributions import Distribution, Normal, constraints

import quivers.formulas
from quivers.family_catalog import FamilyCatalog
from quivers.formulas import (
    AuxParam,
    Domain,
    Elliptical,
    Family,
    Link,
    ResponseModel,
    MultivariateFormula,
    custom_family,
    families,
    fit,
    formula_to_qvr,
    links,
    mixture,
    set_rescor,
)
from quivers.formulas._brms_priors import scale_prior_arguments, sdgp_prior
from quivers.formulas._fit import _initial_values, _prepare_formula
from quivers.formulas._joint_missing import missing_specification
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import REAL

N = 12


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "y": rng.normal(size=N) + 3.0,
            "z": rng.normal(size=N),
            "x": rng.normal(size=N),
            "e": rng.uniform(0.1, 0.5, size=N),
            "st": [0, 1, 0] * 4,
            "t": list(range(N)),
            "two": pd.Categorical(["no", "yes"] * 6),
        }
    )


def _bell(**changes) -> Family:
    """Declare a Normal family with its own name and parameter name."""
    bell = Family(
        name="bell",
        location_link=links["identity"],
        location_links=("identity", "log"),
        observe_family="Normal",
        observe_args=("mu", "spread"),
        recorded_events=True,
        error_scale="spread",
        elliptical=Elliptical(scale="spread"),
        r_squared="model",
        mixture_law="MixtureNormal",
        aux_params=(
            AuxParam(
                name="spread",
                prior="HalfCauchy(2.0)",
                link=links["log"],
                brms_class="sigma",
            ),
        ),
    )
    return bell.with_(**changes) if changes else bell


def _heavy(**changes) -> Family:
    """Declare a Student-t family with its own parameter names."""
    heavy = families["student_t"].with_(
        name="heavy",
        observe_args=("tail", "mu", "spread"),
        error_scale="spread",
        elliptical=Elliptical(scale="spread", df="tail"),
        aux_params=tuple(
            aux.with_(name=name)
            for aux, name in zip(
                families["student_t"].aux_params, ("tail", "spread"), strict=True
            )
        ),
    )
    return heavy.with_(**changes) if changes else heavy


def _renamed(source: str) -> str:
    """Rename the parameters of `_bell` and `_heavy` to the built-in names."""
    return source.replace("spread", "sigma").replace("tail", "nu")


@pytest.mark.parametrize("known", ["se(e)", "se(e, sigma = TRUE)"])
@pytest.mark.parametrize(
    ("make", "builtin"), [(_bell, "gaussian"), (_heavy, "student_t")]
)
def test_a_declared_error_scale_admits_se(make, builtin, known):
    formula = f"y | {known} ~ x"
    emitted = formula_to_qvr(formula, data=_frame(), family=make())
    assert _renamed(emitted) == formula_to_qvr(formula, data=_frame(), family=builtin)
    with pytest.raises(ValueError, match=r"'(bell|heavy)'.*Family\.error_scale"):
        formula_to_qvr(formula, data=_frame(), family=make(error_scale=None))


@pytest.mark.parametrize("term", ["cens(st)", "trunc(lb = 0)"])
def test_declared_recorded_events_admit_cens_and_trunc(term):
    formula = f"y | {term} ~ x"
    emitted = formula_to_qvr(formula, data=_frame(), family=_bell())
    assert _renamed(emitted) == formula_to_qvr(
        formula, data=_frame(), family="gaussian"
    )
    with pytest.raises(ValueError, match=r"'bell'.*Family\.recorded_events"):
        formula_to_qvr(formula, data=_frame(), family=_bell(recorded_events=False))


def test_a_mixture_admits_recorded_events_when_every_component_does():
    assert mixture(_bell(), "student_t").recorded_events
    assert not mixture(_bell(recorded_events=False), "student_t").recorded_events


def test_a_declared_mixture_law_is_emitted_on_the_stacked_arguments():
    emitted = formula_to_qvr("y ~ x", data=_frame(), family=mixture(_bell(), nmix=2))
    builtin = formula_to_qvr("y ~ x", data=_frame(), family=mixture("gaussian", nmix=2))
    assert _renamed(emitted) == builtin
    assert "MixtureNormal(" in builtin
    generic = formula_to_qvr(
        "y ~ x", data=_frame(), family=mixture(_bell(mixture_law=None), nmix=2)
    )
    assert "MixtureNormal(" not in generic and "Mixture(" in generic


def test_the_registered_mixture_is_a_template_of_its_component():
    emitted = formula_to_qvr(
        "y ~ x", data=_frame(), family="mixture", mixture_components=3
    )
    assert emitted == formula_to_qvr(
        "y ~ x", data=_frame(), family=mixture("gaussian", nmix=3)
    )
    with pytest.raises(ValueError, match="cannot itself be a mixture"):
        mixture("mixture", "gaussian")


@pytest.mark.parametrize(
    ("make", "builtin"), [(_bell, "gaussian"), (_heavy, "student_t")]
)
def test_a_declared_elliptical_law_correlates_rows(make, builtin):
    formula = "y ~ x + ar(t, p = 1, cov = TRUE)"

    def scored(family):
        fitted = fit(
            formula, data=_frame(), family=family, method="svi", num_steps=0, seed=1
        )
        return _renamed(fitted.qvr_source)

    assert scored(make()) == scored(builtin)
    latent = scored(make(elliptical=None))
    assert "Residual" not in latent.split("observe")[-1]
    assert latent != scored(builtin)


def test_autocorrelation_binding_names_the_missing_elliptical_law():
    from quivers.formulas._autocor import autocor_family, split_autocor_terms
    from quivers.formulas import formula_from_data

    text, terms = split_autocor_terms("y ~ x + ar(t, p = 1, cov = TRUE)")
    parsed = formula_from_data(text, _frame())
    with pytest.raises(ValueError, match=r"Family\.elliptical.*'bell' declares none"):
        autocor_family(
            terms,
            _bell(elliptical=None),
            data=_frame(),
            response=np.asarray(parsed.response_values),
            response_name="y",
            data2=None,
            distributional=None,
            modifiers=parsed.modifiers,
            caller="fit",
        )


def _joint(family: Family | str) -> MultivariateFormula:
    return (
        ResponseModel(formula="y ~ x", family=family)
        + ResponseModel(formula="z ~ x", family=family)
        + set_rescor(True)
    )


@pytest.mark.parametrize(
    ("make", "builtin"), [(_bell, "gaussian"), (_heavy, "student_t")]
)
def test_a_declared_elliptical_law_correlates_responses(make, builtin):
    def emitted(family):
        fitted = fit(_joint(family), data=_frame(), method="svi", num_steps=0, seed=1)
        return _renamed(fitted.qvr_source), fitted

    source, fitted = emitted(make())
    assert source == emitted(builtin)[0]
    named = {_renamed(name) for name in fitted.variables(num_samples=1)}
    other = emitted(builtin)[1].variables(num_samples=1)
    assert named == set(other)
    assert "rescor__y__z" in named
    with pytest.raises(dx.ValidationError, match=r"Family\.elliptical"):
        _joint(make(elliptical=None))


def test_residual_correlation_needs_one_common_elliptical_law():
    with pytest.raises(
        dx.ValidationError, match=r"Family\.elliptical.*'bell', 'student_t'"
    ):
        (
            ResponseModel(formula="y ~ x", family=_bell())
            + ResponseModel(formula="z ~ x", family="student_t")
            + set_rescor(True)
        )


def test_declared_r_squared_selects_the_model_variance():
    def ratio(family, method="auto"):
        fitted = fit(
            "y ~ x", data=_frame(), family=family, method="svi", num_steps=0, seed=1
        )
        return fitted.bayes_r2(method=method, num_samples=5, seed=2)

    assert torch.equal(ratio(_bell()), ratio(_bell(), "model"))
    plain = _bell(r_squared="residual")
    assert torch.equal(ratio(plain), ratio(plain, "residual"))
    assert not torch.equal(ratio(plain), ratio(_bell()))


def test_links_are_admitted_by_declaration():
    bell = _bell()
    assert bell.with_links(mu="log", spread="softplus").location_link == links["log"]
    with pytest.raises(ValueError, match=r"admissible links are \['identity', 'log'\]"):
        bell.with_links(mu="probit")
    with pytest.raises(ValueError, match=r"Family\.location_links"):
        _bell(location_links=None).with_links(mu="log")
    shifted = AuxParam(
        name="spread",
        prior="Normal(0.0, 1.0)",
        link=links["identity"],
        domain=Domain(lower=None),
    )
    with pytest.raises(ValueError, match=r"AuxParam\.links"):
        _bell(aux_params=(shifted,)).with_links(spread="identity")
    declared = shifted.with_(links=("identity", "log1p"))
    assert (
        _bell(aux_params=(declared,)).with_links(spread="log1p").aux_params[0].link
        == links["log1p"]
    )


def test_links_of_a_parameter_default_to_those_of_its_domain():
    def admitted(domain):
        return AuxParam(
            name="p", prior="HalfCauchy(1.0)", link=links["log"], domain=domain
        ).admissible_links

    assert admitted(Domain()) == ("log", "identity", "softplus", "squareplus")
    assert admitted(Domain(upper=1.0)) == ("logit", "identity")
    assert admitted(Domain(lower=1.0)) is None
    assert admitted(Domain(lower=None)) is None


def test_with_links_keeps_every_declaration():
    family = custom_family(
        "Shifted",
        catalog=_shifted_catalog(),
        location="center",
        auxiliary={"width": _WIDTH},
        data={"bias": "e"},
    ).with_(error_scale="spread")
    linked = family.with_links(spread="softplus")
    assert linked.custom == family.custom
    assert linked.error_scale == "spread"
    assert linked.recorded_events


def test_a_link_must_map_into_a_domain_narrower_than_the_native_constraint():
    narrow = families["frechet"]
    nu = narrow.aux_params[0]
    frame = _frame().assign(y=np.exp(_frame()["y"]))
    identity = narrow.with_(aux_params=(nu.with_(link=links["identity"]),))
    with pytest.raises(ValueError, match=r"'nu'.*exceed 1.*'identity'.*Link\.image"):
        formula_to_qvr("y ~ x", data=frame, family=identity)
    own = Link(name="above", inverse_expr="1.0 + exp({eta})")
    with pytest.raises(ValueError, match=r"'above'.*Link\.image"):
        formula_to_qvr(
            "y ~ x",
            data=frame,
            family=narrow.with_(aux_params=(nu.with_(link=own),)),
        )
    declared = own.with_(image=Domain(lower=1.0))
    emitted = formula_to_qvr(
        "y ~ x",
        data=frame,
        family=narrow.with_(aux_params=(nu.with_(link=declared),)),
        distributional={"nu": "~ x"},
    )
    assert "let nu = 1.0 + exp(dpar_nu_eta)" in emitted


def _late(**changes) -> Family:
    """Declare a shifted log-normal family with its own shift name."""
    source = families["shifted_lognormal"]
    sigma, ndt = source.aux_params
    return source.with_(
        name="late",
        observe_args=("mu", "sigma", "onset"),
        aux_params=(sigma, ndt.with_(name="onset", **changes)),
    )


def test_a_shift_below_the_response_is_bounded_by_it():
    emitted = formula_to_qvr("y ~ x", data=_frame(), family=_late())
    assert emitted.replace("onset", "ndt") == formula_to_qvr(
        "y ~ x", data=_frame(), family="shifted_lognormal"
    )
    with pytest.raises(ValueError, match="late responses must exceed fixed onset"):
        formula_to_qvr(
            "y ~ x",
            data=_frame(),
            family=_late(),
            fixed_parameters={"onset": float(_frame()["y"].min())},
        )
    with pytest.raises(ValueError, match=r"'onset'.*AuxParam\.below_response"):
        formula_to_qvr("y ~ x", data=_frame(), family=_late(below_response=False))


@pytest.mark.parametrize(
    ("changes", "distributional", "expected"),
    [
        ({"prior": "LogNormal(-2.0, 0.5)"}, None, {"onset"}),
        ({}, None, set()),
        ({}, {"onset": "~ x"}, {"dpar_onset_intercept", "dpar_onset_beta_x"}),
        ({"below_response": False, "prior": "Uniform(0.0, 1.0)"}, None, set()),
    ],
)
def test_a_shift_below_the_response_starts_below_it(changes, distributional, expected):
    prepared = _prepare_formula(
        "y ~ x",
        data=_frame(),
        family=_late(**changes),
        distributional=distributional,
        caller="fit",
    )
    starts = _initial_values(prepared, None, None)
    assert set(starts) == expected
    quarter = float(_frame()["y"].min()) / 4
    if "onset" in expected:
        assert starts["onset"] == quarter
    if "dpar_onset_intercept" in expected:
        assert starts["dpar_onset_intercept"] == np.log(quarter)


def test_a_mixture_component_keeps_the_start_of_its_shift():
    fitted = fit(
        "y ~ x",
        data=_frame(),
        family=mixture("shifted_lognormal", "lognormal"),
        distributional={"ndt1": "~ x"},
        method="svi",
        num_steps=1,
    )
    assert "dpar_ndt1_intercept" in fitted.draws(num_samples=1)


def test_declarations_name_auxiliary_parameters():
    with pytest.raises(dx.ValidationError, match="error_scale 'sigma'"):
        _bell(error_scale="sigma")
    with pytest.raises(dx.ValidationError, match="elliptical.df 'nu'"):
        _bell(elliptical=Elliptical(scale="spread", df="nu"))
    per_component = families["logistic_normal"].aux_params[0]
    with pytest.raises(dx.ValidationError, match="per-component"):
        _bell(error_scale=None, elliptical=None, aux_params=(per_component,))


def test_the_per_component_parameter_is_declared():
    source = families["logistic_normal"]
    family = source.with_(
        name="alr",
        aux_params=tuple(aux.with_(name="width") for aux in source.aux_params),
    )
    frame = pd.DataFrame(
        {"a": [0.2, 0.3, 0.1], "b": [0.3, 0.2, 0.3], "c": [0.5, 0.5, 0.6]}
    ).assign(x=[0.1, -0.2, 0.4])
    for distributional in (None, {"width": "~ x"}, {"widthb": "~ x"}):
        emitted = formula_to_qvr(
            "cbind(a, b, c) ~ x",
            data=frame,
            family=family,
            distributional=distributional,
        )
        builtin = formula_to_qvr(
            "cbind(a, b, c) ~ x",
            data=frame,
            family=source,
            distributional=None
            if distributional is None
            else {
                name.replace("width", "sigma"): value
                for name, value in distributional.items()
            },
        )
        assert emitted.replace("width", "sigma") == builtin


def test_factor_levels_follow_the_native_support_or_the_declaration():
    assert families["bernoulli"].coded_levels == 2
    assert families["binomial"].coded_levels == 2
    assert families["beta_binomial"].coded_levels is None
    own = families["binomial"].with_(name="yesno")
    assert formula_to_qvr("two ~ x", data=_frame(), family=own) == formula_to_qvr(
        "two ~ x", data=_frame(), family="binomial"
    )
    with pytest.raises(ValueError, match="yesno requires a numeric response"):
        formula_to_qvr("two ~ x", data=_frame(), family=own.with_(factor_levels=None))


def test_brms_default_scales_read_the_declared_response_transform():
    response = np.exp(np.linspace(0.0, 30.0, 20))
    log = links["log"]
    assert scale_prior_arguments(response, log, center=True)[1] > 2.5
    assert scale_prior_arguments(
        response, log.with_(brms_response_expr=None), center=True
    ) == (0.0, 2.5)
    own = Link(name="own_log", inverse_expr="exp({eta})", brms_response_expr="log({y})")
    family = families["gamma"].with_(location_link=own)
    assert sdgp_prior(response, family) == sdgp_prior(response, families["gamma"])


#: Width of the custom law, under its own formula name.
_WIDTH = AuxParam(name="spread", prior="LogNormal(0.0, 0.3)", link=links["log"])


class _Shifted(Normal):
    """A Normal law under another name and parameterization."""

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}


def _shifted_catalog() -> FamilyCatalog:
    record = DistributionFamily(
        name="Shifted",
        parameters=(
            FamilyParameter(name="center", constraint="real", rank=0),
            FamilyParameter(name="width", constraint="positive", rank=0),
            FamilyParameter(name="bias", constraint="real", rank=0),
        ),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
    )

    def build(parameters) -> Distribution:
        return _Shifted(
            *torch.broadcast_tensors(
                parameters["center"] + parameters["bias"], parameters["width"]
            )
        )

    return FamilyCatalog().with_family(record, build)


def _plain_custom(**changes) -> Family:
    record = DistributionFamily(
        name="Plain",
        parameters=(
            FamilyParameter(name="center", constraint="real", rank=0),
            FamilyParameter(name="width", constraint="positive", rank=0),
        ),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
    )
    catalog = FamilyCatalog().with_family(
        record,
        lambda parameters: _Shifted(
            *torch.broadcast_tensors(parameters["center"], parameters["width"])
        ),
    )
    return custom_family(
        "Plain", catalog=catalog, location="center", auxiliary={"width": _WIDTH}
    ).with_(**changes)


def test_a_custom_family_declares_an_error_scale():
    family = _plain_custom(error_scale="spread")
    fitted = fit(
        "y | se(e, sigma = TRUE) ~ x",
        data=_frame(),
        family=family,
        method="svi",
        num_steps=0,
        seed=1,
    )
    assert "width=formula_sigma" in fitted.qvr_source.replace(" ", "")
    draws = fitted.draws(num_samples=3, seed=2)
    frame = _frame()
    total = torch.sqrt(
        draws["spread"][0, :, None].double() ** 2
        + torch.as_tensor(frame["e"].to_numpy()) ** 2
    )
    mean = draws["intercept"][0, :, None].double() + draws["beta_x"][
        0, :, None
    ].double() * torch.as_tensor(frame["x"].to_numpy())
    expected = Normal(mean, total).log_prob(torch.as_tensor(frame["y"].to_numpy()))
    actual = fitted.log_likelihood(num_samples=3, seed=2)
    torch.testing.assert_close(
        actual["y"].reshape(expected.shape).double(), expected, rtol=1e-4, atol=1e-4
    )
    with pytest.raises(ValueError, match=r"'Plain'.*Family\.error_scale"):
        formula_to_qvr(
            "y | se(e) ~ x", data=_frame(), family=family.with_(error_scale=None)
        )


def test_a_custom_family_declares_an_elliptical_law():
    family = _plain_custom(elliptical=Elliptical(scale="spread"))
    joint = fit(_joint(family), data=_frame(), method="svi", num_steps=1, seed=1)
    assert "MultivariateNormal(" in joint.qvr_source
    assert joint.posterior_predict(num_samples=2, seed=1)["y"].shape[-1] == N
    rows = fit(
        "y ~ x + ar(t, p = 1, cov = TRUE)",
        data=_frame(),
        family=family,
        method="svi",
        num_steps=1,
        seed=1,
    )
    assert "ResidualNormal_y" in rows.qvr_source


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("poisson", "discrete"),
        ("cumulative", "discrete"),
        ("categorical", "vector or a category"),
        ("mixture", "mixes component laws"),
        ("hurdle_gamma", "point masses"),
        ("wiener", "decision"),
        ("cox", "baseline hazard"),
        ("shifted_lognormal", "bounded by the recorded responses"),
    ],
)
def test_mi_refusals_name_the_property(name, reason):
    with pytest.raises(ValueError, match=reason):
        missing_specification(
            MultivariateFormula(
                responses={
                    "y": ResponseModel(formula="y | mi() ~ x", family=name),
                    "z": ResponseModel(formula="z ~ mi(y)"),
                },
                rescor=False,
                response_names=("y", "z"),
            )
        )


def test_mi_is_admitted_for_the_families_brms_lists():
    assert {
        name
        for name, family in families.items()
        if family.latent_response_obstacle is None
    } == {
        "gaussian",
        "student_t",
        "skew_normal",
        "lognormal",
        "exgaussian",
        "gamma",
        "weibull",
        "frechet",
        "exponential",
        "inverse_gaussian",
        "beta",
        "von_mises",
        "asym_laplace",
        "gen_extreme_value",
    }


def test_a_custom_continuous_law_admits_missing_responses():
    frame = _frame()
    frame.loc[[2, 5], "y"] = np.nan
    fitted = fit(
        ResponseModel(formula="y | mi() ~ x", family=_plain_custom())
        + ResponseModel(formula="z ~ mi(y)")
        + set_rescor(False),
        data=frame,
        method="svi",
        num_steps=2,
        seed=1,
    )
    draws = fitted.draws(num_samples=2)
    assert any("mi_y__missing" in name for name in draws)


_SOURCES = sorted(Path(quivers.formulas.__file__).parent.glob("*.py"))


def _identity_tests(source: str) -> list[tuple[int, str]]:
    """Find comparisons of a family's name or native law with constants."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        subject = any(
            isinstance(operand, ast.Attribute)
            and (
                operand.attr == "observe_family"
                or operand.attr == "name"
                and "family" in ast.unparse(operand.value).lower()
            )
            for operand in operands
        )
        constant = any(
            isinstance(leaf, ast.Constant) and isinstance(leaf.value, str)
            for operand in operands
            for leaf in ast.walk(operand)
        )
        if subject and constant:
            found.append((node.lineno, ast.unparse(node)))
    return found


def test_no_formula_module_compares_a_family_name():
    assert _identity_tests("if family.name == 'gaussian': pass")
    assert _identity_tests("if self._family.name not in {'a', 'b'}: pass")
    assert _identity_tests("if spec.observe_family != 'Categorical': pass")
    assert not _identity_tests("if target.name == 'x': pass")
    found = {path.name: _identity_tests(path.read_text()) for path in _SOURCES}
    assert {name: where for name, where in found.items() if where} == {}
