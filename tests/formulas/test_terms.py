"""The terms of a predictor declare what its program contains."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
import pytest
import torch

import quivers.formulas
from quivers.dsl.ast_nodes import (
    Module,
    ProgramDecl,
    SampleStep,
    TypeName,
)
from quivers.dsl.emit import module_to_source
from quivers.dsl.program_templates import _bound_names
from quivers.dsl.step_resolution import _step_value_names
from quivers.formulas import FormulaToQVRModule, families
from quivers.formulas._ast import finite_object_sizes
from quivers.formulas._fit import _formula_lens
from quivers.formulas._group_law import law_inputs
from quivers.formulas._predictor import Predictor, response_axis, transition_axis
from quivers.formulas._term import (
    InputDecl,
    OutputAxis,
    bind_inputs,
    column_rms,
    declare_inputs,
    tensor_type,
)
from quivers.formulas.formula import Formula, _qvr_name
from quivers.qiec.families import FAMILIES

ROWS = 24
FRAME = pd.DataFrame(
    {
        "y": np.random.default_rng(1).normal(size=ROWS).round(3),
        "n": np.random.default_rng(2).poisson(3.0, size=ROWS),
        "o": [0, 1, 2, 3] * 6,
        "c": ["A", "B", "C"] * 8,
        "x": np.random.default_rng(3).normal(size=ROWS).round(3),
        "x2": np.random.default_rng(4).normal(size=ROWS).round(3),
        "w": np.linspace(0.5, 1.5, ROWS),
        "z": [0, 1, 2, 3, 2, 1] * 4,
        "g": ["a", "b", "c"] * 8,
        "h": ["a", "c", "b", "b", "a", "c"] * 4,
        "t": list(range(12)) * 2,
        "ser": ["u"] * 12 + ["v"] * 12,
        "loc": [f"l{i % 4}" for i in range(ROWS)],
    }
)
ADJACENCY = pd.DataFrame(
    np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], float),
    index=["l0", "l1", "l2", "l3"],
    columns=["l0", "l1", "l2", "l3"],
)


class Options(TypedDict, total=False):
    """Model options a case sets."""

    family: str
    center: bool
    decomp: Literal["none", "QR"]
    reparameterize: Literal["centered", "noncentered"]
    distributional: dict[str, str]


CASES: dict[str, tuple[str, Options]] = {
    "population": ("y ~ x * x2", {}),
    "no_intercept": ("y ~ 0 + x", {}),
    "centered": ("y ~ x", {"center": True}),
    "qr": ("y ~ x + x2", {"decomp": "QR"}),
    "offset": ("y ~ x + offset(w)", {}),
    "smooth": ("y ~ s(x, k = 5) + x2", {}),
    "shared_smooth": ("y ~ s(x, k = 5, id = 1) + s(x2, k = 5, id = 1)", {}),
    "tensor": ("y ~ t2(x, x2, k = 3)", {}),
    "spectral": ("y ~ gp(x, k = 5, c = 1.25)", {}),
    "exact": ("y ~ gp(x)", {}),
    "monotonic": ("y ~ x + mo(z)", {}),
    "monotonic_shared": ("y ~ mo(z, id = 'a') + mo(z, id = 'a'):x", {}),
    "monotonic_group": ("y ~ mo(z) + (1 + mo(z) | g)", {}),
    "group": ("y ~ x + (1 + x | g)", {}),
    "group_uncorrelated": ("y ~ x + (1 + x || g)", {"reparameterize": "centered"}),
    "group_student": ("y ~ x + (1 | gr(g, dist = 'student'))", {}),
    "membership": ("y ~ x + (1 + x | mm(g, h))", {}),
    "escar": ("n ~ x + car(W, gr = loc)", {"family": "poisson"}),
    "esicar": ("n ~ x + car(W, gr = loc, type = 'esicar')", {"family": "poisson"}),
    "icar": ("n ~ x + car(W, gr = loc, type = 'icar')", {"family": "poisson"}),
    "bym2": ("n ~ x + car(W, gr = loc, type = 'bym2')", {"family": "poisson"}),
    "latent_arma": ("n ~ x + ar(time = t, gr = ser)", {"family": "poisson"}),
    "latent_unstr": ("n ~ x + unstr(time = t, gr = ser)", {"family": "poisson"}),
    "dpar": ("y ~ x", {"distributional": {"sigma": "~ s(x2, k = 4) + (1 | g)"}}),
    "specific": ("o ~ x + cs(x2) + (1 + cs(x) | g)", {"family": "acat"}),
    "categorical": ("c ~ x + s(x2, k = 4) + (1 | g)", {"family": "categorical"}),
    "per_category": ("c ~ x + mo(z)", {"family": "categorical"}),
}


def _emit(name: str) -> tuple[FormulaToQVRModule, Formula, Module, ProgramDecl]:
    formula, options = CASES[name]
    lens, parsed = _formula_lens(formula, data=FRAME, data2={"W": ADJACENCY}, **options)
    module, _ = lens.forward(parsed)
    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    return lens, parsed, module, program


def _terms(lens: FormulaToQVRModule):
    return [term for predictor in lens.predictors for term in predictor.terms]


def _block_sites(lens: FormulaToQVRModule) -> set[str]:
    sites: set[str] = set()
    for law in lens.group_laws:
        sites.update(law.correlation.sites)
        sites.update(
            name
            for member in law.members
            for name in (member.raw_site, member.scale.site)
            if name is not None
        )
        if law.level.mixing is not None:
            sites.update({law.level.mixing.df_site, law.level.mixing.mixing_site})
    return sites


@pytest.mark.parametrize("name", CASES)
def test_declared_sites_are_the_sites_the_program_draws(name):
    """Every sampled site belongs to a term, a group block or the response law."""
    lens, _, _, program = _emit(name)
    drawn = {
        step.vars[0]: step for step in program.draws if isinstance(step, SampleStep)
    }
    declared = {site.name: site for term in _terms(lens) for site in term.sites()}
    # The main formula of a model with per-category formulas is not emitted.
    if name != "per_category":
        assert declared.keys() <= drawn.keys()
    response = {
        site
        for site in drawn
        if site.startswith("cutpoint")
        or site in {auxiliary.name for auxiliary in lens._family.aux_params}
    }
    assert drawn.keys() == (
        (declared.keys() & drawn.keys()) | _block_sites(lens) | response
    )
    for site, declaration in declared.items():
        if site not in drawn:
            continue
        index = drawn[site].index
        if index is None:
            assert declaration.index is None or declaration.event_axis is not None
        else:
            assert isinstance(index, TypeName)
            assert index.name == declaration.index


@pytest.mark.parametrize("name", CASES)
def test_declared_axes_are_objects_of_the_program(name):
    """Every axis a term uses is a finite object of the size it declares."""
    lens, _, module, _ = _emit(name)
    sizes = finite_object_sizes(module)
    for term in _terms(lens):
        for axis in term.axes():
            assert sizes[axis.name] == axis.size


@pytest.mark.parametrize("name", CASES)
def test_declared_inputs_cover_what_the_program_reads(name):
    """Declared inputs are the typed telescope, and their values fit their axes."""
    lens, parsed, module, program = _emit(name)
    sizes = finite_object_sizes(module)
    inputs = [item for term in _terms(lens) for item in term.inputs()]
    parameters = {
        parameter.name: parameter.type_expr for parameter in program.value_parameters
    }
    for item in inputs:
        assert (item.name in parameters) == item.declared
        if item.declared:
            kind = parameters[item.name]
            assert kind == tensor_type(item.element, *item.axes)
        if item.values is not None:
            assert tuple(item.values.shape) == tuple(sizes[axis] for axis in item.axes)
    free = {value for step in program.draws for value in _step_value_names(step)} - set(
        _bound_names(program.draws)
    )
    outside = {
        f"{_qvr_name(group)}_idx"
        for _, _, predictor in lens._predictor_items(parsed)
        for group in predictor.group_levels
    }
    outside |= {item.name for item in law_inputs(lens.group_laws)}
    functions = set(re.findall(r"(\w+)\(", module_to_source(module)))
    assert free <= {item.name for item in inputs} | outside | functions


@pytest.mark.parametrize("name", CASES)
def test_catalogs_hold_the_laws_the_program_draws_from(name):
    """A law that is not built in comes from the catalog of some term."""
    lens, _, _, program = _emit(name)
    used = {
        step.morphism for step in program.draws if isinstance(step, SampleStep)
    } - set(FAMILIES)
    available = {
        record.name
        for term in _terms(lens)
        if (catalog := term.catalog()) is not None
        for record in catalog.definitions
    }
    assert used <= available


CATALOGUE: dict[str, list[tuple[str, str, str | None, str | None, float | None]]] = {
    "centered": [
        ("intercept_centered", "Intercept", "Intercept", None, None),
        ("beta_x", "b", "x", None, None),
        ("sigma", "sigma", None, None, 0.0),
    ],
    "qr": [
        ("intercept_qr", "Intercept", "Intercept", None, None),
        ("beta_x_qr", "b", "x", None, None),
        ("beta_x2_qr", "b", "x2", None, None),
        ("sigma", "sigma", None, None, 0.0),
    ],
    "shared_smooth": [
        ("smooth_id__1_1_scale", "sds", "s(x, k=5, id=1)", None, 0.0),
    ],
    "spectral": [
        ("sdgp_gpx", "sdgp", "gpx", None, 0.0),
        ("lscale_gpx", "lscale", "gpx", None, 0.0),
        ("intercept", "Intercept", "Intercept", None, None),
        ("sigma", "sigma", None, None, 0.0),
    ],
    "monotonic": [
        ("simo_moz1", "simo", "moz1", None, None),
        ("bsp_mo_z", "b", "mo(z)", None, None),
        ("intercept", "Intercept", "Intercept", None, None),
        ("beta_x", "b", "x", None, None),
        ("sigma", "sigma", None, None, 0.0),
    ],
    "group_uncorrelated": [
        ("intercept", "Intercept", "Intercept", None, None),
        ("beta_x", "b", "x", None, None),
        ("sigma_g_Intercept", "sd", "Intercept", "g", 0.0),
        ("sigma_g_x", "sd", "x", "g", 0.0),
        ("sigma", "sigma", None, None, 0.0),
    ],
    "bym2": [
        ("intercept", "Intercept", "Intercept", None, None),
        ("beta_x", "b", "x", None, None),
        ("sdcar", "sdcar", None, None, 0.0),
        ("rhocar", "rhocar", None, None, 0.0),
    ],
}


@pytest.mark.parametrize("name", CATALOGUE)
def test_prior_targets_follow_the_declared_sites(name):
    """The catalogue lists each overridable site once, in its stated order."""
    lens, _, _, _ = _emit(name)
    targets = [
        (target.site, target.class_, target.coef, target.group, target.lower)
        for target in lens.prior_targets
    ]
    if name == "shared_smooth":
        targets = [target for target in targets if target[1] == "sds"]
    assert targets == CATALOGUE[name]


def test_component_coefficients_carry_the_labels_of_their_coordinates():
    """A coefficient vector is labelled by the free categories it indexes."""
    lens, _, _, _ = _emit("categorical")
    labels = {
        target.site: target.component_labels
        for target in lens.prior_targets
        if target.axes == ("FormulaCategoryLogit",)
    }
    assert set(labels) == {
        target.site for target in lens.prior_targets if target.class_ != "cor"
    }
    assert set(labels.values()) == {("B", "C")}


def _inputs(name: str) -> dict[str, torch.Tensor]:
    lens, _, _, _ = _emit(name)
    return bind_inputs(item for term in _terms(lens) for item in term.inputs())


def test_design_values_are_read_from_the_frame():
    """Row data equals the frame's columns under each term's own coding."""
    x = torch.tensor(FRAME["x"].to_numpy(), dtype=torch.float32)
    assert torch.equal(_inputs("population")["x"], x)
    offset = _inputs("offset")["formula_offset_0"]
    assert torch.allclose(offset, torch.tensor(FRAME["w"].to_numpy()).float())
    steps = np.array(FRAME["z"])[:, None] >= np.array([1, 2, 3])
    design = _inputs("monotonic")["mo_z_mo1"]
    assert torch.equal(design, torch.tensor(steps, dtype=torch.float64))
    assert torch.equal(_inputs("group")["random_x"], x)
    codes = torch.tensor([int(label[1]) for label in FRAME["loc"]])
    assert torch.equal(_inputs("escar")["car_idx"], codes)
    members = _inputs("membership")
    position = {label: index for index, label in enumerate("abc")}
    for member, column in enumerate(("g", "h"), start=1):
        levels = [position[label] for label in FRAME[column]]
        assert members[f"mmgh_idx_member{member}"].tolist() == levels
        weighted = members[f"mmgh_x_member{member}"]
        assert torch.allclose(weighted, 0.5 * torch.tensor(FRAME["x"].to_numpy()))


def test_exact_process_inputs_are_the_pairwise_distances():
    """The distance matrix is the scaled distance between the distinct inputs."""
    values = _inputs("exact")
    scaled = FRAME["x"].drop_duplicates().to_numpy()
    distance = np.abs(scaled[:, None] - scaled[None, :])
    assert torch.allclose(
        values["gp_gpx_distance_1"], torch.tensor(distance / distance.max())
    )


def _input(**changes) -> InputDecl:
    return InputDecl(
        name="v",
        element="Real",
        axes=("Resp",),
        declared=True,
        values=torch.tensor([1.0, float("nan")]),
    ).with_(**changes)


def test_a_repeated_input_must_agree():
    """Two terms may share an input only with one type and one value."""
    assert list(bind_inputs([_input(), _input()])) == ["v"]
    for other in (
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, float("nan")], dtype=torch.float64),
    ):
        with pytest.raises(ValueError, match="to two values"):
            bind_inputs([_input(), _input(values=other)])
    with pytest.raises(ValueError, match="two types"):
        declare_inputs([_input(), _input(element="Int")])


def test_column_scale_is_its_root_mean_square():
    """The divisor is the root mean square, and absent for a degenerate column."""
    assert column_rms(np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))
    assert column_rms(np.zeros(4)) is None
    assert column_rms(np.array([1.0, np.inf])) is None


def test_output_axes_hold_the_free_coordinates():
    """A location has one coordinate per free category or transition."""
    axis = response_axis(families["categorical"], ("A", "B", "C"))
    assert axis is not None
    assert (axis.size, axis.labels) == (2, ("B", "C"))
    assert response_axis(families["gaussian"], ("A", "B")) is None
    assert transition_axis(families["acat"], 4).size == 3
    assert transition_axis(families["hurdle_cumulative"], 4).size == 2
    with pytest.raises(ValueError, match="ordinal response family"):
        transition_axis(families["gaussian"], 4)


def test_parts_separate_category_specific_terms():
    """The scalar part and the transitions hold different terms."""
    lens, _, _, _ = _emit("specific")
    location = lens.predictors[0]
    assert isinstance(location, Predictor)
    parts = dict(location.parts())
    transitions = OutputAxis(name="FormulaCutpoint", size=3)
    assert set(parts) == {None, transitions}
    assert [term.label for term in parts[None]] == ["Intercept + x", "Intercept | g"]
    assert [term.label for term in parts[transitions]] == ["cs(x2)", "cs(x) | g"]


#: The class of each kind of term and the module that owns it.
VARIANTS = {
    "PopulationTerm": "_population_coordinates.py",
    "OffsetTerm": "_predictor.py",
    "HostTerm": "_predictor.py",
    "BasisTerm": "_smooth.py",
    "SpectralTerm": "_gp.py",
    "ExactProcessTerm": "_gp.py",
    "MonotonicRow": "_monotonic_emission.py",
    "GroupTerm": "_group_plan.py",
    "FieldTerm": "_latent_effects.py",
}


def test_no_module_tests_the_class_of_a_term():
    """Only the module that owns a kind of term may ask for its class."""
    root = Path(quivers.formulas.__file__).parent
    tests: list[str] = []
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                named = node.args[1:] if node.func.id == "isinstance" else []
            elif isinstance(node, ast.Compare):
                named = [node.left, *node.comparators]
            elif isinstance(node, ast.match_case):
                named = [node.pattern]
            else:
                continue
            names = {
                leaf.id
                for item in named
                for leaf in ast.walk(item)
                if isinstance(leaf, ast.Name)
            }
            tests.extend(
                f"{path.name}: {name}"
                for name in names & VARIANTS.keys()
                if VARIANTS[name] != path.name
            )
    assert not tests


#: Modules that plan and emit predictors.
PLANNED = (
    "_term.py",
    "_predictor.py",
    "_prior_emission.py",
    "_term_emission.py",
    "_category_emission.py",
    "compile.py",
)


def test_parsed_kinds_are_read_only_by_the_plan():
    """The kind of a parsed block is read once, where terms are built."""
    root = Path(quivers.formulas.__file__).parent
    kinds = {"gp", "car_type"}
    for name in PLANNED:
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        owners = {
            function.name
            for function in ast.walk(tree)
            if isinstance(function, ast.FunctionDef)
            for node in ast.walk(function)
            if isinstance(node, ast.Attribute) and node.attr in kinds
        }
        assert owners <= {"design_terms"}, (name, owners)
