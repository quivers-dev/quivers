"""Symbolic design columns keep their canonical tensor and nominal row contracts."""

import pickle

from didactic.fields import ValidationError

import pytest

from quivers.dsl.ast_nodes import LetExprVar
from quivers.formulas._design import SymbolicColumn
from quivers.qiec import BOOL, REAL, NAT, IndexLiteral, ProgramParameter
from quivers.qiec.canonical import tensor_type


def _column(*, element=REAL, shape=(3,), axes=("Rows",), multiplier=1.0):
    return SymbolicColumn(
        term="mi(x)",
        name="mi(x)",
        parameter=ProgramParameter(
            name="latent_x",
            role="data",
            type=tensor_type(
                element, tuple(IndexLiteral(value=n, sort=NAT) for n in shape)
            ),
            axes=axes,
            per_row=True,
        ),
        expression=LetExprVar(name="completed_x"),
        prior_multiplier=multiplier,
    )


def test_symbolic_column_retains_checked_contract_and_expression_without_data():
    column = _column()
    assert column.qvr_name == "latent_x"
    assert column.term == "mi(x)"
    assert column.expression == LetExprVar(name="completed_x")
    assert not hasattr(column, "data")
    assert not column.is_intercept
    assert column.prior_multiplier == 1.0
    assert pickle.loads(pickle.dumps(column)) == column


def test_equal_extent_distinct_nominal_rows_do_not_share_a_column_contract():
    a = _column(axes=("ResponseRows",))
    b = _column(axes=("OtherRows",))
    assert a != b
    assert a.parameter.type == b.parameter.type
    assert a.parameter.axes != b.parameter.axes


@pytest.mark.parametrize(
    "changes",
    [
        {"element": BOOL},
        {"shape": ()},
        {"shape": (3, 2)},
        {"axes": ()},
        {"multiplier": 0.0},
        {"multiplier": -1.0},
        {"multiplier": float("inf")},
        {"multiplier": float("nan")},
    ],
)
def test_invalid_symbolic_column_contracts_are_rejected(changes):
    with pytest.raises((ValueError, TypeError, ValidationError)):
        _column(**changes)


def _formula(column, *, random=False):
    import numpy as np
    from quivers.formulas.formula import Formula, RandomTerm

    return Formula(
        formula="y ~ mi(x)",
        response_name="y",
        fixed_columns=(column,),
        response_values=np.array([0.2, -0.3, 0.5]),
        random_terms=(RandomTerm(slope=column.name, group="g", column=column),)
        if random
        else (),
        group_levels={"g": ("a", "b")} if random else {},
        group_indices={"g": (0, 1, 0)} if random else {},
    )


def test_symbolic_lens_retains_matching_complement_across_interleaved_forward_calls():
    from quivers.formulas import FormulaToQVRModule, families

    a = _formula(_column(axes=("Resp",)))
    b = _formula(_column(axes=("Resp",), multiplier=2.0))
    lens = FormulaToQVRModule(families["gaussian"])
    module, complement = lens.forward(a)
    lens.forward(b)
    assert lens.backward(module, complement) == a
    assert complement.symbolic_columns == {"latent_x": a.fixed_columns[0]}
    assert not complement.fixed_column_data
    assert "latent_x" not in lens.fixed_column_observations(a)


def test_symbolic_priors_use_declared_scale_and_preserve_explicit_override():
    from quivers.formulas import FormulaToQVRModule, families

    value = _formula(_column(axes=("Resp",), multiplier=2.0))
    lens = FormulaToQVRModule(families["gaussian"])
    assert lens.get_prior(value)["beta_latent_x"] == "Normal(loc=0.0, scale=10.0)"
    override = FormulaToQVRModule(
        families["gaussian"], user_priors={"beta_latent_x": "Normal(1.0, 3.0)"}
    )
    assert override.get_prior(value)["beta_latent_x"] == "Normal(loc=1.0, scale=3.0)"


def test_symbolic_runtime_headers_keep_declared_row_contract_and_slope_carrier():
    from quivers.formulas import FormulaToQVRModule, families
    from quivers.dsl.ast_nodes import ProgramDecl

    value = _formula(_column(axes=("Resp",)), random=True)
    lens = FormulaToQVRModule(families["gaussian"])
    module, complement = lens.forward(value)
    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    names = {parameter.name for parameter in program.value_parameters}
    assert {"latent_x", "random_latent_x"} <= names
    assert "latent_x" not in lens.fixed_column_observations(value)
    assert "random_latent_x" not in lens.fixed_column_observations(value)
    assert lens.backward(module, complement) == value
    with pytest.raises(ValueError, match="nominal row axis"):
        lens.forward(_formula(_column(axes=("Other",))))
    with pytest.raises(ValueError, match="nominal row axis"):
        lens.forward(_formula(_column(axes=("Resp",), shape=(2,))))
