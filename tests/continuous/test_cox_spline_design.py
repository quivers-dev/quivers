"""Censored follow-up affects Cox boundaries without moving event knots."""

import math

import pytest

from quivers._spline_design import cox_spline_basis


def test_event_quantiles_and_full_followup_boundary_have_distinct_roles():
    basis = cox_spline_basis([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 50.0], df=6)
    assert basis.interior_knots == pytest.approx((5 / 3, 7 / 3))
    assert basis.boundary_knots == pytest.approx((0.0, 50 + 49 / 50))
    reordered = cox_spline_basis([3.0, 1.0, 2.0], [50.0, 3.0, 1.0, 2.0], df=6)
    assert reordered == basis
    short = cox_spline_basis([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], df=6)
    assert short.interior_knots == basis.interior_knots
    assert short.boundary_knots != basis.boundary_knots


def test_single_time_span_and_no_interior_knots_have_explicit_policies():
    basis = cox_spline_basis([2.0], [2.0])
    assert basis.interior_knots == (2.0,)
    assert basis.boundary_knots == (0.0, 2.04)
    no_events = cox_spline_basis([], [1.0, 3.0], df=4)
    assert no_events.interior_knots == ()
    assert no_events.size == 4


@pytest.mark.parametrize(
    "events,observed,kwargs",
    [
        ([], [], {}),
        ([], [1.0, 2.0], {}),
        ([0.0], [0.0], {}),
        ([0.0], [0.0, 1.0], {}),
        ([1.0], [1.0, 2.0], {"df": 6}),
        ([3.0], [1.0, 2.0], {}),
        ([1.0], [1.0, math.inf], {}),
        ([True], [1.0], {}),
        ([1.0], [1.0], {"df": True}),
        ([1.0], [1.0], {"df": 3}),
        ([1.0], [1.0], {"degree": -1}),
    ],
)
def test_ambiguous_or_invalid_training_design_requires_explicit_structure(
    events, observed, kwargs
):
    with pytest.raises(ValueError):
        cox_spline_basis(events, observed, **kwargs)
