"""Declared Euclidean event axes survive native constructor compilation."""

from copy import deepcopy
import math
import pickle

import didactic.api as dx
import pytest
import torch

from quivers.continuous.inline import make_inline_distribution
from quivers.continuous.spaces import Euclidean
from quivers.dsl import loads
from quivers.dsl.ast_nodes.program_steps import (
    DrawArgDist,
    DrawArgList,
    DrawArgNamed,
    DrawArgScalar,
)


@pytest.mark.parametrize("shape", [(2, 3), (1, 3), (3, 1), (1, 1), (0, 3), ()])
def test_explicit_event_geometry_membership_and_uniform_draws(shape):
    space = Euclidean(
        name="event", dim=math.prod(shape), intrinsic_shape=shape, low=-2.0, high=3.0
    )
    points = torch.zeros((4, 5, *shape))
    assert space.event_shape == shape
    assert space.contains(points).shape == (4, 5)
    assert bool(space.contains(points).all())
    if points.numel():
        points.reshape(20, -1)[7, 0] = 4.0
        expected = torch.ones(20, dtype=torch.bool)
        expected[7] = False
        assert torch.equal(space.contains(points).reshape(-1), expected)
    samples = space.sample_uniform(7)
    assert samples.shape == (7, *shape)
    assert bool(space.contains(samples).all())
    assert deepcopy(space) == space
    assert pickle.loads(pickle.dumps(space)) == space


@pytest.mark.parametrize(
    "shape,dim", [((2, 3), 5), ((-1, 3), -3), ((True, 3), 3), ((1.5, 2), 3)]
)
def test_invalid_declared_event_geometry_fails_at_construction(shape, dim):
    with pytest.raises((ValueError, TypeError, dx.ValidationError)):
        Euclidean(name="bad", dim=dim, intrinsic_shape=shape)


@pytest.mark.parametrize("shape", [(6,), (3, 2), (2,), ()])
def test_equal_entry_counts_do_not_establish_event_axis_compatibility(shape):
    space = Euclidean(name="rectangle", dim=6, intrinsic_shape=(2, 3))
    with pytest.raises(ValueError, match="trailing event shape"):
        space.contains(torch.zeros(shape))


def test_unspecified_event_geometry_keeps_existing_flat_semantics():
    space = Euclidean(name="legacy", dim=6, low=0.0, high=1.0)
    assert space.intrinsic_shape is None
    assert space.event_shape == (6,)
    assert space.sample_uniform(3).shape == (3, 6)
    assert space.contains(torch.zeros(2, 6)).shape == (2,)


@pytest.mark.parametrize("rows,cols", [(2, 3), (1, 3), (3, 1), (1, 1)])
@pytest.mark.parametrize("explicit_axes", [False, True])
def test_checked_matrix_normal_retains_rectangular_event_axes(
    rows, cols, explicit_axes
):
    axes = " [over=[Rows, Cols]]" if explicit_axes else ""
    source = f"""object Out : Real 1
object Rows : FinSet {rows}
object Cols : FinSet {cols}
program p(loc:Tensor[Real]([Rows,Cols]), U:Tensor[Real]([Rows,Rows]), V:Tensor[Real]([Cols,Cols])) : Out -> Out
    sample M <- MatrixNormal(loc=loc,row_covariance=U,col_covariance=V){axes}
    return M
export p
"""
    program = loads(source).morphism
    carrier = program.steps[0].morphism.codomain
    assert isinstance(carrier, Euclidean)
    assert carrier.dim == rows * cols
    assert carrier.intrinsic_shape == (rows, cols)
    assert carrier.event_shape == (rows, cols)


def test_operator_rebuild_keeps_carrier_metadata_and_actual_tensor_axes():
    mean = DrawArgList(
        items=tuple(
            DrawArgList(
                items=tuple(DrawArgScalar(value=float(i + j)) for j in range(3))
            )
            for i in range(2)
        )
    )
    base = DrawArgDist(
        family="Normal",
        args=(
            DrawArgNamed(parameter="loc", value=mean),
            DrawArgNamed(parameter="scale", value=DrawArgScalar(value=1.0)),
        ),
    )
    original = Euclidean(name="bounded", dim=6, low=-5.0, high=5.0, plate_rows=2)
    morphism, _ = make_inline_distribution(
        "Independent",
        (
            DrawArgNamed(parameter="base", value=base),
            DrawArgNamed(
                parameter="reinterpreted_batch_ndims", value=DrawArgScalar(value=2)
            ),
        ),
        original,
    )
    carrier = morphism.codomain
    assert carrier == original.with_(intrinsic_shape=(2, 3))
    assert carrier.event_shape == (2, 3)
    assert carrier.row_width == 3
