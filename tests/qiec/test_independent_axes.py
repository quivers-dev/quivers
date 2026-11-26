"""Independent axes retain their ownership through source and both executors."""

import math
import random

import pytest
import torch

from quivers.continuous.measure import Independent, Normalize, Restrict
from quivers.dsl import loads, parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.inference.trace import trace
from quivers.qiec import (
    INT,
    REAL,
    DistributionValue,
    Evaluator,
    KernelError,
    KernelRegistry,
    LiteralValue,
    Return,
    HostLaw,
    dumps,
    loads as load_qiec,
    family,
    infer_value,
    run_named,
    sampleable_type,
    tensor_type,
)
from quivers.qiec._distribution_axes import distribution_axes
from quivers.qiec.identifiers import SourceOrigin
from quivers.qiec.kinds import NAT
from quivers.qiec.program_runtime import run_program
from quivers.qiec.reference_support import DistributionError
from quivers.qiec.terms import PlateAxis, PlateShape
from quivers.qiec.types import IndexLiteral


ORIGIN = SourceOrigin(
    module="independent", structural_path=("test",), role="test", source_protocol="test"
)


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


def _normal(shape):
    axes = tuple(
        PlateAxis(name=f"batch{i}", size=IndexLiteral(value=n, sort=NAT))
        for i, n in enumerate(shape)
    )
    return DistributionValue(
        family=family("Normal").id,
        name="Normal",
        arguments=(
            ("loc", LiteralValue(value=0.0, type=REAL)),
            ("scale", LiteralValue(value=1.0, type=REAL)),
        ),
        result_type=sampleable_type(tensor_type(REAL, tuple(a.size for a in axes)))
        if axes
        else sampleable_type(REAL),
        origin=ORIGIN,
        plate=PlateShape(batch=axes),
    )


def _independent(base, count, count_type=INT):
    return DistributionValue(
        family=family("Independent").id,
        name="Independent",
        arguments=(
            ("base", base),
            ("reinterpreted_batch_ndims", LiteralValue(value=count, type=count_type)),
        ),
        result_type=base.result_type,
        origin=ORIGIN,
    )


@pytest.mark.parametrize(
    "shape,count", [((2, 3), 0), ((2, 3), 1), ((2, 3), 2), ((3,), 1), ((), 0)]
)
def test_checked_reference_batch_event_partition_and_wire_roundtrip(shape, count):
    term = _independent(_normal(shape), count)
    batch, event = distribution_axes(term)
    assert (
        tuple(a.size.value for a in batch) == shape[: len(shape) - count]
        if count
        else tuple(a.size.value for a in batch) == shape
    )
    assert tuple(a.size.value for a in event) == (shape[-count:] if count else ())
    assert infer_value(term, KernelRegistry()) == term.result_type
    assert load_qiec(dumps(Return(value=term))) == Return(value=term)
    runtime = Evaluator().evaluate(Return(value=term))
    assert isinstance(runtime, HostLaw)
    assert runtime.batch == tuple(a.size.value for a in batch)
    values = (
        torch.arange(math.prod(shape) or 1, dtype=torch.float64).reshape(shape) / 10
    )
    expected = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(shape), torch.ones(shape)), count
    ).log_prob(values)
    actual = runtime.log_prob(_tuples(values.tolist()), keep_batch=True)
    torch.testing.assert_close(torch.tensor(actual, dtype=torch.float64), expected)
    assert runtime.log_mass(batch_axes=len(batch)) == (
        0.0 if not batch else _tuples(torch.zeros(runtime.batch).tolist())
    )
    sample = torch.tensor(runtime.sample(random.Random(4)))
    assert sample.shape == shape


@pytest.mark.parametrize("count", [True, 1.0, 0.5, -1, 3, 2**100])
def test_invalid_axis_counts_fail_native_reference_and_checked_kernel(count):
    base = torch.distributions.Normal(torch.zeros(2, 3), torch.ones(2, 3))
    with pytest.raises((TypeError, ValueError)):
        Independent(base, count)
    runtime_base = HostLaw(
        family="Normal", arguments={"loc": 0.0, "scale": 1.0}, batch=(2, 3)
    )
    runtime = HostLaw(
        family="Independent",
        arguments={"base": runtime_base, "reinterpreted_batch_ndims": count},
    )
    with pytest.raises(DistributionError):
        runtime.log_prob(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    with pytest.raises(KernelError):
        infer_value(
            _independent(
                _normal((2, 3)), count, REAL if isinstance(count, float) else INT
            ),
            KernelRegistry(),
        )


def test_prefix_selection_preserves_unselected_trailing_broadcast_parameters():
    base = HostLaw(
        family="Normal", arguments={"loc": (1.0, 2.0, 3.0), "scale": 1.0}, batch=(2, 3)
    )
    first = base.at((1,))
    assert first.batch == (3,)
    assert first.arguments["loc"] == (1.0, 2.0, 3.0)
    assert first.log_prob((0.0, 0.0, 0.0)) == pytest.approx(
        sum(-0.5 * x * x - 0.5 * math.log(2 * math.pi) for x in (1.0, 2.0, 3.0))
    )


@pytest.mark.parametrize("shape,count", [((2, 3), 0), ((2, 3), 1), ((2, 3), 2)])
def test_native_raw_restriction_mass_density_and_parameter_gradients(shape, count):
    loc = (
        torch.linspace(-0.2, 0.3, math.prod(shape), dtype=torch.float64)
        .reshape(shape)
        .requires_grad_()
    )
    scale = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    base = torch.distributions.Normal(loc, scale)
    law = Normalize(Independent(Restrict(base, -0.5, 1.2), count))
    value = torch.full(shape, 0.1, dtype=torch.float64)
    normalizer = (base.cdf(torch.tensor(1.2)) - base.cdf(torch.tensor(-0.5))).log()
    score = base.log_prob(value) - normalizer
    expected = score.sum(tuple(range(-count, 0))) if count else score
    actual = law.log_prob(value)
    torch.testing.assert_close(actual, expected)
    got = torch.autograd.grad(actual.sum(), (loc, scale), create_graph=True)
    want = torch.autograd.grad(expected.sum(), (loc, scale), create_graph=True)
    for left, right in zip(got, want, strict=True):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize(
    "rows,width,count", [(None, 3, 0), (None, 3, 1), (3, 3, 1), (2, 3, 1), (2, 3, 0)]
)
def test_source_reserved_axes_and_actual_data_agree_with_normal_oracle(
    rows, width, count
):
    shape = (width,) if rows is None else (rows, width)
    loc = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape) / 10
    declaration = f"object Row : FinSet {rows or 1}\nobject K : FinSet {width}\n"
    shape_text = "[K]" if rows is None else "[Row, K]"
    axis = "" if rows is None else " : Row"
    source = (
        declaration
        + f"""program model(x:Tensor[Real]({shape_text})) : Row -> Row
    observe y{axis} <- Independent(base=Normal(loc=x, scale=0.8), reinterpreted_batch_ndims={count})
    return y
export model
"""
    )
    values = torch.full(shape, 0.2, dtype=torch.float64)
    module = lower_qvr_to_qiec(parse(source))
    expected = torch.distributions.Normal(loc, 0.8).log_prob(values).sum()
    reference = run_program(
        module,
        "model",
        data={"x": _tuples(loc.tolist()), "y": _tuples(values.tolist())},
        sites={},
    )
    assert reference.log_joint == pytest.approx(float(expected), abs=1e-7)
    native = trace(
        loads(source).morphism,
        torch.zeros(1, 1, dtype=torch.long),
        {"x": loc, "y": values},
    )
    assert native.log_joint is not None
    torch.testing.assert_close(native.log_joint.squeeze(), expected)


@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("rows", [None, 2])
def test_source_partial_and_matrix_events_preserve_all_axes(rows, count):
    shape = (2, 3) if rows is None else (rows, 2, 3)
    location = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape) / 10
    axes = "[A,B]" if rows is None else "[Row,A,B]"
    index = "" if rows is None else " : Row"
    source = f"""object Row : FinSet {rows or 1}
object A : FinSet 2
object B : FinSet 3
program p(x:Tensor[Real]({axes})) : Row -> Row
    observe y{index} <- Independent(base=Normal(loc=x,scale=1.0),reinterpreted_batch_ndims={count})
    return y
export p
"""
    values = torch.full(shape, 0.1, dtype=torch.float64)
    module = lower_qvr_to_qiec(parse(source))
    reference = run_program(
        module,
        "p",
        data={"x": _tuples(location.tolist()), "y": _tuples(values.tolist())},
        sites={},
    )
    expected = torch.distributions.Normal(location, 1.0).log_prob(values).sum()
    assert reference.log_joint == pytest.approx(float(expected), abs=1e-7)
    native = trace(
        loads(source).morphism,
        torch.zeros(1, 1, dtype=torch.long),
        {"x": location, "y": values},
    )
    torch.testing.assert_close(native.log_joint.squeeze(), expected)


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("normalize", [False, True])
def test_pure_constructor_and_program_share_raw_mass_and_reinterpretation(
    count, normalize
):
    base = "Restrict(Normal([0.0,0.3],0.8),-0.5,1.2)"
    if normalize:
        base = f"Normalize({base})"
    source = f"""define law() : Sampleable[Tensor[Real]([2])] !{{}} =
    return Independent({base},{count})
"""
    module = lower_qvr_to_qiec(parse(source))
    runtime = run_named(module, "law").value
    native_base = torch.distributions.Normal(
        torch.tensor([0.0, 0.3], dtype=torch.float64),
        torch.tensor(0.8, dtype=torch.float64),
    )
    mass = (
        native_base.cdf(torch.tensor(1.2, dtype=torch.float64))
        - native_base.cdf(torch.tensor(-0.5, dtype=torch.float64))
    ).log()
    expected = native_base.log_prob(torch.tensor([0.1, 0.2], dtype=torch.float64)) - (
        mass if normalize else 0
    )
    assert runtime.log_prob((0.1, 0.2)) == pytest.approx(
        float(expected.sum().detach()), abs=1e-10
    )
    assert runtime.batch == ((2,) if count == 0 else ())
    assert runtime.log_mass() == pytest.approx(
        0.0 if normalize else float(mass.sum()), abs=1e-10
    )


def test_mixed_parameter_batch_ownership_retains_pointwise_axes_and_gradients():
    source = """object Row : FinSet 2
object A : FinSet 2
object B : FinSet 3
program p(x:Tensor[Real]([Row,A,B]),s:Tensor[Real]([Row,B])) : Row -> Row
    observe y : Row <- Independent(base=Normal(loc=x,scale=s),reinterpreted_batch_ndims=1)
    return y
export p
"""
    location = (
        torch.linspace(-0.4, 0.7, 12, dtype=torch.float64)
        .reshape(2, 2, 3)
        .requires_grad_()
    )
    scale = (
        torch.linspace(0.5, 1.0, 6, dtype=torch.float64).reshape(2, 3).requires_grad_()
    )
    response = torch.linspace(-0.1, 0.2, 12, dtype=torch.float64).reshape(2, 2, 3)
    program = loads(source).morphism
    observer = program.steps[0].morphism
    theta = torch.cat((location.reshape(2, 6), scale), dim=-1)
    expected = (
        torch.distributions.Normal(location, scale[:, None, :])
        .log_prob(response)
        .sum(-1)
    )
    actual = observer.log_prob_pointwise(theta, response)
    assert actual.shape == (2, 2)
    torch.testing.assert_close(actual, expected)
    got = torch.autograd.grad(actual.sum(), (location, scale), create_graph=True)
    want = torch.autograd.grad(expected.sum(), (location, scale), create_graph=True)
    for left, right in zip(got, want, strict=True):
        torch.testing.assert_close(left, right)
    ref = run_program(
        lower_qvr_to_qiec(parse(source)),
        "p",
        data={
            "x": _tuples(location.tolist()),
            "s": _tuples(scale.tolist()),
            "y": _tuples(response.tolist()),
        },
        sites={},
    )
    assert ref.log_joint == pytest.approx(float(expected.sum().detach()), abs=1e-10)
    torch.manual_seed(31)
    samples = observer.rsample(theta.detach(), torch.Size((10000,)))
    assert samples.shape == (10000, 2, 2, 3)
    torch.testing.assert_close(samples.mean(0), location.detach(), atol=0.04, rtol=0)
    torch.testing.assert_close(
        samples.var(0),
        scale.detach()[:, None, :].expand_as(location).square(),
        atol=0.05,
        rtol=0,
    )


def test_source_latent_partial_batch_uses_the_same_parameter_packing():
    source = """object Row : FinSet 2
object A : FinSet 2
object B : FinSet 3
program p(x:Tensor[Real]([Row,A,B]),s:Tensor[Real]([Row,B])) : Row -> Row
    sample y : Row <- Independent(base=Normal(loc=x,scale=s),reinterpreted_batch_ndims=1)
    return y
export p
"""
    location = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3) / 10
    scale = torch.full((2, 3), 0.8, dtype=torch.float64)
    response = torch.full_like(location, 0.1)
    native = trace(
        loads(source).morphism,
        torch.zeros(1, 1, dtype=torch.long),
        {"x": location, "s": scale, "y": response},
    )
    expected = (
        torch.distributions.Normal(location, scale[:, None, :]).log_prob(response).sum()
    )
    torch.testing.assert_close(native.log_joint.squeeze(), expected)


@pytest.mark.parametrize(
    "count", ["true", "1.5", "-1", "3", "999999999999999999999999999999"]
)
def test_source_counts_do_not_coerce_invalid_values(count):
    source = f"""object Out : Real 1
program p : Out -> Out
    observe y <- Independent(base=Normal(loc=[0.0,0.0],scale=1.0),reinterpreted_batch_ndims={count})
    return y
export p
"""
    with pytest.raises(QiecDiagnosticError, match="Independent"):
        lower_qvr_to_qiec(parse(source))


def test_recorded_event_queries_use_the_effective_scalar_event_partition():
    source = """object Row : FinSet 2
object Coordinate : FinSet 3
program p(x:Tensor[Real]([Row,Coordinate]),y:Tensor[Real]([Row,Coordinate])) : Row -> Row
    observe y : Row <- Independent(base=Normal(loc=x,scale=0.8),reinterpreted_batch_ndims=0) [trunc_lower=-1.0,trunc_upper=1.0]
    return y
export p
"""
    location = torch.linspace(-0.3, 0.3, 6, dtype=torch.float64).reshape(2, 3)
    response = torch.full_like(location, 0.1)
    base = torch.distributions.Normal(location, 0.8)
    expected = (
        base.log_prob(response)
        - (
            base.cdf(location.new_tensor(1.0)) - base.cdf(location.new_tensor(-1.0))
        ).log()
    ).sum()
    reference = run_program(
        lower_qvr_to_qiec(parse(source)),
        "p",
        data={"x": _tuples(location.tolist()), "y": _tuples(response.tolist())},
        sites={},
    )
    assert reference.log_joint == pytest.approx(float(expected), abs=1e-10)
    native = trace(
        loads(source).morphism,
        torch.zeros(1, 1, dtype=torch.long),
        {"x": location, "y": response},
    )
    torch.testing.assert_close(native.log_joint.squeeze(), expected)
    with pytest.raises(QiecDiagnosticError, match="scalar response events"):
        lower_qvr_to_qiec(
            parse(
                source.replace(
                    "reinterpreted_batch_ndims=0", "reinterpreted_batch_ndims=1"
                )
            )
        )


def test_model_binding_infers_rows_without_consuming_the_reserved_event_axis():
    from quivers.modeling import Model

    source = """object Coordinate : FinSet 3
program vectors[n:Nat](x:Tensor[Real]([n,Coordinate]),y:Tensor[Real]([n,Coordinate])) : n -> n
    observe y : n <- Independent(base=Normal(loc=x,scale=0.8),reinterpreted_batch_ndims=1)
    return y
export vectors
"""
    model = Model(source)
    assert model.schema.describe()["y"]["role"] == "observation"
    for rows in (2, 3, 5):
        location = torch.linspace(-0.2, 0.2, rows * 3).reshape(rows, 3)
        response = torch.full_like(location, 0.1)
        bound = model.bind({"x": location, "y": response})
        expected = torch.distributions.Normal(location, 0.8).log_prob(response).sum()
        torch.testing.assert_close(bound.trace({}).log_joint.squeeze(), expected)
    with pytest.raises(ValueError, match="shape|extent|dimension"):
        model.bind({"x": torch.zeros(2, 3), "y": torch.zeros(2, 2)})


@pytest.mark.parametrize("array_shape", [(2,), (2, 3)])
def test_empty_partial_independent_arrays_keep_batch_axes_and_zero_gradients(
    array_shape,
):
    from quivers.continuous.inline import FixedDistribution
    from quivers.continuous.plate import VectorisedObserve
    from quivers.continuous.spaces import Euclidean

    location = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64, requires_grad=True)
    family = FixedDistribution(
        Euclidean(name="event", dim=3),
        lambda batch, device: Independent(torch.distributions.Normal(location, 0.8), 1),
    )
    observation = VectorisedObserve(
        family,
        torch.empty(0, *array_shape, 3),
        array_shape=array_shape,
        array_is_batch=True,
    )
    actual = observation.log_prob_pointwise(torch.empty(0, 1))
    assert actual.shape == (0, *array_shape)
    assert actual.sum().item() == 0
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), location)[0], torch.zeros_like(location)
    )
