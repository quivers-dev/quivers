"""Custom laws share checked source, native probability, and prediction paths."""

from copy import copy, deepcopy
import math
import pickle

import pytest
import torch
from torch.distributions import Distribution, constraints

from quivers import FamilyCatalog
from quivers.dsl import Compiler, lower_qvr_to_qiec, parse
from quivers.modeling import Model, VariationalMethod
from quivers.qiec.families import FAMILIES, FamilyParameter
from quivers.qiec.module import validate_module
from quivers.qiec.serialization import (
    SerializationError,
    dumps,
    loads,
    to_data,
    from_data,
)
from quivers.transpile import UnsupportedConstruct, available_targets
from quivers.transpile.plan import checked_program_plan


class Tilted(Distribution):
    """Normalized density (1 + tilt*y)/2 on [-1,1], with an exact inverse CDF."""

    arg_constraints = {"tilt": constraints.interval(-0.99, 0.99)}
    support = constraints.interval(-1.0, 1.0)
    has_rsample = True

    def __init__(self, tilt):
        self.tilt = tilt
        super().__init__(tilt.shape, validate_args=True)

    def log_prob(self, value):
        score = torch.log1p(self.tilt * value) - math.log(2)
        return torch.where(self.support.check(value), score, -torch.inf)

    def cdf(self, value):
        t = (value.clamp(-1, 1) + 1) / 2
        return (1 - self.tilt) * t + self.tilt * t.square()

    def rsample(self, sample_shape=torch.Size()):
        u = torch.rand(sample_shape + self.batch_shape, device=self.tilt.device)
        root = ((1 - self.tilt).square() + 4 * self.tilt * u).sqrt()
        return 4 * u / (1 - self.tilt + root) - 1

    @property
    def mean(self):
        return self.tilt / 3

    @property
    def variance(self):
        return 1 / 3 - self.tilt.square() / 9


def _record(name="UserTilted"):
    return FAMILIES["Normal"].with_(
        name=name,
        parameters=(FamilyParameter(name="tilt", constraint="real", rank=0),),
        support="interval",
        targets={},
    )


def _catalog(factory=None):
    return FamilyCatalog().with_family(
        _record(),
        factory or (lambda p: Tilted(p["tilt"])),
        support=constraints.interval(-1, 1),
    )


SOURCE = """object Row:FinSet _
object One:FinSet 1
object Out:Real 1
program p(y:Tensor[Real]([Row])) : One -> Out
    sample b <- Normal(loc=0.0,scale=1.0)
    let tilt = 0.9 * tanh(b)
    observe y:Row <- UserTilted(tilt=tilt)
    return b
export p
"""


def test_arbitrary_law_density_every_derivative_and_sampling_share_native_paths():
    catalog = _catalog()
    bound = Model(SOURCE, families=catalog).bind({"y": torch.tensor([-0.4, 0.2, 0.8])})
    b = torch.tensor(0.3, requires_grad=True)
    joint = bound.program.log_joint(bound.input, {**bound.data, "b": b}).sum()
    tilt = 0.9 * b.tanh()
    expected = -0.5 * b.square() - 0.5 * math.log(2 * math.pi)
    expected += (torch.log1p(tilt * bound.data["y"]) - math.log(2)).sum()
    torch.testing.assert_close(joint, expected)
    actual_first = torch.autograd.grad(joint, b, create_graph=True)[0]
    expected_first = torch.autograd.grad(expected, b, create_graph=True)[0]
    torch.testing.assert_close(actual_first, expected_first)
    torch.testing.assert_close(
        torch.autograd.grad(actual_first, b)[0],
        torch.autograd.grad(expected_first, b)[0],
    )
    fitted = Model(SOURCE, families=catalog).fit(
        bound.data, method=VariationalMethod(num_steps=2, seed=5)
    )
    samples = fitted.predict(num_samples=4, seed=9)["y"]
    assert samples.shape == (4, 3)
    assert bool(((samples >= -1) & (samples <= 1)).all())
    draws = fitted.draws(num_samples=4, seed=9)
    expected_mean = (0.9 * draws["b"].tanh() / 3)[..., None].expand(1, 4, 3)
    torch.testing.assert_close(
        fitted.posterior_epred(num_samples=4, seed=9)["y"], expected_mean
    )


def test_catalogs_are_isolated_and_factories_are_not_probed_during_check_or_bind():
    calls = []
    first = _catalog(lambda p: calls.append("first") or Tilted(p["tilt"]))
    second = _catalog(lambda p: calls.append("second") or Tilted(-p["tilt"]))
    models = [Model(SOURCE, families=item) for item in (first, second)]
    bindings = [model.bind({"y": torch.tensor([0.2])}) for model in models]
    assert not calls
    scores = [
        b.program.log_joint(b.input, {**b.data, "b": torch.tensor(0.4)})
        for b in bindings
    ]
    assert calls == ["first", "second"]
    assert not torch.equal(*scores)
    assert "UserTilted" not in FAMILIES
    with pytest.raises(Exception, match="unknown|undefined|no family|names no family"):
        Model(SOURCE)
    assert copy(first) is first and deepcopy(first) is first
    with pytest.raises(TypeError, match="live native"):
        pickle.dumps(first)


def test_native_semantic_metadata_does_not_mutate_factory_owned_distribution():
    law = Tilted(torch.tensor(0.2))
    built = _catalog(lambda p: law).build("UserTilted", {"tilt": law.tilt})
    assert built is not law
    assert built.tilt is law.tilt
    assert not hasattr(law, "_family_contract")
    from quivers.continuous._atoms import atomicity

    assert atomicity(built) == "atomless"


@pytest.mark.parametrize(
    "failure", ["value", "batch", "event", "carrier", "reparameterization"]
)
def test_factory_contract_violations_are_checked_without_sampling(failure):
    record = _record()

    def factory(p):
        if failure == "value":
            return p["tilt"]
        if failure == "batch":
            return Tilted(p["tilt"][0])
        if failure == "event":
            return torch.distributions.Independent(Tilted(p["tilt"]), 1)
        if failure == "carrier":
            return torch.distributions.Poisson(p["tilt"].exp())
        return torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=torch.zeros(2)),
            torch.distributions.Normal(torch.stack((p["tilt"], p["tilt"]), -1), 1),
        )

    catalog = FamilyCatalog().with_family(
        record, factory, support=constraints.interval(-1, 1)
    )
    with pytest.raises(
        (TypeError, ValueError), match="Distribution|shape|rank|carrier|reparameter"
    ):
        catalog.build(record.name, {"tilt": torch.tensor([0.1, 0.2])})


def test_dependent_vector_geometry_checks_exact_event_extents():
    record = FAMILIES["MultivariateNormal"].with_(
        name="UserVector",
        targets={},
        parameters=tuple(
            p
            for p in FAMILIES["MultivariateNormal"].parameters
            if p.name in {"loc", "scale_tril"}
        ),
        event_sources=("loc",),
    )
    catalog = FamilyCatalog().with_family(
        record,
        lambda p: torch.distributions.MultivariateNormal(
            p["loc"], scale_tril=p["scale_tril"]
        ),
    )
    correct = {"loc": torch.zeros(2, 3), "scale_tril": torch.eye(3).expand(2, 3, 3)}
    assert catalog.build(record.name, correct).event_shape == (3,)
    wrong = FamilyCatalog().with_family(
        record,
        lambda p: torch.distributions.MultivariateNormal(
            p["loc"][..., :2], torch.eye(2)
        ),
    )
    with pytest.raises(ValueError, match="extent|shape"):
        wrong.build(record.name, correct)
    source = """object Row:FinSet _
object K:FinSet 3
object One:FinSet 1
object Out:Real 1
program p(loc:Tensor[Real]([Row,K]), chol:Tensor[Real]([K,K]), y:Tensor[Real]([Row,K])) : One -> Out
    sample b <- Normal(loc=0.0,scale=1.0)
    observe y:Row <- UserVector(loc=loc,scale_tril=chol)[over=K]
    return b
export p
"""
    data = {
        "loc": torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.4, 0.1]]),
        "chol": torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [-0.1, 0.3, 0.9]]),
        "y": torch.tensor([[0.3, -0.1, 0.5], [0.7, 0.2, -0.4]]),
    }
    bound = Model(source, families=catalog).bind(data)
    actual = bound.program.log_joint(
        bound.input, {**bound.data, "b": torch.tensor(0.0)}
    ).sum()
    whitened = torch.linalg.solve_triangular(
        data["chol"], (data["y"] - data["loc"]).T, upper=False
    ).T
    expected = (
        -0.5 * whitened.square().sum()
        - 2 * data["chol"].diag().log().sum()
        - 3 * math.log(2 * math.pi)
        - 0.5 * math.log(2 * math.pi)
    )
    torch.testing.assert_close(actual, expected)


def test_custom_sampleable_queries_reuse_mass_and_endpoint_algebra():
    source = """object One:FinSet 1
object Out:Real 1
program p(y:Real) : One -> Out
    let law = UserTilted(0.3)
    let density = log_prob(law,y)
    let restricted = Restrict(law,-0.2,0.7)
    let mass = log_mass(restricted)
    let total = weight_value(density) + weight_value(mass)
    return total
export p
"""
    bound = Model(source, families=_catalog()).bind({"y": torch.tensor(0.2)})
    actual = bound.program.rsample(bound.input, observations=dict(bound.data))

    def cdf(y):
        return (y + 1) / 2 + 0.3 * (y * y - 1) / 4

    expected = math.log1p(0.3 * 0.2) - math.log(2) + math.log(cdf(0.7) - cdf(-0.2))
    torch.testing.assert_close(actual, torch.tensor(expected))


def test_custom_contract_wire_roundtrip_and_strict_versioned_fields():
    module = lower_qvr_to_qiec(parse(SOURCE), families=_catalog())
    encoded = dumps(module)
    restored = loads(encoded)
    assert restored == module
    assert validate_module(restored).distributions["UserTilted"] == _record()
    assert "factory" not in encoded and "Tilted(" not in encoded
    ordinary = lower_qvr_to_qiec(
        parse(SOURCE.replace("UserTilted(tilt=tilt)", "Normal(loc=tilt,scale=1.0)"))
    )
    envelope = to_data(ordinary)
    envelope["$schema"] = "qiec-json/v1"
    with pytest.raises(SerializationError, match="wire format"):
        from_data(envelope)
    root = envelope["root"]
    root["fields"] = [
        item for item in root["fields"] if item["name"] != "distribution_families"
    ]
    with pytest.raises(SerializationError, match="wire format"):
        from_data(envelope)
    envelope["$schema"] = "qiec-json/v2"
    with pytest.raises(SerializationError, match="fields"):
        from_data(envelope)
    root["fields"] = [item for item in root["fields"] if item["name"] != "entries"]
    with pytest.raises(SerializationError, match="fields"):
        from_data(envelope)
    with pytest.raises(SerializationError, match="wire format"):
        from_data({**envelope, "$schema": "qiec-json/future"})


@pytest.mark.parametrize("target", sorted(available_targets()))
def test_custom_native_laws_are_explicit_external_capability_refusals(target):
    source = parse(SOURCE.replace("FinSet _", "FinSet 3"))
    module = lower_qvr_to_qiec(source, families=_catalog())
    with pytest.raises(UnsupportedConstruct, match="UserTilted|family|native-family"):
        checked_program_plan(module, source, target)


def test_static_checked_source_cache_does_not_reuse_another_catalog_contract():
    source = parse(SOURCE)
    checked = lower_qvr_to_qiec(source, families=_catalog())
    compiler = Compiler(source)
    assert not compiler._reuse_checked_source(source, checked)
    catalog = _catalog()
    compiler = Compiler(source, families=catalog)
    assert compiler._reuse_checked_source(source, checked)


@pytest.mark.parametrize("shift,flip", [(-3, True), (2**25 + 1, False)])
def test_custom_signed_integer_and_boolean_parameters_keep_declared_carriers(
    shift, flip
):
    seen = []
    record = FAMILIES["Normal"].with_(
        name="TypedLaw",
        targets={},
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=0),
            FamilyParameter(name="shift", constraint="integer", rank=0),
            FamilyParameter(name="flip", constraint="boolean", rank=0),
        ),
    )

    def factory(p):
        assert p["shift"].dtype == torch.int64
        assert p["flip"].dtype == torch.bool
        seen.append(p["shift"].item())
        delta = torch.where(p["flip"], -p["shift"], p["shift"])
        return torch.distributions.Normal(p["loc"] + delta, 1.0)

    catalog = FamilyCatalog().with_family(record, factory)
    source = """object One:FinSet 1
object Out:Real 1
program p(shift:Int,flip:Bool,y:Real) : One -> Out
    sample b <- Normal(loc=0.0,scale=1.0)
    observe y <- TypedLaw(loc=b,shift=shift,flip=flip)
    return b
export p
"""
    mean = -shift if flip else shift
    data = {
        "shift": torch.tensor(shift),
        "flip": torch.tensor(flip),
        "y": torch.tensor(mean + 0.2, dtype=torch.float64),
    }
    bound = Model(source, families=catalog).bind(data)
    value = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    actual = bound.program.log_joint(bound.input, {**bound.data, "b": value}).sum()
    expected = torch.distributions.Normal(value, 1.0).log_prob(
        value.new_tensor(0.0)
    ) + torch.distributions.Normal(value + mean, 1.0).log_prob(data["y"])
    torch.testing.assert_close(actual, expected)
    assert seen == [shift]
    with pytest.raises(ValueError, match="packed capacity"):
        bound.program.log_joint(
            bound.input, {**bound.data, "shift": torch.tensor(2**53 + 1), "b": value}
        )


def test_catalog_constraints_reject_wrong_carriers_and_invalid_natural_parameters():
    record = FAMILIES["Normal"].with_(name="CheckedNormal", targets={})
    catalog = FamilyCatalog().with_family(
        record, lambda p: torch.distributions.Normal(p["loc"], p["scale"])
    )
    with pytest.raises(TypeError, match="real numeric"):
        catalog.build(
            record.name, {"loc": torch.tensor(True), "scale": torch.tensor(1.0)}
        )
    with pytest.raises(ValueError, match="positive"):
        catalog.build(
            record.name, {"loc": torch.tensor(0.0), "scale": torch.tensor(-1.0)}
        )


def test_catalog_identity_immutability_and_file_loading(tmp_path):
    from dataclasses import FrozenInstanceError

    catalog = _catalog()
    assert catalog != _catalog()
    with pytest.raises(TypeError):
        catalog["UserTilted"].targets["python"] = "undefined"
    with pytest.raises(FrozenInstanceError):
        catalog._definitions = {}
    path = tmp_path / "custom.qvr"
    path.write_text(SOURCE)
    model = Model.from_file(path, families=catalog)
    assert model.families is catalog
    assert model.schema == Model(SOURCE, families=catalog).schema


def test_nested_custom_constructor_retains_native_context_and_sampling_normalizer():
    source = SOURCE.replace(
        "UserTilted(tilt=tilt)",
        "Restrict(base=UserTilted(tilt=tilt),low=-0.2,high=0.7)",
    )
    y = torch.tensor([0.0, 0.3], dtype=torch.float64)
    bound = Model(source, families=_catalog()).bind({"y": y})
    b = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    actual = bound.program.log_joint(bound.input, {**bound.data, "b": b}).sum()
    tilt = 0.9 * b.tanh()
    expected = torch.distributions.Normal(b.new_tensor(0.0), 1.0).log_prob(b)
    expected = expected + (torch.log1p(tilt * y) - math.log(2)).sum()
    # Source draws normalize the restricted measure. Raw log_mass queries
    # retain its mass separately, as checked by the pure-query oracle above.
    mass = 0.45 + 0.1125 * tilt
    expected = expected - y.numel() * mass.log()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, b)[0],
        torch.autograd.grad(expected, b)[0],
    )
