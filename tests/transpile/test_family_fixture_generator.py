"""Family fixtures account for live registrations and exercise valid programs."""

from pathlib import Path

import pytest
import torch

from quivers.continuous.programs import MonadicProgram
from quivers.dsl import loads
from quivers.inference.trace import trace
from tests.transpile.fixtures import _generate_families as generator


def test_every_registration_has_one_explicit_source_strategy():
    sources = generator.fixture_sources()
    assert set(sources).isdisjoint(generator.EXCLUDED_FAMILIES)
    assert set(sources) | set(generator.EXCLUDED_FAMILIES) == set(
        generator._get_family_registry()
    )
    assert all(reason.strip() for reason in generator.EXCLUDED_FAMILIES.values())
    directory = Path(generator.__file__).parent / "families"
    assert {path.stem for path in directory.glob("*.qvr")} == {
        name.lower() for name in sources
    }
    for name, source in sources.items():
        assert (directory / f"{name.lower()}.qvr").read_text() == source


def test_generator_is_idempotent_without_rewriting_unchanged_files(tmp_path):
    assert generator.main(output_dir=tmp_path) == 0
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.glob("*.qvr")
    }
    assert generator.main(output_dir=tmp_path) == 0
    assert before == {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.glob("*.qvr")
    }


@pytest.mark.parametrize("change", ["new", "removed", "overlap", "unexplained"])
def test_generator_rejects_incomplete_accounting_before_writing(
    tmp_path, monkeypatch, change
):
    registry = dict(generator._get_family_registry())
    if change == "new":
        registry["FutureFamily"] = type
    elif change == "removed":
        registry.pop("Normal")
    elif change == "overlap":
        monkeypatch.setitem(generator._INLINE_OBSERVE, "Normal", "loc=0, scale=1")
    else:
        monkeypatch.setitem(generator.EXCLUDED_FAMILIES, "Independent", " ")
    monkeypatch.setattr(generator, "_get_family_registry", lambda: registry)
    destination = tmp_path / "untouched"
    assert generator.main(output_dir=destination) == 1
    assert not destination.exists()


@pytest.mark.parametrize("family", sorted(generator.fixture_sources()), ids=str)
def test_generated_family_source_compiles_natively(family):
    loads(generator.fixture_sources()[family])


@pytest.mark.parametrize(
    "family", ["LogisticNormal", "LKJCholesky", "Mixture", "Transformed"]
)
def test_compositional_and_event_fixtures_score_independent_torch_densities(family):
    if family == "LogisticNormal":
        value = torch.tensor([[0.2, 0.8], [0.5, 0.5], [0.7, 0.3]])
        distribution = torch.distributions.LogisticNormal(torch.zeros(1), torch.ones(1))
    elif family == "LKJCholesky":
        value = torch.eye(2).expand(3, 2, 2)
        distribution = torch.distributions.LKJCholesky(2, torch.tensor(2.0))
    elif family == "Mixture":
        value = torch.tensor([-0.4, 0.2, 1.0])
        distribution = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(torch.tensor([0.3, 0.7])),
            torch.distributions.Normal(
                torch.tensor([-1.0, 1.0]), torch.full((2,), 0.5)
            ),
        )
    else:
        value = torch.tensor([0.5, 1.0, 2.0])
        distribution = torch.distributions.LogNormal(
            torch.tensor(0.0), torch.tensor(1.0)
        )
    program = loads(generator.fixture_sources()[family]).morphism
    assert isinstance(program, MonadicProgram)
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), {"y": value})
    assert recorded.log_joint is not None
    torch.testing.assert_close(
        recorded.log_joint.squeeze(), distribution.log_prob(value).sum()
    )


def test_independent_source_exclusion_reproduces_checked_batch_limitation():
    source = """object Row : FinSet 3
program model : Row -> Row
    observe y : Row <- Independent(base=Normal(loc=[0.0, 0.0], scale=[1.0, 1.0]), reinterpreted_batch_ndims=1)
    return y
export model
"""
    with pytest.raises(ValueError, match="takes a Real"):
        loads(source)
