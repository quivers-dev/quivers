"""Guide tracing resolves its module independently of package import order."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("first", ["quivers.inference", "quivers.effects"])
def test_guide_registry_traces_after_either_package_import(first: str) -> None:
    """A fresh interpreter finds only the latent and preserves the RNG state.

    Parameters
    ----------
    first : str
        The package imported before constructing the guide's registry.
    """
    script = f'''
from importlib import import_module
import_module({first!r})
import torch
from quivers.dsl import loads
from quivers.inference import Guide

torch.manual_seed(0)
program = loads("""object One : FinSet 1
program p : One -> One
    sample z <- Normal(loc=0.0, scale=1.0)
    observe y : One <- Normal(loc=z, scale=1.0)
    return z
export p
""")
before = torch.get_rng_state().clone()
registry = Guide.build_registry(
    program.morphism, {{"y"}}, x=torch.zeros(1, 1),
    observations={{"y": torch.tensor([0.25])}},
)
assert registry.names == ("z",), registry.names
assert "y" not in registry.names
assert torch.equal(before, torch.get_rng_state())
'''
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
