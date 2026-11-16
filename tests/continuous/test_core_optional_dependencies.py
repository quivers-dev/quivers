"""Core laws remain usable without optional array and scientific packages."""

import os
import subprocess
import sys

import pytest
import torch

from quivers.continuous import _quadrature_constants as rules


@pytest.mark.parametrize("order", [32, 128])
def test_fixed_quadrature_integrates_monomials(order):
    nodes = torch.tensor(getattr(rules, f"_NODES_{order}"), dtype=torch.float64)
    weights = torch.tensor(getattr(rules, f"_WEIGHTS_{order}"), dtype=torch.float64)
    assert bool((weights > 0).all())
    torch.testing.assert_close(nodes, -nodes.flip(0), rtol=0, atol=0)
    for degree in range(2 * order):
        integral = (weights * nodes.pow(degree)).sum().item()
        expected = 0.0 if degree % 2 else 2.0 / (degree + 1)
        assert integral == pytest.approx(expected, abs=3e-14)


def test_import_and_skew_normal_tails_without_optional_dependencies():
    # A separate interpreter prevents previously imported test dependencies
    # from concealing a core module's accidental optional import.
    source = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'pandas'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockOptional())
import quivers
import torch
from quivers.continuous import SkewNormal
law = SkewNormal(torch.tensor(0.), torch.tensor(1.), torch.tensor(2.))
assert torch.isfinite(law.log_cdf(torch.tensor(-1.)))
assert torch.isfinite(law.log_survival_function(torch.tensor(1.)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
