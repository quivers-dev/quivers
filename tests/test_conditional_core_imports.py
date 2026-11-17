"""Core prediction grids do not import the optional formula/data stack."""

import os
import subprocess
import sys
import textwrap


def test_core_grid_model_and_summaries_without_frame_dependencies():
    script = """
import importlib.abc
import sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pandas', 'narwhals', 'formulae'}:
            raise ImportError('blocked optional dependency: ' + fullname)

sys.meta_path.insert(0, Block())
import torch
from quivers.modeling import Model
from quivers.diagnostics import prediction_grid

grid = prediction_grid({'x': [0., 1.]}, conditions={'z': [-1., 1.]}, marginalize=True)
assert grid.to_mapping() == {'z': (-1., 1., -1., 1.), 'x': (0., 0., 1., 1.)}
result = grid.result(torch.arange(12.).reshape(1, 3, 4), kind='prior')
assert result.draws.shape == (1, 3, 2)
assert result.summary().mean.shape == (2,)
assert prediction_grid({'x': [0.]}, conditions={}).rows == ((0.,),)
assert not {'pandas', 'narwhals', 'formulae'} & sys.modules.keys()
"""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_polars_grid_adapter_does_not_require_pandas():
    script = """
import importlib.abc
import importlib.util
import sys
original_find_spec = importlib.util.find_spec
def unavailable(name, *args, **kwargs):
    if name.split('.')[0] in {'pandas', 'formulae'}:
        return None
    return original_find_spec(name, *args, **kwargs)
importlib.util.find_spec = unavailable
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pandas', 'formulae'}:
            raise ImportError('blocked optional dependency: ' + fullname)
sys.meta_path.insert(0, Block())
import polars as pl
from quivers.diagnostics import prediction_grid
frame = pl.DataFrame({'z': [-1., 1.], 'g': ['a', None]})
grid = prediction_grid({'x': [0., 1.]}, conditions=frame)
assert grid.to_mapping() == {'z': (-1., 1., -1., 1.), 'g': ('a', None, 'a', None), 'x': (0., 0., 1., 1.)}
assert 'pandas' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
