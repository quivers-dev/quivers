"""A program step drawing through an attention source is refused where it is written.

`[param_source=attention]` pools a sequence of rows into one parameter
vector, and a program step hands a kernel one row per draw, so no draw
reads it as attention. The compiled runtime (`loads`) and the checked
elaboration both refuse such a step with a diagnostic located at the
step; the kernel outside a program still compiles. A program over a
finite domain also refuses an input that is not one element index per
run, naming the shape it expects and the shape it got, rather than
failing inside a network.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from quivers.cli.check import check_file
from quivers.dsl import loads, parse
from quivers.dsl.compiler import CompileError
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.inference import trace

_LAYER = """
object Token : FinSet 6
object Latent : Real 8
morphism tok_embed : Token -> Latent [role=embed]
morphism attn : Latent -> Latent [param_source=attention] ~ Normal
morphism lm_head : Latent -> Token ~ Categorical
define layer = tok_embed >> attn
program lm : Token -> Token
    sample h <- layer
    observe next_token <- lm_head(h)
    return next_token
export lm
"""

_DIRECT = """
object Latent : Real 8
morphism attn : Latent -> Latent [param_source=attention] ~ Normal
program p : Latent -> Latent
    sample h <- attn
    return h
export p
"""

_OUTSIDE = """
object Latent : Real 8
morphism attn : Latent -> Latent [param_source=attention] ~ Normal
export attn
"""

_PLAIN = _LAYER.replace(" [param_source=attention]", "")


@pytest.mark.parametrize(("source", "line", "col"), [(_LAYER, 9, 4), (_DIRECT, 5, 4)])
def test_the_compiled_runtime_refuses_the_step(
    source: str, line: int, col: int
) -> None:
    with pytest.raises(CompileError, match="param_source=attention") as caught:
        loads(source)
    assert (caught.value.line, caught.value.col) == (line, col)


@pytest.mark.parametrize(("source", "line"), [(_LAYER, 9), (_DIRECT, 5)])
def test_the_checked_elaboration_refuses_the_step(source: str, line: int) -> None:
    with pytest.raises(QiecDiagnosticError, match="param_source=attention") as caught:
        lower_qvr_to_qiec(parse(source), module_name="m", file_path="m.qvr")
    assert caught.value.code == "qiec-program"
    assert caught.value.line == line


def test_qvr_check_locates_the_refusal(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "lm.qvr"
    path.write_text(_LAYER)
    diagnostics = check_file(path)
    assert diagnostics
    assert all(d.line == 9 for d in diagnostics)
    assert any("param_source=attention" in d.message for d in diagnostics)


def test_an_attention_kernel_outside_a_program_compiles() -> None:
    program = loads(_OUTSIDE)
    sequence = torch.randn(2, 5, 8)
    assert program.morphism.rsample(sequence).shape[0] == 2


def test_a_finite_domain_input_of_several_elements_per_run_is_refused() -> None:
    program = loads(_PLAIN)
    with pytest.raises(
        ValueError, match=r"\(B,\) or \(B, 1\); got an input of shape \(1, 5\)"
    ):
        trace(
            program.morphism,
            torch.randint(0, 6, (1, 5)),
            observations={"next_token": torch.randint(0, 6, (1,))},
        )


def test_a_finite_domain_index_outside_the_set_is_refused() -> None:
    program = loads(_PLAIN)
    with pytest.raises(ValueError, match="indices from 0 to 6"):
        trace(
            program.morphism,
            torch.tensor([0, 6]),
            observations={"next_token": torch.tensor([1, 2])},
        )


@pytest.mark.parametrize("shape", [(3,), (3, 1)])
def test_one_index_per_run_is_accepted(shape: tuple[int, ...]) -> None:
    program = loads(_PLAIN)
    torch.manual_seed(0)
    recorded = trace(
        program.morphism,
        torch.randint(0, 6, shape),
        observations={"next_token": torch.randint(0, 6, (3,))},
    )
    assert torch.isfinite(recorded.log_joint).all()
