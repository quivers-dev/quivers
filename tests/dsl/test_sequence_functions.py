"""Learned functions over whole sequences: self-attention and position tables.

A ``[role=function]`` morphism with ``param_source=self_attention`` or
``param_source=positions(N)`` maps a sequence of rows to a sequence of
rows. Each claim is checked against a hand-written float64 computation
from the same weights, written with explicit loops over positions so it
shares no code with either execution path. The compiled runtime runs in
float32 and is compared with the oracle at ``rel=1e-5`` (log joints) or
``atol=1e-5`` (entries of order one; float32 rounding over the few dozen
terms of a score is about 1e-6); the reference machine runs in float64
and is compared at ``rel=1e-9``. Every comparison is paired with
deliberately wrong oracles, a non-causal mask and a mask that lets each
position see one position ahead, that the same tolerance rejects.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import pytest
import torch
import torch.distributions as D

from quivers.continuous import (
    Euclidean,
    PositionSource,
    SelfAttentionSource,
    SequenceFunction,
    function_source,
)
from quivers.dsl import CompileError, loads
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.inference import trace
from quivers.program import Program
from quivers.transpile import UnsupportedConstruct, available_targets, transpile

type Host = float | tuple["Host", ...]
"""A value as the reference machine takes it."""

type Mask = Callable[[int, int], bool]
"""Whether position ``i`` attends to position ``j``."""


def _host(value: torch.Tensor) -> Host:
    """A tensor as nested tuples of floats, or a float for a scalar.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    Host
        The host value.
    """
    if value.dim() == 0:
        return float(value)
    return tuple(_host(row) for row in value)


def _causal(i: int, j: int) -> bool:
    """The causal mask: position ``i`` sees itself and the positions before it."""
    return j <= i


def _open(i: int, j: int) -> bool:
    """No mask: every position sees every position."""
    del i, j
    return True


def _one_ahead(i: int, j: int) -> bool:
    """A mask off by one: position ``i`` also sees position ``i + 1``."""
    return j <= i + 1


def _attention_oracle(
    x: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    name: str,
    heads: int,
    mask: Mask,
) -> torch.Tensor:
    """Multi-head scaled dot-product self-attention, by hand, in float64.

    Parameters
    ----------
    x : torch.Tensor
        One sequence, shape ``(T, d)``.
    weights : Mapping[str, torch.Tensor]
        The function's weights by their checked input names.
    name : str
        The function's name.
    heads : int
        The number of heads.
    mask : Mask
        Which positions each position attends to.

    Returns
    -------
    torch.Tensor
        Shape ``(T, m)``.
    """
    w = {k: v.double() for k, v in weights.items()}
    x = x.double()
    length = x.shape[0]

    def project(role: str) -> list[list[float]]:
        weight = w[f"{name}_param_{role}_weight"]
        bias = w[f"{name}_param_{role}_bias"]
        return [
            [
                float(sum(weight[r, c] * x[t, c] for c in range(x.shape[1])) + bias[r])
                for r in range(weight.shape[0])
            ]
            for t in range(length)
        ]

    q, k, v = project("query"), project("key"), project("value")
    model = len(q[0])
    width = model // heads
    context = [[0.0] * model for _ in range(length)]
    for g in range(heads):
        block = range(g * width, (g + 1) * width)
        for i in range(length):
            seen = [j for j in range(length) if mask(i, j) and j < length]
            scores = {
                j: sum(q[i][c] * k[j][c] for c in block) / math.sqrt(width)
                for j in seen
            }
            peak = max(scores.values())
            total = sum(math.exp(s - peak) for s in scores.values())
            for c in block:
                context[i][c] = sum(
                    math.exp(scores[j] - peak) / total * v[j][c] for j in seen
                )
    out_weight = w[f"{name}_param_weight"]
    out_bias = w[f"{name}_param_bias"]
    return torch.tensor(
        [
            [
                float(
                    sum(out_weight[r, c] * context[i][c] for c in range(model))
                    + out_bias[r]
                )
                for r in range(model)
            ]
            for i in range(length)
        ],
        dtype=torch.float64,
    )


def _attention_weights(
    source: SelfAttentionSource, name: str
) -> dict[str, torch.Tensor]:
    """A self-attention network's weights by their checked input names.

    Parameters
    ----------
    source : SelfAttentionSource
        The network.
    name : str
        The function's name.

    Returns
    -------
    dict[str, torch.Tensor]
        ``<name>_param_<role>_weight`` and ``_bias`` for the query, key,
        and value projections, then ``<name>_param_weight`` and
        ``<name>_param_bias`` for the output projection.
    """
    found: dict[str, torch.Tensor] = {}
    for role in ("query", "key", "value"):
        layer = getattr(source, role)
        found[f"{name}_param_{role}_weight"] = layer.weight.detach()
        found[f"{name}_param_{role}_bias"] = layer.bias.detach()
    found[f"{name}_param_weight"] = source.output.weight.detach()
    found[f"{name}_param_bias"] = source.output.bias.detach()
    return found


def _scramble(module: torch.nn.Module, scale: float, seed: int) -> None:
    """Set every parameter of a module to a seeded Normal draw.

    Default initializations leave attention close to uniform, which
    would make masks hard to tell apart; weights of order one do not.

    Parameters
    ----------
    module : torch.nn.Module
        The module.
    scale : float
        The draws' standard deviation.
    seed : int
        The generator's seed.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * scale)


def _space(name: str, dim: int) -> Euclidean:
    """A Euclidean space of the given width."""
    return Euclidean(name=name, dim=dim)


# ---------------------------------------------------------------------------
# The networks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("heads", "causal", "mask", "wrong"),
    [
        (2, True, _causal, (_open, _one_ahead)),
        (1, True, _causal, (_open, _one_ahead)),
        (2, False, _open, (_causal,)),
    ],
)
def test_self_attention_matches_the_hand_computation(
    heads: int, causal: bool, mask: Mask, wrong: tuple[Mask, ...]
) -> None:
    source = SelfAttentionSource(3, 4, heads=heads, causal=causal)
    _scramble(source, 0.8, seed=0)
    x = torch.randn(2, 6, 3, generator=torch.Generator().manual_seed(1))
    out = source(x)
    assert out.shape == (2, 6, 4)
    weights = _attention_weights(source, "f")
    for run in range(2):
        oracle = _attention_oracle(x[run], weights, "f", heads, mask)
        assert torch.allclose(out[run].double(), oracle, atol=1e-5)
        for other in wrong:
            rejected = _attention_oracle(x[run], weights, "f", heads, other)
            assert not torch.allclose(out[run].double(), rejected, atol=1e-5)


def test_causal_attention_reads_no_later_position() -> None:
    torch.manual_seed(2)
    causal = SelfAttentionSource(3, 4, heads=2, causal=True)
    _scramble(causal, 0.8, seed=3)
    x = torch.randn(1, 6, 3)
    later = x.clone()
    later[:, 4:] += 5.0
    # Positions 0 through 3 see only positions 0 through 3, which agree.
    assert torch.equal(causal(x)[:, :4], causal(later)[:, :4])
    assert not torch.allclose(causal(x)[:, 4:], causal(later)[:, 4:])
    # Without the mask every position reads the changed rows.
    unmasked = SelfAttentionSource(3, 4, heads=2, causal=False)
    unmasked.load_state_dict(causal.state_dict())
    assert not torch.allclose(unmasked(x)[:, 0], unmasked(later)[:, 0])


def test_self_attention_refuses_inputs_and_sizes_it_cannot_read() -> None:
    source = SelfAttentionSource(3, 4, heads=2, causal=True)
    with pytest.raises(ValueError, match="sequence of 3-wide rows"):
        source(torch.randn(3))
    with pytest.raises(ValueError, match="sequence of 3-wide rows"):
        source(torch.randn(2, 5, 4))
    with pytest.raises(ValueError, match="3 heads do not divide"):
        SelfAttentionSource(3, 4, heads=3, causal=True)
    with pytest.raises(ValueError, match="positive"):
        SelfAttentionSource(3, 4, heads=0, causal=True)


def test_a_position_table_adds_its_leading_rows() -> None:
    torch.manual_seed(4)
    source = PositionSource(3, 8)
    x = torch.randn(2, 5, 3)
    expected = x + source.table.detach()[:5]
    assert torch.equal(source(x), expected)
    # A table read from the second row onwards is told apart.
    assert not torch.allclose(source(x), x + source.table.detach()[1:6])
    assert source(torch.randn(8, 3)).shape == (8, 3)
    with pytest.raises(ValueError, match="9 positions is longer than the 8"):
        source(torch.randn(1, 9, 3))
    with pytest.raises(ValueError, match="sequence of 3-wide rows"):
        source(torch.randn(3))


def test_function_source_builds_the_sequence_networks() -> None:
    domain = _space("x", 4)
    attention = function_source(domain, 4, "self_attention", heads=2, causal=True)
    assert isinstance(attention, SelfAttentionSource)
    assert (attention.heads, attention.causal) == (2, True)
    positions = function_source(domain, 4, "positions(16)")
    assert isinstance(positions, PositionSource)
    assert positions.max_length == 16
    with pytest.raises(ValueError, match="causal or not by declaration"):
        function_source(domain, 4, "self_attention")
    with pytest.raises(ValueError, match="heads is read by a self_attention"):
        function_source(domain, 4, "linear", heads=2)
    with pytest.raises(ValueError, match="positions takes the number"):
        function_source(domain, 4, "positions")
    with pytest.raises(ValueError, match="cannot write 3"):
        function_source(domain, 3, "positions(8)")


def test_a_sequence_function_keeps_the_position_axis() -> None:
    torch.manual_seed(5)
    space = _space("m", 4)
    function = SequenceFunction(
        space, space, SelfAttentionSource(4, 4, heads=2, causal=True)
    )
    x = torch.randn(3, 5, 4)
    assert torch.equal(function.apply_values(x), function.source(x))
    assert torch.equal(function.rsample(x), function.source(x))
    with pytest.raises(ValueError, match="reads one sequence, but the call passes 2"):
        function.apply_values(x, x)
    with pytest.raises(ValueError, match="sequence of 4-wide rows"):
        function.apply_values(torch.randn(4))
    with pytest.raises(ValueError, match="reads no drawn weights"):
        function.evaluate(x, {"w": torch.zeros(1)})


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

_DECLARED = """\
object Model : Real 4
object Wide : Real 6
morphism f : Model -> Model [{options}]
"""


def test_a_declaration_builds_a_sequence_function() -> None:
    compiler = Compiler(
        parse(
            _DECLARED.format(
                options="role=function, param_source=self_attention, heads=2, "
                "causal=false"
            )
        )
    )
    compiler.compile()
    function = compiler.morphisms["f"]
    assert isinstance(function, SequenceFunction)
    assert isinstance(function.source, SelfAttentionSource)
    assert (function.source.heads, function.source.causal) == (2, False)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ("role=function, param_source=self_attention", "causal or not by declaration"),
        (
            "role=function, param_source=self_attention, causal=maybe",
            "causal=maybe is neither",
        ),
        (
            "role=function, param_source=self_attention, heads=3, causal=true",
            "3 heads do not divide",
        ),
        ("role=function, heads=2", "heads is read by a self_attention source"),
        ("role=function, causal=true", "causal is read by a self_attention source"),
        (
            "role=function, param_source=self_attention, causal=true, output=tanh",
            "applies no output nonlinearity",
        ),
        (
            "role=function, param_source=self_attention, causal=true, activation=relu",
            "activation is read by the hidden layers",
        ),
        (
            "role=function, param_source=self_attention, causal=true, weights=latent",
            "a 'self_attention' source has none to draw",
        ),
        ("role=function, param_source=positions", "positions takes the number"),
    ],
)
def test_a_malformed_sequence_function_is_refused(options: str, message: str) -> None:
    with pytest.raises(CompileError, match=message):
        loads(_DECLARED.format(options=options))


def test_a_position_table_keeps_its_width() -> None:
    source = "object Model : Real 4\nobject Wide : Real 6\nmorphism f : Model -> Wide [role=function, param_source=positions(8)]\n"
    with pytest.raises(CompileError, match="cannot write 6"):
        loads(source)


def test_a_sequence_function_over_a_product_is_refused() -> None:
    source = (
        "object A : Real 2\nobject B : Real 2\nobject Out : Real 4\n"
        "morphism f : A * B -> Out [role=function, param_source=self_attention, "
        "causal=true]\n"
    )
    with pytest.raises(CompileError, match="reads one sequence"):
        loads(source)


# ---------------------------------------------------------------------------
# A let call on a host sequence, on both paths
# ---------------------------------------------------------------------------

HOST_CALL = """\
object Model : Real 4
object Val : Real 1
morphism pos : Model -> Model [role=function, param_source=positions(8)]
morphism attn : Model -> Model [role=function, param_source=self_attention, heads=2, causal=true]
program p : Val -> Val
    let x = pos(seq)
    let h = attn(x)
    let s = sum(sum(h * x))
    observe y <- Normal(loc=s, scale=0.5)
    return y
export p
"""


def _program_function(program: Program, name: str) -> SequenceFunction:
    """The sequence function a compiled program holds under a name.

    Parameters
    ----------
    program : Program
        The compiled program.
    name : str
        The function's name.

    Returns
    -------
    SequenceFunction
        The one module held as ``_function_<name>``.
    """
    morphism = program.morphism
    assert isinstance(morphism, torch.nn.Module)
    found = {
        module
        for path, module in morphism.named_modules()
        if path.split(".")[-1] == f"_function_{name}"
    }
    assert len(found) == 1, f"{name!r} is held {len(found)} times"
    (function,) = found
    assert isinstance(function, SequenceFunction)
    return function


def _host_call_weights(program: Program) -> dict[str, torch.Tensor]:
    """The host-call program's weights by their checked input names."""
    attention = _program_function(program, "attn").source
    assert isinstance(attention, SelfAttentionSource)
    positions = _program_function(program, "pos").source
    assert isinstance(positions, PositionSource)
    return {
        **_attention_weights(attention, "attn"),
        "pos_param_positions": positions.table.detach(),
    }


def _host_call_oracle(
    weights: Mapping[str, torch.Tensor],
    seq: torch.Tensor,
    y: torch.Tensor,
    mask: Mask,
) -> tuple[torch.Tensor, float]:
    """The host-call program's attended sequence and log joint, by hand."""
    x = seq.double() + weights["pos_param_positions"].double()[: seq.shape[0]]
    h = _attention_oracle(x, weights, "attn", 2, mask)
    s = float((h * x).sum())
    return h, float(
        D.Normal(torch.tensor(s, dtype=torch.float64), 0.5).log_prob(
            torch.tensor(float(y), dtype=torch.float64)
        )
    )


def test_a_let_call_scores_alike_on_both_paths_and_by_hand() -> None:
    program = loads(HOST_CALL)
    _scramble(program.morphism, 0.8, seed=6)  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(7)
    seq = torch.randn(2, 5, 4, generator=generator)
    y = torch.randn(2, generator=generator)
    recorded = trace(
        program.morphism, torch.zeros(2, 1), observations={"seq": seq, "y": y}
    )
    weights = _host_call_weights(program)
    for run in range(2):
        h, oracle = _host_call_oracle(weights, seq[run], y[run], _causal)
        assert torch.allclose(recorded.sites["h"].value[run].double(), h, atol=1e-5)
        assert float(recorded.log_joint[run].detach()) == pytest.approx(
            oracle, rel=1e-5
        )
        replay = program.run(
            "p",
            data={
                "val": (0.0,),
                "seq": _host(seq[run]),
                "y": float(y[run]),
                **{k: _host(v.double()) for k, v in weights.items()},
            },
            sites={},
            seed=0,
        )
        assert replay.log_joint == pytest.approx(oracle, rel=1e-9)
        for wrong in (_open, _one_ahead):
            rejected_h, rejected = _host_call_oracle(weights, seq[run], y[run], wrong)
            assert not torch.allclose(
                recorded.sites["h"].value[run].double(), rejected_h, atol=1e-5
            )
            assert abs(rejected - oracle) > 1e-5 * abs(oracle)


def test_one_program_reads_sequences_of_any_length() -> None:
    program = loads(HOST_CALL)
    _scramble(program.morphism, 0.8, seed=8)  # type: ignore[arg-type]
    weights = _host_call_weights(program)
    generator = torch.Generator().manual_seed(9)
    for length in (1, 3, 8):
        seq = torch.randn(1, length, 4, generator=generator)
        y = torch.randn(1, generator=generator)
        recorded = trace(
            program.morphism, torch.zeros(1, 1), observations={"seq": seq, "y": y}
        )
        _, oracle = _host_call_oracle(weights, seq[0], y[0], _causal)
        assert float(recorded.log_joint[0].detach()) == pytest.approx(oracle, rel=1e-5)
    with pytest.raises(ValueError, match="9 positions is longer than the 8"):
        trace(
            program.morphism,
            torch.zeros(1, 1),
            observations={"seq": torch.randn(1, 9, 4), "y": torch.zeros(1)},
        )


def test_the_checked_program_reads_a_sequence_functions_weights_as_inputs() -> None:
    program = loads(HOST_CALL)
    types = {
        parameter.name: (parameter.type, parameter.role)
        for parameter in program.entry("p").parameters
    }
    assert types == {
        "val": ("Tensor[Real]([1])", "domain"),
        "seq": ("Tensor[Real]([seq_extent, 4])", "data"),
        "pos_param_positions": ("Tensor[Real]([8, 4])", "weight"),
        "attn_param_query_weight": ("Tensor[Real]([4, 4])", "weight"),
        "attn_param_query_bias": ("Tensor[Real]([4])", "bias"),
        "attn_param_key_weight": ("Tensor[Real]([4, 4])", "weight"),
        "attn_param_key_bias": ("Tensor[Real]([4])", "bias"),
        "attn_param_value_weight": ("Tensor[Real]([4, 4])", "weight"),
        "attn_param_value_bias": ("Tensor[Real]([4])", "bias"),
        "attn_param_weight": ("Tensor[Real]([4, 4])", "weight"),
        "attn_param_bias": ("Tensor[Real]([4])", "bias"),
        "y": ("Real", "observation"),
    }


@pytest.mark.parametrize(
    ("let", "message"),
    [
        ("let h = attn(x[0])", "reads a sequence of 4-wide real rows"),
        ("let h = attn(x, x)", "'attn': expected 1 positional argument\\(s\\), got 2"),
        ("let h = factor i : K in attn(x)", "called inside a factor or lambda body"),
    ],
)
def test_a_call_that_is_not_one_sequence_is_refused(let: str, message: str) -> None:
    source = HOST_CALL.replace(
        "object Val : Real 1", "object Val : Real 1\nobject K : FinSet 2"
    ).replace("    let h = attn(x)", "    " + let)
    with pytest.raises((QiecDiagnosticError, CompileError), match=message):
        loads(source)


# ---------------------------------------------------------------------------
# A causal attention language model over scanned sequences
# ---------------------------------------------------------------------------

LANGUAGE_MODEL = """\
object Vocab : Real 5
object Model : Real 4
object Logits : Real 5
morphism embed : Vocab -> Model [role=function]
morphism pos : Model -> Model [role=function, param_source=positions(12)]
morphism attn : Model -> Model [role=function, param_source=self_attention, heads=2, causal=true]
morphism lm_head : Model -> Logits [role=function]
program embed_step(x_t, s_prev) : Vocab * Model -> Model
    let e = embed(x_t)
    return e
program head_step(h_t, s_prev) : Model * Model -> Model
    let logits = lm_head(h_t)
    observe next_token <- Categorical(logits=logits)
    return h_t
define embedded = scan(embed_step, outputs=all)
define scored = scan(head_step, outputs=all)
program lm : Vocab -> Model
    sample e <- embedded
    let x = pos(e)
    let h = x + attn(x)
    sample out <- scored(h)
    return out
export lm
"""


def _language_model_weights(program: Program) -> dict[str, torch.Tensor]:
    """The language model's weights by their checked input names."""
    morphism = program.morphism
    assert isinstance(morphism, torch.nn.Module)
    found: dict[str, torch.Tensor] = {}
    for name in ("embed", "lm_head"):
        (linear,) = {
            module.source.linear
            for path, module in morphism.named_modules()
            if path.split(".")[-1] == f"_function_{name}"
        }
        found[f"{name}_param_weight"] = linear.weight.detach()
        found[f"{name}_param_bias"] = linear.bias.detach()
    attention = _program_function(program, "attn").source
    assert isinstance(attention, SelfAttentionSource)
    positions = _program_function(program, "pos").source
    assert isinstance(positions, PositionSource)
    found.update(_attention_weights(attention, "attn"))
    found["pos_param_positions"] = positions.table.detach()
    return found


def _language_model_oracle(
    weights: Mapping[str, torch.Tensor],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    mask: Mask,
) -> float:
    """The language model's log joint for one run, by hand, in float64."""
    w = {k: v.double() for k, v in weights.items()}
    onehot = torch.nn.functional.one_hot(tokens, 5).double()
    e = onehot @ w["embed_param_weight"].T + w["embed_param_bias"]
    x = e + w["pos_param_positions"][: tokens.shape[0]]
    h = x + _attention_oracle(x, w, "attn", 2, mask)
    logits = h @ w["lm_head_param_weight"].T + w["lm_head_param_bias"]
    return sum(
        float(D.Categorical(logits=logits[t]).log_prob(targets[t]))
        for t in range(tokens.shape[0])
    )


def test_the_language_model_scores_alike_on_both_paths_and_by_hand() -> None:
    program = loads(LANGUAGE_MODEL)
    _scramble(program.morphism, 0.8, seed=10)  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randint(0, 5, (2, 7), generator=generator)
    targets = torch.randint(0, 5, (2, 7), generator=generator)
    onehot = torch.nn.functional.one_hot(tokens, 5).float()
    compiled = program.morphism.log_joint(onehot, {"next_token": targets})  # type: ignore[union-attr]
    weights = _language_model_weights(program)
    for run in range(2):
        oracle = _language_model_oracle(weights, tokens[run], targets[run], _causal)
        assert float(compiled[run].detach()) == pytest.approx(oracle, rel=1e-5)
        replay = program.run(
            "lm",
            data={
                "vocab": _host(onehot[run]),
                "next_token": tuple(int(t) for t in targets[run]),
                **{k: _host(v.double()) for k, v in weights.items()},
            },
            sites={},
            seed=0,
        )
        assert replay.log_joint == pytest.approx(oracle, rel=1e-9)
        for wrong in (_open, _one_ahead):
            rejected = _language_model_oracle(weights, tokens[run], targets[run], wrong)
            assert abs(rejected - oracle) > 1e-5 * abs(oracle)


def test_gradients_reach_every_weight_of_the_language_model() -> None:
    program = loads(LANGUAGE_MODEL)
    generator = torch.Generator().manual_seed(12)
    tokens = torch.randint(0, 5, (3, 6), generator=generator)
    targets = torch.randint(0, 5, (3, 6), generator=generator)
    model = program.morphism
    assert isinstance(model, torch.nn.Module)
    joint = model.log_joint(  # type: ignore[operator]
        torch.nn.functional.one_hot(tokens, 5).float(), {"next_token": targets}
    )
    joint.sum().backward()
    parameters = dict(model.named_parameters())
    attention = {name for name in parameters if "_function_attn." in name}
    assert len(attention) == 8
    assert any("_function_pos." in name for name in parameters)
    # Only the first six rows of the twelve-row table are read.
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        if name.endswith("_function_pos.source.table"):
            assert float(parameter.grad[:6].abs().sum()) > 0
            assert float(parameter.grad[6:].abs().sum()) == 0
        else:
            assert float(parameter.grad.abs().sum()) > 0, name


def test_a_sequence_function_in_a_chain_is_refused() -> None:
    source = LANGUAGE_MODEL.replace(
        "define scored = scan(head_step, outputs=all)",
        "define scored = scan(head_step, outputs=all)\n"
        "define chained = scan(embed_step, outputs=all) >> attn >> "
        "scan(head_step, outputs=all)",
    ).replace("    sample out <- scored(h)", "    sample out <- chained")
    with pytest.raises(CompileError, match="chain:sequence-function:attn"):
        loads(source)


def test_drawing_from_a_sequence_function_is_refused() -> None:
    source = HOST_CALL.replace("let h = attn(x)", "sample h <- attn(x)")
    with pytest.raises(
        (QiecDiagnosticError, CompileError), match="'attn' is deterministic"
    ):
        loads(source)


# ---------------------------------------------------------------------------
# Transpilation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declaration", "kind"),
    [
        (
            "param_source=self_attention, heads=2, causal=true",
            "param-source:self_attention",
        ),
        ("param_source=positions(8)", "param-source:positions"),
    ],
)
def test_every_target_refuses_a_sequence_function_by_its_network(
    declaration: str, kind: str
) -> None:
    source = (
        "object Model : Real 4\nobject Val : Real 1\n"
        f"morphism f : Model -> Model [role=function, {declaration}]\n"
        "program p : Val -> Val\n"
        "    let h = f(seq)\n"
        "    let s = sum(sum(h))\n"
        "    observe y <- Normal(loc=s, scale=1.0)\n"
        "    return y\n"
        "export p\n"
    )
    for target in available_targets():
        with pytest.raises(UnsupportedConstruct) as caught:
            transpile(parse(source), target=target)
        reported = {k.partition(": ")[0] for k in caught.value.kinds}
        assert kind in reported, (target, caught.value.kinds)
