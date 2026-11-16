"""Seeded posterior collection shares device-local RNG isolation with calibration."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

import pytest
import torch

from quivers._rng import seeded_rng
from quivers.diagnostics.calibration import _seeded_rng
from quivers.dsl import loads
from quivers.inference import AutoNormalGuide
from quivers.inference._posterior import constrained_draws
from quivers.inference._response import response_rng


DEVICES = [
    pytest.param(torch.device("cpu"), id="cpu"),
    pytest.param(
        torch.device("cuda", 0),
        id="cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA hardware unavailable"
        ),
    ),
    pytest.param(
        torch.device("mps", 0),
        id="mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS hardware unavailable"
        ),
    ),
]


def _guide():
    program = loads("""object Out : Real 1
program model : Out -> Out
    sample z <- Normal(loc=0.0,scale=1.0)
    return z
export model
""").morphism
    return AutoNormalGuide(program, observed_names=set())


def _state(device):
    return (
        torch.get_rng_state().clone()
        if device.type == "cpu"
        else getattr(torch, device.type).get_rng_state(device.index).clone()
    )


@contextmanager
def _restore(device) -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))
        if device.type != "cpu":
            stack.enter_context(
                torch.random.fork_rng(
                    devices=[device.index or 0], device_type=device.type
                )
            )
        yield


def test_calibration_keeps_the_canonical_scope_as_its_private_alias():
    assert _seeded_rng is seeded_rng


def test_cpu_guide_uses_no_accelerator_probe_or_initialization(monkeypatch):
    guide = _guide()

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU posterior collection touched an accelerator")

    for name in ("cuda", "mps", "xpu", "mtia"):
        module = getattr(torch, name, None)
        if module is not None:
            for operation in ("device_count", "get_rng_state", "set_rng_state"):
                if hasattr(module, operation):
                    monkeypatch.setattr(module, operation, forbidden)
    result = constrained_draws(guide, torch.zeros(1, 1), num_samples=3, seed=12)
    assert result["z"].shape[:2] == (1, 3)


@pytest.mark.parametrize("device", DEVICES)
def test_real_guide_draws_restore_all_participating_streams_and_repeat(device):
    guide = _guide().to(device)
    with _restore(device):
        before_cpu = _state(torch.device("cpu"))
        before_device = _state(device)
        first = constrained_draws(guide, torch.zeros(1, 1), num_samples=5, seed=31)
        assert torch.equal(before_cpu, _state(torch.device("cpu")))
        assert torch.equal(before_device, _state(device))
        torch.randn(7, device=device)
        second = constrained_draws(guide, torch.zeros(1, 1), num_samples=5, seed=31)
        torch.testing.assert_close(first["z"], second["z"], rtol=0, atol=0)
        other = constrained_draws(guide, torch.zeros(1, 1), num_samples=5, seed=32)
        assert not torch.equal(first["z"], other["z"])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("source", ["input", "parameters", "buffers"])
def test_requested_input_parameter_and_buffer_devices_match_local_generators(
    device, source, monkeypatch
):
    guide = _guide()
    if source == "parameters":
        guide.to(device)
    elif source == "buffers":
        guide.register_buffer("rng_probe", torch.empty(0, device=device))
    input = torch.zeros(1, 1, device=device if source == "input" else "cpu")
    shape = guide.registry["z"].constrained_shape

    def draw(x):
        torch.randn(2)  # A guide may use both host and device streams.
        return {"z": torch.randn(shape, device=device)}

    monkeypatch.setattr(guide, "rsample", draw)
    with _restore(device):
        before_cpu = _state(torch.device("cpu"))
        before_device = _state(device)
        actual = constrained_draws(guide, input, num_samples=4, seed=17)["z"]
        assert torch.equal(before_cpu, _state(torch.device("cpu")))
        assert torch.equal(before_device, _state(device))
        local = torch.Generator(device=device).manual_seed(17)
        expected = []
        for _ in range(4):
            if device.type == "cpu":
                torch.randn(2, generator=local)
            expected.append(torch.randn(shape, device=device, generator=local))
        torch.testing.assert_close(
            actual, torch.stack(expected).unsqueeze(0), rtol=0, atol=0
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("failure", ["callback", "names", "shape"])
def test_guide_callback_and_validation_errors_restore_cpu_and_device_streams(
    device, failure, monkeypatch
):
    guide = _guide().to(device)

    def fail(x):
        torch.randn(5)
        torch.randn(7, device=device)
        if failure == "callback":
            raise RuntimeError("deliberate guide failure")
        return (
            {"unknown": torch.zeros(1, device=device)}
            if failure == "names"
            else {"z": torch.zeros(19, device=device)}
        )

    monkeypatch.setattr(guide, "rsample", fail)
    with _restore(device):
        before_cpu = _state(torch.device("cpu"))
        before_device = _state(device)
        with pytest.raises(
            RuntimeError if failure == "callback" else ValueError,
            match="failure|sites|shape",
        ):
            constrained_draws(guide, torch.zeros(1, 1), num_samples=2, seed=8)
        assert torch.equal(before_cpu, _state(torch.device("cpu")))
        assert torch.equal(before_device, _state(device))


@pytest.mark.parametrize("device", DEVICES)
def test_none_seed_consumes_existing_device_stream_without_reset(device, monkeypatch):
    guide = _guide().to(device)
    shape = guide.registry["z"].constrained_shape
    monkeypatch.setattr(
        guide, "rsample", lambda x: {"z": torch.randn(shape, device=device)}
    )
    with _restore(device):
        local = torch.Generator(device=device).manual_seed(19)
        if device.type == "cpu":
            torch.set_rng_state(local.get_state())
        else:
            getattr(torch, device.type).set_rng_state(local.get_state(), device.index)
        expected = torch.stack(
            [torch.randn(shape, device=device, generator=local) for _ in range(3)]
        ).unsqueeze(0)
        actual = constrained_draws(guide, torch.zeros(1, 1), num_samples=3, seed=None)[
            "z"
        ]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(_state(device), local.get_state())


def test_nested_response_scopes_restore_the_enclosing_stream():
    cpu = torch.empty(0)
    before = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(15)
    expected = torch.randn(6, generator=generator)
    with response_rng(15, (cpu,)):
        first = torch.randn(3)
        with response_rng(33, (cpu,)):
            torch.testing.assert_close(
                torch.randn(3),
                torch.randn(3, generator=torch.Generator().manual_seed(33)),
                rtol=0,
                atol=0,
            )
        second = torch.randn(3)
    torch.testing.assert_close(torch.cat((first, second)), expected, rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("device", DEVICES)
def test_none_seed_does_not_rewind_caller_stream_when_guide_raises(device, monkeypatch):
    guide = _guide().to(device)

    def fail(x):
        torch.randn(3, device=device)
        raise RuntimeError("caller stream failure")

    monkeypatch.setattr(guide, "rsample", fail)
    with _restore(device):
        local = torch.Generator(device=device).manual_seed(23)
        if device.type == "cpu":
            torch.set_rng_state(local.get_state())
        else:
            getattr(torch, device.type).set_rng_state(local.get_state(), device.index)
        torch.randn(3, device=device, generator=local)
        with pytest.raises(RuntimeError, match="caller stream failure"):
            constrained_draws(guide, torch.zeros(1, 1), num_samples=2, seed=None)
        assert torch.equal(_state(device), local.get_state())
