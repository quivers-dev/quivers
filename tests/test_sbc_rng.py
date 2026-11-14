"""CPU and accelerator random-state isolation during calibration."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

import pytest
import torch

from tests.test_diagnostics_sbc import _gaussian_fit, _run


def _devices() -> tuple[torch.device, ...]:
    """Return CPU and the CUDA or MPS devices available to this test.

    Returns
    -------
    tuple[torch.device, ...]
        Devices on which actual random sampling can run.
    """
    devices = [torch.device("cpu")]
    devices.extend(torch.device("cuda", i) for i in range(torch.cuda.device_count()))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps", 0))
    return tuple(devices)


@contextmanager
def _restore_states() -> Iterator[None]:
    """Restore caller state even when a regression assertion fails.

    Yields
    ------
    None
        A test scope that saves CPU and available accelerator generators.
    """
    with ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng())
        if torch.backends.mps.is_available():
            stack.enter_context(torch.random.fork_rng(device_type="mps", devices=[0]))
        yield


@pytest.mark.parametrize("fail", [False, True])
def test_sbc_restores_caller_rng_states_on_success_and_failure(fail: bool) -> None:
    """A CPU calibration run preserves CPU and actual accelerator streams.

    Parameters
    ----------
    fail : bool
        Whether the fitting callback raises after drawing random values.
    """
    devices = _devices()
    with _restore_states():
        before = {
            device: (
                torch.get_rng_state()
                if device.type == "cpu"
                else getattr(torch, device.type).get_rng_state(device.index)
            )
            for device in devices
        }

        def fit(
            observations: dict[str, torch.Tensor], seed: int
        ) -> dict[str, torch.Tensor]:
            """Draw on every tested device, then fit or raise.

            Parameters
            ----------
            observations : dict[str, torch.Tensor]
                Simulated observed data.
            seed : int
                Seed passed by calibration.

            Returns
            -------
            dict[str, torch.Tensor]
                Exact Gaussian posterior draws.

            Raises
            ------
            RuntimeError
                If the failure path is requested.
            """
            for device in devices:
                torch.randn(5, device=device)
            if fail:
                raise RuntimeError("deliberate fit failure")
            return _gaussian_fit()(observations, seed)

        if fail:
            with pytest.raises(RuntimeError, match="deliberate fit failure"):
                _run(fit, 72, num_simulations=2, thin=1)
        else:
            _run(fit, 72, num_simulations=2, thin=1)
        for device in devices:
            after = (
                torch.get_rng_state()
                if device.type == "cpu"
                else getattr(torch, device.type).get_rng_state(device.index)
            )
            assert torch.equal(before[device], after), str(device)


def test_sbc_callback_streams_match_independently_seeded_device_generators() -> None:
    """Global callback draws equal a fresh local generator on each device."""
    devices = _devices()
    with _restore_states():

        def fit(
            observations: dict[str, torch.Tensor], seed: int
        ) -> dict[str, torch.Tensor]:
            """Check each callback stream against an independent generator.

            Parameters
            ----------
            observations : dict[str, torch.Tensor]
                Simulated observed data.
            seed : int
                The callback's independently derived seed.

            Returns
            -------
            dict[str, torch.Tensor]
                Exact Gaussian posterior draws.
            """
            for device in devices:
                expected = torch.randn(
                    12,
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(seed),
                )
                actual = torch.randn(12, device=device)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            return _gaussian_fit()(observations, seed)

        _run(fit, 73, num_simulations=2, thin=1)
