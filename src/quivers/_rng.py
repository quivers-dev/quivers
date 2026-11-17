"""Seeded random-stream scopes shared by inference and diagnostics."""

from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Protocol, cast

import torch


class _RngBackend(Protocol):
    """The device-state operations used by a seeded scope."""

    def device_count(self) -> int:
        """Return the number of available devices.

        Returns
        -------
        int
            Devices whose random state can be saved and restored.
        """
        ...

    def set_rng_state(self, state: torch.Tensor, device: int) -> None:
        """Set one device's generator state.

        Parameters
        ----------
        state : torch.Tensor
            State produced by a generator for this backend.
        device : int
            Index of the device whose state is set.
        """
        ...


@contextmanager
def seeded_rng(
    seed: int, *, devices: Iterable[torch.device] | None = None
) -> Iterator[None]:
    """Isolate CPU and selected accelerator random streams.

    Local generators provide seeded states for the global generators used
    by distribution samplers and fitting callbacks. Every modified state
    is restored on exit, including when a callback raises.

    Parameters
    ----------
    seed : int
        Seed used independently on CPU and each selected device, modulo
        the generator's 64-bit seed range.
    devices : Iterable[torch.device] or None
        Participating devices; CPU is always included. An explicit sequence
        touches only its accelerator devices, without enumerating or
        initializing other backends. None selects every available accelerator,
        as required for callbacks whose device use is unknown in advance.

    Yields
    ------
    None
        A scope in which global draws use the explicit seed.

    Raises
    ------
    RuntimeError
        If a selected backend cannot construct or restore a generator.
    """
    selected: dict[str, tuple[int, ...]] = {}
    if devices is None:
        backend_names = dict.fromkeys(
            ("cuda", "mps", "xpu", "mtia", torch._C._get_privateuse1_backend_name())
        )
        for name in backend_names:
            module = getattr(torch, name, None)
            if module is not None:
                selected[name] = tuple(range(cast(_RngBackend, module).device_count()))
    else:
        requested: dict[str, set[int]] = {}
        for device in devices:
            if device.type not in {"cpu", "meta"}:
                requested.setdefault(device.type, set()).add(device.index or 0)
        selected = {name: tuple(sorted(indices)) for name, indices in requested.items()}
    with ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))
        backends: list[tuple[str, _RngBackend, tuple[int, ...]]] = []
        for name, indices in selected.items():
            if not indices:
                continue
            backend = cast(_RngBackend, getattr(torch, name))
            stack.enter_context(
                torch.random.fork_rng(devices=indices, device_type=name)
            )
            backends.append((name, backend, indices))
        seed %= 2**64
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        for name, backend, indices in backends:
            for index in indices:
                generator = torch.Generator(device=torch.device(name, index))
                backend.set_rng_state(generator.manual_seed(seed).get_state(), index)
        yield


__all__: tuple[str, ...] = ()
