"""The reference machine and the compiled runtime score programs alike.

A program that compiles to steps can also run as its checked computation:
a `CheckedProgram` of the same module and entry runs it on the reference
machine with tensor values. On every such program the two must agree, at
points that vary every site: the sites a run reaches, each site's density,
the joint, its gradient in every site value and datum, and a replay of a
run's own trace. The programs cover marginalized draws, a scan, a
deduction with learned weights, calls under authored handlers, and
repeated calls of one computation.

Agreement of two library paths is not an oracle; each path is held to an
independent oracle elsewhere (enumeration, closed forms, differences).
This check holds them to each other, so a change of either is seen.
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Mapping

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.dsl import loads
from quivers.effects.checked_program import CheckedProgram
from quivers.effects.replay import replay
from quivers.inference import Trace, trace
from tests.evidence.checks._registry import add
from tests.evidence.checks.consumers import PROBLEMS
from tests.evidence.checks.span import double_precision
from tests.evidence.compare import (
    ELEMENTARY,
    Accuracy,
    Mismatch,
    compare_number,
    compare_tensor,
)
from tests.evidence.sources import SOURCES

FIXTURES = pathlib.Path(__file__).parents[2] / "fixtures" / "qiec"
"""The checked fixture corpus."""

POINTS = 3
"""The points each program is compared at: runs drawn from three seeds."""


class Pair(ValueModel):
    """A program run both ways, with the data it is given.

    Parameters
    ----------
    name : str
        The program.
    compiled : MonadicProgram
        It, on the compiled runtime.
    checked : CheckedProgram
        It, on the reference machine.
    x : torch.Tensor
        The input.
    data : dict of str to torch.Tensor
        The observations and data.
    """

    name: str
    compiled: MonadicProgram = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    checked: CheckedProgram = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    x: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    data: dict[str, torch.Tensor] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


def _source(name: str) -> str:
    """A program's text, from the harness sources or the fixture corpus.

    Parameters
    ----------
    name : str
        The program.

    Returns
    -------
    str
        The source.
    """
    if name in SOURCES:
        return SOURCES[name]
    return (FIXTURES / f"{name}.qvr").read_text()


def _data(name: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """A program's input and the data both runtimes are given.

    Parameters
    ----------
    name : str
        The program.

    Returns
    -------
    tuple of (torch.Tensor, dict of str to torch.Tensor)
        The input and the data.
    """
    if name in PROBLEMS:
        x, data = PROBLEMS[name].data()
        return x, {key: value for key, value in data.items() if key != "features"}
    if name in ("conjugate", "calls"):
        return torch.zeros(1, 1), {"y": torch.tensor([0.25])}
    if name == "grouped":
        return torch.zeros(1, 1), {
            "y": torch.tensor([True, False, True, True]),
            "idx": torch.tensor([0, 1, 1, 0]),
        }
    if name == "discharged_handlers":
        return torch.zeros(1, 1), {"y": torch.tensor([[0.5, -0.25, 1.0, 0.0]])}
    raise KeyError(name)


def pair(name: str) -> Pair:
    """A program on both runtimes.

    Parameters
    ----------
    name : str
        The program.

    Returns
    -------
    Pair
        The pair.
    """
    compiled = loads(_source(name))
    program = compiled.morphism
    module = compiled.qiec
    assert isinstance(program, MonadicProgram) and module is not None
    entry = module.entries[-1].name
    x, data = _data(name)
    return Pair(
        name=name,
        compiled=program,
        checked=CheckedProgram(program.domain, program.codomain, module, entry),
        x=x,
        data=data,
    )


def _latents(recorded: Trace) -> dict[str, torch.Tensor]:
    """The values of a trace's drawn sites.

    Parameters
    ----------
    recorded : Trace
        The trace.

    Returns
    -------
    dict of str to torch.Tensor
        The value of every site drawn, not observed.
    """
    return {
        name: site.value.detach()
        for name, site in recorded.sites.items()
        if site.morphism is not None and not site.is_observed
    }


def _layout(values: Mapping[str, torch.Tensor], like: Trace) -> dict[str, torch.Tensor]:
    """Values laid out as another trace holds them.

    Parameters
    ----------
    values : Mapping of str to torch.Tensor
        Values by site.
    like : Trace
        A trace of the other runtime.

    Returns
    -------
    dict of str to torch.Tensor
        Each value with the shape the other trace gives the site; the
        entries are the same.
    """
    return {
        name: value.reshape(like.sites[name].value.shape)
        for name, value in values.items()
    }


def check_equivalence(name: str) -> None:
    """Sites, densities, joints, gradients and replays agree on a program.

    Parameters
    ----------
    name : str
        The program.

    Raises
    ------
    Mismatch
        If the runtimes reach different sites, or a drawn site's density,
        the joint (which holds every observation's density), a gradient or
        a replayed joint differs beyond rounding.
    """
    with double_precision():
        both = pair(name)
        accuracy = Accuracy(domain="log", operations=400 * ELEMENTARY, magnitude=64.0)
        for seed in range(POINTS):
            torch.manual_seed(seed)
            compiled = trace(both.compiled, both.x, observations=both.data)
            checked = trace(both.checked, both.x, observations=both.data)
            values = _latents(compiled)
            if set(values) != set(_latents(checked)):
                raise Mismatch(
                    "support",
                    math.inf,
                    f"{name}: sites {sorted(values)} and {sorted(_latents(checked))}",
                )
            at = {**both.data, **_layout(values, checked)}
            leaves = {
                key: value.clone().requires_grad_(value.is_floating_point())
                if isinstance(value, torch.Tensor)
                else value
                for key, value in at.items()
            }
            again = trace(both.checked, both.x, observations=leaves)
            if not isinstance(again.output, torch.Tensor) or not isinstance(
                compiled.output, torch.Tensor
            ):
                raise Mismatch("support", math.inf, f"{name}: expected tensor outputs")
            # Each fixture makes one run. Its value is compared after removing
            # singleton axes used for the run or its one-element carrier.
            compare_tensor(
                again.output.squeeze(),
                compiled.output.squeeze(),
                accuracy,
                f"{name} output",
            )
            for site in values:
                compare_number(
                    float(again.sites[site].log_prob.sum()),
                    float(compiled.sites[site].log_prob.sum()),
                    accuracy,
                    f"{name} density of {site}",
                )
            reference = {
                key: value.clone().requires_grad_(value.is_floating_point())
                if isinstance(value, torch.Tensor)
                else value
                for key, value in {**both.data, **values}.items()
            }
            compiled_joint = both.compiled.log_joint(both.x, reference)
            checked_joint = both.checked.log_joint(both.x, leaves)
            compare_number(
                float(checked_joint.sum()),
                float(compiled_joint.sum()),
                accuracy,
                f"{name} joint at seed {seed}",
            )
            _compare_gradients(
                name,
                compiled_joint.sum(),
                reference,
                checked_joint.sum(),
                leaves,
                accuracy,
            )
            with replay(checked):
                replayed = trace(both.checked, both.x, observations=both.data)
            assert replayed.log_joint is not None and checked.log_joint is not None
            compare_number(
                float(replayed.log_joint.sum()),
                float(checked.log_joint.sum()),
                accuracy,
                f"{name} replay of its own trace",
            )


def _compare_gradients(
    name: str,
    compiled: torch.Tensor,
    compiled_leaves: Mapping[str, torch.Tensor],
    checked: torch.Tensor,
    checked_leaves: Mapping[str, torch.Tensor],
    accuracy: Accuracy,
) -> None:
    """The joints' gradients in every real site value and datum, entry by entry.

    Parameters
    ----------
    name : str
        The program.
    compiled : torch.Tensor
        The compiled runtime's joint.
    compiled_leaves : Mapping of str to torch.Tensor
        Its site values and data.
    checked : torch.Tensor
        The machine's joint.
    checked_leaves : Mapping of str to torch.Tensor
        Its site values and data, the same entries in its layout.
    accuracy : Accuracy
        The accuracy.

    Raises
    ------
    Mismatch
        If a gradient is missing on one side or differs.
    """
    names = [
        key
        for key, leaf in compiled_leaves.items()
        if isinstance(leaf, torch.Tensor) and leaf.requires_grad
    ]
    if not names:
        return
    found = torch.autograd.grad(
        compiled, [compiled_leaves[key] for key in names], allow_unused=True
    )
    machine = torch.autograd.grad(
        checked, [checked_leaves[key] for key in names], allow_unused=True
    )
    for key, left, right in zip(names, found, machine, strict=True):
        if (left is None) != (right is None):
            raise Mismatch("missing", math.inf, f"{name} gradient of {key}")
        if left is None or right is None:
            continue
        for entry, (a, b) in enumerate(
            zip(left.reshape(-1).tolist(), right.reshape(-1).tolist(), strict=True)
        ):
            compare_number(b, a, accuracy, f"{name} gradient of {key}[{entry}]")


PROGRAMS = (
    "conjugate",
    "beta",
    "discrete",
    "ungrouped",
    "grouped",
    "hmm",
    "parsed",
    "discharged_handlers",
    "calls",
)
"""The programs both runtimes run."""


def _register() -> None:
    """Register one check per program."""
    for name in PROGRAMS:
        add(
            f"equivalence/{name}",
            "machine-program",
            "Q24",
            lambda name=name: check_equivalence(name),
        )


_register()

__all__ = ["PROGRAMS", "Pair", "check_equivalence", "pair"]
