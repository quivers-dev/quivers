"""Internal helpers shared by the deduction package.

Two responsibilities:

* `build_locator` walks a `DeductionSystem`'s
  ``_axiom_module`` and ``_rule_module`` to produce a
  ``path -> (parent_module, attribute_name)`` lookup plus the
  ordered list of parameter paths and their current
  `torch.nn.Parameter`\\ s. The locator is consumed by
  the shared [`quivers.inference.lifts._swap_named_parameters`][quivers.inference.lifts._swap_named_parameters]
  context manager to swap NUTS-proposed values into the
  deduction's parameter slots.
* `materialise_parameters` runs the deduction once on each
  corpus sentence so that any lazy rule-weight ``ParameterDict``
  observes every binding it will see at fit time and allocates the
  corresponding `torch.nn.Parameter`\\ s. This guarantees
  the parameter set is stable when the fitting / posterior
  machinery enumerates it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from quivers.stochastic.agenda import DeductionSystem


__all__ = ["build_locator", "materialise_parameters"]


def build_locator(
    ded: DeductionSystem,
) -> tuple[
    Callable[[str], tuple[torch.nn.Module, str]],
    list[str],
    list[torch.nn.Parameter],
]:
    """Return ``(locator, paths, params)`` over the deduction's
    learnable parameter set."""
    paths: list[str] = []
    params: list[torch.nn.Parameter] = []
    locations: dict[str, tuple[torch.nn.Module, str]] = {}
    for module_name in ("_axiom_module", "_rule_module"):
        mod = ded.modules.get(module_name)
        if mod is None:
            continue
        for path_in_mod, p in mod.named_parameters():
            full_path = f"{module_name}/{path_in_mod}"
            paths.append(full_path)
            params.append(p)
            parts = path_in_mod.split(".")
            parent: torch.nn.Module = mod
            for part in parts[:-1]:
                parent = getattr(parent, part)
            locations[full_path] = (parent, parts[-1])

    def locator(path: str) -> tuple[torch.nn.Module, str]:
        return locations[path]

    return locator, paths, params


def materialise_parameters(
    ded: DeductionSystem,
    corpus: Sequence[Sequence[str]],
) -> None:
    """Touch every sentence in ``corpus`` once so the deduction's
    lazy rule-weight ``ParameterDict``\\ s observe every binding
    they will see at fit time and allocate the corresponding
    `torch.nn.Parameter`\\ s."""
    for sentence in corpus:
        chart = ded(list(sentence))
        _ = chart.goal_weight()
