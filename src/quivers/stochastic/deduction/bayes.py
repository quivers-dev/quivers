"""Bayesian posterior wrapping for weighted deduction systems.

`nuts_program_from_deduction` lifts the deduction's
learnable log-weights into a
[`quivers.continuous.programs.MonadicProgram`][quivers.continuous.programs.MonadicProgram] whose
``log_joint`` is
:math:`-\\tfrac{1}{2\\sigma^2}\\lVert \\mathbf{w} \\rVert^2
+ \\sum_n \\log Z(s_n; \\mathbf{w})`. The resulting program is
ready for [`quivers.inference.MCMC`][quivers.inference.MCMC] with
[`quivers.inference.NUTSKernel`][quivers.inference.NUTSKernel].

Modelling note
--------------

The sampler targets exactly
:math:`\\pi(\\mathbf{w}) \\propto \\exp(-\\lVert \\mathbf{w}
\\rVert^2/(2\\sigma^2) + \\sum_n \\log Z(s_n; \\mathbf{w}))` with a
deterministic log-density and exact gradients. Whether that joint
is the Bayesian posterior :math:`p(\\mathbf{w} \\mid S)` depends
on the modelling reading:

* **Undirected / globally-normalised** (CRF / log-linear /
  energy-based): :math:`\\pi(\\mathbf{w})` is the posterior; the
  implementation is exact.
* **Directed / locally-normalised PCFG**: the true sentence
  likelihood is :math:`Z(s; \\mathbf{w}) / \\sum_{s'} Z(s';
  \\mathbf{w})`; the global normaliser depends on
  :math:`\\mathbf{w}` and is intractable. The sampler then targets
  a *pseudo-posterior* differing from the true posterior by a
  factor of :math:`\\bigl(\\sum_{s'} Z(s'; \\mathbf{w})\\bigr)^{-N}`.
  Users committed to this reading should constrain rule weights
  to local simplices via a Dirichlet + softmax surface rather
  than the free-parameter Normal lift this function provides.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from quivers.continuous.program_steps import Draw, Score, Step, reading
from quivers.continuous.programs import MonadicProgram
from quivers.core.objects import Unit
from quivers.stochastic.deduction._internal import build_locator, materialise_parameters
from quivers.inference.lifts import (
    _make_normal_prior_morphism,
    _swap_named_parameters,
)
from quivers.stochastic.agenda import DeductionSystem


__all__ = ["nuts_program_from_deduction"]


def nuts_program_from_deduction(
    ded: DeductionSystem,
    corpus: Sequence[Sequence[str]],
    *,
    prior_scale: float = 1.0,
    site_prefix: str = "log_w",
) -> tuple[MonadicProgram, torch.Tensor, dict[str, torch.Tensor]]:
    """Lift a deduction system's learnable parameters to a
    `MonadicProgram` suitable for NUTS / SVI.

    The returned program has one
    `torch.distributions.Normal` sample site per learnable
    parameter (lexicon entries and rule bindings alike) plus one
    score step that substitutes the sampled values into the
    deduction's parameter slots and adds
    :math:`\\sum_n \\log Z(s_n; \\mathbf{w})` to the joint.

    Parameters
    ----------
    ded : DeductionSystem
        Deduction whose parameters are lifted.
    corpus : sequence of sentences
        Corpus the score step closes over.
    prior_scale : float
        Standard deviation of the Normal prior on each parameter.
    site_prefix : str
        Stem of each sample-site's name (the parameter's path is
        appended for round-trip mapping).

    Returns
    -------
    (model, x, observations)
        The lifted program plus a ``(1, 1)`` placeholder input and
        an empty observation dict, ready to feed to
        [`quivers.inference.MCMC`][quivers.inference.MCMC] ``.run``.

    Raises
    ------
    ValueError
        If the deduction has no learnable parameters.
    """
    materialise_parameters(ded, corpus)
    locator, paths, _ = build_locator(ded)
    if not paths:
        raise ValueError(
            "nuts_program_from_deduction: the deduction has no "
            "learnable parameters (neither lexicon nor rules are "
            "marked #[learnable])"
        )
    site_names: list[str] = []
    for path in paths:
        safe = path.replace("/", "__").replace(".", "_")
        site_names.append(f"{site_prefix}__{safe}")

    prior_morph = _make_normal_prior_morphism(prior_scale)
    steps: list[Step] = [
        Draw(names=(site,), morphism=prior_morph) for site in site_names
    ]

    def _score_fn(
        env: dict[str, torch.Tensor],
        _ded: DeductionSystem = ded,
        _corpus: list[list[str]] = [list(s) for s in corpus],
        _site_names: list[str] = list(site_names),
        _paths: list[str] = list(paths),
        _locator: Callable[[str], tuple[torch.nn.Module, str]] = locator,
    ) -> torch.Tensor:
        # A score step reads one run's values, each site a single draw.
        overrides = {
            path: env[name].reshape(()) for path, name in zip(_paths, _site_names)
        }
        with _swap_named_parameters(_locator, overrides):
            log_z = torch.zeros(())
            for sentence in _corpus:
                log_z = log_z + _ded(sentence).goal_weight()
        return log_z.reshape(1)

    steps.append(Score(name="log_Z", score=reading(_score_fn, site_names)))
    model = MonadicProgram(
        domain=Unit,
        codomain=Unit,
        steps=steps,
        return_vars=("log_Z",),
    )
    return model, torch.zeros(1, 1), {}
