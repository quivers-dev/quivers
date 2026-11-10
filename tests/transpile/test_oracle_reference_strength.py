"""Verify QVR reference log densities independently.

Backend equivalence is measured relative to the QVR oracle, so that
comparison cannot also validate the oracle. Every pinned gallery
example must instead have one of two witnesses:

1. A live backend container re-evaluates the model at every point.
2. This module reconstructs the joint with raw
   `torch.distributions` calls when no backend cell is available.

The raw-torch reconstructions compare every stochastic site log-density
as well as the joint, preventing compensating site errors from
canceling. They follow the oracle constrained-space convention, sum
over plates, and use `logsumexp` for marginalized discrete latents. A
mutation catalogue checks that each reconstruction rejects specified
statistical defects by a measured margin.

Examples without a pin are classified separately. Some compositions
produce a reproducible quadrature approximation without an independent
witness; others assign zero density to a composition-bound latent. The
tests re-measure those grounds and check program-free examples against
the structural exemption registry. They also verify that flat
plate-shaped terms do not duplicate scalar site contributions during
joint reduction.
"""

from __future__ import annotations

import functools
import math
import pathlib
from collections.abc import Callable

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F
from torch import distributions as td

from quivers.dsl.ast_nodes.declarations import ProgramDecl
from quivers.dsl.parser import parse
from quivers.effects.trace_types import Trace
from tests.transpile import _gallery_data
from tests.transpile import test_gallery_numeric_equivalence as _gallery_tier
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.qvr import (
    assert_reference_joint_deterministic,
    reference_traces,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SOURCE_DIR = _REPO_ROOT / "docs" / "examples" / "source"

# `ConditionalNormal` clamps its scale at this floor before scoring.
# Reproduced here as a constant rather than imported so the
# reconstruction does not read the value out of the module it is
# checking; a change to the runtime floor must show up as a
# disagreement, not be absorbed silently.
_SCALE_FLOOR = 1e-7


# ---------------------------------------------------------------------
# Point marshalling.
#
# The reconstruction has to score the same numbers the oracle scored,
# so it inflates the wire `Point` the same way. This is deliberately a
# second implementation rather than a call into
# `tests.transpile.probes.qvr.clamping_observations`: the two must
# agree, and a disagreement is a real ambiguity in how a point reaches
# the oracle, not something to paper over by sharing one code path.
# Nothing here computes a density.
# ---------------------------------------------------------------------


def _flat_tensor(
    value: float | int | list[float] | list[int],
) -> Tensor:
    """Inflate one wire entry to the 1-D float32 tensor the oracle
    clamps with."""
    if isinstance(value, (int, float)):
        return torch.tensor([float(value)], dtype=torch.float32)
    return torch.tensor([float(v) for v in value], dtype=torch.float32)


def _clamped_values(
    dataset: _gallery_data.GalleryDataset,
    point: Point,
) -> dict[str, Tensor]:
    """Every clamped value the oracle sees at `point`, by site name.

    Latents arrive flat (the wire discards their axes and the oracle
    re-derives them); observations arrive pre-shaped through
    [`observations_for_point`][tests.transpile._gallery_data.observations_for_point],
    which is the only channel that preserves a multi-axis observation
    and takes precedence over the flat payload in the oracle too.
    """
    values: dict[str, Tensor] = {}
    for name in dataset.params:
        if name not in point.params:
            raise AssertionError(
                f"latent {name!r} is captured in the dataset but absent "
                f"from the point's params section, so the "
                f"reconstruction has no value to score it at."
            )
        values[name] = _flat_tensor(point.params[name])
    for name, tensor in _gallery_data.observations_for_point(
        dataset,
        point,
    ).items():
        values[name] = tensor
    return values


# ---------------------------------------------------------------------
# Raw density primitives.
#
# Written out rather than taken from `quivers.continuous.families`,
# which is the code under test. `torch.distributions` supplies every
# family the six examples name; the two hand-written helpers below
# exist only where the runtime's parameterisation (a network head
# emitting a location and a log-scale side by side) has no
# `torch.distributions` counterpart.
# ---------------------------------------------------------------------


def _affine(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """`x W^T + b`, written as a matmul rather than through
    `torch.nn.functional.linear`."""
    return x @ weight.transpose(-1, -2) + bias


def _normal_head(
    raw: Tensor,
    event_dim: int,
    unit_scale: bool = False,
) -> tuple[Tensor, Tensor]:
    """Split a `(..., 2 * event_dim)` parameter row into `(loc, scale)`.

    The runtime's `ConditionalNormal` reads the first `event_dim`
    columns as the location and exponentiates the rest, clamping the
    result at `_SCALE_FLOOR`. `unit_scale` is the mutant path: a head
    whose scale columns are ignored.
    """
    loc = raw[..., :event_dim]
    if unit_scale:
        return loc, torch.ones_like(loc)
    scale = raw[..., event_dim:].exp().clamp(min=_SCALE_FLOOR)
    return loc, scale


def _unknown_variant(example: str, variant: str) -> AssertionError:
    return AssertionError(
        f"{example!r}: reconstruction asked for unknown variant "
        f"{variant!r}. A mutant whose variant name the reconstruction "
        f"does not branch on would silently score the faithful joint "
        f"and be reported as rejected-with-margin-zero; raising here "
        f"names the typo instead."
    )


# ---------------------------------------------------------------------
# Per-example reconstructions.
#
# Each returns the joint as a dict of per-site scalar contributions,
# keyed by the trace's own site names, so the comparison is summand by
# summand. `variant` selects a mutant; the empty string is the
# faithful reconstruction. Mutants live inside the same function as
# the term they corrupt so that the defect reads as a one-line
# difference from the correct term and cannot drift away from it.
# ---------------------------------------------------------------------


#: Factor widths of one `seq2seq` transformer block, in source order:
#: `fan(head)`, `attn_proj`, `residual_attn`, `ff_up`, `ff_down`,
#: `residual_ff`. `object Latent : Real 16`, `object FFHidden : Real 32`.
_SEQ2SEQ_BLOCK_WIDTHS: tuple[int, ...] = (16, 16, 16, 32, 16, 16)


def _seq2seq_block_prefixes(root: str) -> tuple[tuple[str, ...], ...]:
    """Parameter prefixes of one block, in the order the source composes.

    A block is `fan(head) >> attn_proj >> residual_attn >> ff_up >>
    ff_down >> residual_ff`, composed left-associatively, so the last
    factor sits at `right` and each earlier one gains a `left`. The
    `fan` is the head of the chain and carries its four
    `[replicate=4]` components. Each entry is the tuple of prefixes
    that factor draws its parameters from: one for an ordinary factor,
    four for the fan.
    """
    return (
        tuple(
            f"{root}.left.left.left.left.left._components.{index}.param_source"
            for index in range(4)
        ),
        (f"{root}.left.left.left.left.right.param_source",),
        (f"{root}.left.left.left.right.param_source",),
        (f"{root}.left.left.right.param_source",),
        (f"{root}.left.right.param_source",),
        (f"{root}.right.param_source",),
    )


def _seq2seq_side(
    root: str,
    tokens: Tensor,
    weights: dict[str, Tensor],
    *,
    activated: bool,
) -> tuple[Tensor, Tensor]:
    """Score one `seq2seq` tower along its canonical path.

    The tower is `embed >> stack(block, 2)`. Its embedding is a
    `Normal` whose location is the token's own centre row, so at the
    origin it is scored there and contributes its normalizer; every
    later factor is likewise bound to its own location, which is what
    the base measure's origin pushes to. Returns the summed density
    and the tower's output, which is the row the cross-attention
    factor reads.
    """
    centres = weights[root + ".left.centers"]
    scales = weights[root + ".left.log_sigma"].exp()
    embedded = centres[tokens]
    term = td.Normal(centres[tokens], scales[tokens]).log_prob(embedded)
    total = term.sum(tuple(range(1, term.dim())))

    current = embedded
    factors: list[tuple[tuple[str, ...], int]] = []
    for block in ("right.left", "right.right"):
        prefixes = _seq2seq_block_prefixes(f"{root}.{block}")
        factors += list(zip(prefixes, _SEQ2SEQ_BLOCK_WIDTHS, strict=True))

    for prefixes, width in factors:
        if len(prefixes) > 1:
            # `[replicate=4]`: four heads of `object HeadOut : Real 4`,
            # concatenated back into the 16 columns of `Latent`.
            locations, scale_columns = [], []
            for prefix in prefixes:
                head_loc, head_scale = _normal_head(
                    _mlp3_linear(
                        current,
                        weights,
                        prefix + ".net.",
                        activated=activated,
                    ),
                    4,
                )
                locations.append(head_loc)
                scale_columns.append(head_scale)
            location = torch.cat(locations, dim=-1)
            scale = torch.cat(scale_columns, dim=-1)
        else:
            location, scale = _normal_head(
                _mlp3_linear(
                    current,
                    weights,
                    prefixes[0] + ".net.",
                    activated=activated,
                ),
                width,
            )
        factor_term = td.Normal(location, scale).log_prob(location)
        total = total + factor_term.sum(tuple(range(1, factor_term.dim())))
        current = location
    return total, current


def _reconstruct_seq2seq(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`seq2seq.qvr`:

        sample h          <- backbone
        observe next_token : Resp <- lm_head(h)

    with `backbone = (encoder @ decoder) >> cross`, each tower
    `embed >> stack(block, 2)`, and each block
    `fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >>
    residual_ff`. There is no scan here: the chain is feed-forward, so
    the canonical path is a plain fold and the reconstruction agrees
    with the oracle within the calibrated float32 roundoff budget.

    The product scores both towers at the image of the base measure's
    origin and hands their concatenated outputs to `cross`, which is
    where `h` enters. `object Latent : Real 16`, `object Combined :
    Real 32`, and `enc_head` / `dec_head` carry `[replicate=4]` over
    `object HeadOut : Real 4`, so each fan rebuilds `Latent` from four
    four-column heads. The MLP and embedding weights are compile-time
    draws with no spelling in the source and are taken as given.
    """
    x_input = dataset.x_input
    if x_input is None:
        raise AssertionError(
            "seq2seq's reconstruction needs the snippet's "
            "`(source, target)` column pair, which the dataset did "
            "not carry."
        )
    known = (
        "",
        "plate_mean",
        "drop_encoder_tower",
        "drop_cross_factor",
    )
    if variant not in known:
        raise _unknown_variant("seq2seq", variant)

    activated = True
    source_tokens = x_input[..., 0]
    target_tokens = x_input[..., 1]
    encoder_term, encoder_out = _seq2seq_side(
        "_step_h.left.left",
        source_tokens,
        weights,
        activated=activated,
    )
    decoder_term, decoder_out = _seq2seq_side(
        "_step_h.left.right",
        target_tokens,
        weights,
        activated=activated,
    )

    combined = torch.cat([encoder_out, decoder_out], dim=-1)
    loc_cross, scale_cross = _normal_head(
        _mlp3_linear(
            combined,
            weights,
            "_step_h.right.param_source.net.",
            activated=activated,
        ),
        32,
    )
    cross_term = (
        td.Normal(loc_cross, scale_cross)
        .log_prob(
            values["h"],
        )
        .sum(-1)
    )

    head = "_step_next_token._family.param_source.linear."
    logits = F.linear(
        values["h"],
        weights[head + "weight"],
        weights[head + "bias"],
    )
    per_token = td.Categorical(logits=logits).log_prob(
        values["next_token"],
    )

    if variant == "drop_encoder_tower":
        encoder_term = torch.zeros_like(encoder_term)
    if variant == "drop_cross_factor":
        cross_term = torch.zeros_like(cross_term)
    latent = encoder_term + decoder_term + cross_term
    if variant == "plate_mean":
        return {"h": latent.mean(), "next_token": per_token.mean()}
    return {"h": latent.sum(), "next_token": per_token.sum()}


def _mlp3_linear(
    x: Tensor,
    weights: dict[str, Tensor],
    prefix: str,
    *,
    activated: bool = True,
) -> Tensor:
    """`_mlp3` spelled through `torch.nn.functional.linear`.

    The two differ only in floating-point re-association, which for a
    feed-forward chain stays below the pin tolerance. A reconstruction
    that scores a chain of many factors calls what the parameter
    source itself calls, so that what the value is compared against is
    the same implemented map.
    """
    layer = F.linear(
        x,
        weights[prefix + "0.weight"],
        weights[prefix + "0.bias"],
    )
    if activated:
        layer = torch.tanh(layer)
    layer = F.linear(
        layer,
        weights[prefix + "2.weight"],
        weights[prefix + "2.bias"],
    )
    if activated:
        layer = torch.tanh(layer)
    return F.linear(
        layer,
        weights[prefix + "4.weight"],
        weights[prefix + "4.bias"],
    )


def _mlp3(
    x: Tensor,
    weights: dict[str, Tensor],
    prefix: str,
    *,
    activated: bool = True,
) -> Tensor:
    """The parameter row a three-layer `mlp` source computes.

    `[param_source=mlp]` builds `Linear, tanh, Linear, tanh, Linear`,
    whose modules are numbered `0`, `2` and `4`. `activated=False` is
    the mutant path: the same three matrices with the nonlinearities
    dropped, which collapses the network to a single affine map.
    """
    layer = _affine(
        x,
        weights[prefix + "0.weight"],
        weights[prefix + "0.bias"],
    )
    if activated:
        layer = torch.tanh(layer)
    layer = _affine(
        layer,
        weights[prefix + "2.weight"],
        weights[prefix + "2.bias"],
    )
    if activated:
        layer = torch.tanh(layer)
    return _affine(
        layer,
        weights[prefix + "4.weight"],
        weights[prefix + "4.bias"],
    )


def _reconstruct_deep_markov(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """Reconstruct the path density declared by `deep_markov.qvr`.

    The transition and emission sites are two-factor Normal chains. The
    reconstruction follows each chain through the base-measure image,
    scores all four factors, and sums over the 32-row plate. Compile-time
    MLP weights are taken from the compiled model; wiring, layer order,
    head splits, and densities are rebuilt.

    Only whole-term removals appear in the mutant catalogue. The other
    measured variants move this path density by less than the catalogue
    floor, even though they remain above the pin tolerance.
    """
    x_input = dataset.x_input
    if x_input is None:
        raise AssertionError(
            "deep_markov's reconstruction needs the snippet's "
            "`(driver, state)` row, which the dataset did not carry."
        )
    known = (
        "",
        "drop_transition_prefix",
        "drop_emission_prefix",
        "drop_emission_term",
    )
    if variant not in known:
        raise _unknown_variant("deep_markov", variant)

    raw_hidden = _mlp3(
        x_input,
        weights,
        "_step_s_new.left.param_source.net.",
    )
    loc_hidden, scale_hidden = _normal_head(raw_hidden, 32)
    per_hidden = td.Normal(loc_hidden, scale_hidden).log_prob(loc_hidden)
    raw_state = _mlp3(
        loc_hidden,
        weights,
        "_step_s_new.right.param_source.net.",
    )
    loc_state, scale_state = _normal_head(raw_state, 8)
    per_state = td.Normal(loc_state, scale_state).log_prob(values["s_new"])

    per_transition_row = per_hidden.sum(-1) + per_state.sum(-1)
    if variant == "drop_transition_prefix":
        transition = per_state.sum()
    else:
        transition = per_transition_row.sum()

    raw_emit = _mlp3(
        values["s_new"],
        weights,
        "_step_o.left.param_source.net.",
    )
    loc_emit, scale_emit = _normal_head(raw_emit, 32)
    per_emit = td.Normal(loc_emit, scale_emit).log_prob(loc_emit)
    raw_obs = _mlp3(
        loc_emit,
        weights,
        "_step_o.right.param_source.net.",
    )
    loc_obs, scale_obs = _normal_head(raw_obs, 4)
    per_obs = td.Normal(loc_obs, scale_obs).log_prob(values["o"])
    if variant == "drop_emission_prefix":
        observation = per_obs.sum()
    else:
        observation = (per_emit.sum(-1) + per_obs.sum(-1)).sum()

    if variant == "drop_emission_term":
        return {"s_new": transition, "o": torch.zeros(())}
    return {"s_new": transition, "o": observation}


def _reconstruct_vae(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`vae.qvr`:

        sample z <- prior
        observe Y <- decoder(z)

    `prior : UnitSpace -> Latent ~ Normal` is one affine map to a
    `(loc, log_scale)` pair over `Latent : Real 4`. `decoder` is the
    chain `dec_1 >> stack(dec_deep, 1) >> dec_to_obs`, whose three
    factors each carry a conditional density, so
    [`SampledComposition`][quivers.continuous.morphisms.SampledComposition]
    scores every one of them along its canonical path: each
    intermediate is bound to the image of the base measure's origin,
    which for a Normal factor is its own location, and the observed
    `Y` enters through the last factor. A factor scored at its own
    location contributes its normalizer alone, and writing that out
    makes this an independent statement of the path density
    rather than a call back into the object under test.

    The affine weights are compile-time draws with no spelling in the
    source and are taken as given; the wiring, the layer order of the
    `mlp` source, the head split, the three densities and the sum
    over the 32-row plate are rebuilt.
    """
    x_input = dataset.x_input
    if x_input is None:
        raise AssertionError(
            "vae's reconstruction needs the snippet's `UnitSpace` "
            "driver column, which the dataset did not carry."
        )
    raw_z = _affine(
        x_input,
        weights["_step_z.param_source.linear.weight"],
        weights["_step_z.param_source.linear.bias"],
    )
    # `object Latent : Real 4`.
    loc_z, scale_z = _normal_head(raw_z, 4)
    per_z = td.Normal(loc_z, scale_z).log_prob(values["z"])

    # `dec_1 : Latent -> DecoderHidden`, `object DecoderHidden : Real 16`.
    raw_1 = _affine(
        values["z"],
        weights["_step_Y.left.left.param_source.linear.weight"],
        weights["_step_Y.left.left.param_source.linear.bias"],
    )
    loc_1, scale_1 = _normal_head(raw_1, 16)
    if variant == "prefix_at_zero":
        hidden_1 = torch.zeros_like(loc_1)
    else:
        hidden_1 = loc_1
    per_1 = td.Normal(loc_1, scale_1).log_prob(hidden_1)

    # `dec_deep : DecoderHidden -> DecoderHidden [param_source=mlp]`.
    prefix = "_step_Y.left.right.param_source.net."
    layer = _affine(
        hidden_1,
        weights[prefix + "0.weight"],
        weights[prefix + "0.bias"],
    )
    layer = _affine(
        torch.tanh(layer),
        weights[prefix + "2.weight"],
        weights[prefix + "2.bias"],
    )
    layer = torch.tanh(layer)
    raw_2 = _affine(
        layer,
        weights[prefix + "4.weight"],
        weights[prefix + "4.bias"],
    )
    loc_2, scale_2 = _normal_head(raw_2, 16)
    per_2 = td.Normal(loc_2, scale_2).log_prob(loc_2)

    # `dec_to_obs : DecoderHidden -> ObsSpace`, `object ObsSpace : Real 8`.
    raw_3 = _affine(
        loc_2,
        weights["_step_Y.right.param_source.linear.weight"],
        weights["_step_Y.right.param_source.linear.bias"],
    )
    loc_3, scale_3 = _normal_head(
        raw_3,
        8,
        unit_scale=variant == "decoder_unit_scale",
    )
    per_obs = td.Normal(loc_3, scale_3).log_prob(values["Y"])

    if variant == "drop_decoder_prefix":
        observation = per_obs.sum()
    elif variant in (
        "",
        "prefix_at_zero",
        "decoder_unit_scale",
        "drop_latent_prior",
    ):
        observation = (per_1.sum(-1) + per_2.sum(-1) + per_obs.sum(-1)).sum()
    else:
        raise _unknown_variant("vae", variant)

    if variant == "drop_latent_prior":
        return {"z": torch.zeros(()), "Y": observation}
    return {"z": per_z.sum(), "Y": observation}


def _reconstruct_bnn(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`bnn.qvr`: `observe y : Resp <- net(x)`, `net` a heteroscedastic
    `~ Normal` morphism with `[param_source=mlp, hidden_dim=64]`.

    `net`'s weights are drawn at compile time from torch's default
    `nn.Linear` initialiser. They appear nowhere in the `.qvr` text
    and nowhere in the `.md` snippet, so the reconstruction takes the
    four tensors as given and rebuilds everything else: the layer
    order (affine, `tanh`, affine), the `(loc, log_scale)` split of
    the two output columns, the Normal density, and the sum over the
    200-row `Resp` plate. That is the whole of what the oracle
    computes here apart from the numbers in the weight tensors, which
    no independent source can supply.
    """
    del dataset
    prefix = "_step_y._family.param_source.net."
    hidden = _affine(
        values["x"],
        weights[prefix + "0.weight"],
        weights[prefix + "0.bias"],
    )
    if variant == "mlp_without_activation":
        activated = hidden
    elif variant in ("", "homoscedastic", "plate_mean"):
        activated = torch.tanh(hidden)
    else:
        raise _unknown_variant("bnn", variant)
    raw = _affine(
        activated,
        weights[prefix + "2.weight"],
        weights[prefix + "2.bias"],
    )
    # `object Target : Real 1`: a one-dimensional response, so the
    # head emits one location column and one log-scale column.
    loc, scale = _normal_head(
        raw,
        1,
        unit_scale=variant == "homoscedastic",
    )
    per_row = td.Normal(loc, scale).log_prob(values["y"])
    if variant == "plate_mean":
        return {"y": per_row.mean()}
    return {"y": per_row.sum()}


def _reconstruct_state_space(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
    state_dim: int,
    obs_dim: int,
) -> dict[str, Tensor]:
    """`continuous_hmm.qvr` and `linear_gaussian_ssm.qvr`, which share
    a program shape:

        sample s_new <- transition(<program input>)
        observe o    <- emission(s_new)

    Both morphisms are `~ Normal` Kleisli arrows over a default linear
    parameter source, so each step is a single affine map from its
    conditioning row to a `(loc, log_scale)` pair. As with `bnn` the
    affine weights are compile-time draws with no source spelling and
    are taken as given; the wiring (which row conditions which step),
    the head split, the density, and the per-step sum over the
    32-element scan are rebuilt.
    """
    x_input = dataset.x_input
    if x_input is None:
        raise AssertionError(
            "a state-space reconstruction needs the snippet's "
            "`state_prev` / `x_in` driver matrix, which the dataset "
            "did not carry."
        )
    state = values["s_new"]

    raw_state = _affine(
        x_input,
        weights["_step_s_new.param_source.linear.weight"],
        weights["_step_s_new.param_source.linear.bias"],
    )
    loc_state, scale_state = _normal_head(
        raw_state,
        state_dim,
        unit_scale=variant == "unit_transition_scale",
    )
    per_step_state = td.Normal(loc_state, scale_state).log_prob(state)

    if variant == "emission_reads_previous_state":
        emission_input = x_input
    elif variant in (
        "",
        "unit_transition_scale",
        "drop_transition_term",
        "swap_loc_and_log_scale",
    ):
        emission_input = state
    else:
        raise _unknown_variant("state-space", variant)
    raw_obs = _affine(
        emission_input,
        weights["_step_o.param_source.linear.weight"],
        weights["_step_o.param_source.linear.bias"],
    )
    if variant == "swap_loc_and_log_scale":
        loc_obs = raw_obs[..., obs_dim:]
        scale_obs = raw_obs[..., :obs_dim].exp().clamp(min=_SCALE_FLOOR)
    else:
        loc_obs, scale_obs = _normal_head(raw_obs, obs_dim)
    per_step_obs = td.Normal(loc_obs, scale_obs).log_prob(values["o"])

    if variant == "drop_transition_term":
        return {"s_new": torch.zeros(()), "o": per_step_obs.sum()}
    return {"s_new": per_step_state.sum(), "o": per_step_obs.sum()}


def _reconstruct_continuous_hmm(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`object State : Real 16`, `object Obs : Real 8`."""
    return _reconstruct_state_space(
        dataset,
        values,
        weights,
        variant,
        state_dim=16,
        obs_dim=8,
    )


def _reconstruct_linear_gaussian_ssm(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`object State : Real 4`, `object Obs : Real 2`; the transition
    additionally reads a `Driver : Real 2` row, which the snippet
    concatenates onto the previous state before the call."""
    return _reconstruct_state_space(
        dataset,
        values,
        weights,
        variant,
        state_dim=4,
        obs_dim=2,
    )


def _reconstruct_mixture_model(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`mixture_model.qvr`, instantiated at `gmm(alpha=1.0)`:

        sample probs <- Dirichlet(concentration=alpha) [over=Component]
        sample mu    : Component <- Normal(loc=0.0, scale=5.0)
        sample sigma : Component <- HalfNormal(scale=1.0)
        observe r    : Resp <- MixtureNormal(weights=probs, loc=mu, scale=sigma)

    Wholly reconstructible from the source: every family is named,
    every hyper-parameter is a literal, and the one non-elementary
    step is the closed-form marginalisation the `MixtureNormal`
    likelihood performs,

        log p(r_n) = logsumexp_k [ log probs_k + log N(r_n; mu_k, sigma_k) ],

    summed over the `Resp` plate. No component assignment is drawn.
    """
    del weights
    alpha = dataset.scalar_params["alpha"]
    probs, mu, sigma = values["probs"], values["mu"], values["sigma"]
    response = values["r"]

    concentration = torch.full_like(probs, float(alpha))
    if variant == "drop_dirichlet_prior":
        probs_term = torch.zeros(())
    else:
        probs_term = td.Dirichlet(concentration).log_prob(probs).sum()

    mu_term = td.Normal(0.0, 5.0).log_prob(mu).sum()

    sigma_term = td.HalfNormal(1.0).log_prob(sigma).sum()
    if variant == "unconstrain_sigma":
        # Scoring a positive latent in log space adds a
        # `log |d sigma / d log sigma| = log sigma` Jacobian. The
        # oracle scores in the constrained space and adds none.
        sigma_term = sigma_term + sigma.log().sum()

    per_component = td.Normal(
        mu.unsqueeze(0),
        sigma.unsqueeze(0),
    ).log_prob(response.unsqueeze(-1))
    weighted = probs.log().unsqueeze(0) + per_component
    if variant == "first_component_only":
        per_row = weighted[..., 0]
    elif variant in (
        "",
        "drop_dirichlet_prior",
        "unconstrain_sigma",
        "likelihood_mean",
    ):
        per_row = torch.logsumexp(weighted, dim=-1)
    else:
        raise _unknown_variant("mixture_model", variant)
    response_term = per_row.mean() if variant == "likelihood_mean" else per_row.sum()

    return {
        "probs": probs_term,
        "mu": mu_term,
        "sigma": sigma_term,
        "r": response_term,
    }


def _reconstruct_parametric_pooling(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`parametric_pooling.qvr`, exporting `pooled_tight`:

        sample z : K <- Normal(loc=0.0, scale=1.0)      (inside school_effects)
        let effect = spread * z               (spread = 0.6)
        sample sigma <- LogNormal(loc=0.0, scale=0.5)
        let total_effect = sum(theta)
        score centering = -50.0 * total_effect * total_effect
        observe y : School <- Normal(loc=theta, scale=sigma)

    Wholly reconstructible. The `score` step is a log-density summand
    in its own right, not a normalised family: it enters the joint as
    its own value. `theta` and `total_effect` are deterministic `let`
    bindings and score zero.
    """
    del dataset, weights
    z, sigma, y = values["theta$z"], values["sigma"], values["y"]
    # `sample theta <- school_effects(0.6, School)` instantiates the
    # template's `spread` at 0.6.
    spread = 1.0 if variant == "ignore_spread" else 0.6
    theta = spread * z

    z_term = td.Normal(0.0, 1.0).log_prob(z).sum()

    sigma_term = td.LogNormal(0.0, 0.5).log_prob(sigma).sum()
    if variant == "unconstrain_sigma":
        sigma_term = sigma_term + sigma.log().sum()

    total_effect = theta.sum()
    if variant == "drop_centering_score":
        centering = torch.zeros(())
    elif variant in ("", "ignore_spread", "unconstrain_sigma", "plate_mean"):
        centering = -50.0 * total_effect * total_effect
    else:
        raise _unknown_variant("parametric_pooling", variant)

    per_school = td.Normal(theta, sigma).log_prob(y)
    y_term = per_school.mean() if variant == "plate_mean" else per_school.sum()

    return {
        "theta$z": z_term,
        "sigma": sigma_term,
        "centering": centering,
        "y": y_term,
    }


def _reconstruct_pmf(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`pmf.qvr`, an algebra-level example whose source declares a
    mean surface rather than a measure:

        object LatentDim : FinSet 2
        object User, Movie : FinSet 8
        morphism U : LatentDim -> User   [role=latent]
        morphism V : LatentDim -> Movie  [role=latent]
        define pmf = U.dagger >> V

    The `.md` snippet supplies the probabilistic surface the source
    leaves open, and it is the standard PMF one: an entrywise
    `Normal(0, 1)` prior over each factor matrix, and a
    `Normal(S, 0.5)` likelihood over every cell of the dense rating
    matrix, with `S` rebuilt from the sampled factors by the same
    composition the source names.

    Wholly reconstructible. An arrow's tensor is indexed
    `(domain, codomain)`, so each factor is `(2, 8)`, and
    `U.dagger >> V` under the `real` algebra contracts the shared
    `LatentDim` index: the `(u, m)` entry is
    `sum_k U[k, u] * V[k, m]`, written here as an `einsum` rather
    than through the compiled composition, which is the code under
    test. `mu` is a deterministic `let` and scores zero.
    """
    del dataset, weights
    if variant == "factors_read_column_major":
        # The wire payload arrives flat; inflating it as `(User,
        # LatentDim)` and transposing keeps every entry and every
        # shape while permuting which factor coordinate meets which.
        user_factor = values["U"].reshape(8, 2).transpose(0, 1)
        movie_factor = values["V"].reshape(8, 2).transpose(0, 1)
    elif variant in (
        "",
        "score_transposed",
        "drop_movie_factor_prior",
        "unit_rating_scale",
        "plate_mean",
    ):
        user_factor = values["U"].reshape(2, 8)
        movie_factor = values["V"].reshape(2, 8)
    else:
        raise _unknown_variant("pmf", variant)

    score = torch.einsum("ku,km->um", user_factor, movie_factor)
    if variant == "score_transposed":
        score = score.transpose(0, 1)

    unit = td.Normal(0.0, 1.0)
    movie_term = (
        torch.zeros(())
        if variant == "drop_movie_factor_prior"
        else unit.log_prob(movie_factor).sum()
    )
    # `sigma = 0.5` in the snippet: the source fixes the mean surface
    # and says nothing about the observation scale.
    scale = 1.0 if variant == "unit_rating_scale" else 0.5
    per_cell = td.Normal(score, scale).log_prob(values["rating"])

    return {
        "U": unit.log_prob(user_factor).sum(),
        "V": movie_term,
        "rating": (per_cell.mean() if variant == "plate_mean" else per_cell.sum()),
    }


def _reconstruct_tensor_contraction(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`tensor_contraction.qvr`, the other algebra-level example:

        object Item : FinSet 4
        object PredDim, ArgDim : FinSet 2
        object Judgment : FinSet 3
        morphism pred_embed  : Item -> PredDim
        morphism arg_embed   : Item -> ArgDim
        morphism interaction : (PredDim * ArgDim) -> Judgment
        define plausibility = bilinear_score(
            pred_embed, arg_embed, interaction,
        )

    The `.md` snippet gives each declared arrow an entrywise
    `Normal(0, 1)` prior and scores the judgment plate under
    `Normal(S, 0.5)`, with `S` the contraction of the three sampled
    tensors.

    Wholly reconstructible. The contraction's wiring is fixed by the
    typed signature rather than spelled out: `PredDim` and `ArgDim`
    each appear in two inputs and not in the output, so both are
    summed over, while `Item` and `Judgment` appear in the output and
    propagate. That is `sum_b sum_c p[i, b] * a[i, c] * w[b, c, s]`,
    written here as an `einsum` rather than by calling the compiled
    wiring. `mu` is a deterministic `let` and scores zero.
    """
    del dataset, weights
    pred = values["pred_embed"].reshape(4, 2)
    arg = values["arg_embed"].reshape(4, 2)
    interaction = values["interaction"].reshape(2, 2, 3)

    if variant == "interaction_axes_swapped":
        # `PredDim` and `ArgDim` are both `FinSet 2`, so reading the
        # interaction tensor's two contracted axes in the wrong order
        # passes every shape check the wiring performs.
        spec = "ib,ic,cbs->is"
    elif variant in (
        "",
        "drop_interaction_prior",
        "unit_judgment_scale",
        "plate_mean",
    ):
        spec = "ib,ic,bcs->is"
    else:
        raise _unknown_variant("tensor_contraction", variant)
    score = torch.einsum(spec, pred, arg, interaction)

    unit = td.Normal(0.0, 1.0)
    interaction_term = (
        torch.zeros(())
        if variant == "drop_interaction_prior"
        else unit.log_prob(interaction).sum()
    )
    scale = 1.0 if variant == "unit_judgment_scale" else 0.5
    per_cell = td.Normal(score, scale).log_prob(values["judgment"])

    return {
        "pred_embed": unit.log_prob(pred).sum(),
        "arg_embed": unit.log_prob(arg).sum(),
        "interaction": interaction_term,
        "judgment": (per_cell.mean() if variant == "plate_mean" else per_cell.sum()),
    }


def _reconstruct_tree_categorical(
    dataset: _gallery_data.GalleryDataset,
    values: dict[str, Tensor],
    weights: dict[str, Tensor],
    variant: str,
) -> dict[str, Tensor]:
    """`tree_categorical.qvr`:

        sample p_root, p_left, p_right <- Beta(concentration1=1.0, concentration0=1.0)
        let leaf_log = factor cls : Class in { 0 -> log(1 - p_root) + log(1 - p_left), ... }
        sample sigma_v <- HalfNormal(scale=1.0)
        sample delta : Verb <- Normal(loc=0.0, scale=sigma_v)
        sample mu : Class <- Normal(loc=0.0, scale=1.0)
        let cell_score = factor v : Verb, cls : Class in delta[v] + mu[cls] + leaf_log[cls]
        let cell0 = cell_score[0, 0]
        observe y : Resp <- Normal(loc=cell0, scale=0.5)

    Wholly reconstructible. Only cell `(0, 0)` of the rank-2 factor
    reaches the likelihood, so the reconstruction evaluates that one
    cell: `delta[0] + mu[0] + leaf_log[0]`, with
    `leaf_log[0] = log(1 - p_root) + log(1 - p_left)` from the case
    table's `0 ->` arm. The three `Beta(1, 1)` sites score exactly
    zero, which
    [`test_zero_scoring_sites_are_zero_by_identity`][tests.transpile.test_oracle_reference_strength.test_zero_scoring_sites_are_zero_by_identity]
    asserts is an identity rather than a coincidence.

    `y` carries one response per element of `object Resp : FinSet
    200`, every one of them scored against the same `cell0` location,
    so the likelihood is a 200-term sum over a plate of width 200.
    """
    del dataset, weights
    p_root, p_left, p_right = (
        values["p_root"],
        values["p_left"],
        values["p_right"],
    )
    sigma_v, delta, mu, y = (
        values["sigma_v"],
        values["delta"],
        values["mu"],
        values["y"],
    )
    unit = td.Beta(1.0, 1.0)

    if variant == "delta_unit_scale":
        delta_scale = torch.ones_like(sigma_v)
    elif variant in (
        "",
        "wrong_leaf_branch",
        "drop_leaf_offset",
        "plate_mean",
    ):
        delta_scale = sigma_v
    else:
        raise _unknown_variant("tree_categorical", variant)

    if variant == "wrong_leaf_branch":
        leaf_zero = torch.log(p_root) + torch.log(p_left)
    else:
        leaf_zero = torch.log(1.0 - p_root) + torch.log(1.0 - p_left)
    cell_zero = delta[0] + mu[0]
    if variant != "drop_leaf_offset":
        cell_zero = cell_zero + leaf_zero

    # `y` is summed, never averaged: the joint is a product over the
    # 200-element `Resp` plate. Every response is scored against the
    # same `cell0`, so an averaged plate returns the per-response
    # density itself and the defect is worth two orders of magnitude.
    per_response = td.Normal(cell_zero, 0.5).log_prob(y)
    y_term = per_response.mean() if variant == "plate_mean" else per_response.sum()

    return {
        "p_root": unit.log_prob(p_root).sum(),
        "p_left": unit.log_prob(p_left).sum(),
        "p_right": unit.log_prob(p_right).sum(),
        "sigma_v": td.HalfNormal(1.0).log_prob(sigma_v).sum(),
        "delta": td.Normal(
            torch.zeros_like(delta),
            delta_scale,
        )
        .log_prob(delta)
        .sum(),
        "mu": td.Normal(0.0, 1.0).log_prob(mu).sum(),
        "y": y_term,
    }


_Builder = Callable[
    [
        _gallery_data.GalleryDataset,
        dict[str, Tensor],
        dict[str, Tensor],
        str,
    ],
    dict[str, Tensor],
]
"""Signature every reconstruction shares: `(dataset, clamped values,
compiled weights, variant) -> per-site log-density terms`."""


class _Reconstruction:
    """One example's raw-torch reconstruction and what it takes as given.

    A plain class rather than a `dx.Model`: it holds a callable and is
    never serialised, compared structurally, or round-tripped through
    didactic's encode / decode.
    """

    def __init__(
        self,
        build: _Builder,
        opaque_weights: tuple[str, ...],
        opaque_reason: str,
    ) -> None:
        self.build = build
        self.opaque_weights = opaque_weights
        self.opaque_reason = opaque_reason


_RECONSTRUCTIONS: dict[str, _Reconstruction] = {
    "seq2seq": _Reconstruction(
        _reconstruct_seq2seq,
        (
            "_step_h.left.left.left.centers",
            "_step_h.left.left.left.log_sigma",
            "_step_h.left.right.left.centers",
            "_step_h.left.right.left.log_sigma",
            "_step_next_token._family.param_source.linear.weight",
            "_step_next_token._family.param_source.linear.bias",
        ),
        "Both embedding tables, the twenty-four block factors "
        "of the two towers, the cross factor and the "
        "language-model head take their weights from "
        "compile-time draws, named in neither the `.qvr` "
        "source nor the `.md` snippet.",
    ),
    "deep_markov": _Reconstruction(
        _reconstruct_deep_markov,
        (
            "_step_s_new.left.param_source.net.0.weight",
            "_step_s_new.left.param_source.net.0.bias",
            "_step_s_new.left.param_source.net.2.weight",
            "_step_s_new.left.param_source.net.2.bias",
            "_step_s_new.left.param_source.net.4.weight",
            "_step_s_new.left.param_source.net.4.bias",
            "_step_s_new.right.param_source.net.0.weight",
            "_step_s_new.right.param_source.net.0.bias",
            "_step_s_new.right.param_source.net.2.weight",
            "_step_s_new.right.param_source.net.2.bias",
            "_step_s_new.right.param_source.net.4.weight",
            "_step_s_new.right.param_source.net.4.bias",
            "_step_o.left.param_source.net.0.weight",
            "_step_o.left.param_source.net.0.bias",
            "_step_o.left.param_source.net.2.weight",
            "_step_o.left.param_source.net.2.bias",
            "_step_o.left.param_source.net.4.weight",
            "_step_o.left.param_source.net.4.bias",
            "_step_o.right.param_source.net.0.weight",
            "_step_o.right.param_source.net.0.bias",
            "_step_o.right.param_source.net.2.weight",
            "_step_o.right.param_source.net.2.bias",
            "_step_o.right.param_source.net.4.weight",
            "_step_o.right.param_source.net.4.bias",
        ),
        "All four MLPs take their weights from compile-time "
        "draws, which are named in neither the `.qvr` source "
        "nor the `.md` snippet.",
    ),
    "vae": _Reconstruction(
        _reconstruct_vae,
        (
            "_step_z.param_source.linear.weight",
            "_step_z.param_source.linear.bias",
            "_step_Y.left.left.param_source.linear.weight",
            "_step_Y.left.left.param_source.linear.bias",
            "_step_Y.left.right.param_source.net.0.weight",
            "_step_Y.left.right.param_source.net.0.bias",
            "_step_Y.left.right.param_source.net.2.weight",
            "_step_Y.left.right.param_source.net.2.bias",
            "_step_Y.left.right.param_source.net.4.weight",
            "_step_Y.left.right.param_source.net.4.bias",
            "_step_Y.right.param_source.linear.weight",
            "_step_Y.right.param_source.linear.bias",
        ),
        "`prior` and the three decoder factors take their affine "
        "weights from compile-time draws, which are named in neither "
        "the `.qvr` source nor the `.md` snippet.",
    ),
    "bnn": _Reconstruction(
        _reconstruct_bnn,
        (
            "_step_y._family.param_source.net.0.weight",
            "_step_y._family.param_source.net.0.bias",
            "_step_y._family.param_source.net.2.weight",
            "_step_y._family.param_source.net.2.bias",
        ),
        "`net`'s MLP weights are drawn by torch's default `nn.Linear` "
        "initialiser when the module compiles. They are named in "
        "neither the `.qvr` source nor the `.md` snippet, so no "
        "independent source can supply their values.",
    ),
    "continuous_hmm": _Reconstruction(
        _reconstruct_continuous_hmm,
        (
            "_step_s_new.param_source.linear.weight",
            "_step_s_new.param_source.linear.bias",
            "_step_o.param_source.linear.weight",
            "_step_o.param_source.linear.bias",
        ),
        "`transition` and `emission` take the default linear "
        "parameter source, whose affine weights are compile-time "
        "draws with no spelling in the source.",
    ),
    "linear_gaussian_ssm": _Reconstruction(
        _reconstruct_linear_gaussian_ssm,
        (
            "_step_s_new.param_source.linear.weight",
            "_step_s_new.param_source.linear.bias",
            "_step_o.param_source.linear.weight",
            "_step_o.param_source.linear.bias",
        ),
        "`transition_cell` and `emission` take the default linear "
        "parameter source, whose affine weights are compile-time "
        "draws with no spelling in the source.",
    ),
    "mixture_model": _Reconstruction(_reconstruct_mixture_model, (), ""),
    "parametric_pooling": _Reconstruction(
        _reconstruct_parametric_pooling,
        (),
        "",
    ),
    "pmf": _Reconstruction(_reconstruct_pmf, (), ""),
    "tensor_contraction": _Reconstruction(
        _reconstruct_tensor_contraction,
        (),
        "",
    ),
    "tree_categorical": _Reconstruction(
        _reconstruct_tree_categorical,
        (),
        "",
    ),
}


class _Mutant:
    """A named statistical defect the reconstruction must reject.

    A plain class rather than a `dx.Model`, for the same reason as
    `_Reconstruction`.
    """

    def __init__(
        self,
        example: str,
        variant: str,
        defect: str,
        floor: float,
    ) -> None:
        self.example = example
        self.variant = variant
        self.defect = defect
        self.floor = floor

    @property
    def ident(self) -> str:
        return f"{self.example}:{self.variant}"


_MUTANTS: tuple[_Mutant, ...] = (
    _Mutant(
        "seq2seq",
        "plate_mean",
        "the 32-row plate is averaged instead of summed.",
        10000.0,
    ),
    _Mutant(
        "seq2seq",
        "drop_encoder_tower",
        "the encoder tower's thirteen factors are dropped and only "
        "the decoder's are scored, which is the joint a chain that "
        "scores one arm of a product would report.",
        7000.0,
    ),
    _Mutant(
        "seq2seq",
        "drop_cross_factor",
        "the cross factor is dropped, so `h` enters no density at "
        "all and the two towers are scored without the step that "
        "relates them.",
        1000.0,
    ),
    _Mutant(
        "deep_markov",
        "drop_emission_prefix",
        "the emission chain's first factor is dropped and only its "
        "second is scored, which is the joint a chain that scores its "
        "endpoint's marginal alone would report.",
        1.0,
    ),
    _Mutant(
        "deep_markov",
        "drop_transition_prefix",
        "the transition chain's first factor is dropped and only its "
        "second is scored, which is what a chain that scores its "
        "endpoint's marginal alone would report.",
        30.0,
    ),
    _Mutant(
        "deep_markov",
        "drop_emission_term",
        "the `observe o` term is dropped, leaving the transition "
        "alone in a joint that claims to carry both.",
        33.0,
    ),
    _Mutant(
        "vae",
        "drop_decoder_prefix",
        "the two decoder prefix factors are dropped and only the "
        "observation factor is scored, which is what a chain that "
        "scores its endpoint's marginal alone would report.",
        25.0,
    ),
    _Mutant(
        "vae",
        "prefix_at_zero",
        "the first decoder intermediate is bound to the origin of "
        "the codomain rather than to the image of the base measure's "
        "origin, which is the off-by-one a chain that forgets to "
        "push its own location through would make.",
        43.0,
    ),
    _Mutant(
        "vae",
        "decoder_unit_scale",
        "the observation head's log-scale columns are ignored and "
        "`Y` is scored at unit scale.",
        0.35,
    ),
    _Mutant(
        "vae",
        "drop_latent_prior",
        "the `sample z <- prior` term is dropped, leaving the "
        "likelihood alone in a joint that claims to carry both.",
        6.3,
    ),
    _Mutant(
        "bnn",
        "mlp_without_activation",
        "the hidden layer's `tanh` is dropped, collapsing the network "
        "to a single affine map: the defect class of a renderer that "
        "emits a linear head for a nonlinear parameter source.",
        5000.0,
    ),
    _Mutant(
        "bnn",
        "homoscedastic",
        "the head's log-scale columns are ignored and the response is "
        "scored at unit scale, dropping the heteroscedasticity the "
        "model exists to express.",
        80.0,
    ),
    _Mutant(
        "bnn",
        "plate_mean",
        "the 200-row `Resp` plate is averaged instead of summed.",
        200.0,
    ),
    _Mutant(
        "continuous_hmm",
        "drop_transition_term",
        "the `sample s_new <- transition` prior term is dropped and "
        "only the emission likelihood is scored.",
        14.0,
    ),
    _Mutant(
        "continuous_hmm",
        "emission_reads_previous_state",
        "`emission` is conditioned on the program input rather than "
        "on the freshly-drawn `s_new`, an off-by-one in the scan's "
        "wiring that leaves every shape valid.",
        0.65,
    ),
    _Mutant(
        "continuous_hmm",
        "unit_transition_scale",
        "the transition head's log-scale columns are ignored.",
        0.7,
    ),
    _Mutant(
        "linear_gaussian_ssm",
        "swap_loc_and_log_scale",
        "the emission head's location and log-scale columns are "
        "transposed, which every shape check still accepts.",
        0.16,
    ),
    _Mutant(
        "linear_gaussian_ssm",
        "drop_transition_term",
        "the `sample s_new <- transition_cell` prior term is dropped "
        "and only the emission likelihood is scored.",
        3.5,
    ),
    _Mutant(
        "linear_gaussian_ssm",
        "unit_transition_scale",
        "the transition head's log-scale columns are ignored, so the "
        "process noise is scored at unit scale instead of at the "
        "kernel's own.",
        0.3,
    ),
    _Mutant(
        "mixture_model",
        "first_component_only",
        "the closed-form marginalisation is replaced by component 0 "
        "alone, so the `logsumexp` over the discrete latent is lost.",
        1000.0,
    ),
    _Mutant(
        "mixture_model",
        "drop_dirichlet_prior",
        "the `Dirichlet(alpha)` prior on the mixing weights is dropped.",
        0.3,
    ),
    _Mutant(
        "mixture_model",
        "unconstrain_sigma",
        "the `HalfNormal` scale is scored in log space, adding the "
        "`log sigma` Jacobian the constrained-space convention "
        "forbids.",
        1.0,
    ),
    _Mutant(
        "mixture_model",
        "likelihood_mean",
        "the 100-row `Resp` plate is averaged instead of summed.",
        100.0,
    ),
    _Mutant(
        "parametric_pooling",
        "drop_centering_score",
        "the soft sum-to-zero `score` factor is dropped. Its value is "
        "~0 at the ground truth, where the snippet centres the group "
        "effects exactly, so this mutant is invisible at point 0 and "
        "only the perturbed points reject it.",
        5.0,
    ),
    _Mutant(
        "parametric_pooling",
        "ignore_spread",
        "the template's `spread = 0.6` scaling is dropped, so the "
        "non-centred parameterisation collapses to `theta = z`.",
        10.0,
    ),
    _Mutant(
        "parametric_pooling",
        "unconstrain_sigma",
        "the `LogNormal` observation scale is scored in log space, "
        "adding a `log sigma` Jacobian.",
        0.5,
    ),
    _Mutant(
        "parametric_pooling",
        "plate_mean",
        "the 8-school plate is averaged instead of summed.",
        2.0,
    ),
    _Mutant(
        "pmf",
        "score_transposed",
        "the rating mean is read as `S[m, u]` rather than `S[u, m]`, "
        "the orientation error a mis-taken dagger produces. The score "
        "matrix is square, so no shape check objects.",
        1000.0,
    ),
    _Mutant(
        "pmf",
        "factors_read_column_major",
        "each flat factor payload is inflated as `(User, LatentDim)` "
        "and transposed rather than read as `(LatentDim, User)`, "
        "permuting which latent coordinate meets which user while "
        "leaving both entrywise priors, and every shape, untouched.",
        1000.0,
    ),
    _Mutant(
        "pmf",
        "drop_movie_factor_prior",
        "the entrywise `Normal(0, 1)` prior on the movie factor matrix is dropped.",
        27.0,
    ),
    _Mutant(
        "pmf",
        "unit_rating_scale",
        "the rating likelihood is scored at unit scale instead of at "
        "the snippet's `sigma = 0.5`.",
        18.0,
    ),
    _Mutant(
        "pmf",
        "plate_mean",
        "the 64-cell `(User, Movie)` rating plate is averaged instead of summed.",
        85.0,
    ),
    _Mutant(
        "tensor_contraction",
        "interaction_axes_swapped",
        "the interaction tensor's `PredDim` and `ArgDim` axes are "
        "read in the wrong order. Both are `FinSet 2`, so the "
        "contraction stays well-typed and every shape check passes.",
        70.0,
    ),
    _Mutant(
        "tensor_contraction",
        "drop_interaction_prior",
        "the entrywise `Normal(0, 1)` prior on the third-order "
        "interaction tensor is dropped.",
        19.0,
    ),
    _Mutant(
        "tensor_contraction",
        "unit_judgment_scale",
        "the judgment likelihood is scored at unit scale instead of "
        "at the snippet's `sigma = 0.5`. The ground truth sits close "
        "to the bilinear score, so this mutant is nearly invisible "
        "there and only the perturbed points reject it firmly.",
        9.5,
    ),
    _Mutant(
        "tensor_contraction",
        "plate_mean",
        "the 12-cell `(Item, Judgment)` plate is averaged instead of summed.",
        24.0,
    ),
    _Mutant(
        "tree_categorical",
        "wrong_leaf_branch",
        "leaf 0 of the case table reads the `p_root` / `p_left` arm "
        "instead of its complement, the classic tree-traversal "
        "polarity error.",
        40.0,
    ),
    _Mutant(
        "tree_categorical",
        "delta_unit_scale",
        "the per-verb effects are scored at unit scale instead of at "
        "the sampled `sigma_v`, severing the hierarchy.",
        1.5,
    ),
    _Mutant(
        "tree_categorical",
        "drop_leaf_offset",
        "the tree-structured leaf log-probability is dropped from the "
        "score cell, leaving `delta[0] + mu[0]`.",
        400.0,
    ),
    _Mutant(
        "tree_categorical",
        "plate_mean",
        "the 200-response `Resp` plate is averaged instead of summed.",
        100.0,
    ),
)


# ---------------------------------------------------------------------
# Oracle fixtures, memoised per example.
#
# Every test below scores the same six points of the same six
# examples; tracing them once per test would multiply the module's
# cost by the number of tests without adding a single independent
# observation.
# ---------------------------------------------------------------------


class _Fixture:
    """One example's dataset, point set, compiled weights, and the
    oracle's per-point joint and per-site summands.

    A plain class rather than a `dx.Model`: it holds torch tensors, a
    compiled
    [`GalleryDataset`][tests.transpile._gallery_data.GalleryDataset],
    and a list of [`Point`][tests.transpile.probes._protocol.Point]s,
    none of which survive didactic's encode / decode round trip.
    """

    def __init__(
        self,
        dataset: _gallery_data.GalleryDataset,
        points: list[Point],
        weights: dict[str, Tensor],
        joints: list[float],
        sites: list[dict[str, float]],
    ) -> None:
        self.dataset = dataset
        self.points = points
        self.weights = weights
        self.joints = joints
        self.sites = sites


_FIXTURES: dict[str, _Fixture] = {}


def _fixture(example: str) -> _Fixture:
    """Build (or return the memoised) oracle fixture for `example`."""
    cached = _FIXTURES.get(example)
    if cached is not None:
        return cached

    source_path = _SOURCE_DIR / f"{example}.qvr"
    dataset = _gallery_data.load_gallery_data(source_path)
    if dataset is None:
        raise AssertionError(
            f"{example!r}: `load_gallery_data` returned None, so this "
            f"module has no point set to reconstruct against. Either "
            f"the `.md` synthetic-data block broke, or the "
            f"reconstruction registry names an example that never had "
            f"one."
        )
    monadic = dataset.monadic
    if monadic is None:
        raise AssertionError(
            f"{example!r}: the synthetic-data block bound no compiled "
            f"`MonadicProgram`, so there is no oracle to check."
        )

    points = _gallery_data.points_from_dataset(dataset)
    weights = {name: tensor.detach() for name, tensor in monadic.state_dict().items()}
    joints: list[float] = []
    sites: list[dict[str, float]] = []
    for point in points:
        traces = reference_traces(
            monadic,
            point,
            x_input=dataset.x_input,
            observations=_gallery_data.observations_for_point(
                dataset,
                point,
            ),
        )
        # A reconstruction compared against a redrawn "joint" would be
        # comparing against a sample, and the disagreement would read
        # as a defect in whichever side happened to be checked second.
        # `reference_traces` already ran the program under two distinct
        # global RNG states; requiring them to agree bit for bit costs
        # nothing here and rules that reading out.
        assert_reference_joint_deterministic(traces, example)
        tr = traces[0]
        joint = tr.log_joint
        if joint is None:
            raise AssertionError(f"{example!r}: the trace returned no `log_joint`.")
        joints.append(float(joint.sum().item()))
        sites.append(
            {
                name: float(site.log_prob.sum().item())
                for name, site in tr.sites.items()
                if site.log_prob is not None
            }
        )

    fixture = _Fixture(dataset, points, weights, joints, sites)
    _FIXTURES[example] = fixture
    return fixture


def _reconstruct(
    example: str,
    index: int,
    variant: str = "",
) -> dict[str, float]:
    """Per-site reconstruction of `example` at point `index`."""
    fixture = _fixture(example)
    values = _clamped_values(fixture.dataset, fixture.points[index])
    terms = _RECONSTRUCTIONS[example].build(
        fixture.dataset,
        values,
        fixture.weights,
        variant,
    )
    return {name: float(term.item()) for name, term in terms.items()}


def _cells() -> list[tuple[str, int]]:
    """Every `(example, point index)` pair the reconstruction covers."""
    n_points = len(_gallery_data.perturbation_labels())
    return [
        (example, index)
        for example in sorted(_RECONSTRUCTIONS)
        for index in range(n_points)
    ]


def _cell_id(cell: tuple[str, int]) -> str:
    example, index = cell
    return f"{example}-pt{index}"


def _live_backend_cells(example: str) -> list[str]:
    """Backends whose container actually scores `example`.

        A backend is live for an example when the gallery tier neither
        pins its transpile as a raise nor parks the cell in a skip
        registry: exactly the cells that reach
    `assert_log_density_match` and thus re-derive the oracle in a
        foreign runtime.
    """
    return sorted(
        backend
        for backend in _gallery_tier._BACKENDS_WITH_IMAGES
        if (backend, example) not in _gallery_tier._EXPECTED_TRANSPILE_RAISES
        and (backend, example) not in _gallery_tier._SKIP_PROBE_INCOMPATIBLE
        and example not in _gallery_tier._SKIP_DATASET_LOAD_FAILED
        and example not in _gallery_tier._SKIP_QVR_INCOMPATIBLE
    )


# ---------------------------------------------------------------------
# Coverage of the pinned registry.
# ---------------------------------------------------------------------


def test_reconstruction_registry_is_well_formed() -> None:
    """Every reconstruction names a pinned example, and every tensor it
    takes as given exists in that example's compiled module.

    The second half is what keeps `opaque_weights` honest. A
    reconstruction that quietly reached for a fifth weight tensor, or
    that kept naming a tensor the runtime no longer builds, would
    still run; the declared list would then understate what the check
    assumes and overstate what it verifies.
    """
    pinned = set(_gallery_tier._QVR_REFERENCE_JOINT)
    unknown = sorted(set(_RECONSTRUCTIONS) - pinned)
    assert not unknown, (
        f"{unknown!r} carry a raw-torch reconstruction but no "
        f"`_QVR_REFERENCE_JOINT` entry, so the reconstruction verifies "
        f"a value nothing else asserts. Pin the example or drop the "
        f"reconstruction."
    )

    for example in sorted(_RECONSTRUCTIONS):
        entry = _RECONSTRUCTIONS[example]
        available = set(_fixture(example).weights)
        missing = sorted(set(entry.opaque_weights) - available)
        assert not missing, (
            f"{example!r}: the reconstruction declares {missing!r} as "
            f"compile-time weights it takes as given, but the "
            f"compiled module has no such entries. The runtime's "
            f"parameter layout moved under the reconstruction."
        )
        if entry.opaque_weights:
            assert entry.opaque_reason.strip(), (
                f"{example!r}: takes {len(entry.opaque_weights)} "
                f"tensor(s) as given without saying why they cannot "
                f"be derived independently. An undeclared assumption "
                f"is the part of the joint nothing checks."
            )
        else:
            assert not entry.opaque_reason, (
                f"{example!r}: declares no opaque weights but carries "
                f"a reason for them. Drop the stale prose."
            )


def test_every_pinned_example_has_an_independent_witness() -> None:
    """Require an independent witness for every pinned reference.

    A pin must be re-derived by a live backend cell or by a raw-torch
    reconstruction in this module. Reconstructed examples must also still
    occupy the no-live-backend gap they were added to cover.
    """
    pinned = sorted(_gallery_tier._QVR_REFERENCE_JOINT)
    unwitnessed = [
        example
        for example in pinned
        if not _live_backend_cells(example) and example not in _RECONSTRUCTIONS
    ]
    assert not unwitnessed, (
        f"{unwitnessed!r} carry a pinned oracle joint that nothing "
        f"outside the oracle reproduces: every backend cell is a "
        f"pinned raise or a skip, and there is no raw-torch "
        f"reconstruction here. Theorem 4.1's constant-spread quotient "
        f"cannot see a point-independent oracle error, so for these "
        f"examples the pin is a transcript of whatever the oracle "
        f"printed. Add a reconstruction to `_RECONSTRUCTIONS`, or fix "
        f"the defect blocking a backend cell."
    )

    idle = sorted(
        example for example in _RECONSTRUCTIONS if not _live_backend_cells(example)
    )
    assert idle, (
        f"every reconstructed example now has a live backend cell, so "
        f"this module is asserting nothing a container does not "
        f"already assert. That is good news about the backends and "
        f"bad news about this test: re-target the reconstructions at "
        f"whatever the current zero-cell set is, or the coverage "
        f"claim in the module docstring is stale. Reconstructed: "
        f"{sorted(_RECONSTRUCTIONS)!r}."
    )


# ---------------------------------------------------------------------
# The reconstruction itself.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cell", _cells(), ids=_cell_id)
def test_reconstruction_matches_the_oracle_per_site(
    cell: tuple[str, int],
) -> None:
    """Match the oracle at every reconstructed site and point.

    Per-site comparison prevents compensating errors from canceling in the
    joint. Unmodeled deterministic bindings must contribute exactly zero.
    """
    example, index = cell
    fixture = _fixture(example)
    labels = _gallery_data.perturbation_labels(len(fixture.points))
    terms = _reconstruct(example, index)
    oracle_sites = fixture.sites[index]

    unknown = sorted(set(terms) - set(oracle_sites))
    assert not unknown, (
        f"{example!r} point {index} ({labels[index]}): the "
        f"reconstruction scores {unknown!r}, which the trace does not "
        f"record as sites. The reconstruction is modelling a step the "
        f"program does not have."
    )

    unmodelled = sorted(
        name
        for name, value in oracle_sites.items()
        if name not in terms and value != 0.0
    )
    assert not unmodelled, (
        f"{example!r} point {index} ({labels[index]}): the oracle "
        f"scores {unmodelled!r} with a non-zero log-density and the "
        f"reconstruction does not model them, so that part of the "
        f"joint is verified by nothing. Add the term."
    )

    for name in sorted(terms):
        expected = oracle_sites[name]
        measured = terms[name]
        assert math.isfinite(measured), (
            f"{example!r} point {index} ({labels[index]}) site "
            f"{name!r}: the reconstruction is {measured!r}."
        )
        atol = _gallery_tier.reference_roundoff_atol(expected)
        assert abs(measured - expected) <= atol, (
            f"{example!r} point {index} ({labels[index]}) site "
            f"{name!r}: oracle {expected!r} against independent "
            f"reconstruction {measured!r}, a gap of "
            f"{abs(measured - expected):.6g} nats past the "
            f"{atol:.6g} per-site round-off budget. One of the two computes a "
            f"different density. Re-derive the term from the `.qvr` "
            f"source before touching either side; the ULP budget does "
            f"not move."
        )

    total = sum(terms.values())
    joint = fixture.joints[index]
    atol = _gallery_tier.reference_pin_atol(joint)
    assert abs(total - joint) <= atol, (
        f"{example!r} point {index} ({labels[index]}): every site "
        f"agreed but the joints do not ({total!r} against {joint!r}). "
        f"The trace is summing something its per-site log-densities do "
        f"not account for."
    )


@pytest.mark.parametrize(
    "example",
    sorted(_RECONSTRUCTIONS),
    ids=lambda name: name,
)
def test_reconstruction_matches_the_pinned_reference(
    example: str,
) -> None:
    """Re-derive each pinned joint from raw torch distributions."""
    reference = _gallery_tier._QVR_REFERENCE_JOINT[example]
    fixture = _fixture(example)
    labels = _gallery_data.perturbation_labels(len(fixture.points))
    assert len(reference) == len(fixture.points), (
        f"{example!r}: {len(reference)} pinned value(s) against "
        f"{len(fixture.points)} point(s)."
    )

    for index, expected in enumerate(reference):
        total = sum(_reconstruct(example, index).values())
        atol = _gallery_tier.reference_pin_atol(expected)
        assert abs(total - expected) <= atol, (
            f"{example!r} point {index} ({labels[index]}): pinned "
            f"reference {expected!r} against independent raw-torch "
            f"reconstruction {total!r}, a gap of "
            f"{abs(total - expected):.6g} nats past the {atol:.6g} "
            f"round-off budget. The pinned number is wrong, or the "
            f"program changed. Do not re-pin from the oracle's output "
            f"alone: the reconstruction is the only thing here that "
            f"can tell those two cases apart."
        )


#: Sites that contribute nothing at some points and something at the
#: rest, keyed by example and pinned to the point indices where they
#: vanish.
#:
#: A score whose argument is zero at a point contributes nothing there,
#: and that is the term working rather than the fixture failing to
#: exercise it, so long as the term is visible at the other points.
#: What has to be pinned is where the vanishing happens: a term that
#: starts vanishing at a point not listed here, or stops vanishing at
#: one that is, has changed, whether or not the joint still matches.
_PARTIALLY_INVISIBLE_SITES: dict[str, dict[str, tuple[int, ...]]] = {
    # `centering` scores `-50 * sum(theta)**2`, and the ground-truth
    # `theta` is a centered vector. Points 0, 2 and 5 leave the latents
    # at ground truth and perturb the data alone, so the penalty is
    # zero there; at 1, 3 and 4, which perturb the latents, it reaches
    # 1.85, 2.09 and 10.88 nats.
    "parametric_pooling": {"centering": (0, 2, 5)},
}


@pytest.mark.parametrize(
    "example",
    sorted(_RECONSTRUCTIONS),
    ids=lambda name: name,
)
def test_zero_scoring_sites_are_zero_by_identity(example: str) -> None:
    """Account for every site that the joint comparison cannot see.

    A summand smaller than the tolerance that comparison allows is a
    summand it cannot distinguish from absent, and two cases follow
    that need different evidence.

    A site invisible at *every* point cannot be dropped detectably at
    all, so its flatness has to hold as a family identity rather than
    as a property of the values the fixture happens to carry. That is
    checked directly, on a grid across the support.

    A site invisible at only *some* points is visible at the rest, and
    what needs pinning there is which points those are, so that a term
    whose vanishing spreads is caught by name rather than left to show
    up as a joint that still happens to match.

    Visibility is measured against
    [`reference_pin_atol`][tests.transpile.test_gallery_numeric_equivalence.reference_pin_atol]
    rather than against exact zero. A term equal to zero up to
    round-off is already invisible to the pin, and whether the last
    bits cancel to `0.0` or to `-7e-13` is a property of the
    platform's summation order, not of the model.
    """
    fixture = _fixture(example)
    reference = _gallery_tier._QVR_REFERENCE_JOINT[example]
    invisible: dict[str, set[int]] = {}
    for index in range(len(fixture.points)):
        atol = _gallery_tier.reference_pin_atol(reference[index])
        for name, value in _reconstruct(example, index).items():
            if abs(float(value)) > atol:
                continue
            invisible.setdefault(name, set()).add(index)
            assert abs(fixture.sites[index][name]) <= atol, (
                f"{example!r} site {name!r}: the reconstruction scores "
                f"{float(value)!r}, under the {atol:.6g} the pin "
                f"tolerates, and the oracle scores "
                f"{fixture.sites[index][name]!r} at point {index}."
            )

    n_points = len(fixture.points)
    everywhere = sorted(n for n, at in invisible.items() if len(at) == n_points)
    sometimes = {
        name: tuple(sorted(at)) for name, at in invisible.items() if len(at) < n_points
    }
    assert sometimes == _PARTIALLY_INVISIBLE_SITES.get(example, {}), (
        f"{example!r}: sites vanishing at some points but not others "
        f"are {sometimes!r}, against the pinned "
        f"{_PARTIALLY_INVISIBLE_SITES.get(example, {})!r}. Either the "
        f"reconstruction changed, or the fixture stopped exercising a "
        f"term it used to."
    )
    for name, at in sometimes.items():
        peak = max(
            abs(float(_reconstruct(example, index)[name]))
            for index in range(n_points)
            if index not in at
        )
        assert peak >= _SMALLEST_SCORED_SITE, (
            f"{example!r} site {name!r} vanishes at {at!r} and peaks "
            f"at {peak:.6g} nats over the points that remain, under "
            f"the {_SMALLEST_SCORED_SITE} floor. A term visible "
            f"nowhere by much is a term the pins barely constrain."
        )

    if example != "tree_categorical":
        assert not everywhere, (
            f"{example!r}: {everywhere!r} contribute nothing to the "
            f"joint at every point, so dropping them from the "
            f"reconstruction would be undetectable. Only "
            f"`tree_categorical`'s `Beta(1, 1)` splits are flat by "
            f"identity; state why these are, or the check is weaker "
            f"than its site count suggests."
        )
        return

    assert everywhere == ["p_left", "p_right", "p_root"], (
        f"tree_categorical: expected exactly the three `Beta(1, 1)` "
        f"splits to score zero; got {everywhere!r}."
    )
    grid = torch.linspace(0.05, 0.95, 19)
    flat = td.Beta(1.0, 1.0).log_prob(grid)
    assert torch.equal(flat, torch.zeros_like(flat)), (
        f"`Beta(1, 1)` is not flat across (0, 1) under this torch "
        f"build ({flat.tolist()!r}), so the three splits scoring zero "
        f"is a coincidence of the fixture's values rather than an "
        f"identity, and dropping one of them would be an undetected "
        f"defect."
    )


_SMALLEST_SCORED_SITE = 0.35
"""Floor on the smallest non-flat per-site contribution any
reconstruction makes at any point, in nats.

Each site is taken at its largest contribution across the six points,
and the floor bounds the smallest of those peaks over every site of
every reconstruction. Measured: 0.3827 nats, `tree_categorical`'s
`sigma_v` under `HalfNormal(1)`. The pin tolerance at that example's
magnitude is 7.63e-06, so even the least visible site the
reconstruction checks sits about 50000 times above the noise the
comparison tolerates. The floor sits just under the measurement so
that a term shrinking toward the tolerance trips here first."""


@pytest.mark.parametrize(
    "example",
    sorted(_RECONSTRUCTIONS),
    ids=lambda name: name,
)
def test_dropping_any_scored_site_is_rejected(example: str) -> None:
    """Require every non-flat site to affect the joint at some point.

    A term may vanish at one point, so visibility is required somewhere
    in the evaluation set. The reported minimum records the sensitivity
    margin.
    """
    fixture = _fixture(example)
    labels = _gallery_data.perturbation_labels(len(fixture.points))
    per_point = [_reconstruct(example, index) for index in range(len(fixture.points))]
    atols = [
        _gallery_tier.reference_pin_atol(sum(terms.values())) for terms in per_point
    ]

    names = sorted({name for terms in per_point for name in terms})
    tightest = math.inf
    tightest_at = ""
    for name in names:
        contributions = [abs(terms[name]) for terms in per_point]
        best = max(contributions)
        if best == 0.0:
            continue
        best_index = contributions.index(best)
        assert best > atols[best_index], (
            f"{example!r}: dropping site {name!r} shifts the joint by "
            f"at most {best:.6g} nats across the whole point set, "
            f"inside the {atols[best_index]:.6g} pin tolerance. That "
            f"site is invisible to the reference pin, so nothing here "
            f"would catch the oracle forgetting it."
        )
        if best < tightest:
            tightest = best
            tightest_at = f"{name!r} at point {best_index} ({labels[best_index]})"

    assert tightest is not math.inf, (
        f"{example!r}: every reconstructed term is exactly zero at "
        f"every point, so the comparison asserts nothing about the "
        f"joint."
    )
    assert tightest >= _SMALLEST_SCORED_SITE, (
        f"{example!r}: the least visible non-flat site now peaks at "
        f"{tightest:.6g} nats ({tightest_at}), below the pinned floor "
        f"{_SMALLEST_SCORED_SITE:.6g}. A shrinking term is a check "
        f"losing its grip; find out why the site stopped carrying "
        f"weight rather than lowering the floor."
    )


# ---------------------------------------------------------------------
# What the reconstruction rejects.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutant",
    _MUTANTS,
    ids=lambda m: m.ident,
)
def test_reconstruction_rejects_mutant(mutant: _Mutant) -> None:
    """Reject each catalogued defect above its recorded margin.

    The faithful reconstruction is accepted at the same tolerance. A
    defect need only be visible at one point because some terms vanish at
    the ground truth.
    """
    fixture = _fixture(mutant.example)
    labels = _gallery_data.perturbation_labels(len(fixture.points))
    margins: list[float] = []
    for index in range(len(fixture.points)):
        total = sum(_reconstruct(mutant.example, index, mutant.variant).values())
        assert math.isfinite(total), (
            f"{mutant.ident}: the mutant scores {total!r} at point "
            f"{index} ({labels[index]}). A mutant that leaves the "
            f"support is a different failure from one that denotes a "
            f"different measure, and is not what this catalogue "
            f"measures."
        )
        margins.append(abs(total - fixture.joints[index]))

    best = max(margins)
    best_index = margins.index(best)
    atol = _gallery_tier.reference_pin_atol(fixture.joints[best_index])
    assert best > atol, (
        f"{mutant.ident}: {mutant.defect} The mutant's largest "
        f"deviation from the oracle across the point set is "
        f"{best:.6g} nats, inside the {atol:.6g} pin tolerance, so "
        f"this defect would pass the reference pin unnoticed."
    )
    assert best >= mutant.floor, (
        f"{mutant.ident}: {mutant.defect} Rejected with margin "
        f"{best:.6g} nats at point {best_index} "
        f"({labels[best_index]}), below the pinned floor "
        f"{mutant.floor:.6g}. The margin shrank, which means the "
        f"fixture exercises this defect less than it used to. Fix the "
        f"fixture; never lower the floor to restore green."
    )


_TIGHTEST_MUTANT_MARGIN_IN_TOLERANCES = 4000.0
"""Floor on the *tightest* rejection in the mutant catalogue, measured
in multiples of the reference pin tolerance at the rejecting point.

Measured: 5678, `mixture_model`'s dropped `Dirichlet(1, 1, 1)` prior,
which shifts the joint by exactly `log Gamma(3) = 0.6931` nats against
a 1.221e-04 tolerance. The next-narrowest is 9122
(`continuous_hmm`'s emission conditioned on the wrong state) and the
catalogue runs out to 4.9e07 (`bnn`'s MLP stripped of its `tanh`).
Declaring the tightest one is what keeps the catalogue from decaying
into a set of enormous, uninformative deviations while the small,
realistic defect quietly stops being covered."""


def test_tightest_mutant_rejection_is_declared_and_holds() -> None:
    """Keep the smallest mutant margin above its declared floor."""
    ratios: list[tuple[float, str]] = []
    for mutant in _MUTANTS:
        fixture = _fixture(mutant.example)
        best = 0.0
        best_index = 0
        for index in range(len(fixture.points)):
            total = sum(_reconstruct(mutant.example, index, mutant.variant).values())
            margin = abs(total - fixture.joints[index])
            if margin > best:
                best = margin
                best_index = index
        atol = _gallery_tier.reference_pin_atol(fixture.joints[best_index])
        ratios.append((best / atol, mutant.ident))

    ratios.sort()
    tightest, ident = ratios[0]
    assert tightest >= _TIGHTEST_MUTANT_MARGIN_IN_TOLERANCES, (
        f"the catalogue's narrowest rejection is now {ident} at "
        f"{tightest:.0f} tolerances, below the declared floor "
        f"{_TIGHTEST_MUTANT_MARGIN_IN_TOLERANCES:.0f}. The check has "
        f"lost grip on its smallest covered defect. Find out why that "
        f"defect moved the joint less than it used to; do not restate "
        f"the floor."
    )


def test_mutant_catalogue_covers_every_reconstruction() -> None:
    """Require at least three live mutations for each reconstruction."""
    covered: dict[str, list[str]] = {}
    for mutant in _MUTANTS:
        covered.setdefault(mutant.example, []).append(mutant.variant)

    unknown = sorted(set(covered) - set(_RECONSTRUCTIONS))
    assert not unknown, (
        f"{unknown!r} appear in the mutant catalogue but have no "
        f"reconstruction to mutate."
    )

    thin = sorted(
        example for example in _RECONSTRUCTIONS if len(covered.get(example, [])) < 3
    )
    assert not thin, (
        f"{thin!r} carry fewer than three catalogued defects, so the "
        f"claim that their reconstruction can fail rests on almost "
        f"nothing. Add mutants covering the terms the example "
        f"actually has: a dropped prior, a mis-parameterised family, "
        f"a plate averaged instead of summed."
    )

    for example, variants in sorted(covered.items()):
        duplicates = sorted(
            variant for variant in set(variants) if variants.count(variant) > 1
        )
        assert not duplicates, (
            f"{example!r}: mutant variant(s) {duplicates!r} registered "
            f"more than once, so the catalogue's size overstates the "
            f"number of distinct defects it covers."
        )
        for variant in sorted(set(variants)):
            assert variant, (
                f"{example!r}: a mutant registered the empty variant, "
                f"which is the faithful reconstruction. It would be "
                f"'rejected' with margin zero."
            )


@pytest.mark.parametrize(
    "example",
    sorted(_RECONSTRUCTIONS),
    ids=lambda name: name,
)
def test_reconstruction_rejects_an_unknown_variant(example: str) -> None:
    """Raise for mutation variants that a reconstruction does not define."""
    with pytest.raises(AssertionError, match="unknown variant"):
        _reconstruct(example, 0, "no_such_defect")


@pytest.mark.parametrize(
    "example",
    sorted(_RECONSTRUCTIONS),
    ids=lambda name: name,
)
def test_pin_comparison_boundary_is_the_tolerance(example: str) -> None:
    """Check acceptance below and rejection above `reference_pin_atol`."""
    fixture = _fixture(example)
    for index, joint in enumerate(fixture.joints):
        atol = _gallery_tier.reference_pin_atol(joint)
        assert atol > 0.0, (
            f"{example!r} point {index}: pin tolerance is {atol!r}, so "
            f"the comparison demands bit equality of two independently "
            f"ordered float32 accumulations."
        )
        inside = joint + 0.5 * atol
        outside = joint + 2.0 * atol
        assert abs(inside - joint) <= atol, (
            f"{example!r} point {index}: a deviation of half the "
            f"tolerance was rejected, so the pin is stricter than it "
            f"claims and round-off will make it flaky."
        )
        assert not abs(outside - joint) <= atol, (
            f"{example!r} point {index}: a deviation of twice the "
            f"tolerance was accepted, so the comparison is not the "
            f"absolute band it is documented to be."
        )


def test_reference_pin_tracks_float32_roundoff_at_every_point() -> None:
    """Tie every whole-joint pin to its documented float32 ULP budget."""
    assert _gallery_tier._REFERENCE_PIN_ULP_BUDGET == 8
    mismatches: list[str] = []
    for example, values in sorted(_gallery_tier._QVR_REFERENCE_JOINT.items()):
        for index, value in enumerate(values):
            atol = _gallery_tier.reference_pin_atol(value)
            expected = _gallery_tier._REFERENCE_PIN_ULP_BUDGET * max(
                _gallery_tier._float32_ulp(value),
                _gallery_tier._float32_ulp(1.0),
            )
            if atol != expected:
                mismatches.append(
                    f"{example}[{index}]={atol:.3e}, expected {expected:.3e}"
                )
    assert not mismatches, (
        f"{mismatches!r} do not use the calibrated float32 roundoff "
        f"budget. A separate scalar cap can make a large-value pin "
        f"stricter than one representable ULP."
    )


# ---------------------------------------------------------------------
# The exemptions.
# ---------------------------------------------------------------------


def _exempt_by(registry: frozenset[str]) -> list[str]:
    return sorted(set(_gallery_tier._REFERENCE_PIN_EXEMPT) & registry)


_DECLARED_STRUCTURAL_EXEMPT: frozenset[str] = frozenset()
"""Examples claiming exemption from the reference pin on the grounds
that they declare no probabilistic program at all.

Empty, and empty on purpose. The ground is available: an example whose
`.qvr` exports a `define`d composition denotes a linear map rather
than a measure, so its source names no joint a pin could hold. But the
source is not the only place a program can come from, and both
algebra-level examples take the other route. `pmf` and
`tensor_contraction` each carry a `.md` snippet that wraps the
compiled composition in a `MonadicProgram` with entrywise
`Normal(0, 1)` priors over the declared arrows and a
`Normal(score, 0.5)` likelihood, so both score a joint at every point
of the set and both are pinned and reconstructed in this module
instead. `schema_chart_parser` and `term_autoencoder` also parse to no
`ProgramDecl` and build no dataset either, so the numeric tier never
reaches them and they need no exemption.

Declaring the emptiness rather than deriving it is the point. An
exemption added on this ground has to be written down here, next to
the reason the category was empty, and
[`test_program_free_examples_are_exempt_or_pinned`][tests.transpile.test_oracle_reference_strength.test_program_free_examples_are_exempt_or_pinned]
then has to establish it example by example."""


def _declared_programs(example: str) -> list[str]:
    """Names of the `program`s one example's `.qvr` declares."""
    source = (_SOURCE_DIR / f"{example}.qvr").read_text(encoding="utf-8")
    return [
        statement.name
        for statement in parse(source).statements
        if isinstance(statement, ProgramDecl)
    ]


@functools.cache
def _program_free_examples() -> tuple[str, ...]:
    """Gallery examples whose `.qvr` parses to no `ProgramDecl`.

    Derived from the sources rather than listed, so an example that
    loses its program declaration joins the set without anyone
    remembering to add it. This is the candidate set for the
    structural exemption: no other example could claim that ground,
    and every one of these has to be shown to fall on one side of it.
    """
    return tuple(
        path.stem
        for path in sorted(_SOURCE_DIR.glob("*.qvr"))
        if not _declared_programs(path.stem)
    )


def test_reference_pin_exemptions_are_each_covered_by_a_check() -> None:
    """Partition every reference-pin exemption into a checked category.

    The structural category is compared with its declared empty registry;
    the composition categories are checked by their per-example
    registries.
    """
    exempt = set(_gallery_tier._REFERENCE_PIN_EXEMPT)
    structural = set(_exempt_by(_gallery_tier._SKIP_DATASET_LOAD_FAILED))
    nondeterministic = set(_exempt_by(_gallery_tier._SKIP_QVR_INCOMPATIBLE))

    unchecked = sorted(exempt - structural - nondeterministic)
    assert not unchecked, (
        f"{unchecked!r} claim exemption from the reference pin without "
        f"falling into either checked category: no probabilistic "
        f"program at all, or a demonstrably non-deterministic oracle. "
        f"An exemption of a new kind needs a test that establishes it, "
        f"not a comment that asserts it."
    )
    overlap = sorted(structural & nondeterministic)
    assert not overlap, (
        f"{overlap!r} are exempt on both grounds at once. An example "
        f"either has no program or has a non-deterministic one; "
        f"claiming both means one of the two skip registries is stale."
    )

    assert structural == set(_DECLARED_STRUCTURAL_EXEMPT), (
        f"the structural exemption category holds "
        f"{sorted(structural)!r}, against the declared "
        f"{sorted(_DECLARED_STRUCTURAL_EXEMPT)!r}. Membership of this "
        f"category is a claim about which examples score no joint at "
        f"all, and it is written down rather than read off so that "
        f"the category emptying, or filling, is a decision someone "
        f"records here with its reason."
    )
    candidates = _program_free_examples()
    assert candidates, (
        "no gallery example parses to a module without a "
        "`ProgramDecl`, so nothing exercises the structural "
        "exemption's discriminator and its emptiness is untested "
        "rather than measured. Every source now declares a program: "
        "either the algebra-level examples left the gallery, or the "
        "parse is finding declarations that are not there."
    )

    assert nondeterministic, (
        "the unwitnessed exemption category is empty, so "
        "`test_unwitnessable_exempt_examples_have_no_independent_source` "
        "and `test_unwitnessable_exempt_examples_sum_each_site_once` "
        "parametrize over nothing. Every sequence model gained a "
        "witnessed joint, which is a real change worth reading the "
        "measurements for, not a green run."
    )


@pytest.mark.parametrize(
    "example",
    sorted(
        set(_program_free_examples())
        | set(_exempt_by(_gallery_tier._SKIP_DATASET_LOAD_FAILED)),
    ),
    ids=lambda name: name,
)
def test_program_free_examples_are_exempt_or_pinned(example: str) -> None:
    """Classify every source without a `ProgramDecl`.

    An example may be structurally exempt, may acquire a program from its
    synthetic-data snippet and be pinned, or may fall outside the numeric
    tier because no program is built.
    """
    programs = _declared_programs(example)
    exempt = example in _exempt_by(_gallery_tier._SKIP_DATASET_LOAD_FAILED)
    pinned = example in _gallery_tier._QVR_REFERENCE_JOINT
    dataset = _gallery_data.load_gallery_data(
        _SOURCE_DIR / f"{example}.qvr",
    )

    if exempt:
        assert not programs, (
            f"{example!r} is exempt from the reference pin on the "
            f"grounds that it declares no probabilistic program, but "
            f"it declares program(s) {programs!r}. Either the example "
            f"gained a joint and needs a `_QVR_REFERENCE_JOINT` row, "
            f"or the exemption names the wrong file."
        )
        assert dataset is None, (
            f"{example!r} declares no program yet `load_gallery_data` "
            f"built a dataset for it, so something does score. "
            f"Re-derive the exemption."
        )
        assert not pinned, (
            f"{example!r} is exempt from the reference pin and pinned "
            f"at the same time. One of the two registries describes an "
            f"example that no longer exists."
        )
        return

    if dataset is not None:
        assert pinned, (
            f"{example!r} declares no probabilistic program, but its "
            f"`.md` snippet builds one anyway and `load_gallery_data` "
            f"returns a dataset, so the oracle scores a joint for it "
            f"at every point. It carries neither a "
            f"`_QVR_REFERENCE_JOINT` row nor a structural exemption, "
            f"so that joint is asserted by nothing. Pin it, or add it "
            f"to `_DECLARED_STRUCTURAL_EXEMPT` and show that it "
            f"scores nothing."
        )
        return

    assert not pinned, (
        f"{example!r} declares no probabilistic program and "
        f"`load_gallery_data` builds no dataset for it, so there is no "
        f"joint at any point, yet `_QVR_REFERENCE_JOINT` pins one. "
        f"Nothing can re-derive that row: drop it, or restore the data "
        f"block that made the example score."
    )


# ---------------------------------------------------------------------
# The sequence-model exemptions.
#
# The five sequence models score a joint no pin can be held to, for
# want of a second source rather than of a density.
# `bidirectional_rnn_lm` and `transformer_lm` bind a latent to a
# `SampledComposition` whose factors are each scored once along the
# canonical path. `gru_lm`, `lstm_lm`, and `vanilla_rnn_lm` scan a
# program cell, whose embedding and cell draws are sites of the
# program at every position, so their joint is the sum of every
# position's densities and the emission's. Every backend refuses the
# scan or the composition, and no reconstruction re-derives these
# joints, which is measured below per example.
# ---------------------------------------------------------------------


_UNWITNESSABLE_JOINT: frozenset[str] = frozenset(
    {
        "bidirectional_rnn_lm",
        "gru_lm",
        "lstm_lm",
        "transformer_lm",
        "vanilla_rnn_lm",
    }
)
"""Exempt examples whose joint a pin could hold, but which nothing
outside the oracle reproduces.

Every factor of each composition, and every site a scan reaches at
each position, carries a density scored once, so the joint is exact,
needs no quadrature rule, and is bitwise reproducible. What is missing
is a second source for the number. All five refuse on every backend, so
no container re-derives them, and none carries a reconstruction here.

A reconstruction is what closes each row, and for the two compositions
it is not merely unwritten. Their chains run through a scan whose cell
carries its own density, so the prefix is a recurrence, and as
[`ScanMorphism.log_prob`][quivers.continuous.scan.ScanMorphism.log_prob]
records, a perturbation of the state grows by roughly an order of
magnitude per step.

That was measured on `bidirectional_rnn_lm` rather than assumed. A
reconstruction of it written from the source, calling the same
``torch.nn.functional.linear`` the parameter source calls, reproduces
the embedding, both recurrent paths, and the canonical prefix they
produce, bitwise. What it does not reproduce bitwise is the last
factor, which scores `h` against a combiner whose predicted scale is
small compared with the distance it covers. That term is near -1.5e08,
and the assembled site lands 16 nats away.

Sixteen nats is not a discrepancy to chase. The float32 grid spacing
at that magnitude *is* 16, so the reconstruction differs from the
oracle by one unit in the last place, the smallest non-zero difference
either can express.
`test_reconstruction_matches_the_oracle_per_site` holds an absolute
budget some thirty thousand times finer than that spacing, so nothing
short of a bitwise-identical accumulation clears it.

`transformer_lm` was written too, and lands in the same place from a
different direction. It carries no recurrence at all: it is one
feed-forward tower, the same shape as either arm of `seq2seq`, which
*is* pinned here because its reconstruction agrees within the
calibrated float32 roundoff budget. What
separates them is where the chain ends. `seq2seq` hands its towers'
origin outputs to a further factor, so every factor is scored at its
own location and nothing is evaluated far from it; `transformer_lm`
ends at `h`, and its last factor scores a value the canonical path did
not produce. That single term is near -4e06 and its sensitivity to the
prefix runs as `(h - mu) / sigma^2`, so the reconstruction's token
likelihood reproduces exactly while its `h` lands 0.5 nats out. The
float32 spacing at -4e06 is 0.25, so that too is two units in the last
place, and still far finer than any budget an absolute check states.

So the missing witness in both rows is not a missing afternoon.
`test_reconstruction_matches_the_oracle_per_site` holds an absolute
budget, and agreement against a term of this size would need the
reconstruction to accumulate in the same order as the object under
test, which is mirroring rather than witnessing: a pin resting on it
would certify the implementation against itself. What would close
these rows is the joint's conditioning improved, so that what a factor
scores lies within reach of what it predicts, or the comparison recast
onto the per-factor terms, each of which does agree exactly.

`gru_lm`, `lstm_lm`, and `vanilla_rnn_lm` carry no reconstruction yet,
and their case is different. A scan of a program cell scores every
position's embedding and cell draw at its own value, so each term is
evaluated where it was drawn and a reconstruction from raw
`torch.distributions` is within reach; until one lands, these rows rest
on the missing witness alone."""


@functools.cache
def _exempt_trace(example: str, index: int) -> Trace:
    """One trace of an exempt example at one point, memoised.

    [`_fixture`][tests.transpile.test_oracle_reference_strength._fixture]
    keeps only the numbers it needs; the checks below need the
    `SampleSite` records themselves, to see which morphism each site
    was drawn from.
    """
    return _exempt_traces(example, _fixture(example).points[index])[0]


def _exempt_traces(example: str, point: Point) -> list[Trace]:
    """Trace an exempt example at an arbitrary point, unmemoised."""
    fixture = _fixture(example)
    monadic = fixture.dataset.monadic
    if monadic is None:
        raise AssertionError(
            f"{example!r}: the synthetic-data block bound no compiled "
            f"`MonadicProgram`, so there is nothing to trace."
        )
    return reference_traces(
        monadic,
        point,
        x_input=fixture.dataset.x_input,
        observations=_gallery_data.observations_for_point(
            fixture.dataset,
            point,
        ),
    )


def test_composition_exemption_grounds_partition_the_registry() -> None:
    """Assign every composition exemption its measured ground."""
    declared = set(_exempt_by(_gallery_tier._SKIP_QVR_INCOMPATIBLE))
    unwitnessed = set(_UNWITNESSABLE_JOINT)

    unchecked = sorted(declared - unwitnessed)
    assert not unchecked, (
        f"{unchecked!r} are exempt from the reference pin, but no "
        f"check below establishes why their joint carries no pin. "
        f"Measure the example: if nothing outside the oracle "
        f"re-derives its joint, add it to `_UNWITNESSABLE_JOINT`; "
        f"otherwise the joint is a value and the example wants a "
        f"`_QVR_REFERENCE_JOINT` row rather than an exemption."
    )
    stale = sorted(unwitnessed - declared)
    assert not stale, (
        f"{stale!r} carry a ground for exemption but are not exempt: "
        f"either they left `_SKIP_QVR_INCOMPATIBLE` or the ground "
        f"names the wrong example."
    )
    assert unwitnessed, (
        "the unwitnessed ground is empty, so the checks below "
        "parametrize over nothing and pass vacuously."
    )


@pytest.mark.parametrize(
    "example",
    sorted(_UNWITNESSABLE_JOINT),
    ids=lambda name: name,
)
def test_unwitnessable_exempt_examples_have_no_independent_source(
    example: str,
) -> None:
    """Confirm that unwitnessed exemptions lack a live backend or reconstruction.

    These programs produce reproducible, point-varying path densities, but
    no independent implementation currently re-derives those values.
    """
    live = _live_backend_cells(example)
    assert not live, (
        f"{example!r} now has live backend cells {live!r}, so its "
        f"joint is re-derived outside the oracle and the "
        f"want-of-a-witness ground is gone. Pin the joint at every "
        f"point and drop the row from `_UNWITNESSABLE_JOINT` and "
        f"from `_REFERENCE_PIN_EXEMPT`."
    )
    assert example not in _RECONSTRUCTIONS, (
        f"{example!r} now carries a reconstruction, which is a "
        f"witness. Pin the joint at every point and drop the row "
        f"from `_UNWITNESSABLE_JOINT` and from "
        f"`_REFERENCE_PIN_EXEMPT`."
    )

    fixture = _fixture(example)
    joints = [
        float(_exempt_trace(example, index).log_joint.sum().item())
        for index in range(len(fixture.points))
    ]
    assert all(math.isfinite(value) for value in joints), (
        f"{example!r} scores a non-finite joint {joints!r}, so the "
        f"ground for its exemption is not the missing witness this "
        f"registry names but something worse. Re-derive it."
    )
    assert len(set(joints)) > 1, (
        f"{example!r} scores the same joint {joints[0]!r} at every "
        f"point of the set, so the perturbation schedule does not "
        f"move it and a pin here would say nothing about the data. "
        f"That is a different exemption from this one; re-derive it."
    )


@pytest.mark.parametrize(
    "example",
    sorted(_UNWITNESSABLE_JOINT),
    ids=lambda name: name,
)
def test_unwitnessable_exempt_examples_sum_each_site_once(
    example: str,
) -> None:
    """Require the joint to add each site density once.

    Plate-shaped terms must not broadcast a scalar density across the
    plate before reduction, and a scan's per-position sites must each
    reach the joint once.
    """
    fixture = _fixture(example)
    labels = _gallery_data.perturbation_labels(len(fixture.points))

    for index in range(len(fixture.points)):
        traced = _exempt_trace(example, index)
        joint = float(traced.log_joint.sum().item())
        per_site = sum(
            float(site.log_prob.sum().item()) for site in traced.sites.values()
        )
        tolerance = _gallery_tier.reference_pin_atol(abs(joint))
        assert abs(joint - per_site) <= tolerance, (
            f"{example!r} point {index} ({labels[index]}): the joint "
            f"{joint!r} differs from the sum of its per-site "
            f"densities {per_site!r} by {abs(joint - per_site)!r} "
            f"nats, more than the {tolerance!r} round-off this "
            f"magnitude allows. A site is being counted more than "
            f"once, or not at all: a plate-shaped term broadcasting a "
            f"scalar density across the row plate is one way that "
            f"happens. Find the reduction that widened rather than "
            f"widening this tolerance."
        )
