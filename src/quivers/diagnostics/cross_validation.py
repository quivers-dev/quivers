"""Explicit refit partitions and held-out posterior predictive scores."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
import heapq
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.diagnostics._numeric import scaled_samples

type FoldLabel = str | int
"""A retained group or stratum label; labels are never inferred from outcomes."""


class FoldSplit(ValueModel):
    """Original row identities for one refit and its held-out prediction.

    Parameters
    ----------
    fold : int
        Zero-based fold identity in the complete plan.
    train : tuple[int, ...]
        Original row positions supplied to the refitting procedure.
    test : tuple[int, ...]
        Original row positions to predict, in their original order.
    omitted : tuple[int, ...]
        Rows excluded from training, possibly including other group members.
    seed : int
        Fold-specific seed independent of evaluation order or selected folds.
    """

    fold: int
    train: tuple[int, ...]
    test: tuple[int, ...]
    omitted: tuple[int, ...]
    seed: int


class FoldPlan(ValueModel):
    """A reusable complete partition, independent of model and fitting backend.

    Parameters
    ----------
    assignments : tuple[int, ...]
        One zero-based fold per original included row. Every fold must occur.
    seed : int
        Nonnegative seed used to derive independent refit streams.
    omission_groups : tuple[str or int, ...]
        Optional grouping identities. A fold also omits every member of a
        held-out row's group, while predicting only its designated test rows.
        This expresses leave-one-out prediction with whole-group exclusion.
    """

    assignments: tuple[int, ...]
    seed: int = 0
    omission_groups: tuple[FoldLabel, ...] = ()

    @dx.model_validator()
    def _validate(self) -> None:
        """Check an explicit partition without changing its row identities.

        Raises
        ------
        ValueError
            If labels, folds, or the seed cannot describe a complete partition.
        """
        if (
            len(self.assignments) < 2
            or any(type(value) is not int or value < 0 for value in self.assignments)
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError(
                "fold plans require nonnegative integer folds and a seed in [0, 2**63)"
            )
        labels = set(self.assignments)
        if len(labels) < 2 or labels != set(range(len(labels))):
            raise ValueError(
                "fold assignments must contain every zero-based fold, with at least two folds"
            )
        if self.omission_groups:
            _labels(self.omission_groups, len(self.assignments), "omission groups")

    @property
    def num_folds(self) -> int:
        """Return the number of nonempty folds.

        Returns
        -------
        int
            Complete partition size.
        """
        return max(self.assignments) + 1

    def split(self, fold: int) -> FoldSplit:
        """Resolve training, omitted, and prediction rows for one fold.

        Parameters
        ----------
        fold : int
            Zero-based fold identity.

        Returns
        -------
        FoldSplit
            Original positions; an explicitly omitted whole group can make
            training empty, in which case the refitter must support prior fits.

        Raises
        ------
        ValueError
            If the requested fold is outside the complete partition.
        """
        if type(fold) is not int or not 0 <= fold < self.num_folds:
            raise ValueError("fold index is outside the complete partition")
        test = tuple(i for i, value in enumerate(self.assignments) if value == fold)
        if self.omission_groups:
            excluded = {self.omission_groups[i] for i in test}
            omitted = tuple(
                i for i, group in enumerate(self.omission_groups) if group in excluded
            )
        else:
            omitted = test
        omit = set(omitted)
        return FoldSplit(
            fold=fold,
            train=tuple(i for i in range(len(self.assignments)) if i not in omit),
            test=test,
            omitted=omitted,
            seed=(self.seed + fold + 1) % 2**63,
        )


def _labels(values: Sequence[FoldLabel], size: int, name: str) -> tuple[FoldLabel, ...]:
    result = tuple(values)
    if len(result) != size or any(type(value) not in {str, int} for value in result):
        raise ValueError(f"{name} require one string or integer label per original row")
    return result


def fold_plan(
    size: int,
    *,
    folds: int = 10,
    groups: Sequence[FoldLabel] | None = None,
    strata: Sequence[FoldLabel] | None = None,
    seed: int = 0,
) -> FoldPlan:
    """Construct seeded balanced random, grouped, or stratified folds.

    Parameters
    ----------
    size : int
        Number of original included response rows.
    folds : int
        Number of nonempty folds, at most the number of independent groups.
        Setting this to size gives leave-one-out when groups are absent.
    groups : Sequence[str or int] or None
        Indivisible units assigned wholly to one fold.
    strata : Sequence[str or int] or None
        Labels balanced across folds. Every indivisible group must have one
        stratum; small strata need not occur in every fold.
    seed : int
        Local random stream; the caller's RNG is preserved.

    Returns
    -------
    FoldPlan
        Original row order and exact assignments for reuse across models.

    Raises
    ------
    ValueError
        If counts, labels, or group/stratum combinations are invalid.
    """
    if type(size) is not int or size < 2 or type(folds) is not int or folds < 2:
        raise ValueError("fold construction requires at least two rows and two folds")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("fold seed must lie in [0, 2**63)")
    group_values = (
        tuple(range(size)) if groups is None else _labels(groups, size, "groups")
    )
    stratum_values = (0,) * size if strata is None else _labels(strata, size, "strata")
    units: dict[FoldLabel, list[int]] = {}
    for row, group in enumerate(group_values):
        units.setdefault(group, []).append(row)
    if folds > len(units):
        raise ValueError("fold count exceeds the number of indivisible groups")
    by_stratum: dict[FoldLabel, list[list[int]]] = {}
    for rows in units.values():
        stratum = stratum_values[rows[0]]
        if any(stratum_values[row] != stratum for row in rows):
            raise ValueError("each indivisible group must belong to one stratum")
        by_stratum.setdefault(stratum, []).append(rows)
    generator = torch.Generator().manual_seed(seed)
    totals = [0] * folds
    assignments = [0] * size
    priorities = torch.randperm(folds, generator=generator).tolist()
    global_heap = [(0, priorities[i], i) for i in range(folds)]
    heapq.heapify(global_heap)
    for values in by_stratum.values():
        # Shuffle equal-size units before balancing large groups first.
        order = torch.randperm(len(values), generator=generator).tolist()
        ordered = sorted((values[i] for i in order), key=len, reverse=True)
        if len(ordered) < folds:
            # Every group can occupy a distinct fold in this stratum. Holding
            # selected folds out of the heap avoids O(K) work per small stratum.
            used = []
            for rows in ordered:
                _, priority, fold = heapq.heappop(global_heap)
                totals[fold] += len(rows)
                used.append((totals[fold], priority, fold))
                for row in rows:
                    assignments[row] = fold
            for item in used:
                heapq.heappush(global_heap, item)
        else:
            local_heap = [(0, totals[i], priorities[i], i) for i in range(folds)]
            heapq.heapify(local_heap)
            for rows in ordered:
                count, _, priority, fold = heapq.heappop(local_heap)
                totals[fold] += len(rows)
                heapq.heappush(
                    local_heap, (count + len(rows), totals[fold], priority, fold)
                )
                for row in rows:
                    assignments[row] = fold
            global_heap = [(totals[i], priorities[i], i) for i in range(folds)]
            heapq.heapify(global_heap)
    return FoldPlan(assignments=tuple(assignments), seed=seed)


class FoldPrediction(ValueModel):
    """Raw held-out likelihoods returned by a refitting adapter.

    Parameters
    ----------
    log_likelihood : Mapping[str, torch.Tensor]
        Response scores of shape (chain, draw, test row), unweighted and with
        each response event already reduced. Joint responses share draw axes.
    fit : object or None
        Optional actual fitted result, retained only when save_fits is true.
    """

    log_likelihood: Mapping[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    fit: object | None = dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)


class CrossValidationResult(ValueModel):
    """Predictive scores from explicit refits on a retained partition.

    Parameters
    ----------
    plan : FoldPlan
        Complete original partition.
    fold_ids : tuple[int, ...]
        Evaluated folds; a subset is never extrapolated to a full result.
    responses : tuple[str, ...]
        Selected response names, summed within each joint posterior draw.
    joint : str
        Predictive unit: observation, whole fold, or group within fold.
    units : tuple[tuple[int, tuple[int, ...]], ...]
        Fold identity and original response positions for every scored unit.
    log_predictive_density : torch.Tensor
        Float64 log-mean likelihood per unit, preserving zero densities as -inf.
    fits : Mapping[int, object]
        Optional actual refits keyed by complete-plan fold identity.
    """

    plan: FoldPlan
    fold_ids: tuple[int, ...]
    responses: tuple[str, ...]
    joint: Literal["observation", "fold", "group"]
    units: tuple[tuple[int, tuple[int, ...]], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    log_predictive_density: torch.Tensor = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    fits: Mapping[int, object] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @dx.model_validator()
    def _validate(self) -> None:
        """Require scores to cover exactly the selected original row units.

        Raises
        ------
        ValueError
            If scores, selected folds, or original response positions disagree.
        """
        selected = set(self.fold_ids)
        count = self.plan.num_folds
        if (
            not selected
            or len(selected) != len(self.fold_ids)
            or any(type(fold) is not int or not 0 <= fold < count for fold in selected)
            or not self.responses
            or len(set(self.responses)) != len(self.responses)
            or any(not isinstance(name, str) or not name for name in self.responses)
        ):
            raise ValueError(
                "cross-validation selections do not describe a nonempty result"
            )
        scores = self.log_predictive_density
        if (
            not isinstance(scores, torch.Tensor)
            or not scores.is_floating_point()
            or scores.shape != (len(self.units),)
            or not self.units
            or bool((torch.isnan(scores) | torch.isposinf(scores)).any())
        ):
            raise ValueError(
                "cross-validation scores require one real log density per unit"
            )
        expected: dict[int, set[int]] = {fold: set() for fold in selected}
        for row, fold in enumerate(self.plan.assignments):
            if fold in expected:
                expected[fold].add(row)
        seen: dict[int, set[int]] = {fold: set() for fold in selected}
        for fold, rows in self.units:
            if (
                fold not in selected
                or not rows
                or len(set(rows)) != len(rows)
                or any(type(row) is not int for row in rows)
                or not set(rows) <= expected[fold]
                or bool(seen[fold] & set(rows))
                or self.joint == "observation"
                and len(rows) != 1
                or self.joint == "fold"
                and set(rows) != expected[fold]
            ):
                raise ValueError(
                    "predictive units must partition the designated held-out rows"
                )
            seen[fold].update(rows)
        if seen != expected or not set(self.fits) <= selected:
            raise ValueError(
                "result rows or retained fits do not match the selected folds"
            )

    @property
    def complete(self) -> bool:
        """Report whether every fold in the retained partition was evaluated.

        Returns
        -------
        bool
            False for a partial fold selection, regardless of score values.
        """
        return set(self.fold_ids) == set(range(self.plan.num_folds))

    @property
    def elpd(self) -> torch.Tensor:
        """Sum evaluated predictive units without extrapolating missing folds.

        Returns
        -------
        torch.Tensor
            Scalar score on the natural-log scale; -inf remains -inf.
        """
        return self.log_predictive_density.sum()

    @property
    def se(self) -> torch.Tensor:
        """Return sqrt(N times sample variance) across evaluated score units.

        Returns
        -------
        torch.Tensor
            Descriptive sampling-unit SE, not a Monte Carlo SE or a correction
            for overlapping training folds. Undefined for fewer than two units
            or nonfinite scores, in which case it is NaN.
        """
        scores = self.log_predictive_density
        if scores.numel() < 2 or not bool(torch.isfinite(scores).all()):
            return scores.new_tensor(float("nan"))
        normalized, _, scale = scaled_samples(scores)
        return normalized.std(correction=1) * math.sqrt(scores.numel()) * scale


def cross_validate(
    plan: FoldPlan,
    refit: Callable[[FoldSplit], FoldPrediction],
    *,
    fold_ids: Sequence[int] | None = None,
    responses: Sequence[str] | None = None,
    joint: Literal["observation", "fold", "group"] = "observation",
    groups: Sequence[FoldLabel] | None = None,
    save_fits: bool = False,
) -> CrossValidationResult:
    """Refit selected partitions and score held-out units within joint draws.

    Parameters
    ----------
    plan : FoldPlan
        Explicit reusable partition of original included rows.
    refit : Callable[[FoldSplit], FoldPrediction]
        Adapter that fits only split.train and scores exactly split.test.
        It owns preprocessing, prior/basis policy, inference quality, and raw
        observation semantics; the runner does not fit another model itself.
    fold_ids : Sequence[int] or None
        Nonempty unique subset of complete-plan folds, or all folds.
    responses : Sequence[str] or None
        Explicit joint response selection; required with multiple scored names.
    joint : {"observation", "fold", "group"}
        Sum row/response log likelihoods within a draw before log-mean-exp.
    groups : Sequence[str or int] or None
        Original row labels for joint group-within-fold prediction units.
    save_fits : bool
        Retain actual callback results and their omitted-row metadata in plan.

    Returns
    -------
    CrossValidationResult
        Conditional predictive scores from refits, retaining exact row units.
        Leave-one-out partitions still have the supplied fitter's approximation.

    Raises
    ------
    ValueError
        If selections, response shapes, or scores are invalid or inconsistent.
    TypeError
        If a refitter does not return raw real floating-point likelihoods.
    """
    from quivers._rng import seeded_rng

    chosen = tuple(range(plan.num_folds)) if fold_ids is None else tuple(fold_ids)
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("select a nonempty set of unique fold identities")
    count = plan.num_folds
    if any(type(fold) is not int or not 0 <= fold < count for fold in chosen):
        raise ValueError("fold index is outside the complete partition")
    if joint not in {"observation", "fold", "group"} or type(save_fits) is not bool:
        raise ValueError("invalid predictive unit or save_fits policy")
    labels = (
        None
        if groups is None
        else _labels(groups, len(plan.assignments), "score groups")
    )
    if (joint == "group") != (labels is not None):
        raise ValueError("score groups must be supplied exactly when joint='group'")
    selected = None if responses is None else tuple(responses)
    if selected is not None and (
        not selected
        or len(set(selected)) != len(selected)
        or any(not isinstance(name, str) or not name for name in selected)
    ):
        raise ValueError("response selection must contain unique nonempty names")
    values: list[torch.Tensor] = []
    units: list[tuple[int, tuple[int, ...]]] = []
    fits: dict[int, object] = {}
    device: torch.device | None = None
    for fold in chosen:
        split = plan.split(fold)
        with seeded_rng(split.seed):
            prediction = refit(split)
        if not isinstance(prediction, FoldPrediction):
            raise TypeError("refit must return FoldPrediction with raw held-out scores")
        if selected is None:
            if len(prediction.log_likelihood) != 1:
                raise ValueError(
                    "select responses explicitly when a refit scores multiple names"
                )
            selected = tuple(prediction.log_likelihood)
        score: torch.Tensor | None = None
        for name in selected:
            if name not in prediction.log_likelihood:
                raise ValueError(f"refit lacks selected response {name!r}")
            raw = prediction.log_likelihood[name]
            if not isinstance(raw, torch.Tensor) or not raw.is_floating_point():
                raise TypeError(
                    "held-out log likelihoods must be real floating-point tensors"
                )
            if (
                raw.ndim != 3
                or min(raw.shape[:2]) < 1
                or raw.shape[2] != len(split.test)
            ):
                raise ValueError(
                    "held-out scores require nonempty chain/draw axes and exactly the designated test rows"
                )
            value = raw.detach().to(dtype=torch.float64)
            if bool((torch.isnan(value) | torch.isposinf(value)).any()):
                raise ValueError(
                    "held-out scores cannot contain NaN or positive infinity"
                )
            if device is not None and value.device != device:
                raise ValueError("all refit scores must use the same device")
            device = value.device
            if score is not None and score.shape != value.shape:
                raise ValueError(
                    "joint responses must share aligned chain, draw, and row axes"
                )
            score = value if score is None else score + value
        assert score is not None
        positions: list[list[int]]
        if joint == "observation":
            positions = [[i] for i in range(len(split.test))]
        elif joint == "fold":
            positions = [list(range(len(split.test)))]
        else:
            assert labels is not None
            buckets: dict[FoldLabel, list[int]] = {}
            for i, original in enumerate(split.test):
                buckets.setdefault(labels[original], []).append(i)
            positions = list(buckets.values())
        # Pool sample axes only after summing each predictive unit within draw.
        per_unit = (
            score
            if joint == "observation"
            else torch.stack([score[..., rows].sum(-1) for rows in positions], -1)
        )
        pooled = per_unit.flatten(0, 1)
        values.append(torch.logsumexp(pooled, 0) - math.log(pooled.shape[0]))
        units.extend(
            (split.fold, tuple(split.test[i] for i in rows)) for rows in positions
        )
        if save_fits:
            if prediction.fit is None:
                raise ValueError(
                    "save_fits requires an actual fitted result from every refitter"
                )
            fits[split.fold] = prediction.fit
        del prediction
    assert selected is not None
    return CrossValidationResult(
        plan=plan,
        fold_ids=chosen,
        responses=selected,
        joint=joint,
        units=tuple(units),
        log_predictive_density=torch.cat(values),
        fits=fits,
    )


__all__ = [
    "FoldLabel",
    "FoldSplit",
    "FoldPlan",
    "FoldPrediction",
    "CrossValidationResult",
    "fold_plan",
    "cross_validate",
]
