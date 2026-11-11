"""Expected log-density offsets between QVR and backend probes.

The equivalence relation permits one point-independent additive
constant. This module records those constants for supported gallery
cells and checks them separately from the residual-spread test. It
also verifies that offset registries cover live cells, that cached
probe results are isolated by inputs and harness identity, and that
changes to probe scripts or points cannot reuse stale measurements.

Offsets are evaluated on deterministic multi-point sets. A shifted
backend vector must move the fitted constant by the same amount while
leaving its spread unchanged. Mutations that change the density by a
nonconstant term must still fail the equivalence check.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from typing import Literal

import didactic.api as dx
import pytest

from quivers.dsl.ast_nodes import (
    DiscreteConstructor,
    DrawArgName,
    ExportDecl,
    ExprIdent,
    LetExprVar,
    MarginalizeStep,
    Module,
    ObjectDecl,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    SampleStep,
    ScoreStep,
    TypeFromExpr,
)
from quivers.dsl.parser import parse
from quivers.dsl.program_templates import instantiate_program, template_bindings
from quivers.transpile import UnsupportedConstruct, transpile
from quivers.transpile.family_meta import FAMILY_META
from tests.transpile import _docker, _equivalence, _gallery_data
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.qvr import QvrProbe
from tests.transpile.test_gallery_numeric_equivalence import (
    _BACKENDS_WITH_IMAGES,
    _EXPECTED_TRANSPILE_RAISES,
    _SKIP_DATASET_LOAD_FAILED,
    _SKIP_PROBE_INCOMPATIBLE,
    _SKIP_QVR_INCOMPATIBLE,
    _dtypes_from_dataset,
    _gallery_cells,
    _shapes_from_dataset,
)


_LOG_2 = math.log(2.0)
"""The truncation renormalizer of a folded half-family, `log 2`.

`HalfCauchy`, `HalfNormal` and `HalfStudentT` are the base density
restricted to the non-negative half-line and rescaled by two. A target
that emits the symmetric base without a renormalization scores this
much below the reference at every point of the support."""

_HALF_FAMILIES: frozenset[str] = frozenset({"HalfCauchy", "HalfNormal", "HalfStudentT"})
"""QVR families whose density carries a folding factor of two.

Membership is a statement about the QVR-side density, independent of
how any target spells the family. A new folded family added to
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] belongs here
and needs its own row in the drop table below.

`HalfStudentT` folds `student_t(df, 0, scale)` over the non-negative
half-line exactly as `HalfCauchy` folds `cauchy(0, gamma)`, so it
carries the same `log 2`, and it is spelled as the bare `student_t` on
Stan; its drop set is therefore narrower than the other two, which is
why the table below is keyed by family.

`Truncated` / `TruncatedNormal` stay outside the set: they renormalize
by `log(F(b) - F(a))`, which is a function of the bounds and of the
family's own arguments rather than a constant, so a target that drops
it fails the constant-spread leg whenever those arguments are latent
and needs a per-family term here when they are not."""

_DROPS_HALF_NORMALIZER: dict[str, frozenset[str]] = {
    "HalfCauchy": frozenset({"stan", "gen", "webppl"}),
    "HalfNormal": frozenset({"stan", "gen", "webppl"}),
    "HalfStudentT": frozenset({"stan", "edward2"}),
}
"""Per folded family, the targets whose emit is the symmetric base
density with no truncation renormalization.

Read off the *effective* spelling of the family at the draw site, which
is [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] amended
by
[`_RENDERER_SPELLING_OVERRIDES`][tests.transpile.test_expected_offsets._RENDERER_SPELLING_OVERRIDES].
`HalfCauchy.target_names` maps `stan -> "cauchy"`, `gen -> "cauchy"`,
`webppl -> "Cauchy"`, all symmetric bases emitted without a truncation
wrapper, against `numpyro / pyro / pymc / edward2 -> "HalfCauchy"`
(native folded class), `turing -> "truncated"` (Distributions.jl
renormalizes) and `bugs / jags -> "dt"` carrying a one-sided truncation
suffix that JAGS renormalizes over.

`HalfStudentT` splits differently. `stan -> "student_t"` is a bare
base, and so is `edward2`: TFP ships no `HalfStudentT` class, so
`renderers/edward2.py` rewrites the site to the location-scale
`edward2.StudentT(df, 0, scale)` and leans on the fold being an
additive-constant shift, which is exactly a dropped `log 2` per site.
`gen -> "half_student_t"`, `turing -> "HalfStudentT"` and
`webppl -> "HalfStudentT"` all resolve to a grafted runtime helper
whose scorer adds `log 2` explicitly, so those three targets keep the
renormalizer for this family while dropping it for the other two. A
single per-target flag cannot express that, hence the keying by family.

`bugs` and `jags` keep a zero here for `HalfStudentT` for the same
reason they do for the other two folded families: the renderer writes
`sigma ~ dt(0,1/(scale*scale),df) T (0 ,)`, the symmetric base under a
one-sided truncation suffix, and JAGS renormalizes over the truncation
interval. The suffix is the whole of the entitlement question on these
two targets, and it is what
`test_drop_table_agrees_with_the_family_registry` cannot read off the
name table: `dt` and `dnorm` are symmetric spellings whether or not a
`T ( , )` follows, so the assertion below keeps them out of
`bare_base` on the strength of the suffix rather than of the name.

`test_drop_table_agrees_with_the_family_registry` re-derives each set
from the effective spelling and checks every override against the
emitted program, so a renderer that switches to a native or truncated
spelling cannot leave a stale entry behind."""


_RENDERER_SPELLING_OVERRIDES: dict[tuple[str, str], str] = {
    ("edward2", "HalfStudentT"): "StudentT",
}
"""Draw-site spellings a renderer substitutes for the registered one.

[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] records the
name a family resolves to on each target, but a renderer may replace
that name at the site when the target ships no such class. Reading the
name table alone would then charge the cell nothing while its emit
drops a renormalizer at every site, which is precisely the
point-independent error this module exists to catch.

`edward2` is the one such case. TFP has no `HalfStudentT`, so
`renderers/edward2.py` emits `edward2.StudentT(df, 0, scale)` and
carries the non-negativity nowhere at all, while `FAMILY_META` still
reads `edward2 -> "HalfStudentT"`.

Each entry is checked against the emitted program by
`test_drop_table_agrees_with_the_family_registry`: the substituted name
must appear in the emit and the registered name must not, so an
override cannot outlive the renderer branch that motivated it."""


_HALF_FAMILY_PROBE_ARGS: dict[str, str] = {
    "HalfCauchy": "HalfCauchy(scale=1.0)",
    "HalfNormal": "HalfNormal(scale=1.0)",
    "HalfStudentT": "HalfStudentT(df=3.0, scale=1.0)",
}
"""Call spelling used to make each folded family emit a draw site.

Argument values are irrelevant to the spelling the renderer picks; the
arity is not, since a family is resolved by name and arity together."""


_HALF_FAMILY_PROBE_SOURCE = """object Resp : FinSet 3
object Val : Real 1

program probe : Resp -> Val
    sample v <- {call}
    observe y : Resp <- Normal(loc=v, scale=1.0)
    return v

export probe
"""
"""Smallest QVR module that puts one folded-family draw on every target.

A single scalar latent scored by one plated observation. Every backend
in the Docker matrix renders it, so the emitted text is available for
each `(target, family)` pair without a container and without depending
on which gallery example happens to use the family."""


_ZEROS_TRICK_OFFSET: float = 1.0e6
"""The constant `C` a lifted BUGS / JAGS zeros-trick row subtracts.

`zeros[n] ~ dpois(C - log f(y_n))` scored against a host-bound
`zeros[n] = 0` adds `log f(y_n) - C` to the joint, so a program that
lowers a site this way scores `C` below the reference at every row,
whatever the parameter point. The lift exists only to keep the Poisson
rate positive where the negated closed form is not, and `1e6` is the
literal `renderers/jags.py` picks for it.

The value is not taken on trust:
`test_zeros_trick_table_agrees_with_the_emit` parses it back out of the
`phi_<site>[n] <- <C>-log(...)` relation the renderer emits, so a
renderer that changes the lift, or drops it as
`_emit_beta_binomial` already has, moves the pin instead of leaving
this number stale."""

_ZEROS_TRICK_FAMILIES: frozenset[str] = frozenset(
    {"MixtureNormal", "BetaBinomial", "Kumaraswamy", "score"}
)
"""QVR families the BUGS / JAGS renderers score through the zeros
trick rather than through a named distribution, and the `score` step,
which both renderers lower through the same trick as one row.

Membership is a statement about the lowering, not about the lift: both
families reach the engine as `zeros[n] ~ dpois(phi[n])`, and only the
subset in
[`_ZEROS_TRICK_OFFSET_FAMILIES`][tests.transpile.test_expected_offsets._ZEROS_TRICK_OFFSET_FAMILIES]
pays a constant for it. The set is kept apart from that table so the
emit check has a negative control: a family that goes through the
trick with no lift is the evidence that the lift is a choice the
renderer makes per family rather than a property of the idiom."""

_ZEROS_TRICK_OFFSET_FAMILIES: dict[str, frozenset[str]] = {
    "MixtureNormal": frozenset({"jags"}),
    "BetaBinomial": frozenset(),
    "Kumaraswamy": frozenset({"jags"}),
    "score": frozenset({"jags", "bugs"}),
}
"""Per zeros-trick family, the targets whose emit lifts the Poisson
rate by [`_ZEROS_TRICK_OFFSET`][tests.transpile.test_expected_offsets._ZEROS_TRICK_OFFSET].

`MixtureNormal` lowers to `phi[n] <- 1e6-log(<mixture density>)`, whose
inner term is a density value and so may exceed one; the lift is what
keeps the rate positive when it does. `Kumaraswamy` lowers the same
way and for the same reason: it is a density on `(0, 1)` rather than a
mass function, so its log form exceeds zero wherever the density does.
A `score` step is no family, but both renderers lower it through the
same trick as one lifted row, `C_<name> <- 1e6 - (<expr>)` scored by
`zero_<name> ~ dpois(C_<name>)`, so the table carries it under the
name `score` and every `score` step of a source counts one row.
`BetaBinomial` lowers to `phi[n] <- -(<log pmf>)`, already positive
because a pmf is at most one, and `renderers/jags.py` emits it with no
lift at all. The three therefore do not sit on one side of this table
even though they share the idiom.

`bugs` appears nowhere: it has no target name for any of the three and
raises before reaching the trick, which
`test_zeros_trick_table_agrees_with_the_emit` confirms rather than
assumes. The `bugs` renderer does carry the same lift on its `score`
statement path, and no gallery example currently exercises it;
[`zeros_trick_factor_counts`][tests.transpile.test_expected_offsets.zeros_trick_factor_counts]
raises rather than undercount if one ever does."""

_ZEROS_TRICK_PROBE_SOURCES: dict[str, str] = {
    "MixtureNormal": """object Component : FinSet 3
object Resp : FinSet 5
object Weights : Real 3

program probe : Resp -> Weights
    sample probs <- Dirichlet(concentration=1.0) [over=Component]
    sample mu : Component <- Normal(loc=0.0, scale=5.0)
    sample sigma : Component <- HalfNormal(scale=1.0)
    observe r : Resp <- MixtureNormal(weights=probs, loc=mu, scale=sigma)
    return probs

export probe
""",
    "BetaBinomial": """object Arm : FinSet 2
object Batch : FinSet 4
object Val : Real 1

program probe : Batch -> Val
    sample conc1 : Arm <- HalfCauchy(scale=2.0)
    sample conc0 : Arm <- HalfCauchy(scale=2.0)

    let a = conc1[arm_idx]
    let b = conc0[arm_idx]

    observe y : Batch <- BetaBinomial(total_count=n_trials, concentration1=a, concentration0=b)
    return a

export probe
""",
    "Kumaraswamy": """object Resp : FinSet 5
object Val : Real 1

program probe : Resp -> Val
    sample a <- HalfNormal(scale=2.0)
    sample b <- HalfNormal(scale=2.0)
    observe y : Resp <- Kumaraswamy(concentration1=a, concentration0=b)
    return a

export probe
""",
    "score": """object Resp : FinSet 5
object Val : Real 1

program probe : Resp -> Val
    sample a <- Normal(loc=0.0, scale=1.0)
    score penalty = -0.5 * a * a
    observe y : Resp <- Normal(loc=a, scale=1.0)
    return a

export probe
""",
}
"""Smallest QVR module that puts one zeros-trick observation on every
target, per family.

Each family needs its own module because the three take different
argument shapes: a mixture site needs a weight simplex and per-
component location and scale vectors, a beta-binomial site needs a
trial count and two gathered concentrations, a Kumaraswamy site needs
two positive shapes. All three are self-contained, so the emit check
does not depend on which gallery example happens to use the family."""

_ZEROS_TRICK_PHI_RE = re.compile(
    r"(?:phi_\w+\[[^\]]*\]|C_\w+) <- ?(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)-"
)
"""Matches the lift out of a `phi_<site>[n] <- <C>-log(...)` relation,
or of the `C_<name> <- <C>-(...)` relation a `score` step lowers to.

The negative case is what makes the pattern the right shape. An
unlifted row reads `phi_<site>[n] <-- (<term>)`, the assignment arrow
run together with a unary minus, where this pattern finds no leading
number and reports no lift. A lifted row reads
`phi_<site>[n] <- 1000000-log(...)`, where the captured group is the
constant the renderer chose."""


class OffsetJustification(dx.TaggedUnion, discriminator="kind"):
    """Why a registered cell carries the offset it carries.

    Two variants, and the distinction is load-bearing.
    [`Derived`][tests.transpile.test_expected_offsets.Derived] states
    that the module docstring's closed form accounts for the value, and
    the derivation test checks that claim against the QVR source.
    [`Unexplained`][tests.transpile.test_expected_offsets.Unexplained]
    records a measurement nobody has accounted for, which is a finding
    rather than a fact about the contract.
    """

    @property
    def summary(self) -> str:
        """One-line rendering for a failure message."""
        raise NotImplementedError


class Derived(OffsetJustification):
    """The docstring's closed form accounts for this offset.

    Attributes
    ----------
    kind : Literal["derived"]
        Discriminator.
    half_sites : int
        Number of scalar folded-family density factors in the example's
        QVR source, summed over every family in
        [`_HALF_FAMILIES`][tests.transpile.test_expected_offsets._HALF_FAMILIES].
        A property of the source alone, so it is identical across the
        targets of one example and is recorded even for a target that
        drops nothing: it pins the count the derivation test recomputes
        from the source.
    dropped_sites : int
        How many of those factors this target emits as a bare symmetric
        base, i.e. the subset whose family lists the target in
        [`_DROPS_HALF_NORMALIZER`][tests.transpile.test_expected_offsets._DROPS_HALF_NORMALIZER].
        The expected offset is `dropped_sites * log 2`. The two counts
        are kept apart rather than collapsed into one flag because a
        single example may mix families whose drop sets differ, as
        `HalfCauchy` and `HalfStudentT` do on Gen, Turing and WebPPL.
    lifted_rows : int
        How many observed rows this target scores through a
        zeros-trick relation whose Poisson rate the renderer lifts by
        [`_ZEROS_TRICK_OFFSET`][tests.transpile.test_expected_offsets._ZEROS_TRICK_OFFSET].
        Each contributes that constant to the offset. Zero on every
        target that names the family as a distribution, and zero on
        the two BUGS / JAGS zeros-trick families whose emit carries no
        lift, so the default states the common case and an entry only
        names the count when the example actually pays it.
    """

    kind: Literal["derived"] = "derived"
    half_sites: int
    dropped_sites: int
    lifted_rows: int = 0

    @property
    def summary(self) -> str:
        return (
            f"derived from {self.half_sites} folded-family density "
            f"factor(s) in the QVR source, {self.dropped_sites} of them "
            f"emitted without a truncation renormalizer, plus "
            f"{self.lifted_rows} zeros-trick row(s) whose Poisson rate "
            f"is lifted by {_ZEROS_TRICK_OFFSET:g}"
        )


class Unexplained(OffsetJustification):
    """A measured offset the derivation does not account for.

    Attributes
    ----------
    kind : Literal["unexplained"]
        Discriminator.
    measured : float
        The offset observed in the pinned runtime.
    missing : str
        What it would take to explain the value: the term that would
        have to be identified in the renderer's emit, or the scoring-API
        normalizer that would have to be shown absent.
    """

    kind: Literal["unexplained"] = "unexplained"
    measured: float
    missing: str

    @property
    def summary(self) -> str:
        return (
            f"UNEXPLAINED: measured {self.measured:+.6f}, unaccounted "
            f"for by the derivation ({self.missing})"
        )


class ExpectedOffset(dx.Model):
    """One registry entry: the constant a cell is entitled to.

    Attributes
    ----------
    offset : float
        Expected value of `mean_i (log p_QVR(z_i) - log p_T(z_i))`. A
        positive value means the target scores below the reference,
        which is what dropping a term looks like.
    justification : OffsetJustification
        Why the value is what it is.
    """

    offset: float
    justification: OffsetJustification


def _derived(
    *,
    half_sites: int,
    dropped_sites: int,
    lifted_rows: int = 0,
) -> ExpectedOffset:
    """Registry entry for a cell the derivation accounts for.

    Parameters
    ----------
    half_sites
        Scalar folded-family density factors in the example's QVR
        source, over every family in
        [`_HALF_FAMILIES`][tests.transpile.test_expected_offsets._HALF_FAMILIES].
    dropped_sites
        How many of those the target emits as a bare symmetric base,
        per
        [`_DROPS_HALF_NORMALIZER`][tests.transpile.test_expected_offsets._DROPS_HALF_NORMALIZER].
    lifted_rows
        Observed rows the target scores through a lifted zeros-trick
        relation, per
        [`_ZEROS_TRICK_OFFSET_FAMILIES`][tests.transpile.test_expected_offsets._ZEROS_TRICK_OFFSET_FAMILIES].
        Defaults to zero, which is the count for every cell whose
        target names each of the example's families as a distribution.
    """
    return ExpectedOffset(
        offset=(dropped_sites * _LOG_2 + lifted_rows * _ZEROS_TRICK_OFFSET),
        justification=Derived(
            half_sites=half_sites,
            dropped_sites=dropped_sites,
            lifted_rows=lifted_rows,
        ),
    )


# ----------------------------------------------------------------------
# The registry.
#
# Grouped by example. Within a group the `half_sites` count is a
# property of the QVR source and is therefore identical across targets;
# only `dropped_sites` varies, and it varies exactly with the family's
# entry in `_DROPS_HALF_NORMALIZER`.
#
# Population workflow for a newly-live cell: derive `half_sites` from
# the example's `.qvr` (every `sample` / `observe` step naming a folded
# family contributes the product of the cardinalities of the axes
# attached to it, nested `marginalize` scopes included), restrict to the
# families whose drop set contains the target to get `dropped_sites`,
# add the entry, and confirm the measured offset agrees. Never read the
# measurement first and write it down: the derivation is the authority
# and the measurement is its test.
# ----------------------------------------------------------------------
_EXPECTED_OFFSET: dict[tuple[str, str], ExpectedOffset] = {
    # ar1: 1 HalfCauchy folded factor(s).
    ("bugs", "ar1"): _derived(half_sites=1, dropped_sites=0),
    ("edward2", "ar1"): _derived(half_sites=1, dropped_sites=0),
    ("gen", "ar1"): _derived(half_sites=1, dropped_sites=1),
    ("jags", "ar1"): _derived(half_sites=1, dropped_sites=0),
    ("numpyro", "ar1"): _derived(half_sites=1, dropped_sites=0),
    ("pymc", "ar1"): _derived(half_sites=1, dropped_sites=0),
    ("pyro", "ar1"): _derived(half_sites=1, dropped_sites=0),
    ("stan", "ar1"): _derived(half_sites=1, dropped_sites=1),
    ("turing", "ar1"): _derived(half_sites=1, dropped_sites=0),
    ("webppl", "ar1"): _derived(half_sites=1, dropped_sites=1),
    # bayesian_regression: 1 HalfCauchy folded factor(s).
    ("bugs", "bayesian_regression"): _derived(half_sites=1, dropped_sites=0),
    ("edward2", "bayesian_regression"): _derived(half_sites=1, dropped_sites=0),
    ("gen", "bayesian_regression"): _derived(half_sites=1, dropped_sites=1),
    ("jags", "bayesian_regression"): _derived(half_sites=1, dropped_sites=0),
    ("numpyro", "bayesian_regression"): _derived(half_sites=1, dropped_sites=0),
    ("pymc", "bayesian_regression"): _derived(half_sites=1, dropped_sites=0),
    ("pyro", "bayesian_regression"): _derived(half_sites=1, dropped_sites=0),
    ("stan", "bayesian_regression"): _derived(half_sites=1, dropped_sites=1),
    ("turing", "bayesian_regression"): _derived(half_sites=1, dropped_sites=0),
    ("webppl", "bayesian_regression"): _derived(half_sites=1, dropped_sites=1),
    # beta_binomial_ab_test: 4 HalfCauchy folded factor(s). `bugs` is
    # absent because the cell raises in transpile: the BUGS family
    # registry has no target name for `BetaBinomial`, and unlike JAGS
    # the renderer has no zeros-trick path to fall back on.
    #
    # `edward2` keeps a zero on the strength of a corrected emit
    # rather than of the current one. `renderers/edward2.py` passes the
    # exogenous trial count straight through as
    # `total_count=n_trials`, which reaches TFP as an int32 beside
    # float32 concentrations, and `BetaBinomial.__init__` rejects the
    # pair through `dtype_util.common_dtype` before scoring anything.
    # The entry states what a float-valued `total_count` would be
    # entitled to, which is nothing;
    # `test_backend_offset_matches_registry` stays red for this cell
    # until the renderer emits one.
    ("edward2", "beta_binomial_ab_test"): _derived(half_sites=4, dropped_sites=0),
    ("gen", "beta_binomial_ab_test"): _derived(half_sites=4, dropped_sites=4),
    ("jags", "beta_binomial_ab_test"): _derived(half_sites=4, dropped_sites=0),
    ("numpyro", "beta_binomial_ab_test"): _derived(half_sites=4, dropped_sites=0),
    ("pymc", "beta_binomial_ab_test"): _derived(half_sites=4, dropped_sites=0),
    ("pyro", "beta_binomial_ab_test"): _derived(half_sites=4, dropped_sites=0),
    ("stan", "beta_binomial_ab_test"): _derived(half_sites=4, dropped_sites=4),
    ("turing", "beta_binomial_ab_test"): _derived(half_sites=4, dropped_sites=0),
    ("webppl", "beta_binomial_ab_test"): _derived(half_sites=4, dropped_sites=4),
    # beta_regression: 3 HalfCauchy folded factor(s).
    ("bugs", "beta_regression"): _derived(half_sites=3, dropped_sites=0),
    ("edward2", "beta_regression"): _derived(half_sites=3, dropped_sites=0),
    ("gen", "beta_regression"): _derived(half_sites=3, dropped_sites=3),
    ("jags", "beta_regression"): _derived(half_sites=3, dropped_sites=0),
    ("numpyro", "beta_regression"): _derived(half_sites=3, dropped_sites=0),
    ("pymc", "beta_regression"): _derived(half_sites=3, dropped_sites=0),
    ("pyro", "beta_regression"): _derived(half_sites=3, dropped_sites=0),
    ("stan", "beta_regression"): _derived(half_sites=3, dropped_sites=3),
    ("turing", "beta_regression"): _derived(half_sites=3, dropped_sites=0),
    ("webppl", "beta_regression"): _derived(half_sites=3, dropped_sites=3),
    # ccg: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("bugs", "ccg"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "ccg"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "ccg"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "ccg"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "ccg"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "ccg"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "ccg"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "ccg"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "ccg"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "ccg"): _derived(half_sites=0, dropped_sites=0),
    # changepoint: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("bugs", "changepoint"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "changepoint"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "changepoint"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "changepoint"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "changepoint"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "changepoint"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "changepoint"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "changepoint"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "changepoint"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "changepoint"): _derived(half_sites=0, dropped_sites=0),
    # continuous_hmm / linear_gaussian_ssm: both programs score two
    # `~ Normal` steps and name no folded family, so no target is
    # entitled to a truncation renormalizer and none reaches its
    # families through the zeros trick. Every cell owes nothing and
    # scores the reference exactly. Both programs are declared
    # without a plate, so the fixture presents the one row they
    # denote and the emitted program reads it directly.
    ("bugs", "continuous_hmm"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "continuous_hmm"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "continuous_hmm"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "continuous_hmm"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "continuous_hmm"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "continuous_hmm"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "continuous_hmm"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "continuous_hmm"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "continuous_hmm"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "continuous_hmm"): _derived(half_sites=0, dropped_sites=0),
    ("bugs", "linear_gaussian_ssm"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "linear_gaussian_ssm"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "linear_gaussian_ssm"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "linear_gaussian_ssm"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "linear_gaussian_ssm"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "linear_gaussian_ssm"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "linear_gaussian_ssm"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "linear_gaussian_ssm"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "linear_gaussian_ssm"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "linear_gaussian_ssm"): _derived(half_sites=0, dropped_sites=0),
    # custom_rules: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("bugs", "custom_rules"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "custom_rules"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "custom_rules"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "custom_rules"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "custom_rules"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "custom_rules"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "custom_rules"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "custom_rules"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "custom_rules"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "custom_rules"): _derived(half_sites=0, dropped_sites=0),
    # factor_analysis: 1 HalfCauchy folded factor(s).
    ("bugs", "factor_analysis"): _derived(half_sites=1, dropped_sites=0),
    ("edward2", "factor_analysis"): _derived(half_sites=1, dropped_sites=0),
    ("gen", "factor_analysis"): _derived(half_sites=1, dropped_sites=1),
    ("jags", "factor_analysis"): _derived(half_sites=1, dropped_sites=0),
    ("numpyro", "factor_analysis"): _derived(half_sites=1, dropped_sites=0),
    ("pymc", "factor_analysis"): _derived(half_sites=1, dropped_sites=0),
    ("pyro", "factor_analysis"): _derived(half_sites=1, dropped_sites=0),
    ("stan", "factor_analysis"): _derived(half_sites=1, dropped_sites=1),
    ("turing", "factor_analysis"): _derived(half_sites=1, dropped_sites=0),
    ("webppl", "factor_analysis"): _derived(half_sites=1, dropped_sites=1),
    # gamma_regression: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("bugs", "gamma_regression"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "gamma_regression"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "gamma_regression"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "gamma_regression"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "gamma_regression"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "gamma_regression"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "gamma_regression"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "gamma_regression"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "gamma_regression"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "gamma_regression"): _derived(half_sites=0, dropped_sites=0),
    # half_student_t_hierarchical: 2 HalfStudentT folded factor(s).
    #
    # `turing` keeps its zero on the strength of a corrected harness
    # rather than of the current one. The emit is right, and it is the
    # only Turing cell that grafts `runtime_turing.jl`, so it is the
    # only one whose source file carries top-level statements above the
    # `@model` macrocall. `probes/_scripts/turing.jl` feeds that file
    # to `Meta.parse`, which reads one expression and rejects the rest
    # ("extra token after end of expression"), so the container never
    # scores it. The entry states what the emitted program is entitled
    # to, which is nothing: `runtime_turing.jl` folds `TDist` with an
    # explicit `+ log(2)`.
    ("bugs", "half_student_t_hierarchical"): _derived(half_sites=2, dropped_sites=0),
    ("edward2", "half_student_t_hierarchical"): _derived(half_sites=2, dropped_sites=2),
    ("gen", "half_student_t_hierarchical"): _derived(half_sites=2, dropped_sites=0),
    ("jags", "half_student_t_hierarchical"): _derived(half_sites=2, dropped_sites=0),
    ("numpyro", "half_student_t_hierarchical"): _derived(half_sites=2, dropped_sites=0),
    ("pymc", "half_student_t_hierarchical"): _derived(half_sites=2, dropped_sites=0),
    ("pyro", "half_student_t_hierarchical"): _derived(half_sites=2, dropped_sites=0),
    ("stan", "half_student_t_hierarchical"): _derived(half_sites=2, dropped_sites=2),
    ("turing", "half_student_t_hierarchical"): _derived(half_sites=2, dropped_sites=0),
    ("webppl", "half_student_t_hierarchical"): _derived(half_sites=2, dropped_sites=0),
    # hmm: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("webppl", "hmm"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "hmm"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "hmm"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "hmm"): _derived(half_sites=0, dropped_sites=0),
    # horseshoe_regression: 6 HalfCauchy folded factor(s).
    ("bugs", "horseshoe_regression"): _derived(half_sites=6, dropped_sites=0),
    ("edward2", "horseshoe_regression"): _derived(half_sites=6, dropped_sites=0),
    ("gen", "horseshoe_regression"): _derived(half_sites=6, dropped_sites=6),
    ("jags", "horseshoe_regression"): _derived(half_sites=6, dropped_sites=0),
    ("numpyro", "horseshoe_regression"): _derived(half_sites=6, dropped_sites=0),
    ("pymc", "horseshoe_regression"): _derived(half_sites=6, dropped_sites=0),
    ("pyro", "horseshoe_regression"): _derived(half_sites=6, dropped_sites=0),
    ("stan", "horseshoe_regression"): _derived(half_sites=6, dropped_sites=6),
    ("turing", "horseshoe_regression"): _derived(half_sites=6, dropped_sites=0),
    ("webppl", "horseshoe_regression"): _derived(half_sites=6, dropped_sites=6),
    # irt_2pl: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("bugs", "irt_2pl"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "irt_2pl"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "irt_2pl"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "irt_2pl"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "irt_2pl"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "irt_2pl"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "irt_2pl"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "irt_2pl"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "irt_2pl"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "irt_2pl"): _derived(half_sites=0, dropped_sites=0),
    # kumaraswamy_bounded_outcome: 1 HalfNormal folded factor(s).
    # `jags` also owes the zeros-trick lift: it has no `Kumaraswamy`
    # distribution, so `_emit_kumaraswamy` writes the closed-form
    # density into a Poisson rate and lifts that rate by
    # `_ZEROS_TRICK_OFFSET` on each of the 64 rows of `Resp`. `bugs` is
    # absent because the cell raises in transpile: the BUGS family
    # registry has no target name for `Kumaraswamy` and its renderer
    # carries no closed-form path to reach one.
    ("edward2", "kumaraswamy_bounded_outcome"): _derived(half_sites=1, dropped_sites=0),
    ("gen", "kumaraswamy_bounded_outcome"): _derived(half_sites=1, dropped_sites=1),
    ("jags", "kumaraswamy_bounded_outcome"): _derived(
        half_sites=1,
        dropped_sites=0,
        lifted_rows=64,
    ),
    ("numpyro", "kumaraswamy_bounded_outcome"): _derived(half_sites=1, dropped_sites=0),
    ("pymc", "kumaraswamy_bounded_outcome"): _derived(half_sites=1, dropped_sites=0),
    ("pyro", "kumaraswamy_bounded_outcome"): _derived(half_sites=1, dropped_sites=0),
    ("stan", "kumaraswamy_bounded_outcome"): _derived(half_sites=1, dropped_sites=1),
    ("turing", "kumaraswamy_bounded_outcome"): _derived(half_sites=1, dropped_sites=0),
    ("webppl", "kumaraswamy_bounded_outcome"): _derived(half_sites=1, dropped_sites=1),
    # lda: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    #
    # `bugs` refuses the example: its word codes are observed class
    # codes, which BUGS cannot shift to `dcat`'s one-based classes.
    ("jags", "lda"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "lda"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "lda"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "lda"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "lda"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "lda"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "lda"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "lda"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "lda"): _derived(half_sites=0, dropped_sites=0),
    # logistic_noise_regression: 1 HalfNormal folded factor(s). The
    # engines' `dlogis(mu, tau)` is rate-parameterised, so `bugs` and
    # `jags` emit `y[n] ~ dlogis(mu[n], 1/scale)`; the reciprocal is an
    # algebraic identity on the density and carries no constant.
    ("bugs", "logistic_noise_regression"): _derived(half_sites=1, dropped_sites=0),
    ("edward2", "logistic_noise_regression"): _derived(half_sites=1, dropped_sites=0),
    ("gen", "logistic_noise_regression"): _derived(half_sites=1, dropped_sites=1),
    ("jags", "logistic_noise_regression"): _derived(half_sites=1, dropped_sites=0),
    ("numpyro", "logistic_noise_regression"): _derived(half_sites=1, dropped_sites=0),
    ("pymc", "logistic_noise_regression"): _derived(half_sites=1, dropped_sites=0),
    ("pyro", "logistic_noise_regression"): _derived(half_sites=1, dropped_sites=0),
    ("stan", "logistic_noise_regression"): _derived(half_sites=1, dropped_sites=1),
    ("turing", "logistic_noise_regression"): _derived(half_sites=1, dropped_sites=0),
    ("webppl", "logistic_noise_regression"): _derived(half_sites=1, dropped_sites=1),
    # mixture_model: 3 HalfNormal folded factor(s). `jags` also owes
    # the zeros-trick lift: it has no `MixtureNormal` distribution, so
    # it writes the closed-form mixture density into a Poisson rate and
    # lifts that rate by `_ZEROS_TRICK_OFFSET` on each of the 100 rows
    # of `Resp`. `bugs` is absent because the cell raises in transpile.
    ("edward2", "mixture_model"): _derived(half_sites=3, dropped_sites=0),
    ("gen", "mixture_model"): _derived(half_sites=3, dropped_sites=3),
    ("jags", "mixture_model"): _derived(
        half_sites=3,
        dropped_sites=0,
        lifted_rows=100,
    ),
    ("numpyro", "mixture_model"): _derived(half_sites=3, dropped_sites=0),
    ("pymc", "mixture_model"): _derived(half_sites=3, dropped_sites=0),
    ("pyro", "mixture_model"): _derived(half_sites=3, dropped_sites=0),
    ("stan", "mixture_model"): _derived(half_sites=3, dropped_sites=3),
    ("turing", "mixture_model"): _derived(half_sites=3, dropped_sites=0),
    ("webppl", "mixture_model"): _derived(half_sites=3, dropped_sites=3),
    # multimodal_tlg: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("bugs", "multimodal_tlg"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "multimodal_tlg"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "multimodal_tlg"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "multimodal_tlg"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "multimodal_tlg"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "multimodal_tlg"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "multimodal_tlg"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "multimodal_tlg"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "multimodal_tlg"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "multimodal_tlg"): _derived(half_sites=0, dropped_sites=0),
    # negbin_regression: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("bugs", "negbin_regression"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "negbin_regression"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "negbin_regression"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "negbin_regression"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "negbin_regression"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "negbin_regression"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "negbin_regression"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "negbin_regression"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "negbin_regression"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "negbin_regression"): _derived(half_sites=0, dropped_sites=0),
    # pcfg: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("bugs", "pcfg"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "pcfg"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "pcfg"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "pcfg"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "pcfg"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "pcfg"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "pcfg"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "pcfg"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "pcfg"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "pcfg"): _derived(half_sites=0, dropped_sites=0),
    # pmcfg: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("bugs", "pmcfg"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "pmcfg"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "pmcfg"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "pmcfg"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "pmcfg"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "pmcfg"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "pmcfg"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "pmcfg"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "pmcfg"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "pmcfg"): _derived(half_sites=0, dropped_sites=0),
    # ppca: 1 HalfCauchy folded factor(s).
    ("bugs", "ppca"): _derived(half_sites=1, dropped_sites=0),
    ("edward2", "ppca"): _derived(half_sites=1, dropped_sites=0),
    ("gen", "ppca"): _derived(half_sites=1, dropped_sites=1),
    ("jags", "ppca"): _derived(half_sites=1, dropped_sites=0),
    ("numpyro", "ppca"): _derived(half_sites=1, dropped_sites=0),
    ("pymc", "ppca"): _derived(half_sites=1, dropped_sites=0),
    ("pyro", "ppca"): _derived(half_sites=1, dropped_sites=0),
    ("stan", "ppca"): _derived(half_sites=1, dropped_sites=1),
    ("turing", "ppca"): _derived(half_sites=1, dropped_sites=0),
    ("webppl", "ppca"): _derived(half_sites=1, dropped_sites=1),
    # quantifier_scope: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("bugs", "quantifier_scope"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "quantifier_scope"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "quantifier_scope"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "quantifier_scope"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "quantifier_scope"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "quantifier_scope"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "quantifier_scope"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "quantifier_scope"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "quantifier_scope"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "quantifier_scope"): _derived(half_sites=0, dropped_sites=0),
    # stochastic_volatility: 1 HalfCauchy folded factor(s).
    ("webppl", "stochastic_volatility"): _derived(
        half_sites=1,
        dropped_sites=1,
    ),
    ("bugs", "stochastic_volatility"): _derived(half_sites=1, dropped_sites=0),
    ("edward2", "stochastic_volatility"): _derived(half_sites=1, dropped_sites=0),
    ("gen", "stochastic_volatility"): _derived(half_sites=1, dropped_sites=1),
    ("jags", "stochastic_volatility"): _derived(half_sites=1, dropped_sites=0),
    ("numpyro", "stochastic_volatility"): _derived(half_sites=1, dropped_sites=0),
    ("pymc", "stochastic_volatility"): _derived(half_sites=1, dropped_sites=0),
    ("pyro", "stochastic_volatility"): _derived(half_sites=1, dropped_sites=0),
    ("stan", "stochastic_volatility"): _derived(half_sites=1, dropped_sites=1),
    ("turing", "stochastic_volatility"): _derived(half_sites=1, dropped_sites=0),
    # survival_weibull: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("bugs", "survival_weibull"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "survival_weibull"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "survival_weibull"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "survival_weibull"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "survival_weibull"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "survival_weibull"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "survival_weibull"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "survival_weibull"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "survival_weibull"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "survival_weibull"): _derived(half_sites=0, dropped_sites=0),
    # tree_categorical: 1 HalfNormal folded factor(s). `sigma_v` is
    # the example's only folded site and it carries no axis, so the
    # count is one whatever `Verb`, `Class` and `Resp` are sized at.
    # `stan` is absent because the cell raises in transpile.
    ("stan", "tree_categorical"): _derived(
        half_sites=1,
        dropped_sites=1,
    ),
    ("bugs", "tree_categorical"): _derived(half_sites=1, dropped_sites=0),
    ("edward2", "tree_categorical"): _derived(half_sites=1, dropped_sites=0),
    ("gen", "tree_categorical"): _derived(half_sites=1, dropped_sites=1),
    ("jags", "tree_categorical"): _derived(half_sites=1, dropped_sites=0),
    ("numpyro", "tree_categorical"): _derived(half_sites=1, dropped_sites=0),
    ("pymc", "tree_categorical"): _derived(half_sites=1, dropped_sites=0),
    ("pyro", "tree_categorical"): _derived(half_sites=1, dropped_sites=0),
    ("turing", "tree_categorical"): _derived(half_sites=1, dropped_sites=0),
    ("webppl", "tree_categorical"): _derived(half_sites=1, dropped_sites=1),
    # type_logical: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ("bugs", "type_logical"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "type_logical"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "type_logical"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "type_logical"): _derived(half_sites=0, dropped_sites=0),
    ("numpyro", "type_logical"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "type_logical"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "type_logical"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "type_logical"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "type_logical"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "type_logical"): _derived(half_sites=0, dropped_sites=0),
    # parametric_pooling: a draw from a program template inlines its
    # `Normal` plate under `theta$z`, spelled `theta__z` on every
    # target; no folded-family site anywhere. Its `score` step is one
    # lifted zeros-trick row on `jags` and `bugs`, and nothing
    # elsewhere.
    ("bugs", "parametric_pooling"): _derived(
        half_sites=0, dropped_sites=0, lifted_rows=1
    ),
    ("edward2", "parametric_pooling"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "parametric_pooling"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "parametric_pooling"): _derived(
        half_sites=0, dropped_sites=0, lifted_rows=1
    ),
    ("numpyro", "parametric_pooling"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "parametric_pooling"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "parametric_pooling"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "parametric_pooling"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "parametric_pooling"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "parametric_pooling"): _derived(half_sites=0, dropped_sites=0),
    # zip_regression: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    #
    # `jags` has no `ContinuousBernoulli` distribution and reaches the
    # family through the zeros trick, but owes no lift for it. The
    # trick's rate is lifted only where a site is emitted directly,
    # and this one is enumerated by a `marginalize z : Resp` block,
    # whose emitter writes the integrated density into `dpois` with no
    # offset: the rendered source carries no lift constant on any of
    # the 400 rows. `bugs` is absent because its cell raises in
    # transpile, its renderer carrying no path to the family at all.
    ("numpyro", "zip_regression"): _derived(half_sites=0, dropped_sites=0),
    ("stan", "zip_regression"): _derived(half_sites=0, dropped_sites=0),
    ("edward2", "zip_regression"): _derived(half_sites=0, dropped_sites=0),
    ("jags", "zip_regression"): _derived(half_sites=0, dropped_sites=0),
    ("pymc", "zip_regression"): _derived(half_sites=0, dropped_sites=0),
    ("pyro", "zip_regression"): _derived(half_sites=0, dropped_sites=0),
    ("turing", "zip_regression"): _derived(half_sites=0, dropped_sites=0),
    ("gen", "zip_regression"): _derived(half_sites=0, dropped_sites=0),
    ("webppl", "zip_regression"): _derived(half_sites=0, dropped_sites=0),
}


_ACKNOWLEDGED_UNEXPLAINED: frozenset[tuple[str, str]] = frozenset()
"""Cells whose registered offset the derivation does not explain.

An entry here is an admission that a backend adds or drops a term
nobody has accounted for, which is a correctness finding and not a
property of the contract. The set is empty, and
`test_no_unexplained_offsets_are_registered` keeps it that way unless
someone deliberately writes a cell into both this set and the registry.
"""


def _live_cells() -> list[tuple[str, str]]:
    """Every `(backend, example)` cell the gallery equivalence test
    currently evaluates numerically.

    Derived from the same registries
    [`test_gallery_numeric_equivalence`][tests.transpile.test_gallery_numeric_equivalence]
    consults, so a cell that becomes live there becomes live here on the
    same commit and
    `test_every_live_cell_has_a_registered_offset` fails until its
    offset is derived and registered.
    """
    out: list[tuple[str, str]] = []
    for example in _gallery_cells():
        stem = example.stem
        if stem in _SKIP_DATASET_LOAD_FAILED:
            continue
        if stem in _SKIP_QVR_INCOMPATIBLE:
            continue
        for backend in sorted(_BACKENDS_WITH_IMAGES):
            if (backend, stem) in _EXPECTED_TRANSPILE_RAISES:
                continue
            if (backend, stem) in _SKIP_PROBE_INCOMPATIBLE:
                continue
            out.append((backend, stem))
    return out


def _example_path(stem: str) -> pathlib.Path:
    """The gallery `.qvr` source for `stem`."""
    for example in _gallery_cells():
        if example.stem == stem:
            return example
    raise AssertionError(
        f"{stem!r} is not a gallery example with a synthetic-data "
        f"block; `_live_cells` should never have produced it"
    )


def _finset_cardinalities(module: Module) -> dict[str, int]:
    """Cardinality of every `object <name> : FinSet <n>` declaration.

    Only plain `FinSet` declarations are collected. An axis declared as
    a product, coproduct or free monoid is absent from the map, and
    [`half_family_factor_counts`][tests.transpile.test_expected_offsets.half_family_factor_counts]
    raises rather than guess when a folded site indexes one.
    """
    out: dict[str, int] = {}
    for statement in module.statements:
        if not isinstance(statement, ObjectDecl):
            continue
        init = statement.init
        if not isinstance(init, TypeFromExpr):
            continue
        expr = init.expr
        if not isinstance(expr, DiscreteConstructor):
            continue
        if expr.constructor != "FinSet" or len(expr.args) != 1:
            continue
        for name in statement.names:
            out[name] = int(expr.args[0])
    return out


def _exported_program(module: Module, stem: str) -> ProgramDecl:
    """The `ProgramDecl` the module's `export` names.

    The exported program's name need not match the file stem
    (`ppca.qvr` exports `ppca_program`), so the export declaration is
    the authority. A module may export more than one name, since the
    probabilistic surface often sits alongside the pure composite it
    scores (`hmm.qvr` exports both `hmm` and `hmm_program`); only the
    exports naming a `program` declaration are candidates, and exactly
    one of those must exist for the count to be well defined.
    """
    programs = {
        statement.name: statement
        for statement in module.statements
        if isinstance(statement, ProgramDecl)
    }
    exported: list[str] = []
    for statement in module.statements:
        if not isinstance(statement, ExportDecl):
            continue
        expr = statement.expr
        if isinstance(expr, ExprIdent) and expr.name in programs:
            exported.append(expr.name)
    if len(exported) != 1:
        raise AssertionError(
            f"{stem!r}: expected exactly one `export <ident>` naming a "
            f"`program` declaration, found {exported!r}. The "
            f"folded-family count is taken off the exported program and "
            f"cannot be resolved without one."
        )
    return programs[exported[0]]


def _axis_names(step: SampleStep | ObserveStep) -> tuple[str, ...]:
    """Every axis attached to a draw step, batch and event alike.

        A folded family is scored elementwise, so the folding factor of two
        applies once per coordinate whether the coordinate sits on a batch
        axis (`sample s : Coef <- HalfCauchy(scale=1.0)`) or on an event axis
    (`[over=...]`). Both thus multiply the factor count. When an
        `[over=..., iid_over=...]` clause is present it is authoritative and
        the bare index is not counted again, since the index axis reappears
        inside `iid_over`.
    """
    axes = step.axes
    if axes is not None:
        return tuple(str(axis) for axis in axes.over) + tuple(
            str(axis) for axis in axes.iid_over
        )
    index = step.index
    if index is None:
        return ()
    return (str(index.name),)


def _flatten_steps(
    steps: Sequence[ProgramStep],
) -> list[ProgramStep]:
    """Every step of a program body, including those nested in a
    `marginalize` scope.

    A folded-family site inside a marginalized scope still contributes
    its `log 2`. Under a `logsumexp` reduction the constant factor is
    common to every branch of the sum and so factors straight back out
    of the reduction; under a fibred reduction it factors out of each
    fibre alike. Either way the site counts exactly once per coordinate,
    as it would at the top level, so a flat walk of the nested steps is
    the right count.
    """
    out: list[ProgramStep] = []
    for step in steps:
        out.append(step)
        if isinstance(step, MarginalizeStep):
            out.extend(_flatten_steps(step.scope))
    return out


def _inlined_steps(module: Module, program: ProgramDecl) -> tuple[ProgramStep, ...]:
    """A program's steps with every draw from a program run in place.

    Parameters
    ----------
    module : Module
        The source module, whose other programs the draws name.
    program : ProgramDecl
        The program.

    Returns
    -------
    tuple[ProgramStep, ...]
        The steps, a draw from a program replaced by the callee's steps
        instantiated at the draw's objects as the elaboration runs
        them, recursively.
    """
    templates = {
        statement.name: statement
        for statement in module.statements
        if isinstance(statement, ProgramDecl) and statement.name != program.name
    }

    def inlined(steps: tuple[ProgramStep, ...]) -> tuple[ProgramStep, ...]:
        """The steps with every draw from a program run in place.

        Parameters
        ----------
        steps : tuple[ProgramStep, ...]
            The steps.

        Returns
        -------
        tuple[ProgramStep, ...]
            The steps, recursively instantiated.
        """
        out: list[ProgramStep] = []
        for step in steps:
            template = (
                templates.get(step.morphism)
                if isinstance(step, (SampleStep, ObserveStep))
                else None
            )
            if template is None:
                out.append(step)
                continue
            arguments = step.args or ()
            names = {
                argument.text: LetExprVar(name=argument.text)
                for argument in arguments
                if isinstance(argument, DrawArgName)
            }
            bindings = template_bindings(template, arguments, names)
            out.extend(inlined(instantiate_program(template, step.vars, bindings)))
        return tuple(out)

    return inlined(program.draws)


def half_family_factor_counts(example: pathlib.Path) -> dict[str, int]:
    """Scalar folded-family density factors in a QVR source, by family.

    Sums, over every `sample` / `observe` step of the exported program
    that names a family in
    [`_HALF_FAMILIES`][tests.transpile.test_expected_offsets._HALF_FAMILIES],
    the number of variables the step binds times the product of the
    cardinalities of the axes attached to it. Steps nested in a
    `marginalize` scope are walked too. Families with no site in the
    source are absent from the returned map.

    The breakdown is per family rather than a single total because the
    targets that drop the renormalizer differ between families: Gen,
    Turing and WebPPL emit a bare base for `HalfCauchy` and `HalfNormal`
    but a folded runtime helper for `HalfStudentT`.

    A step drawing from another program of the module is counted as
    the callee's steps run in place, instantiated at the draw's objects
    as the elaboration runs them, so a folded site inside the callee
    counts under the axes the draw names.

    Raises
    ------
    AssertionError
        If a `marginalize` head names a folded family (the reduction's
        own treatment of the constant would need its own derivation),
        or if a folded site indexes an axis that is not a plain
        `FinSet`.
    """
    module = parse(example.read_text())
    program = _exported_program(module, example.stem)
    cardinalities = _finset_cardinalities(module)
    counts: dict[str, int] = {}
    for step in _flatten_steps(_inlined_steps(module, program)):
        if isinstance(step, MarginalizeStep):
            if step.morphism in _HALF_FAMILIES:
                raise AssertionError(
                    f"{example.stem!r}: `marginalize {step.var}` draws "
                    f"from the folded family {step.morphism!r}. Whether "
                    f"the folding factor survives the reduction depends "
                    f"on the reduction, so the flat per-site count does "
                    f"not apply; derive the term for this step before "
                    f"registering an offset for this example."
                )
            continue
        if not isinstance(step, (SampleStep, ObserveStep)):
            continue
        if step.morphism not in _HALF_FAMILIES:
            continue
        multiplicity = len(step.vars)
        for axis in _axis_names(step):
            if axis not in cardinalities:
                raise AssertionError(
                    f"{example.stem!r}: folded step {step.vars!r} "
                    f"indexes axis {axis!r}, which is not declared as a "
                    f"plain `FinSet`. Its cardinality decides how many "
                    f"`log 2` factors the site carries, so the "
                    f"derivation cannot proceed without it."
                )
            multiplicity *= cardinalities[axis]
        counts[step.morphism] = counts.get(step.morphism, 0) + multiplicity
    return counts


def zeros_trick_factor_counts(example: pathlib.Path) -> dict[str, int]:
    """Observed rows a QVR source offers to the zeros trick, by family.

    Sums, over every `observe` step of the exported program naming a
    family in
    [`_ZEROS_TRICK_FAMILIES`][tests.transpile.test_expected_offsets._ZEROS_TRICK_FAMILIES],
    the number of variables the step binds times the product of the
    cardinalities of the axes attached to it. That is one row per
    `zeros[n] ~ dpois(phi[n])` relation the BUGS / JAGS renderers
    write, which is the granularity the lift is paid at.

    Only `observe` steps count. Both renderers reject a *latent* draw
    from a zeros-trick family outright, on the ground that the idiom
    contributes a density term without declaring a node the engine can
    sample, so a latent site never reaches an emit and never pays a
    lift.

    Families with no observed site in the source are absent from the
    returned map, and the count is a property of the source alone: it
    is the same for every target, and
    [`_ZEROS_TRICK_OFFSET_FAMILIES`][tests.transpile.test_expected_offsets._ZEROS_TRICK_OFFSET_FAMILIES]
    decides which targets pay for it.

    A `score` step counts one row under the name `score`, the lifted
    relation both renderers lower it to. A draw from another program of
    the module counts the callee's steps run in place.

    Raises
    ------
    AssertionError
        If a zeros-trick site indexes an axis that is not a plain
        `FinSet`.
    """
    module = parse(example.read_text())
    program = _exported_program(module, example.stem)
    cardinalities = _finset_cardinalities(module)

    counts: dict[str, int] = {}
    for step in _flatten_steps(_inlined_steps(module, program)):
        if isinstance(step, ScoreStep):
            counts["score"] = counts.get("score", 0) + 1
            continue
        if not isinstance(step, ObserveStep):
            continue
        if step.morphism not in _ZEROS_TRICK_FAMILIES:
            continue
        multiplicity = len(step.vars)
        for axis in _axis_names(step):
            if axis not in cardinalities:
                raise AssertionError(
                    f"{example.stem!r}: zeros-trick step {step.vars!r} "
                    f"indexes axis {axis!r}, which is not declared as a "
                    f"plain `FinSet`. Its cardinality decides how many "
                    f"lifted rows the site carries, so the derivation "
                    f"cannot proceed without it."
                )
            multiplicity *= cardinalities[axis]
        counts[step.morphism] = counts.get(step.morphism, 0) + multiplicity
    return counts


def _dropped_factor_count(backend: str, stem: str) -> int:
    """Folded factors `backend` emits without a renormalizer, for
    `stem`.

    The per-family counts of the source restricted to the families whose
    drop set contains the target.
    """
    counts = half_family_factor_counts(_example_path(stem))
    return sum(
        count
        for family, count in counts.items()
        if backend in _DROPS_HALF_NORMALIZER[family]
    )


def _lifted_row_count(backend: str, stem: str) -> int:
    """Rows `backend` scores through a lifted zeros-trick relation, for
    `stem`.

    The per-family observed-row counts of the source restricted to the
    families whose lift set contains the target.
    """
    counts = zeros_trick_factor_counts(_example_path(stem))
    return sum(
        count
        for family, count in counts.items()
        if backend in _ZEROS_TRICK_OFFSET_FAMILIES[family]
    )


def _derived_offset(backend: str, stem: str) -> float:
    """The closed-form expected offset for a cell.

    `sum_f drops_half(T, f) * n_f(M) * log 2 + sum_f lifts(T, f) *
    m_f(M) * C` per the module docstring.
    """
    return (
        _dropped_factor_count(backend, stem) * _LOG_2
        + _lifted_row_count(backend, stem) * _ZEROS_TRICK_OFFSET
    )


def _offset_atol(dataset: _gallery_data.GalleryDataset) -> float:
    """Tolerance on the measured-versus-registered offset comparison.

    The same estimator the constant-spread check uses, driven by the
    example's observed-data count. The offset is a mean of pointwise
    differences, so it inherits the per-observation round-off the
    estimator models plus a small systematic bias between the
    reference's and the backend's summation order, which is what the
    `5e-4` floor absorbs. The floor is roughly `log(2) / 1386`, so a
    single unaccounted truncation renormalizer, and any other
    point-independent term above a milli-nat, still fails the cell.
    """
    n_obs = sum(
        int(dataset.observations[name].numel())
        for name in _gallery_data.observed_data_names(dataset)
    )
    return _equivalence.adaptive_atol(n_obs=n_obs)


_PROBE_SCRIPTS_AT_IMPORT = _gallery_data.probe_script_digests()
"""Identity of the probe sources as this module was imported.

[`run_probe`][tests.transpile._docker.run_probe] copies them into the
container once per cell, so a mid-session edit to the tree splits the
run into cells measured under one harness and cells measured under
another. Each Docker cell re-checks the digests on both sides of its
container launch, and a move fails the cell as a *session* fault rather
than reporting the resulting number as a property of the model.

This is not hypothetical. `_reshape.py` is imported by every
Python-side probe, so a change to how a flat point payload is inflated
back into the shapes the target declares moves the data all of them
score. The visible signature is a whole row of one model's cells
failing together, which is indistinguishable by eye from a genuine
finding about that model."""


_RESHAPE_HELPERS = frozenset({"_reshape.py", "_reshape.jl"})
"""Helpers [`run_probe`][tests.transpile._docker.run_probe] copies into
every container alongside the backend's own entrypoint, whatever the
target language."""


def _copied_probe_sources(script_name: str) -> frozenset[str]:
    """Names of the probe sources one cell's container actually
    executes: its backend entrypoint plus the reshape helpers.

    Scoping the integrity check to this set keeps the diagnosis exact.
    A cell is compromised by an edit to a file it copied, and by nothing
    else; a change to another backend's entrypoint is reported by the
    cells that copied *that*.
    """
    return _RESHAPE_HELPERS | {script_name}


_HARNESS_WRITTEN_INPUTS = ("points.json", "shapes.json", "dtypes.json")
"""Container inputs the harness serialises rather than copies verbatim.

Checked back after the run by
[`_assert_container_read_our_inputs`][tests.transpile.test_expected_offsets._assert_container_read_our_inputs];
`probe.py` and the reshape helpers are covered instead by the
[`PROBE_SCRIPT_DIR`][tests.transpile._gallery_data.PROBE_SCRIPT_DIR]
digest check, which catches an edit at its source."""


def _assert_container_read_our_inputs(
    scratch: pathlib.Path,
    emitted: bytes,
    source_ext: str,
    payload: list[dict[str, dict[str, float | int | list[float] | list[int]]]],
    context: str,
) -> None:
    """The inputs still in `scratch` are the ones this cell wrote.

    Turns the isolation
    [`probe_scratch`][tests.transpile._gallery_data.probe_scratch]
    claims into something measured. A scratch nothing else can name
    cannot be rewritten under the harness, so the check holds trivially
    while the path stays private; it earns its place the moment any
    caller hands `run_probe` a path some other process can guess, which
    is a one-line change away and otherwise reintroduces the fault in
    silence.

    The failure it names is precise: the container scored a point set
    or a program that this cell did not choose, so the log-densities
    coming back answer a different question than the reference does. The
    resulting offset is then a difference between two unrelated
    measures, which is exactly the kind of number the named-constant
    registry exists to refuse.
    """
    source_path = scratch / f"source.{source_ext}"
    actual_source = source_path.read_bytes()
    assert actual_source == emitted, (
        f"{context}: {source_path} holds {len(actual_source)} bytes of "
        f"program source but this cell emitted {len(emitted)}. Something "
        f"outside this cell wrote the program the container scored, so "
        f"the measured offset compares the reference against a program "
        f"nobody in this test chose."
    )
    written = json.loads((scratch / "points.json").read_text())
    assert written == _docker.spelled_points(payload), (
        f"{context}: {scratch / 'points.json'} no longer holds the point "
        f"set this cell built. The container therefore scored one set of "
        f"points while the QVR reference scored another, and the "
        f"difference between them is not an additive constant of any "
        f"program. This is the shared-scratch fault: it lands on every "
        f"backend of the affected model at once and reads as a "
        f"model-specific finding."
    )
    for name in _HARNESS_WRITTEN_INPUTS[1:]:
        table = scratch / name
        if not table.exists():
            continue
        assert json.loads(table.read_text()), (
            f"{context}: {table} is present but empty. `run_probe` "
            f"writes it only when the caller supplies a non-None table, "
            f"so an empty one is a leftover the probe will read as this "
            f"cell's shape declaration."
        )


_QVR_LOG_DENSITY_CACHE: dict[str, tuple[str, list[float]]] = {}
"""Per-example QVR reference values, under the point set they score.

The reference does not depend on the backend, so the ten cells of one
example share a single in-process evaluation. Each entry therefore
carries the [`_points_key`][tests.transpile.test_expected_offsets._points_key]
of the points it was measured at, and a later cell asking for the same
example under a different point set is refused rather than served.

Without that key the cache is a silent coupling. Every cell rebuilds
its own points, and only the first one's reach the reference; if point
generation ever stopped being reproducible within a session, cells two
through ten would difference this example's reference at one point set
against a container's log-densities at another. The result is a large
offset with a large spread, identical across every backend of that
example, which reads as a finding about the model rather than as a
harness fault."""


def _points_key(points: list[Point]) -> str:
    """Content digest of a point set, over the wire payload itself.

    Digesting the serialised `params` / `data` rather than object
    identity makes the key refer to the same numbers: two runs
    that rebuild equal points hit the cache, and any coordinate that
    moved misses it.
    """
    payload = [{"params": point.params, "data": point.data} for point in points]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _qvr_log_densities(
    example: pathlib.Path,
    dataset: _gallery_data.GalleryDataset,
    points: list[Point],
    scratch: pathlib.Path,
) -> list[float]:
    """QVR reference log-densities at every point of `points`.

    One probe call per point. The probe's `observations` keyword
    overrides the flat per-point payload, so a perturbed point needs its
    own pre-shaped observation dict; passing the dataset's ground-truth
    observations once for the whole set would score the reference at the
    unperturbed data while the container scored the perturbed data, and
    manufacture an offset out of the mismatch.

    A cached value is returned only to a caller asking at the very same
    points; see
    [`_QVR_LOG_DENSITY_CACHE`][tests.transpile.test_expected_offsets._QVR_LOG_DENSITY_CACHE]
    for why serving it to any other caller would be a defect rather than
    a memoisation.
    """
    key = _points_key(points)
    cached = _QVR_LOG_DENSITY_CACHE.get(example.stem)
    if cached is not None:
        cached_key, cached_values = cached
        if cached_key != key:
            raise RuntimeError(
                f"{example.stem!r}: the reference log-densities cached "
                f"for this example were measured at point set "
                f"{cached_key[:16]}, but this cell built {key[:16]}. "
                f"Point generation is not reproducible within this "
                f"session, so the ten cells of this example are no "
                f"longer measuring one thing: each container would be "
                f"scored at its own points and differenced against a "
                f"reference scored at the first cell's. Fix the source "
                f"of the divergence; nothing about the backends can be "
                f"concluded until the point set is stable."
            )
        return cached_values
    probe = QvrProbe()
    source = example.read_bytes()
    values: list[float] = []
    for point in points:
        values.extend(
            probe.evaluate(
                source,
                example.stem,
                [point],
                scratch=scratch,
                monadic=dataset.monadic,
                x_input=dataset.x_input,
                observations=_gallery_data.observations_for_point(
                    dataset,
                    point,
                ),
            ).log_densities
        )
    _QVR_LOG_DENSITY_CACHE[example.stem] = (key, values)
    return values


def _half_family_emit(backend: str, family: str) -> str:
    """The program `backend` emits for one folded-family draw.

    Renders
    [`_HALF_FAMILY_PROBE_SOURCE`][tests.transpile.test_expected_offsets._HALF_FAMILY_PROBE_SOURCE]
    with `family`'s call spelling. The result is the renderer's own
    account of how it writes the family, which is the authority the
    drop table has to answer to: a name table records the resolution a
    renderer *usually* takes, and a renderer that rewrites the site is
    invisible in it.
    """
    source = _HALF_FAMILY_PROBE_SOURCE.format(call=_HALF_FAMILY_PROBE_ARGS[family])
    return transpile(parse(source), target=backend).decode("utf-8")


def _zeros_trick_lift(backend: str, family: str) -> float | None:
    """The constant `backend` lifts a `family` zeros-trick rate by, or
    `None` when the site is not lowered through a lifted trick.

    Renders the family's entry in
    [`_ZEROS_TRICK_PROBE_SOURCES`][tests.transpile.test_expected_offsets._ZEROS_TRICK_PROBE_SOURCES]
    and reads the lift straight out of the emitted
    `phi_<site>[n] <- <C>-log(...)` relation. A target that raises on
    the family cannot pay a lift at all and reports `None`, and so does
    one whose emit names the family as a distribution or writes the
    unlifted `phi_<site>[n] <-- (<term>)`.
    """
    try:
        emitted = transpile(
            parse(_ZEROS_TRICK_PROBE_SOURCES[family]), target=backend
        ).decode("utf-8")
    except UnsupportedConstruct:
        return None
    match = _ZEROS_TRICK_PHI_RE.search(emitted)
    if match is None:
        return None
    return float(match.group(1))


def test_zeros_trick_table_agrees_with_the_emit() -> None:
    """
    [`_ZEROS_TRICK_OFFSET_FAMILIES`][tests.transpile.test_expected_offsets._ZEROS_TRICK_OFFSET_FAMILIES]
    names exactly the targets that lift a zeros-trick rate, and
    [`_ZEROS_TRICK_OFFSET`][tests.transpile.test_expected_offsets._ZEROS_TRICK_OFFSET]
    is the constant they lift it by.

    The lift is worth a hundred thousand nats a row, so it dwarfs every
    other term in the derivation, and it is invisible to the
    constant-spread check by construction. Pinning the table as a
    literal and never reading it back would let the renderer drop the
    lift (as `_emit_beta_binomial` already did for the other
    zeros-trick family) while the registry kept charging a cell `1e6`
    per row it no longer pays, or add one where the registry charges
    nothing. Both directions are checked, and the value is parsed out
    of the emitted relation rather than compared against a number
    copied from the renderer, so the pin answers to the program the
    container actually scores.
    """
    assert set(_ZEROS_TRICK_OFFSET_FAMILIES) == set(_ZEROS_TRICK_FAMILIES), (
        f"the lift table keys "
        f"{sorted(_ZEROS_TRICK_OFFSET_FAMILIES)} but the zeros-trick "
        f"families are {sorted(_ZEROS_TRICK_FAMILIES)}. Every family "
        f"lowered through the trick needs its own lift set, or a cell "
        f"using it is charged by a table that never considered it."
    )
    assert set(_ZEROS_TRICK_PROBE_SOURCES) == set(_ZEROS_TRICK_FAMILIES), (
        f"the zeros-trick probe sources key "
        f"{sorted(_ZEROS_TRICK_PROBE_SOURCES)} but the families are "
        f"{sorted(_ZEROS_TRICK_FAMILIES)}. Without a probe module the "
        f"lift cannot be read off an emit and the table falls back to "
        f"being asserted from memory."
    )

    lifted_somewhere = False
    for family in sorted(_ZEROS_TRICK_FAMILIES):
        lifting: set[str] = set()
        for backend in sorted(_BACKENDS_WITH_IMAGES):
            lift = _zeros_trick_lift(backend, family)
            if lift is None:
                continue
            lifting.add(backend)
            lifted_somewhere = True
            assert lift == pytest.approx(_ZEROS_TRICK_OFFSET, rel=0, abs=0), (
                f"{backend!r} lifts the {family!r} zeros-trick rate by "
                f"{lift!r}, but the derivation charges every lifted row "
                f"{_ZEROS_TRICK_OFFSET!r}. Every cell of every example "
                f"observing this family on this target is off by "
                f"{(lift - _ZEROS_TRICK_OFFSET)!r} per row; re-derive "
                f"before editing either side."
            )
        assert lifting == _ZEROS_TRICK_OFFSET_FAMILIES[family], (
            f"targets lifting the {family!r} zeros-trick rate are "
            f"{sorted(lifting)}, but the table claims "
            f"{sorted(_ZEROS_TRICK_OFFSET_FAMILIES[family])}. A "
            f"renderer changed how it writes this family's Poisson "
            f"rate; re-derive the entitled constant for every affected "
            f"cell before editing either side."
        )

    assert lifted_somewhere, (
        "no target in the Docker matrix lifts any zeros-trick rate, so "
        "this test would pass on a pattern that matches nothing at "
        "all. Either every renderer dropped the lift (in which case "
        "the term leaves the derivation and every `lifted_rows` count "
        "goes to zero) or "
        "`_ZEROS_TRICK_PHI_RE` no longer matches the relation the "
        "renderers emit."
    )


def test_drop_table_agrees_with_the_family_registry() -> None:
    """Every target in
        [`_DROPS_HALF_NORMALIZER`][tests.transpile.test_expected_offsets._DROPS_HALF_NORMALIZER]
        spells the half-families as a symmetric base, and no target outside
        it does.

        The drop table is the hinge of the whole derivation: it decides
        which cells expect `n * log 2` and which expect zero. Pinning it as
        a literal set and never re-checking it would let a renderer switch
        `HalfCauchy` from `cauchy` to a native or truncated spelling while
        the registry kept charging the cell a `log 2` it no longer drops.

        The reconciliation is against the *effective* spelling: the name
        [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] resolves
        the family to, amended by
        [`_RENDERER_SPELLING_OVERRIDES`][tests.transpile.test_expected_offsets._RENDERER_SPELLING_OVERRIDES]
        where the renderer rewrites the draw site rather than take that
        resolution. The name table alone is not enough, and the way it
        fails is exactly the failure this module is built around: Edward2's
        `HalfStudentT` reads as a native folded class in `FAMILY_META` and
        emits as the bare `edward2.StudentT`, so a derivation off the table
        would name zero for a cell that drops `log 2` at every site and the
        constant-spread check would never notice. Each override is
    thus held against the emitted program rather than taken on
        trust.
    """
    symmetric_bases = {
        "cauchy",
        "Cauchy",
        "normal",
        "Normal",
        "gaussian",
        "Gaussian",
        "student_t",
        "StudentT",
    }
    assert set(_DROPS_HALF_NORMALIZER) == set(_HALF_FAMILIES), (
        f"the drop table keys {sorted(_DROPS_HALF_NORMALIZER)} but the "
        f"folded families are {sorted(_HALF_FAMILIES)}. Every folded "
        f"family needs its own drop set; the two differ."
    )
    assert set(_HALF_FAMILY_PROBE_ARGS) == set(_HALF_FAMILIES), (
        f"the probe-call table keys {sorted(_HALF_FAMILY_PROBE_ARGS)} "
        f"but the folded families are {sorted(_HALF_FAMILIES)}. Every "
        f"folded family needs a call spelling, or its effective "
        f"spelling cannot be read off an emit."
    )
    for family in sorted(_HALF_FAMILIES):
        meta = FAMILY_META[family]
        bare_base: set[str] = set()
        for backend in sorted(_BACKENDS_WITH_IMAGES):
            registered = meta.target_names.get(backend)
            assert registered is not None, (
                f"{family!r} has no {backend!r} target name in "
                f"`FAMILY_META`, yet {backend!r} is in the Docker "
                f"matrix. Either the family lost its spelling or the "
                f"matrix gained a backend; the derivation cannot "
                f"decide whether the cell drops a `log 2` without it."
            )
            override = _RENDERER_SPELLING_OVERRIDES.get((backend, family))
            emitted = _half_family_emit(backend, family)
            if override is not None:
                assert override in emitted, (
                    f"`_RENDERER_SPELLING_OVERRIDES` claims "
                    f"{backend!r} rewrites {family!r} to {override!r}, "
                    f"but that name is absent from the emitted "
                    f"program. Read the renderer's current spelling "
                    f"off the emit and re-derive the entitled constant "
                    f"for every cell of every example using this "
                    f"family."
                )
                assert registered not in emitted, (
                    f"`_RENDERER_SPELLING_OVERRIDES` claims "
                    f"{backend!r} replaces the registered spelling "
                    f"{registered!r} of {family!r} with "
                    f"{override!r}, but {registered!r} still appears "
                    f"in the emitted program. The renderer took the "
                    f"`FAMILY_META` resolution after all, so the "
                    f"override is stale and the cells of every example "
                    f"using this family are charged a constant they no "
                    f"longer drop."
                )
            effective = override if override is not None else registered
            if effective in symmetric_bases:
                bare_base.add(backend)

        # BUGS and JAGS name the symmetric `dt` / `dnorm` and attach a
        # one-sided truncation suffix at every folded site, which JAGS
        # renormalizes over; the spelling alone therefore does not
        # settle them and the `dt` / `dnorm` names keep them out of
        # `bare_base` above. Gen, Turing and WebPPL name
        # `half_student_t` / `HalfStudentT`, each a grafted runtime
        # helper whose scorer adds the `log 2` itself.
        assert bare_base == _DROPS_HALF_NORMALIZER[family], (
            f"targets spelling {family!r} as a bare symmetric base are "
            f"{sorted(bare_base)}, but `_DROPS_HALF_NORMALIZER` claims "
            f"{sorted(_DROPS_HALF_NORMALIZER[family])}. A renderer "
            f"changed its emit for this family; re-derive the entitled "
            f"constant for every affected cell before editing either "
            f"side."
        )


@pytest.mark.parametrize("cell", _live_cells(), ids=lambda c: f"{c[0]}-{c[1]}")
def test_every_live_cell_has_a_registered_offset(
    cell: tuple[str, str],
) -> None:
    """Every numerically-evaluated gallery cell names its constant.

        A cell that starts passing without an entry here would be checked
        only for constant spread, which is the existential statement this
        module exists to strengthen. Growing the coverage of
        [`test_gallery_numeric_equivalence`][tests.transpile.test_gallery_numeric_equivalence]
    thus fails this test until the new cell's offset is derived
        and registered, and new coverage cannot bypass the named-constant
        criterion by arriving unannounced.
    """
    backend, stem = cell
    assert cell in _EXPECTED_OFFSET, (
        f"{backend!r} on {stem!r} is evaluated numerically by the "
        f"gallery equivalence suite but has no entry in "
        f"`_EXPECTED_OFFSET`, so its additive constant is unnamed and "
        f"only its spread is checked. Derive the constant: count the "
        f"scalar folded-family density factors in "
        f"`docs/examples/source/{stem}.qvr` (each `sample` / `observe` "
        f"step naming a family in `_HALF_FAMILIES` contributes the "
        f"product of the cardinalities of its axes), keep the ones "
        f"whose family lists {backend!r} in `_DROPS_HALF_NORMALIZER`, "
        f"multiply that subtotal by `log 2`, and "
        f"register the result with a `Derived` justification. If the "
        f"measured offset disagrees with that number, register it as "
        f"`Unexplained` and report the discrepancy rather than tuning "
        f"the entry to match."
    )


def test_registry_has_no_entries_for_dead_cells() -> None:
    """The registry names no cell the gallery suite stopped evaluating.

    A stale entry is a claim about a measurement nobody takes any more.
    It would survive a renderer regression that pushed the cell into
    `_EXPECTED_TRANSPILE_RAISES` or `_SKIP_PROBE_INCOMPATIBLE`, and
    would read as coverage that no longer exists.
    """
    live = set(_live_cells())
    stale = sorted(set(_EXPECTED_OFFSET) - live)
    assert not stale, (
        f"`_EXPECTED_OFFSET` registers {stale!r}, which the gallery "
        f"equivalence suite no longer evaluates (the cell moved into "
        f"`_EXPECTED_TRANSPILE_RAISES`, one of the `_SKIP_*` "
        f"registries, or the example lost its synthetic-data block). "
        f"Drop the entries; a registered constant nobody measures is "
        f"not coverage."
    )


@pytest.mark.parametrize(
    "cell", sorted(_EXPECTED_OFFSET), ids=lambda c: f"{c[0]}-{c[1]}"
)
def test_registry_offset_matches_the_closed_form_derivation(
    cell: tuple[str, str],
) -> None:
    """A registered constant equals what the derivation predicts.

    This is the leg that keeps the registry honest without a container.
    The prediction is recomputed from the example's QVR source, so the
    pinned number cannot drift toward whatever a runtime happened to
    return: it has to stay equal to `drops_half(T) * n_half(M) * log 2`.
    An entry the derivation does not cover is registered as
    `Unexplained`, which fails here and in
    `test_no_unexplained_offsets_are_registered`.
    """
    backend, stem = cell
    entry = _EXPECTED_OFFSET[cell]
    justification = entry.justification
    assert not isinstance(justification, Unexplained), (
        f"{backend!r} on {stem!r}: registered as unexplained "
        f"({justification.missing}); derive the constant and register it."
    )
    assert isinstance(justification, Derived), (
        f"{backend!r} on {stem!r}: justification "
        f"{type(justification).__name__!r} is neither `Derived` nor "
        f"`Unexplained`; the registry admits no third status."
    )

    counts = half_family_factor_counts(_example_path(stem))
    counted = sum(counts.values())
    assert counted == justification.half_sites, (
        f"{backend!r} on {stem!r}: the registry records "
        f"{justification.half_sites} folded-family density factor(s), "
        f"but `docs/examples/source/{stem}.qvr` now carries {counted} "
        f"({counts!r}). The example changed; re-derive the entitled "
        f"constant for every cell of this example."
    )
    dropped = _dropped_factor_count(backend, stem)
    assert dropped == justification.dropped_sites, (
        f"{backend!r} on {stem!r}: the registry records "
        f"{justification.dropped_sites} factor(s) emitted without a "
        f"renormalizer, but the drop table gives {dropped} for the "
        f"source's {counts!r}. Re-derive the entitled constant."
    )
    zeros_counts = zeros_trick_factor_counts(_example_path(stem))
    lifted = _lifted_row_count(backend, stem)
    assert lifted == justification.lifted_rows, (
        f"{backend!r} on {stem!r}: the registry records "
        f"{justification.lifted_rows} lifted zeros-trick row(s), but "
        f"the lift table gives {lifted} for the source's "
        f"{zeros_counts!r}. Each row is worth "
        f"{_ZEROS_TRICK_OFFSET:g} nats, so this is the largest term the "
        f"derivation carries; re-derive the entitled constant."
    )
    predicted = _derived_offset(backend, stem)
    assert entry.offset == pytest.approx(predicted, abs=1e-12), (
        f"{backend!r} on {stem!r}: registry pins offset "
        f"{entry.offset!r} but the derivation predicts {predicted!r} "
        f"({counted} folded-family factor(s) of which {dropped} lose "
        f"the truncation renormalizer on {backend!r}, and {lifted} "
        f"zeros-trick row(s) whose Poisson rate it lifts). The registry "
        f"must state the constant the target is entitled to, not the "
        f"constant it happened to produce."
    )


def test_no_unexplained_offsets_are_registered() -> None:
    """No cell carries an offset the derivation cannot account for.

    An `Unexplained` entry means the backend adds or drops a term
    nobody has identified, which is a correctness finding: the emitted
    program may denote a different measure and the constant-spread
    check would never say so. Recording one is allowed, so the finding
    can be pinned with its measured value while it is investigated, but
    it must also be written into `_ACKNOWLEDGED_UNEXPLAINED`, and this
    test fails until it is. The acknowledgement set is empty today.
    """
    unexplained = {
        cell: entry.justification
        for cell, entry in _EXPECTED_OFFSET.items()
        if isinstance(entry.justification, Unexplained)
    }
    surprises = sorted(set(unexplained) - _ACKNOWLEDGED_UNEXPLAINED)
    assert not surprises, (
        "cells registered with an offset the derivation does not "
        "explain: "
        + "; ".join(
            f"{backend}@{stem} {unexplained[(backend, stem)].summary}"
            for backend, stem in surprises
        )
        + ". Each is a backend term nobody has accounted for. Identify "
        "the term in the renderer's emit and re-register the cell as "
        "`Derived`, or, to carry the finding while it is investigated, "
        "add the cell to `_ACKNOWLEDGED_UNEXPLAINED`."
    )
    resolved = sorted(_ACKNOWLEDGED_UNEXPLAINED - set(unexplained))
    assert not resolved, (
        f"`_ACKNOWLEDGED_UNEXPLAINED` still lists {resolved!r}, but "
        f"those cells no longer carry an `Unexplained` justification. "
        f"Drop them from the acknowledgement set so it keeps naming "
        f"only live findings."
    )


def _stage_probe_inputs(
    scratch: pathlib.Path,
    payload: list[dict[str, dict[str, float | int | list[float] | list[int]]]],
) -> None:
    """Write a point set into `scratch` the way
    [`run_probe`][tests.transpile._docker.run_probe] does.

    Only the one file the demonstration below turns on. `run_probe`
    writes it unconditionally with `write_text`, so a second writer
    aimed at the same directory replaces it rather than colliding with
    it, and nothing in the layout records who wrote it.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "points.json").write_text(json.dumps(payload))


def test_a_shared_scratch_swaps_the_point_set_under_its_writer() -> None:
    """The corruption channel a cell-derived scratch path opens, shown
    on the file layout itself, and shown closed by
    [`probe_scratch`][tests.transpile._gallery_data.probe_scratch].

    Two processes measuring the same cell derive the same path from
    `(model, backend)`, so the second one's `points.json` lands on top
    of the first one's. The first process then launches its container
    against the second's point set, reads back log-densities for points
    it never chose, and differences them against a QVR reference scored
    at its own. Nothing raises: the offset simply comes out wrong, by
    whatever the two point sets differ by, on *every* backend of that
    model at once.

    The private path forbids the same sequence, which is the whole
    content of the fix: the isolation is structural rather than a
    convention callers are asked to respect.
    """
    ours: list[dict[str, dict[str, float | int | list[float] | list[int]]]] = [
        {"params": {"tau": 50.5}, "data": {"y": [3.0, 4.0]}}
    ]
    theirs: list[dict[str, dict[str, float | int | list[float] | list[int]]]] = [
        {"params": {"tau": 12.25}, "data": {"y": [9.0, 1.0]}}
    ]

    shared_root = _gallery_data.probe_scratch("shared-scratch-demo")
    shared_ours = shared_root / "qvr_offset_changepoint_stan"
    shared_theirs = shared_root / "qvr_offset_changepoint_stan"
    _stage_probe_inputs(shared_ours, ours)
    _stage_probe_inputs(shared_theirs, theirs)
    assert json.loads((shared_ours / "points.json").read_text()) == theirs, (
        "the cell-derived path did not actually alias, so this "
        "demonstration proves nothing. Both writers must resolve to one "
        "directory for the swap to be the fault it is."
    )

    private_ours = _gallery_data.probe_scratch("offset-changepoint-stan")
    private_theirs = _gallery_data.probe_scratch("offset-changepoint-stan")
    assert private_ours != private_theirs, (
        f"two runs of one cell were handed the same directory "
        f"{private_ours}, so the label is doing the work the freshness "
        f"of the directory is supposed to do. The path must not be a "
        f"function of the cell."
    )
    _stage_probe_inputs(private_ours, ours)
    _stage_probe_inputs(private_theirs, theirs)
    assert json.loads((private_ours / "points.json").read_text()) == ours, (
        "a second writer using the same label reached this cell's "
        "scratch. `probe_scratch` must return a directory no other call "
        "names, or the harness is back to trading point sets between "
        "concurrent runs."
    )


def test_the_input_check_catches_a_rewritten_program_or_point_set() -> None:
    """
    [`_assert_container_read_our_inputs`][tests.transpile.test_expected_offsets._assert_container_read_our_inputs]
    fires on exactly the two rewrites a foreign writer performs.

    The check sits on the path that is expected never to trip, so
    nothing else in the suite would notice if it stopped asserting. Both
    rewrites are staged here because they are separately reachable: a
    concurrent run of the same cell on a different tree replaces the
    program, and one at a different point of the schedule replaces the
    point set, and either alone is enough to make the container answer a
    question the reference never asked.
    """
    scratch = _gallery_data.probe_scratch("input-check-demo")
    emitted = b"// emitted by this cell\n"
    ours: list[dict[str, dict[str, float | int | list[float] | list[int]]]] = [
        {"params": {"tau": 50.5}, "data": {"y": [3.0, 4.0]}}
    ]
    theirs: list[dict[str, dict[str, float | int | list[float] | list[int]]]] = [
        {"params": {"tau": 12.25}, "data": {"y": [9.0, 1.0]}}
    ]

    (scratch / "source.js").write_bytes(emitted)
    _stage_probe_inputs(scratch, ours)
    _assert_container_read_our_inputs(scratch, emitted, "js", ours, "demo@demo")

    _stage_probe_inputs(scratch, theirs)
    with pytest.raises(AssertionError, match=r"points\.json"):
        _assert_container_read_our_inputs(scratch, emitted, "js", ours, "demo@demo")

    _stage_probe_inputs(scratch, ours)
    (scratch / "source.js").write_bytes(b"// emitted by somebody else\n")
    with pytest.raises(AssertionError, match=r"program source"):
        _assert_container_read_our_inputs(scratch, emitted, "js", ours, "demo@demo")

    (scratch / "source.js").write_bytes(emitted)
    (scratch / "shapes.json").write_text("{}")
    with pytest.raises(AssertionError, match=r"present but empty"):
        _assert_container_read_our_inputs(scratch, emitted, "js", ours, "demo@demo")


def test_the_reference_cache_refuses_a_second_point_set() -> None:
    """The shared QVR reference is served only to the point set it was
    measured at.

    Ten cells of one example difference their container against a single
    in-process reference evaluation, and each of them rebuilds its own
    points. That coupling is safe exactly as long as those rebuilds
    agree, and it is silent when they do not: the offsets would come out
    wrong by whatever the two point sets differ by, on every backend of
    that example at once. The guard turns the agreement from an
    assumption into a measured precondition, so a divergence is reported
    where it happens instead of surfacing as a row of model failures.
    """
    stem = "_reference_cache_guard"
    example = pathlib.Path(stem).with_suffix(".qvr")
    dataset = _gallery_data.load_gallery_data(_example_path("changepoint"))
    assert dataset is not None
    scratch = _gallery_data.probe_scratch("reference-cache-guard")

    mine = [Point(params={"tau": 50.5}, data={"y": [3.0, 4.0]})]
    theirs = [Point(params={"tau": 50.5}, data={"y": [3.0, 4.5]})]
    assert _points_key(mine) != _points_key(theirs)
    assert _points_key(mine) == _points_key(
        [Point(params={"tau": 50.5}, data={"y": [3.0, 4.0]})]
    ), (
        "the point-set key is not a content digest, so it would miss "
        "for a rebuilt but identical point set and the ten cells would "
        "each pay for their own reference evaluation."
    )

    _QVR_LOG_DENSITY_CACHE[stem] = (_points_key(mine), [-131.5])
    try:
        assert _qvr_log_densities(example, dataset, mine, scratch) == [-131.5]
        with pytest.raises(RuntimeError, match=r"point set"):
            _qvr_log_densities(example, dataset, theirs, scratch)
    finally:
        del _QVR_LOG_DENSITY_CACHE[stem]


def test_probe_scratch_is_fresh_and_unshared_for_every_cell() -> None:
    """Every cell of the matrix, and every repeat of a cell, gets its
    own empty directory.

    The repeats are the load-bearing half. A path derived from
    `(model, backend)` also satisfies "pairwise distinct across cells";
    what it fails is being distinct across *runs* of one cell, which is
    what lets a stale compiled model or a leftover shape table from the
    previous run reach this run's container.
    """
    labels = [f"offset-{stem}-{backend}" for backend, stem in sorted(_EXPECTED_OFFSET)]
    labels.extend(labels[:16])
    handed_out: list[pathlib.Path] = []
    for label in labels:
        scratch = _gallery_data.probe_scratch(label)
        assert scratch.is_dir(), f"{label}: {scratch} was not created"
        assert not sorted(scratch.iterdir()), (
            f"{label}: {scratch} came back holding "
            f"{sorted(p.name for p in scratch.iterdir())!r}. A probe "
            f"scratch must be empty, or the container reads an artefact "
            f"of some earlier run as an input of this one."
        )
        handed_out.append(scratch)

    assert len(set(handed_out)) == len(handed_out), (
        f"{len(handed_out) - len(set(handed_out))} of {len(handed_out)} "
        f"scratch directories were handed out more than once. Two runs "
        f"sharing a directory exchange `points.json`, `source.*` and "
        f"`result.json`."
    )
    for scratch in handed_out:
        nested = [
            other
            for other in handed_out
            if other != scratch and scratch in other.parents
        ]
        assert not nested, (
            f"{scratch} contains {nested!r}. A nested scratch is not "
            f"empty from its parent's point of view, so the parent's "
            f"emptiness check would pass on artefacts the child wrote."
        )


def test_the_root_sweep_spares_live_roots_and_reclaims_dead_ones() -> None:
    """
    [`sweep_abandoned_probe_roots`][tests.transpile._gallery_data.sweep_abandoned_probe_roots]
    removes a root whose owner is gone and leaves every other directory
    alone.

    A per-run directory that is never reclaimed is the cost of the
    isolation, and a killed run never reaches its exit hook. The sweep
    pays that cost back, but it runs while other sessions may be
    measuring, so what it must *not* do carries as much weight as what
    it must: this process's own root, a root owned by a live pid, and
    anything whose name it does not recognise all have to survive it.
    """
    mine = _gallery_data.probe_scratch("root-sweep-live")
    my_root = _gallery_data.probe_scratch_root()
    assert my_root is not None, "no per-process root to protect"
    assert my_root.parent == _gallery_data.PROBE_SCRATCH_PARENT, (
        f"the process root {my_root} does not sit under "
        f"{_gallery_data.PROBE_SCRATCH_PARENT}, which is the directory "
        f"the default sweep walks and the one a container bind-mount "
        f"can reach. A root elsewhere is swept by nobody."
    )

    # The sweep is pointed at a directory of this test's own so that
    # exercising it cannot reach a root a concurrent session is
    # measuring under. That is not a convenience: deleting a live root
    # destroys another run's probe inputs mid-container.
    parent = pathlib.Path(tempfile.mkdtemp(prefix="root-sweep-", dir=my_root))
    dead = parent / "quivers-probe-99999999-deadowner"
    (dead / "leftover").mkdir(parents=True)
    live = parent / f"quivers-probe-{os.getpid()}-liveowner"
    (live / "in_use").mkdir(parents=True)
    unrelated = parent / "quivers-probe-not-a-pid"
    unrelated.mkdir(parents=True)

    swept = _gallery_data.sweep_abandoned_probe_roots(parent)
    assert dead in swept and not dead.exists(), (
        f"the sweep left {dead}, whose owning pid names no process. "
        f"Per-run directories that nothing reclaims accumulate for "
        f"every killed run, and the isolation this buys is what makes "
        f"per-run directories necessary in the first place."
    )
    assert live.exists(), (
        f"the sweep removed {live}, which a live pid owns. Removing a "
        f"root in use deletes another session's probe inputs "
        f"mid-measurement, which is a worse fault than the sharing "
        f"private roots exist to prevent."
    )
    assert unrelated.exists(), (
        f"the sweep removed {unrelated}, whose name carries no pid. It "
        f"must claim only directories it can prove are abandoned."
    )
    assert my_root.exists() and mine.exists(), (
        f"the sweep reached outside {parent} and touched this process's "
        f"own root {my_root}."
    )


def test_probe_scratch_root_differs_between_processes() -> None:
    """A second interpreter gets a different root.

    This is the dimension a fixed `/tmp/<prefix>_<model>_<backend>`
    path fails outright, and it is the dimension that matters on a
    machine where more than one session runs the tier at once. Measured
    by spawning an interpreter rather than argued from the naming
    scheme, because the naming scheme is what a regression would
    change.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tests.transpile._gallery_data as g;"
            "print(g.probe_scratch('offset-probe-root'))",
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"the spawned interpreter exited {completed.returncode}\n"
        f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
    )
    other = pathlib.Path(completed.stdout.strip())
    mine = _gallery_data.probe_scratch("offset-probe-root")
    root = _gallery_data.probe_scratch_root()
    assert root is not None, (
        f"no per-process scratch root exists after handing out {mine}, "
        f"so the path came from a location every process on the machine "
        f"can name and reach. That is the whole fault: two sessions "
        f"running one cell then write the same `points.json` and read "
        f"the same `result.json`."
    )
    assert mine.parent == root, (
        f"{mine} is not under this process's root {root}, so the root "
        f"does not in fact contain what the process hands out and "
        f"proves nothing about isolation."
    )
    assert other.parent != root, (
        f"a separate interpreter placed its scratch {other} under the "
        f"same root {root} this process uses, so two concurrent runs of "
        f"one cell would trade probe inputs. The root must be private "
        f"to the process."
    )
    assert other != mine


def test_the_probe_script_guard_names_the_helper_that_moved() -> None:
    """
    [`assert_probe_scripts_unchanged`][tests.transpile._gallery_data.assert_probe_scripts_unchanged]
    rejects a rewritten, a removed, and an added probe source, and
    accepts an unchanged tree.

    The guard is what turns "someone edited the harness mid-run" from an
    unattributable row of failures into a named session fault, so it has
    to be shown firing rather than assumed to. All three mutations are
    exercised because the edit includes more than a rewrite: a
    helper that appears or disappears changes what the container
    imports just as much.
    """
    baseline = _gallery_data.probe_script_digests()
    assert "_reshape.py" in baseline, (
        f"the shared reshape helper is missing from "
        f"{_gallery_data.PROBE_SCRIPT_DIR}; the digest map covers "
        f"{sorted(baseline)!r}. Every Python-side probe imports it, so "
        f"its absence is itself the fault this guard exists to name."
    )
    _gallery_data.assert_probe_scripts_unchanged(baseline)

    rewritten = dict(baseline)
    rewritten["_reshape.py"] = "0" * 64
    with pytest.raises(RuntimeError, match=r"_reshape\.py"):
        _gallery_data.assert_probe_scripts_unchanged(rewritten)

    removed = dict(baseline)
    del removed["_reshape.py"]
    with pytest.raises(RuntimeError, match=r"_reshape\.py"):
        _gallery_data.assert_probe_scripts_unchanged(removed)

    added = dict(baseline)
    added["_not_a_probe_source.py"] = "0" * 64
    with pytest.raises(RuntimeError, match=r"_not_a_probe_source\.py"):
        _gallery_data.assert_probe_scripts_unchanged(added)

    # Scoping is what keeps the diagnosis on the cells an edit actually
    # reached. A backend entrypoint no cell copied is out of scope for
    # that cell; the shared reshape helper never is.
    scope = _copied_probe_sources("stan.py")
    assert "_reshape.py" in scope and "_reshape.jl" in scope
    moved_elsewhere = dict(baseline)
    moved_elsewhere["webppl.py"] = "0" * 64
    _gallery_data.assert_probe_scripts_unchanged(moved_elsewhere, scope)
    with pytest.raises(RuntimeError, match=r"_reshape\.py"):
        _gallery_data.assert_probe_scripts_unchanged(rewritten, scope)
    moved_entrypoint = dict(baseline)
    moved_entrypoint["stan.py"] = "0" * 64
    with pytest.raises(RuntimeError, match=r"stan\.py"):
        _gallery_data.assert_probe_scripts_unchanged(moved_entrypoint, scope)

    _gallery_data.assert_probe_scripts_unchanged(baseline)


@pytest.mark.requires_docker
@pytest.mark.parametrize(
    "cell", sorted(_EXPECTED_OFFSET), ids=lambda c: f"{c[0]}-{c[1]}"
)
def test_backend_offset_matches_registry(cell: tuple[str, str]) -> None:
    """The measured constant equals the registered one.

    Runs the pinned runtime over the same deterministic point set the
    gallery equivalence test uses, takes the mean of
    `log p_QVR - log p_T`, and requires it to equal the registry value.
    This upgrades the cell's contract from "some constant" to "this
    constant": a backend that drops a prior factor, double-counts a
    normalizer, or scores a family at a shifted parameterization moves
    the mean without moving the spread, and passes the existential
    check while failing this one.

    The spread is asserted first, since a non-constant difference makes
    the mean meaningless.
    """
    backend, stem = cell
    image, source_ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.docker_available():
        raise RuntimeError(
            "docker daemon not reachable; the session-scope "
            "`_ensure_docker_environment` autouse fixture should have "
            "started it"
        )
    if not _docker.image_available(image):
        raise RuntimeError(
            f"docker image {image!r} not available; the session-scope "
            f"`_ensure_docker_environment` autouse fixture should have "
            f"built it"
        )

    example = _example_path(stem)
    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{stem!r}: `load_gallery_data` returned None even though the "
        f"example is not in `_SKIP_DATASET_LOAD_FAILED`."
    )

    points = _gallery_data.points_from_dataset(dataset)
    labels = _gallery_data.perturbation_labels(len(points))
    copied = _copied_probe_sources(script_name)
    _gallery_data.assert_probe_scripts_unchanged(_PROBE_SCRIPTS_AT_IMPORT, copied)
    scratch = _gallery_data.probe_scratch(f"offset-{stem}-{backend}")
    assert not sorted(scratch.iterdir()), (
        f"{backend}@{stem}: the scratch {scratch} already holds "
        f"{sorted(p.name for p in scratch.iterdir())!r} before the probe "
        f"has written anything. A probe scratch must be created fresh "
        f"for the run; an inherited artefact (a compiled model, a "
        f"`shapes.json` this call does not overwrite) is read by the "
        f"container as if this cell had produced it."
    )

    qvr_lps = _qvr_log_densities(example, dataset, points, scratch)
    emitted = transpile(parse(example.read_text()), target=backend)
    script_path = _gallery_data.PROBE_SCRIPT_DIR / script_name
    payload = [_gallery_data.target_point(dataset, point) for point in points]
    raw_result = _docker.run_probe(
        image=image,
        script=script_path,
        source=emitted,
        source_ext=source_ext,
        points=payload,
        scratch=scratch,
        shapes=_gallery_data.target_shapes(dataset, _shapes_from_dataset(dataset)),
        dtypes=_dtypes_from_dataset(dataset),
        timeout=600.0,
    )
    _assert_container_read_our_inputs(
        scratch, emitted, source_ext, payload, f"{backend}@{stem}"
    )
    _gallery_data.assert_probe_scripts_unchanged(_PROBE_SCRIPTS_AT_IMPORT, copied)
    backend_lps = [float(x) for x in raw_result["log_densities"]]

    atol = _offset_atol(dataset)
    # `assert_log_density_match` returns mean(target - qvr); the
    # registry states the reference-minus-target sign, so negate.
    measured = -_equivalence.assert_log_density_match(
        qvr_lps,
        backend_lps,
        atol=atol,
        context=f"{backend}@{stem} (named-constant)",
        labels=labels,
        min_points=2,
    )

    entry = _EXPECTED_OFFSET[cell]
    assert abs(measured - entry.offset) <= atol, (
        f"{backend}@{stem}: measured additive constant "
        f"{measured:+.6f} but the cell is entitled to "
        f"{entry.offset:+.6f} ({entry.justification.summary}); "
        f"discrepancy {measured - entry.offset:+.6e} exceeds "
        f"{atol:.3e}. The difference is point-independent, so the "
        f"constant-spread check cannot see it: the backend scores a "
        f"term the reference does not, or drops one the derivation "
        f"does not entitle it to drop. In units of a truncation "
        f"renormalizer the discrepancy is "
        f"{(measured - entry.offset) / _LOG_2:+.4f} x log 2."
    )
