"""Numeric equivalence checks for gallery programs.

For each gallery example with synthetic data, the test builds a
support-preserving multi-point set and evaluates the QVR reference and
each available backend probe. Supported cells must agree up to a
point-independent additive constant under
[`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match].
Known unsupported constructs and probe limitations are recorded in
explicit registries.

The suite checks registry coverage, point-set variation, reference
values, probe isolation, and pairwise backend transitivity. Container
cells are skipped when their image is unavailable.
"""

from __future__ import annotations

import math
import pathlib

import pytest
import torch

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence, _gallery_data
from tests.transpile.probes.qvr import QvrProbe


_BACKENDS_WITH_IMAGES = {
    "stan": ("panproto-test-stan", "stan", "stan.py"),
    "numpyro": ("panproto-test-numpyro", "py", "numpyro.py"),
    "pyro": ("panproto-test-pyro", "py", "pyro.py"),
    "pymc": ("panproto-test-pymc", "py", "pymc.py"),
    "edward2": ("panproto-test-edward2", "py", "edward2.py"),
    "turing": ("panproto-test-julia", "jl", "turing.jl"),
    "gen": ("panproto-test-julia", "jl", "gen.jl"),
    "webppl": ("panproto-test-node", "js", "webppl.py"),
    "jags": ("panproto-test-jags", "jags", "jags.py"),
    "bugs": ("panproto-test-bugs", "bugs", "jags.py"),
}


def _gallery_cells() -> list[pathlib.Path]:
    return _gallery_data.gallery_examples_with_data()


def _backend_cells() -> list[tuple[pathlib.Path, str]]:
    """Every ``(example, backend)`` cell the backend comparison is a test
    case for: the cells the probe scores and the cells whose transpile
    is a pinned refusal. A cell in one of the ``_SKIP_*`` registries is
    not a test case; the registry tests keep those registries honest.
    """
    cells: list[tuple[pathlib.Path, str]] = []
    for example in _gallery_cells():
        stem = example.stem
        if stem in _SKIP_DATASET_LOAD_FAILED or stem in _SKIP_QVR_INCOMPATIBLE:
            continue
        for backend in sorted(_BACKENDS_WITH_IMAGES):
            if (backend, stem) in _SKIP_PROBE_INCOMPATIBLE:
                continue
            cells.append((example, backend))
    return cells


# ----------------------------------------------------------------------
# Pre-declared cell outcomes.
#
# Population workflow when a previously-skipped infrastructure gap
# closes (probe script gains shape registration; QVR trace grows a
# primitive): drop the cell from the corresponding registry, re-run,
# pass.
#
# Population workflow when a renderer regression introduces a
# transpile gap: pin the cell in `_EXPECTED_TRANSPILE_RAISES` with
# the new kind-prefix; the test then makes that raise a contract
# instead of a silent xfail.
# ----------------------------------------------------------------------

# Cells where the pipeline raises `UnsupportedConstruct` and the
# raise is the documented support boundary. `pytest.raises` asserts
# the kind-prefix; a closed gap surfaces as a test failure and a
# regression (different kind) surfaces as a mismatch. Each group
# carries a one-line reason for the boundary.
_EXPECTED_TRANSPILE_RAISES: dict[tuple[str, str], str] = {
    # mixture_model names the `MixtureNormal` likelihood, which the
    # BUGS lower family registry has no target name for. Every other
    # backend resolves it and its cell is live.
    ("bugs", "mixture_model"): "family:MixtureNormal",
    # tensor_contraction carries `composition` and `contraction`
    # declarations; no PPL backend has a surface for them. The example
    # still scores a joint, because its `.md` snippet wraps the
    # compiled contraction in a `MonadicProgram`, so it carries a
    # `_QVR_REFERENCE_JOINT` row. Every one of its cells being a pinned
    # raise means no container re-derives that row, and the witness
    # therefore has to come from the raw-`torch.distributions` side in
    # `test_oracle_reference_strength.py`.
    ("bugs", "tensor_contraction"): "composition_decl",
    ("edward2", "tensor_contraction"): "composition_decl",
    ("gen", "tensor_contraction"): "composition_decl",
    ("jags", "tensor_contraction"): "composition_decl",
    ("numpyro", "tensor_contraction"): "composition_decl",
    ("pymc", "tensor_contraction"): "composition_decl",
    ("pyro", "tensor_contraction"): "composition_decl",
    ("stan", "tensor_contraction"): "composition_decl",
    ("turing", "tensor_contraction"): "composition_decl",
    ("webppl", "tensor_contraction"): "composition_decl",
    # kumaraswamy_bounded_outcome names `Kumaraswamy`, which has no
    # JAGS or BUGS target name. The two engines part company: the JAGS
    # renderer writes the density out in `log` and `pow` alone and
    # adds it through the zeros trick, so its cell is live and scores
    # the model up to the lift the trick pays. The BUGS renderer
    # carries no such path and its cell stays a raise. zip_regression
    # marginalizes over a Poisson scope, which is no categorical
    # mixture, and BUGS has no statement that adds the integrated
    # log-density to the joint.
    ("bugs", "kumaraswamy_bounded_outcome"): "family:Kumaraswamy",
    ("bugs", "zip_regression"): "marginalize:no-collapse",
    # beta_binomial_ab_test observes `BetaBinomial`. Every other
    # backend either has the family natively or reaches it through
    # the closed-form marginal: JAGS writes that marginal into the
    # joint with the zeros trick, whose Poisson carrier has to be
    # bound to data, and JAGS binds it in the `data { ... }` block
    # the emitted source opens. The BUGS language has no such block,
    # so the carrier has nowhere to be declared inside a
    # self-contained model file, and
    # `renderers/bugs.py::_NO_BUGS_DISTRIBUTION` records the family
    # as unreachable rather than emitting source that references an
    # undeclared node. The boundary is the BUGS *language*, not the
    # engine: the `panproto-test-bugs` image runs JAGS, which is why
    # the sibling `jags` cell is live and scores the same model.
    ("bugs", "beta_binomial_ab_test"): ("family:BetaBinomial:no-bugs-distribution"),
    # lda observes its word codes, which the data holds counting from
    # zero, and `dcat` counts from one. JAGS binds the shifted codes in
    # its `data { ... }` block; the BUGS language has no such block, no
    # logical node computed from data can also carry a distribution,
    # and the zeros trick that would score the shifted code needs a
    # data-bound carrier, so BUGS refuses the observed code.
    ("bugs", "lda"): "observe:class-code",
    # gru_lm and lstm_lm build their gate arguments with `+` between
    # two `Hidden`-wide operands. The BUGS model language has no
    # elementwise vector arithmetic: only the contracted product
    # lowers, to `inprod(a, b)`. Both engines therefore reject the
    # `let` before any site is emitted, which is a statement about the
    # target language rather than about the oracle. Both examples sit
    # in `_SKIP_QVR_INCOMPATIBLE`, and the test consults the raises
    # first, so pinning these four turns four cells that asserted
    # nothing into four that assert the boundary. The remaining eight
    # targets render both examples and stay carried by that registry.
}

# A morphism that draws its mean from an `mlp` param source keeps its
# network weights model-internal, absent from both the wire form and
# the sample sites, so no backend can reconstruct the mean; the
# transpiler raises on every backend. Each model below hides part of
# its structure in a `param_source` network rather than writing it as
# a program whose steps are declared sites.
# The targets that reduce an ungrouped `marginalize` row by row
# refuse it rather than emit it. An ungrouped block shares one latent
# across the body's rows, so its density accumulates the rows and
# reduces once; scoring each row on its own gives every row a draw the
# source never declared, and the difference moves with the data, so it
# is a different measure rather than a different base measure. The
# reference was itself reducing per row until it was corrected against
# `docs/semantics/programs.md` §2.6, which is why these four were
# built to agree with it, and `gen` marginalizes through the same
# emission as `turing`. `stan`, `numpyro`, `pyro` and `webppl` emit
# the accumulated order and score the corrected reference.
for _ungrouped_backend in (
    "bugs",
    "church",
    "edward2",
    "gen",
    "jags",
    "pymc",
    "turing",
):
    for _ungrouped_example in ("changepoint",):
        _EXPECTED_TRANSPILE_RAISES[(_ungrouped_backend, _ungrouped_example)] = (
            "marginalize:ungrouped-over-plate"
        )

# `scan(cell)` denotes one draw per sequence position over
# intermediate states the program never names, over a sequence axis
# that is not a declared object. The expansion pass refuses before any
# renderer runs, so the kind is the same on every target, and pinning
# it asserts the boundary where a skip asserted nothing.
for _scan_model in (
    "bidirectional_rnn_lm",
    "gru_lm",
    "lstm_lm",
    "vanilla_rnn_lm",
):
    for _scan_backend in _BACKENDS_WITH_IMAGES:
        _EXPECTED_TRANSPILE_RAISES[(_scan_backend, _scan_model)] = "scan:no-lowering"

# tree_categorical builds its leaf table with a `factor` over the
# `Class` axis, which no plate of the program carries; the Gen helper
# reads factor extents from plate cardinalities and refuses.
_EXPECTED_TRANSPILE_RAISES[("gen", "tree_categorical")] = "let-expr:LetExprFactor:gen"

# BUGS has no data block to shift tree_categorical's zero-based class
# codes to the one-based codes `dcat` reads, and no free log-density
# statement to score them otherwise.
_EXPECTED_TRANSPILE_RAISES[("bugs", "tree_categorical")] = "observe:class-code"

for _neural_model in (
    "deep_markov",
    "seq2seq",
    "transformer_lm",
    "vae",
):
    for _neural_backend in _BACKENDS_WITH_IMAGES:
        _EXPECTED_TRANSPILE_RAISES[(_neural_backend, _neural_model)] = (
            "param-source:mlp"
        )

# Gallery examples whose `.md` synthetic-data snippet fails to
# `exec` (raises at runtime, or never sets the `observations`
# dict). Numeric evaluation has no point set, so the cell skips.
#
# The registry is empty: every gallery example's snippet executes and
# leaves an `observations` dict behind, including the structural
# algebra example `tensor_contraction`, whose snippet wraps the
# compiled contraction in a `MonadicProgram` with entrywise
# `Normal(0, 1)` priors over each declared arrow and a
# `Normal(score, 0.5)` likelihood over replicate judgments. Its joint
# is pinned like every other example's. Keeping the registry (rather
# than deleting it) keeps the skip path a named, testable branch, so a
# snippet that stops executing is recorded here rather than turning
# into a bare load failure.
_SKIP_DATASET_LOAD_FAILED: frozenset[str] = frozenset()

# Gallery examples whose oracle joint no pin can be held to, for want
# of a source outside the oracle.
#
# `bidirectional_rnn_lm` and `transformer_lm` carry a `sample h <-
# backbone` latent whose backbone is a `SampledComposition` over
# continuous intermediate objects; each factor is scored once along the
# composition's canonical path. `gru_lm`, `lstm_lm`, and
# `vanilla_rnn_lm` scan a program cell: the embedding and every cell
# draw are sites of the program at every position, `x_in@t` and the
# cell's labels at `@t`, so their joint is the sum of every position's
# densities and the emission's. All five joints are bitwise stable
# across global RNG seeds, which
# `test_oracle_determinism.py::test_composition_marginalised_models_are_bitwise_deterministic`
# measures; what they lack is a witness. Every backend refuses the scan
# or the composition, so no container re-derives the number, and no
# reconstruction in `test_oracle_reference_strength.py` does either.
# `test_oracle_reference_strength.py::test_unwitnessable_exempt_examples_have_no_independent_source`
# measures that ground per example, and
# `test_oracle_reference_strength.py::test_composition_exemption_grounds_partition_the_registry`
# requires it to cover this registry exactly. A reconstruction or a
# live backend cell is what moves a row to `_QVR_REFERENCE_JOINT`.
_SKIP_QVR_INCOMPATIBLE: frozenset[str] = frozenset(
    {
        "bidirectional_rnn_lm",
        "gru_lm",
        "lstm_lm",
        "transformer_lm",
        "vanilla_rnn_lm",
    }
)

# ----------------------------------------------------------------------
# The reference pin.
#
# Theorem 4.1's quotient by an additive constant is what makes the
# backend comparison robust to differing base measures, and it is also
# what makes it blind: an oracle that is wrong by the same amount at
# every point produces exactly the spread of a correct one. Nothing on
# the backend side can see such an error, so the reference carries its
# own pin, and the pin has to hold everywhere the comparison does.
#
# Three properties, each asserted rather than assumed:
#
# 1. **Total.** Every gallery example that scores a joint has an entry
#    here or a justified row in `_REFERENCE_PIN_EXEMPT`, and
#    `test_gallery_reference_pin_registry_is_total` fails when the two
#    registries stop covering the gallery exactly. A deleted row, or a
#    new example, is a failure rather than a silently weaker suite.
# 2. **Per point.** An entry pins the joint at every point of
#    `points_from_dataset`, in schedule order, not only at the ground
#    truth. A single-point pin cannot see a *data-dependent* oracle
#    error, and neither can the constant-spread check when the same
#    error rides on both sides, so between them that class would be
#    invisible.
# 3. **Representable.** The comparison runs at `reference_pin_atol`,
#    an eight-ULP bound on absolute float32 replay. The backend check
#    separately bounds the centred spread of pairwise differences;
#    its scalar floor cannot cap an absolute pin below one float32 ULP.
#
# Every value here is reproduced from an independent computation on
# each run: an example with live backend cells is re-derived by those
# containers, and every example without one is re-derived from raw
# `torch.distributions` in `test_oracle_reference_strength.py`.
# `test_oracle_reference_strength.py::test_every_pinned_example_has_an_independent_witness`
# asserts that split covers the registry with nothing left over.
#
# Update an entry only after re-deriving the joint independently; a
# drift here is either an oracle regression or a deliberate
# ground-truth change, and the two are told apart by the raw-torch
# reconstruction, not by the number's plausibility.
# ----------------------------------------------------------------------
_QVR_REFERENCE_JOINT: dict[str, tuple[float, ...]] = {
    "ar1": (
        -24.030813217163086,
        -121.17807006835938,
        -44.41229248046875,
        -37.35548400878906,
        -44.10717010498047,
        -38.02667236328125,
    ),
    "bayesian_regression": (
        -54.560791015625,
        -67.99243927001953,
        -81.71551513671875,
        -117.940673828125,
        -164.50860595703125,
        -83.58819580078125,
    ),
    "beta_binomial_ab_test": (
        -192.71270751953125,
        -227.93560791015625,
        -213.17494201660156,
        -300.0035400390625,
        -194.8147430419922,
        -200.92127990722656,
    ),
    "beta_regression": (
        224.82302856445312,
        168.67962646484375,
        166.23532104492188,
        147.81861877441406,
        205.65419006347656,
        183.39932250976562,
    ),
    "bnn": (
        92.09696960449219,
        48.172828674316406,
        -266.59625244140625,
        -614.60498046875,
        -1252.4736328125,
        -316.6635437011719,
    ),
    "changepoint": (
        -131.08474731445312,
        -155.9451446533203,
        -133.2001953125,
        -145.42013549804688,
        -188.2016143798828,
        -135.9104766845703,
    ),
    "continuous_hmm": (
        -21.395313262939453,
        -21.497692108154297,
        -21.478761672973633,
        -22.007896423339844,
        -22.696565628051758,
        -21.73342514038086,
    ),
    "factor_analysis": (
        -3338.30810546875,
        -4454.0234375,
        -3609.645263671875,
        -4188.2109375,
        -3860.51318359375,
        -3656.01806640625,
    ),
    "gamma_regression": (
        -296.97430419921875,
        -488.4329528808594,
        -341.32977294921875,
        -396.873046875,
        -327.86138916015625,
        -330.0488586425781,
    ),
    "half_student_t_hierarchical": (
        -65.16757202148438,
        -134.65968322753906,
        -142.0482940673828,
        -178.59840393066406,
        -113.85769653320312,
        -151.25946044921875,
    ),
    "horseshoe_regression": (
        -197.0647735595703,
        -257.0046691894531,
        -225.5227813720703,
        -430.2098083496094,
        -672.554931640625,
        -241.2408447265625,
    ),
    "irt_2pl": (
        -1307.60009765625,
        -1328.0120849609375,
        -1623.36962890625,
        -1703.003662109375,
        -1334.9044189453125,
        -1599.81591796875,
    ),
    "kumaraswamy_bounded_outcome": (
        48.15655517578125,
        -1.2505619525909424,
        45.79613494873047,
        25.943862915039062,
        28.047801971435547,
        41.459083557128906,
    ),
    "lda": (
        -18.173583984375,
        -18.84844970703125,
        -444.3291015625,
        -437.18743896484375,
        -20.41473388671875,
        -442.911376953125,
    ),
    "linear_gaussian_ssm": (
        -6.176448822021484,
        -6.644833564758301,
        -6.308502197265625,
        -6.5248260498046875,
        -6.4006500244140625,
        -6.19073486328125,
    ),
    "logistic_noise_regression": (
        -80.29877471923828,
        -94.24589538574219,
        -92.63084411621094,
        -86.77716827392578,
        -107.02436065673828,
        -84.29177856445312,
    ),
    "mixture_model": (
        -391.03955078125,
        -424.0065002441406,
        -408.49810791015625,
        -417.8617248535156,
        -435.7660217285156,
        -400.8327331542969,
    ),
    "negbin_regression": (
        -687.5454711914062,
        -739.462646484375,
        -735.5672607421875,
        -751.8236083984375,
        -697.0734252929688,
        -719.3248901367188,
    ),
    "parametric_pooling": (
        -210.12452697753906,
        -212.90829467773438,
        -210.09971618652344,
        -210.6964111328125,
        -211.61328125,
        -210.4910430908203,
    ),
    "pcfg": (
        -23.014850616455078,
        -26.371784210205078,
        -25.505828857421875,
        -23.08839225769043,
        -22.200294494628906,
        -27.879863739013672,
    ),
    "pmf": (
        -409.01629638671875,
        -902.4696044921875,
        -596.490234375,
        -1430.7745361328125,
        -952.38037109375,
        -595.33740234375,
    ),
    "ppca": (
        -2898.3642578125,
        -3589.615234375,
        -3093.763427734375,
        -3519.74853515625,
        -3539.537841796875,
        -3080.2548828125,
    ),
    "stochastic_volatility": (
        -307.8003845214844,
        -497.64605712890625,
        -374.3330078125,
        -449.9366760253906,
        -448.69561767578125,
        -387.8696594238281,
    ),
    "survival_weibull": (
        -155.67352294921875,
        -184.38619995117188,
        -175.67501831054688,
        -172.9993896484375,
        -176.0693359375,
        -159.7487335205078,
    ),
    "tensor_contraction": (
        -482.0543212890625,
        -804.6907958984375,
        -516.5264282226562,
        -971.0435791015625,
        -1093.5211181640625,
        -514.5181274414062,
    ),
    "tree_categorical": (
        -2984.506103515625,
        -3060.587646484375,
        -3399.37646484375,
        -3582.632568359375,
        -2993.26953125,
        -3461.560546875,
    ),
    "zip_regression": (
        -583.2901611328125,
        -580.0896606445312,
        -725.8516235351562,
        -756.9186401367188,
        -606.8170166015625,
        -747.0103759765625,
    ),
    "deep_markov": (
        -70.9559326171875,
        -71.23564910888672,
        -70.98413848876953,
        -70.98515319824219,
        -70.94621276855469,
        -71.1627197265625,
    ),
    "seq2seq": (
        -488.5443420410156,
        -487.85491943359375,
        -489.553466796875,
        -490.8770751953125,
        -489.0007629394531,
        -489.3353576660156,
    ),
    "vae": (
        -4664.140625,
        -4679.982421875,
        -4722.1455078125,
        -4720.44921875,
        -4680.5263671875,
        -4712.373046875,
    ),
}

# Gallery examples that carry synthetic data but score no joint the
# pin could hold, each with the reason. An entry is an assertion, not
# an escape hatch: `test_gallery_reference_pin_registry_is_total`
# requires the reason to name a registry that independently agrees the
# example has no reference, so an exemption cannot outlive the gap it
# describes.
_REFERENCE_PIN_EXEMPT: dict[str, str] = {
    # Sequence models whose exact joint nothing outside the oracle
    # re-derives: every backend refuses the scan or the composition,
    # and no reconstruction exists. Each sits in
    # `_SKIP_QVR_INCOMPATIBLE`, whose comment carries the diagnosis
    # and names the test that measures the ground.
    "bidirectional_rnn_lm": "no backend or reconstruction witnesses the joint",
    "gru_lm": "no backend or reconstruction witnesses the joint",
    "lstm_lm": "no backend or reconstruction witnesses the joint",
    "transformer_lm": "no backend or reconstruction witnesses the joint",
    "vanilla_rnn_lm": "no backend or reconstruction witnesses the joint",
}

_REFERENCE_PIN_ULP_BUDGET = 8
"""Round-off budget for
[`reference_pin_atol`][tests.transpile.test_gallery_numeric_equivalence.reference_pin_atol],
in units of the float32 grid spacing at the pinned magnitude.

The reference joint reaches the harness as
``float(trace.log_joint.sum().item())``: the exact float64 widening of
a float32 accumulator. Two evaluations of the *same* density therefore
agree exactly whenever they accumulate in the same order, and differ
only in the low bits of that accumulator when they do not (a different
reduction kernel, SIMD width, or BLAS). The budget is calibrated
against direct measurements of that re-association effect rather than
against a guess. The raw-`torch.distributions` reconstructions in
`test_oracle_reference_strength.py` sum the same per-site terms in a
different order and grouping, while the CI probe replays the trace on
independent runners. The largest observed difference is **two**
float32 ULPs (1.95e-03, on a feed-forward chain's joint of magnitude
15,836);
`continuous_hmm`'s independent reconstruction differs by one ULP
(6.10e-05 at magnitude 736). Eight ULPs is two bits of headroom above
the measured worst case.

This is a headroom figure, not a necessity: within each measured
runtime and platform, the oracle is bit-exact run to run and across
`torch.set_num_threads`. A failure at this budget is a real change in
the density, not noise."""


def _float32_ulp(value: float) -> float:
    """Spacing of the float32 grid at `value`.

    `math.frexp` returns ``(m, e)`` with ``|value| = m * 2**e`` and
    ``m`` in ``[0.5, 1)``, so the leading significand bit has weight
    ``2**(e - 1)`` and the trailing bit of a 24-bit significand has
    weight ``2**(e - 24)``.
    """
    magnitude = abs(value)
    if not math.isfinite(magnitude):
        raise ValueError(
            f"_float32_ulp needs a finite value; got {value!r}. A "
            f"non-finite reference is a broken pin, not a tolerance "
            f"question."
        )
    if magnitude == 0.0:
        return math.ldexp(1.0, -24)
    _, exponent = math.frexp(magnitude)
    return math.ldexp(1.0, exponent - 24)


def reference_roundoff_atol(reference: float) -> float:
    """Float32 reassociation budget at `reference`.

    This is the tolerance for an independently grouped summand or
    joint, where the only admissible difference is low-bit movement on
    the float32 grid. It is not capped by the backend-equivalence
    floor: a large float32 value may have an ULP wider than that scalar
    floor.
    """
    return _REFERENCE_PIN_ULP_BUDGET * max(
        _float32_ulp(reference),
        _float32_ulp(1.0),
    )


def reference_pin_atol(reference: float) -> float:
    """Absolute tolerance for a whole-joint reference pin.

    The bound is
    [`_REFERENCE_PIN_ULP_BUDGET`][tests.transpile.test_gallery_numeric_equivalence._REFERENCE_PIN_ULP_BUDGET]
    ULPs of the float32 grid at the pinned magnitude, floored at the
    same budget taken at magnitude 1. The floor keeps a joint that
    happens to land near zero from demanding bit equality: such a
    joint is still a sum of order-one terms, and the grid at 1 is the
    finest resolution those terms carry.

    This absolute replay bound and the tolerance used by
    [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]
    measure different quantities. The former bounds absolute drift
    from an independently derived scalar. The latter bounds the
    centred spread of pointwise backend-minus-QVR differences, after
    quotienting out an additive constant. Its scalar floor therefore
    cannot cap this function below a representable float32 step.

    The largest pinned magnitudes are `zip_regression`'s roughly
    760-nat joints, where eight float32 ULPs are 4.88e-04. The tightest bound is
    9.54e-07, the magnitude-1 floor, which
    binds wherever a joint lands below 2
    (`kumaraswamy_bounded_outcome` at its first latents point,
    magnitude 1.25). Thus the pin remains magnitude-aware without
    demanding a sub-ULP comparison that no float32 accumulator can
    satisfy portably.
    """
    return reference_roundoff_atol(reference)


# Gallery examples that genuinely carry no perturbable observation, so
# their point set moves the latents alone. Each entry states why the
# example's observed data cannot move, and
# `test_gallery_multipoint_set_is_in_support_and_varies` asserts the
# claim: an example listed here must have a frozen data section, and an
# example not listed here must have a data section that moves.
#
# Populate an entry only for a model whose every observation is an
# index-valued covariate or a degenerate single-valued response, never
# to quiet a data section the harness merely fails to perturb. A frozen
# data section makes the constant-spread check blind to a backend that
# drops a data-dependent term, which is the single failure mode the
# multi-point set exists to catch.
_NO_PERTURBABLE_OBSERVATION: dict[str, str] = {}

# (backend, example) cells whose measured log-density cannot yet be
# compared against the QVR reference, each grouped under the single
# defect that blocks it. Every entry has been run against its
# container and carries the error the run actually produced, so the
# comment names a live blocker rather than a historical one.
#
# Closure path: fix the named defect in its owning file, re-measure
# the cell with the six-point set, drop it from this registry. A cell
# whose spread is constant to within `adaptive_atol` belongs outside
# the registry; nothing else does.
#: Cells whose transpile succeeds and whose container cannot score
#: the result.
#:
#: **Currently empty.** Every gallery cell that transpiles is scored
#: against the reference by its backend's container. A row here is a
#: cell asserting nothing, so each one carries the error that
#: container actually returned, re-measured rather than inherited: a
#: rationale that outlives its defect reads as a reason to leave the
#: cell alone, which is how a tier stops asserting anything without
#: anyone noticing.
_SKIP_PROBE_INCOMPATIBLE: frozenset[tuple[str, str]] = frozenset()


def test_gallery_reference_pin_registry_is_total() -> None:
    """Every gallery example is either pinned or justifiably exempt,
        and neither registry may carry a row the gallery does not.

        This is the test that makes the pin a guarantee rather than an
        opt-in. The failure mode it exists to prevent is silent: a
        `dict.get` that returns `None` and a guard that skips the
        assertion turn a deleted row, or a newly-added example, into a
        check that passes while asserting nothing about the value it was
        written to protect. Requiring the two registries to *partition*
        the gallery makes both directions loud. Deleting a row fails here.
        Adding an example without deriving its reference fails here.
        Retiring an example without dropping its row fails here too, so a
        stale pin cannot sit in the registry looking like coverage.

        The exemption side is checked against the registries that
        independently agree the example has no reference, not taken on its
        word: an exempt example must appear in `_SKIP_DATASET_LOAD_FAILED`
        (no dataset at all) or in `_SKIP_QVR_INCOMPATIBLE` (no
    deterministic joint). An exemption whose gap has closed thus
        fails rather than quietly suppressing a pin the example could now
        carry.
    """
    gallery = {example.stem for example in _gallery_cells()}
    pinned = set(_QVR_REFERENCE_JOINT)
    exempt = set(_REFERENCE_PIN_EXEMPT)

    overlap = sorted(pinned & exempt)
    assert not overlap, (
        f"{overlap!r} are both pinned and exempt. An example carries a "
        f"reference or it does not; the two registries must be "
        f"disjoint so a reader can tell which examples are actually "
        f"guarded."
    )

    unpinned = sorted(gallery - pinned - exempt)
    assert not unpinned, (
        f"{unpinned!r} ship synthetic data but appear in neither "
        f"`_QVR_REFERENCE_JOINT` nor `_REFERENCE_PIN_EXEMPT`, so "
        f"nothing asserts their oracle joint is right. Derive the "
        f"reference from raw `torch.distributions` and pin every "
        f"point, or record why the example has no reference."
    )

    stale = sorted((pinned | exempt) - gallery)
    assert not stale, (
        f"{stale!r} are registered but are no longer gallery examples "
        f"with synthetic data. Drop the rows: a pin on an example that "
        f"no longer runs reads as coverage and is not."
    )

    expected_points = len(_gallery_data.perturbation_labels())
    for stem, values in sorted(_QVR_REFERENCE_JOINT.items()):
        assert len(values) == expected_points, (
            f"{stem!r}: pinned at {len(values)} point(s) against a "
            f"{expected_points}-point set. The pin has to cover every "
            f"point the equivalence check evaluates, or a "
            f"data-dependent oracle error hides in the unpinned tail."
        )
        for index, value in enumerate(values):
            assert math.isfinite(value), (
                f"{stem!r}: pinned value at point {index} is "
                f"{value!r}. A non-finite pin asserts nothing: every "
                f"comparison against it is `nan`, which no `<=` "
                f"rejects."
            )

    for stem, reason in sorted(_REFERENCE_PIN_EXEMPT.items()):
        assert reason.strip(), (
            f"{stem!r}: exempt from the reference pin with an empty "
            f"reason. An exemption without a stated cause is an "
            f"unexplained hole in the guarantee."
        )
        assert stem in _SKIP_DATASET_LOAD_FAILED or stem in _SKIP_QVR_INCOMPATIBLE, (
            f"{stem!r}: claims exemption from the reference pin "
            f"({reason}) but is in neither `_SKIP_DATASET_LOAD_FAILED` "
            f"nor `_SKIP_QVR_INCOMPATIBLE`, so the gallery tier does "
            f"score a joint for it. Either the gap closed and the row "
            f"belongs in `_QVR_REFERENCE_JOINT`, or the skip registry "
            f"is missing the entry that justifies the exemption."
        )


@pytest.mark.parametrize("example", _gallery_cells(), ids=lambda p: p.stem)
def test_gallery_qvr_reference_pin_holds_at_every_point(
    example: pathlib.Path,
) -> None:
    """The QVR reference reproduces its pinned value at **every**
        point of the set as well as at the ground truth.

        A ground-truth-only pin and the constant-spread check have
        complementary blind spots that overlap exactly on the class of
        error that matters most here. The pin at point 0 sees a constant
        offset but says nothing about points 1..5. The spread check sees a
        varying offset but is invariant to a constant one. An oracle error
        that is *zero at the ground truth and constant across the
    perturbed points* thus passes both: the pin holds where it
        looks, and the spread is unchanged because the same wrong value
        feeds both sides of every difference. Pinning per point removes
        that overlap, because a per-point pin is violated by any error
        that moves at all.

        The concrete shape of that error class in this codebase is a
        data-dependent term the oracle drops. The dropped term is zero at
        the ground truth for a fixture whose ground truth sits at the
        term's zero (a centred residual, a sum-to-zero score, a
        log-normaliser that cancels at the generating parameters) and
        non-zero once the data moves.
    """
    assert example.stem not in _SKIP_DATASET_LOAD_FAILED, (
        f"{example.stem!r} is registered as failing to load its synthetic "
        "data, which is a gap to close rather than a cell to pass over"
    )
    exempt_reason = _REFERENCE_PIN_EXEMPT.get(example.stem)
    if exempt_reason is not None:
        # No pin to hold, so what is asserted is that the exemption is
        # the registry's and not this test's: an example may sit here
        # only with a recorded reason, and
        # `test_gallery_reference_pin_registry_is_total` requires that
        # reason to name a registry which independently agrees. A
        # silent skip would let an example lose its ground and still
        # pass.
        assert example.stem not in _QVR_REFERENCE_JOINT, (
            f"{example.stem!r} carries both a pinned reference joint "
            f"and a row in `_REFERENCE_PIN_EXEMPT`. One of the two is "
            f"stale: pin it or exempt it, not both."
        )
        assert exempt_reason.strip(), (
            f"{example.stem!r} is exempt from the reference pin with "
            f"an empty reason, which records nothing anyone can check."
        )
        return

    reference = _QVR_REFERENCE_JOINT.get(example.stem)
    assert reference is not None, (
        f"{example.stem!r}: reaches the multi-point scoring path with "
        f"no `_QVR_REFERENCE_JOINT` entry. See "
        f"`test_gallery_reference_pin_registry_is_total`."
    )

    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example was not in `_SKIP_DATASET_LOAD_FAILED`."
    )

    points = _gallery_data.points_from_dataset(dataset)
    labels = _gallery_data.perturbation_labels(len(points))
    assert len(points) == len(reference), (
        f"{example.stem!r}: {len(points)} point(s) against "
        f"{len(reference)} pinned value(s). The point schedule moved "
        f"under the pin; re-derive every value before re-pinning."
    )

    probe = QvrProbe()
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_pin_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    source = example.read_bytes()
    for index, point in enumerate(points):
        measured = probe.evaluate(
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
        ).log_densities[0]
        expected = reference[index]
        atol = reference_pin_atol(expected)
        assert abs(measured - expected) <= atol, (
            f"{example.stem!r} point {index} ({labels[index]}): QVR "
            f"joint {measured!r} against pinned reference "
            f"{expected!r}, a gap of {abs(measured - expected):.6g} "
            f"nats past the {atol:.6g} round-off budget. The oracle's "
            f"density changed at this point. Re-derive it from raw "
            f"`torch.distributions` before touching the pin; widening "
            f"the tolerance would restore exactly the blindness the "
            f"per-point pin exists to remove."
        )


@pytest.mark.parametrize("example", _gallery_cells(), ids=lambda p: p.stem)
def test_gallery_qvr_logdensity_finite(example: pathlib.Path) -> None:
    """The QVR-side log-density at the ground-truth (θ_true, y)
    point evaluates to a finite real number.

    This is the always-on Tier-4 invariant: regardless of backend
    Docker availability, every gallery example with synthetic data
    must produce a well-defined joint log-density under the
    reference QVR trace. A non-finite value (`-inf`, `nan`)
    indicates that either (a) the data-gen snippet's ground-truth
    parameters fall outside the model's support, or (b) the QVR
    program has a structural defect that the trace surfaces only
    on real data.
    """
    assert example.stem not in _SKIP_DATASET_LOAD_FAILED, (
        f"{example.stem!r} is registered as failing to load its synthetic "
        "data, which is a gap to close rather than a cell to pass over"
    )

    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example was not in `_SKIP_DATASET_LOAD_FAILED`. "
        f"Add the example's stem to that registry, or fix the "
        f"`.md` snippet so it produces an `observations` dict."
    )

    point = _gallery_data.point_from_dataset(dataset)
    probe = QvrProbe()
    source = example.read_bytes()
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    result = probe.evaluate(
        source,
        example.stem,
        [point],
        scratch=scratch,
        monadic=dataset.monadic,
        x_input=dataset.x_input,
        observations=dataset.observations,
    )
    assert len(result.log_densities) == 1, (
        f"{example.stem!r}: expected exactly one log-density, got "
        f"{result.log_densities!r}"
    )
    lp = result.log_densities[0]
    assert math.isfinite(lp), (
        f"{example.stem!r}: QVR log p(θ_true, y) is non-finite ({lp!r}); "
        f"observations={list(dataset.observations)} params={list(dataset.params)}"
    )

    # Finiteness alone is a weak oracle check: a measure bug can return a
    # finite-but-wrong joint (an unclamped latent resampling, a
    # marginalized site double-counted, a plate broadcast inflating the
    # sum). The reference pin is what turns it into a correctness
    # check, and it is mandatory: an example that reaches this line has
    # a joint, so a missing entry is a hole in the guarantee and fails
    # here rather than reverting to the finiteness-only check.
    exempt_reason = _REFERENCE_PIN_EXEMPT.get(example.stem)
    if exempt_reason is not None:
        # The example carries a recorded reason for having no pin, and
        # `test_gallery_reference_pin_registry_is_total` holds that
        # reason to a registry that independently agrees. What this
        # tier still asserts is its own invariant, which needs no pin:
        # the joint is a finite real number. Skipping instead would
        # drop that too, and a joint that turned non-finite is exactly
        # the failure this line exists to catch.
        return
    reference = _QVR_REFERENCE_JOINT.get(example.stem)
    assert reference is not None, (
        f"{example.stem!r}: scores a finite QVR joint ({lp!r}) but has "
        f"no `_QVR_REFERENCE_JOINT` entry, so nothing checks that the "
        f"value is *right*. Theorem 4.1's constant-spread quotient "
        f"cannot see a point-independent oracle error, which makes "
        f"this pin the only thing that can. Re-derive the joint from "
        f"raw `torch.distributions` (see "
        f"`test_oracle_reference_strength.py`), pin every point of "
        f"the set, and add the row. If this example genuinely has no "
        f"reference, record the reason in `_REFERENCE_PIN_EXEMPT` "
        f"instead."
    )
    atol = reference_pin_atol(reference[0])
    assert abs(lp - reference[0]) <= atol, (
        f"{example.stem!r}: QVR joint {lp!r} drifted from its "
        f"independently-verified reference {reference[0]!r} by "
        f"{abs(lp - reference[0]):.6g} nats, past the "
        f"{atol:.6g} round-off budget. Either the oracle regressed or "
        f"the ground-truth point changed; re-derive the joint "
        f"independently before updating the `_QVR_REFERENCE_JOINT` "
        f"entry, and never widen the tolerance to absorb the drift."
    )


@pytest.mark.parametrize("example", _gallery_cells(), ids=lambda p: p.stem)
def test_gallery_multipoint_set_is_in_support_and_varies(
    example: pathlib.Path,
) -> None:
    """The multi-point set stays in support and actually moves.

        Three properties make
        [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]
        a real test rather than a tautology, and each is asserted here so
        a regression surfaces without needing a Docker image:

        1. Every point scores a **finite** QVR joint. A perturbation that
           steps outside the support sends both evaluators to `-inf`, and
           two `-inf` values differ by `nan` rather than by a constant, so
           the comparison would be meaningless.
        2. The joint **varies** across the set. A point set that collapses
           to repeats of the ground truth restores the single-point
           vacuity the multi-point check exists to remove: the spread of a
           constant difference sequence is zero whatever the backend
           computed. Variation in the joint is the observable form of
           "the latents or the data really moved", and it holds for every
           gallery shape, including examples that capture no latents (only
           the data moves) and examples whose data is entirely
           integer-valued covariates (only the latents move).
        3. The **observed data** varies across the set. This is strictly
           stronger than (2) and it is the property the constant-spread
           contract actually needs: a latents-only perturbation moves the
           joint while leaving every observation at ground truth, so (2)
           passes on a point set whose data section is byte-identical
           throughout. Against such a set a backend that drops a
           data-dependent term keeps a perfectly constant offset and the
           equivalence assertion is vacuous. An example whose data
    cannot move states and is in `_NO_PERTURBABLE_OBSERVATION`
           and has the frozen data section asserted rather than assumed.
    """
    assert example.stem not in _SKIP_DATASET_LOAD_FAILED, (
        f"{example.stem!r} is registered as failing to load its synthetic "
        "data, which is a gap to close rather than a cell to pass over"
    )

    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example was not in `_SKIP_DATASET_LOAD_FAILED`."
    )

    points = _gallery_data.points_from_dataset(dataset)
    labels = _gallery_data.perturbation_labels(len(points))
    assert len(points) >= 2, (
        f"{example.stem!r}: {len(points)} point(s); the constant-spread "
        f"contract needs at least two to be testable."
    )

    probe = QvrProbe()
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_points_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    source = example.read_bytes()
    lps: list[float] = []
    for point in points:
        lps.extend(
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

    for index, lp in enumerate(lps):
        assert math.isfinite(lp), (
            f"{example.stem!r}: point {index} ({labels[index]}) scores a "
            f"non-finite QVR joint ({lp!r}), so it left the model's "
            f"support. The perturbation for one of its sites does not "
            f"respect the site's declared constraint."
        )

    assert len({round(lp, 6) for lp in lps}) > 1, (
        f"{example.stem!r}: every point scores the same QVR joint "
        f"({lps[0]!r}), so no site moved and the constant-spread check "
        f"would pass unconditionally. Either the dataset captured "
        f"nothing perturbable, or a constraint this example needs is "
        f"missing from `_perturb_by_support`."
    )

    observed = _gallery_data.observed_data_names(dataset)
    moved = _gallery_data.varying_observation_names(dataset, points)
    frozen_reason = _NO_PERTURBABLE_OBSERVATION.get(example.stem)
    if frozen_reason is None:
        assert moved, (
            f"{example.stem!r}: the observed data is byte-identical at "
            f"every point ({sorted(observed)} all frozen), so the "
            f"constant-spread check for this example is a "
            f"latents-only test: a backend that drops a "
            f"data-dependent term would keep a constant offset and "
            f"pass. Give the data section a support the perturber can "
            f"step in (an observed count moves inside its attested "
            f"range, a simplex-valued observation renormalises, a "
            f"bounded one moves in its own space), or, if this "
            f"example truly has no perturbable observation, record "
            f"the reason in `_NO_PERTURBABLE_OBSERVATION`."
        )
    else:
        assert not moved, (
            f"{example.stem!r}: `_NO_PERTURBABLE_OBSERVATION` claims "
            f"this example has no perturbable observation "
            f"({frozen_reason}), but {sorted(moved)} moved across the "
            f"point set. Drop the entry: the data section is testable "
            f"and the claim is stale."
        )


@pytest.mark.parametrize(
    ("example", "backend"),
    _backend_cells(),
    ids=lambda value: value.stem if isinstance(value, pathlib.Path) else value,
)
def test_gallery_backend_logdensity_matches_qvr(
    example: pathlib.Path, backend: str
) -> None:
    """Constant-spread equivalence between the QVR reference and the
    backend's native log-density at the ground-truth (θ_true, y)
    point set for `example`."""
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
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

    cell = (backend, example.stem)

    expected_raise = _EXPECTED_TRANSPILE_RAISES.get(cell)
    if expected_raise is not None:
        source = example.read_bytes()
        with pytest.raises(UnsupportedConstruct) as exc_info:
            transpile(parse(source.decode("utf-8")), target=backend)
        kinds = exc_info.value.kinds
        assert any(k.startswith(expected_raise) for k in kinds), (
            f"{backend!r} on {example.stem!r}: expected raise with "
            f"kind prefix {expected_raise!r}, got kinds={kinds!r}. "
            f"Either the renderer changed (update the entry in "
            f"`_EXPECTED_TRANSPILE_RAISES`) or a different gap fired."
        )
        return

    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example was not in `_SKIP_DATASET_LOAD_FAILED`. "
        f"Add the example's stem to that registry, or fix the "
        f"`.md` snippet so it produces an `observations` dict."
    )

    source = example.read_bytes()
    emitted = transpile(parse(source.decode("utf-8")), target=backend)

    points = _gallery_data.points_from_dataset(dataset)
    labels = _gallery_data.perturbation_labels(len(points))
    qvr_probe = QvrProbe()
    # Isolate the bind-mounted scratch per (backend, example) so a
    # probe never mounts a source or helper file another backend's run
    # left behind in a shared directory.
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_eq_{example.stem}_{backend}"
    scratch.mkdir(exist_ok=True, parents=True)
    # One probe call per point: the probe's `observations` keyword
    # overrides the flat per-point payload (it is the only channel that
    # preserves multi-axis shapes), so a perturbed point needs its own
    # pre-shaped observation dict. Passing the dataset's ground-truth
    # observations once for the whole set would score the QVR side at
    # the unperturbed data while the container scored the perturbed
    # data.
    qvr_lps: list[float] = []
    for point in points:
        qvr_lps.extend(
            qvr_probe.evaluate(
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

    script_path = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    raw_result = _docker.run_probe(
        image=image,
        script=script_path,
        source=emitted,
        source_ext=ext,
        points=[_gallery_data.target_point(dataset, point) for point in points],
        scratch=scratch,
        shapes=_gallery_data.target_shapes(dataset, _shapes_from_dataset(dataset)),
        dtypes=_dtypes_from_dataset(dataset),
    )

    backend_lps = [float(x) for x in raw_result["log_densities"]]

    _equivalence.assert_log_density_match(
        qvr_lps,
        backend_lps,
        context=f"{backend}@{example.stem}",
        labels=labels,
        min_points=2,
    )


def _shapes_from_dataset(
    dataset: _gallery_data.GalleryDataset,
) -> dict[str, list[int]]:
    """Per-name shape table the per-backend probe scripts read to
    rebuild nested arrays from the row-major flat lists in `Point`.

    Scalar tensors (zero-dim) map to ``[]`` rather than ``[1]``;
    `point_from_dataset` collapses length-1 lists to bare scalars
    and `_reshape.reshape_value` reads the empty shape as a no-op
    after the dtype cast.
    """

    def _emitted_shape(value: torch.Tensor) -> list[int]:
        """The shape the emitted program declares for this name.

        A plate-less program is emitted for one row, so the fixture's
        length-one batch axis has no counterpart in the declaration:
        Stan writes `vector[16] state`, not `array[1] vector[16]`.
        The flat list is identical, so dropping the axis here is what
        lets the probe rebuild it against what the program declares.
        """
        shape = list(value.shape)
        if dataset.single_row and len(shape) > 1 and shape[0] == 1:
            return shape[1:]
        return shape

    shapes: dict[str, list[int]] = {}
    for k, v in dataset.observations.items():
        shapes[k] = _emitted_shape(v)
    for k, v in dataset.params.items():
        shapes[k] = _emitted_shape(v)
    # A compiled parameter map reaches the container in the data
    # section, so the probe needs its shape to rebuild the matrix from
    # the flat row-major list it travels as.
    for k, v in dataset.param_wires.items():
        shapes[k] = _emitted_shape(v)
    # A scalar type-parameter reaches the container as a bare float in
    # the point's data section; the empty shape casts it through the
    # dtype table without rebuilding a nested container around it.
    for name in dataset.scalar_params:
        shapes[name] = []
    return shapes


def _dtypes_from_dataset(
    dataset: _gallery_data.GalleryDataset,
) -> dict[str, str]:
    """Per-name dtype tag (``"int"`` or ``"float"``) so backends that
    distinguish integer and real declarations (Stan, JAGS, BUGS, PyMC)
    get the right native type after reshape.

    Resolution order, most authoritative first:

    1. **The declared family.** When the name answers to a stochastic
       site of the compiled program, its
       [`site_supports`][tests.transpile._gallery_data.site_supports]
       constraint decides: an integer-supported site (a Poisson /
       Binomial / Categorical / Bernoulli observation) is ``"int"``
       and a continuous-supported one is ``"float"``, whatever its
       ground-truth value happens to look like. This is what keeps a
       continuous latent whose ground truth is integer-valued (a
       horseshoe global scale pinned at ``tau = 1.0``, unit local
       scales) out of the ``int`` bucket: the in-container reshape
       casts every ``int``-tagged leaf through Python's `int`, which
       truncates the value and makes the backend's density piecewise
       constant in integer buckets rather than equal to the QVR
       reference.
    2. **The declared torch dtype**, for a name with no site: an
       integer-kinded covariate (a plate subscript such as
       ``coef_idx``) is ``"int"``.
    3. **The value domain**, for a float-kinded covariate: an
       all-integer-valued one is ``"int"``. This last rule is the only
       place the value heuristic survives, and it applies to
       covariates alone: an observation's kind now comes from its
       family instead.
    """
    integer_dtypes = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    )
    supports = _gallery_data.site_supports(dataset)
    out: dict[str, str] = {}
    for section in (dataset.observations, dataset.params):
        for name, tensor in section.items():
            support = supports.get(name)
            if support is not None:
                out[name] = (
                    "int" if _gallery_data.is_discrete_support(support) else "float"
                )
            elif tensor.dtype in integer_dtypes:
                out[name] = "int"
            elif tensor.numel() > 0 and torch.equal(
                tensor,
                tensor.round(),
            ):
                out[name] = "int"
            else:
                out[name] = "float"
    # A scalar type-parameter is declared `Real` in the program header
    # and rendered as a real input (`real alpha;` in Stan's data block,
    # `rep_vector(alpha, 3)` as a Dirichlet concentration), so it is
    # float-tagged whatever its instantiated value looks like: an
    # integer cast of `alpha = 1.0` would still parse and still be
    # wrong the moment the snippet instantiates at a fractional value.
    for name in dataset.scalar_params:
        out[name] = "float"
    # A compiled parameter map is an affine weight table: real on
    # every backend, and never a site whose family could say
    # otherwise.
    for name in dataset.param_wires:
        out[name] = "float"

    return out
