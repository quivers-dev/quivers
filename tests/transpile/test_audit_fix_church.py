"""Regression tests for audit-confirmed Church transpilation defects.

Church is the abstract stochastic-lambda-calculus target of Goodman,
Mansinghka, Roy, Bonawitz, and Tenenbaum (2008); it has no single
canonical interpreter, so the emitted program grafts a self-contained
Scheme reference runtime (`sample` / `observe` / `factor` on a
distribution-object protocol, plus one correctly parameterised
distribution constructor per family). These tests assert on the
emitted text and on the grafted runtime definitions.

Covered defects:

* `half-family-loc-and-fold`: `HalfNormal` / `HalfCauchy` prepend
  ``loc=0`` and fold onto the nonnegative reals through the runtime
  ``half`` wrapper; `HalfCauchy` maps to `cauchy`, not `gaussian`.
* `deterministic-let-batch-axis`: a batched let lifts its body into a
  per-row `map` and indexes batched references by the loop variable.
* `via-gather-index`: a gather ``beta_0[out_idx]`` indexes the data
  list by the enclosing row variable, not the whole list.
* `sample-event-dims`: a scalar family stamped with event axes wraps
  an inner `map` so each draw is the declared vector.
* `marginalize-over-batch-axis`: the marginalized latent is integrated
  out, its prior gathered through the via index map and each atom's
  observe indexed by the per-row loop variable.
* `undefined-let-builtins`: `sigmoid` and `sum` are defined in the
  grafted runtime.
* `nonexistent-church-distributions`: every emitted distribution name
  has a correct ERP definition in the runtime.
* `matrix-normal-covariance-and-distribution`: `matrix-normal` is a
  distribution object with the row-major `U (x) V` covariance.
* `bernoulli-flip-numeric`: `flip` is a numeric 0/1 distribution so an
  observed integer scores against the same coding.
* `gp-sample-site`: the GP realisation is routed through `sample`.
"""

from __future__ import annotations

import pathlib
import subprocess


from quivers.dsl.parser import parse
from quivers.transpile import transpile
from quivers.transpile._builtin_spellings import BUILTIN_SPELLINGS, NativeCall
from quivers.transpile._pipeline import parser_registry
from tests.transpile import _equivalence, _gallery_data
from tests.transpile.probes._protocol import Point
from tests.transpile._tools import require_scheme
from tests.transpile.probes.church import ChurchProbe
from tests.transpile.probes.qvr import QvrProbe

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FAMILIES = _REPO_ROOT / "tests" / "transpile" / "fixtures" / "families"
_AUDIT = _REPO_ROOT / "tests" / "transpile" / "fixtures" / "audit"
_GALLERY = _REPO_ROOT / "docs" / "examples" / "source"
_RUNTIME = _REPO_ROOT / "src" / "quivers" / "transpile" / "runtime_church.scm"


def _church(source: str) -> str:
    """Transpile a QVR source string to Church and decode the bytes."""
    return transpile(parse(source), target="church").decode()


def _church_file(path: pathlib.Path) -> str:
    """Transpile a QVR fixture file to Church and decode the bytes."""
    return _church(path.read_text())


def _model(text: str) -> str:
    """Return the model form (drops the grafted runtime prelude)."""
    idx = text.rfind("(define(model")
    assert idx >= 0, "no model form in emitted Church source"
    return text[idx:]


def _nospace(text: str) -> str:
    """Drop spaces so assertions ignore the formatter's spacing."""
    return text.replace(" ", "")


_RUNTIME_TEXT = _RUNTIME.read_text()
_RUNTIME_NS = _nospace(_RUNTIME_TEXT)


#: Scheme interpreter binaries the runtime evaluates under; the same
#: preference order the Church probe uses.
def _scheme_interpreter() -> str:
    """The Scheme interpreter the audit runs the emitted programs through.

    Returns
    -------
    str
        The interpreter's path.

    Raises
    ------
    RuntimeError
        If no Chez Scheme interpreter is on ``PATH``.
    """
    return require_scheme()


def _run_scheme(interpreter: str, program: str, scratch: pathlib.Path) -> str:
    """Run `program` through `interpreter` in ``--script`` mode; return
    stdout. Raises on a non-zero exit so a runtime abort surfaces."""
    scratch.mkdir(parents=True, exist_ok=True)
    scm = scratch / "_church_runtime_check.scm"
    scm.write_text(program)
    completed = subprocess.run(
        [interpreter, "--script", str(scm)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{interpreter} exited {completed.returncode}: "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )
    return completed.stdout


# ---------------------------------------------------------------------------
# Half-support families.
# ---------------------------------------------------------------------------


def test_halfnormal_loc_zero_and_positive_fold() -> None:
    """HalfNormal emits `(half (gaussian 0 scale))`: a ``loc=0``
    argument, the base scale, and the runtime positivity fold, not the
    bare `(gaussian scale)` that bound the scale to the mean."""
    model = _nospace(_model(_church_file(_FAMILIES / "halfnormal.qvr")))
    assert "(sample(half(gaussian01)))" in model
    assert "(sample(gaussian1))" not in model


def test_halfcauchy_maps_to_cauchy_not_gaussian() -> None:
    """HalfCauchy emits `(half (cauchy 0 scale))`: the cauchy family
    folded onto the nonnegative reals, never `gaussian`."""
    model = _nospace(_model(_church_file(_FAMILIES / "halfcauchy.qvr")))
    assert "(sample(half(cauchy01)))" in model
    assert "gaussian" not in model


def test_horseshoe_tau_lambda_inherit_half_cauchy() -> None:
    """The horseshoe's global and local scales are HalfCauchy draws, so
    they fold a cauchy; only the raw coefficient stays a gaussian."""
    model = _nospace(_model(_church_file(_AUDIT / "horseshoe_decomposed.qvr")))
    assert "(definetau(sample(half(cauchy01))))" in model
    assert "(sample(half(cauchy01)))" in model
    assert "(sample(gaussian01))" in model


# ---------------------------------------------------------------------------
# Deterministic-let batch axis.
# ---------------------------------------------------------------------------


def test_deterministic_let_lifts_batch_axis() -> None:
    """A batched let lifts its body into a per-row `map` and indexes the
    batched data input by the loop variable, so the batch axis is not
    collapsed to a single shared value."""
    model = _nospace(_model(_church_file(_GALLERY / "bayesian_regression.qvr")))
    assert "(definemu(map(lambda(m_Resp)" in model
    assert "(*beta_1(list-refxm_Resp))" in model
    # The observe indexes the now-batched mu and the response by the row.
    assert "(observe(gaussian(list-refmum_Resp)sigma)(list-refym_Resp))" in model


def test_deterministic_via_gather_indexes_row_variable() -> None:
    """A gather ``beta_0[out_idx]`` indexes the data list by the
    enclosing row variable rather than passing the whole list to
    `list-ref`."""
    model = _nospace(_model(_church_file(_GALLERY / "negbin_regression.qvr")))
    assert "(list-refbeta_0(list-refout_idxm_Resp))" in model
    assert "(list-refbeta_0out_idx)" not in model


# ---------------------------------------------------------------------------
# Sample event dims.
# ---------------------------------------------------------------------------


def test_sample_event_dims_wrap_inner_map() -> None:
    """A scalar Normal stamped with an event axis via ``over=`` wraps an
    inner `map` so each of the 32 items is a length-2 vector, not a
    flat scalar."""
    model = _nospace(_model(_church_file(_GALLERY / "factor_analysis.qvr")))
    assert (
        "(defineZ_mat(map(lambda(m_Item)"
        "(map(lambda(m_LatentDim)(sample(gaussian01)))(iota2)))(iota32)))"
    ) in model


def test_intrinsic_vector_family_has_no_inner_event_map() -> None:
    """A Dirichlet draws its simplex natively, so an ``over=`` event
    axis carries intrinsically with no inner per-event map."""
    model = _nospace(_model(_church_file(_GALLERY / "lda.qvr")))
    assert "(definetheta(map(lambda(m_Doc)(sample(dirichlet" in model
    assert "m_Topic)(sample" not in model.split("definephi")[0]


# ---------------------------------------------------------------------------
# Marginalize over batch axis + via index map.
# ---------------------------------------------------------------------------


def test_marginalize_latent_reads_its_prior_through_the_index_map() -> None:
    """The marginalized categorical declares no site: its per-document
    topic mixture is gathered one row per token through the ``via``
    index map and read as the log-weight of each atom."""
    model = _nospace(_model(_church_file(_GALLERY / "lda.qvr")))
    assert "(sample(categorical" not in model
    assert (
        "(define__marg_z_w(broadcast1log(map(lambda(i)(list-refthetai))word_idx)))"
        in model
    )
    assert "(define__marg_z_w_0(take-last__marg_z_w0))" in model
    assert "(factor(sum-leaves__marg_z))" in model


def test_via_index_map_threads_through_group_plate() -> None:
    """Each atom scores the observed word against the per-topic word
    distribution the atom pins: `phi[k]` for atom `k`.

    The observation plate is `Token`, the corpus-wide set of 200 word
    positions declared by `object Token : FinSet 200`, so the loop
    variable is `m_Token`. `word_idx` carries a token position to its
    document, whose prior weights the atoms; `phi` reads the pinned
    topic's distribution over `Vocab`.
    """
    model = _nospace(_model(_church_file(_GALLERY / "lda.qvr")))
    for atom in range(3):
        assert (
            f"(define(__marg_z_atom_{atom})(map(lambda(m_Token)(dist-score"
            f"(categorical(list-refphi{atom}))(list-refwm_Token)))(iota200)))"
        ) in model
    assert (
        "(define__marg_z(log-sum-exp(list__marg_z_t_0__marg_z_t_1__marg_z_t_2)))"
        in model
    )


# ---------------------------------------------------------------------------
# Undefined let builtins.
# ---------------------------------------------------------------------------


def test_every_church_builtin_spelling_is_defined_in_runtime() -> None:
    """Every math builtin the shared table spells as a Church procedure
    is defined in the grafted runtime, as are the reductions a let
    calls, and an emitted let calls the table's spelling."""
    runtime = _nospace(_RUNTIME.read_text())
    natives = {
        spelling.path[0]
        for spelling in BUILTIN_SPELLINGS["church"].values()
        if isinstance(spelling, NativeCall)
    }
    assert natives
    missing = sorted(name for name in natives if f"(define({name}" not in runtime)
    assert missing == []
    assert "(define(sumlst)" in runtime
    emitted = _nospace(_church_file(_GALLERY / "beta_regression.qvr"))
    assert "(qvr-exp(-(list-refetam_Resp)))" in emitted
    assert "(define(qvr-expx)" in emitted


# ---------------------------------------------------------------------------
# Nonexistent Church distributions get correct ERP definitions.
# ---------------------------------------------------------------------------


def test_previously_dangling_distributions_are_defined() -> None:
    """Every family name the emitter references resolves to a runtime
    ERP definition rather than a dangling symbol."""
    for name in (
        "cauchy",
        "lognormal",
        "student-t",
        "weibull",
        "pareto",
        "categorical",
        "negative-binomial",
        "geometric",
        "multivariate-gaussian",
        "matrix-normal",
    ):
        assert f"(define({name}" in _RUNTIME_NS or (
            f"(define ({name} " in _RUNTIME_TEXT
        ), name


def test_negative_binomial_uses_torch_probs_convention() -> None:
    """The negative-binomial ERP scores with torch's ``probs^k`` and
    ``(1 - probs)^r`` convention (a Gamma-Poisson mixture draw)."""
    assert "(define (negative-binomial r p)" in _RUNTIME_TEXT
    assert "(draw-poisson (draw-gamma r (/ (- 1.0 p) p)))" in _RUNTIME_TEXT
    assert "(* r (log (- 1.0 p)))" in _RUNTIME_TEXT


def test_gamma_uses_shape_rate_not_shape_scale() -> None:
    """The gamma ERP scores in torch's shape / rate parameterisation:
    the log-density carries ``shape * log(rate)`` and ``-rate * x``."""
    assert "(define (gamma shape rate)" in _RUNTIME_TEXT
    assert "(* shape (log rate))" in _RUNTIME_TEXT
    assert "(- (* rate x))" in _RUNTIME_TEXT


# ---------------------------------------------------------------------------
# MatrixNormal.
# ---------------------------------------------------------------------------


def test_matrix_normal_is_sampled_distribution_object() -> None:
    """`matrix-normal` is wrapped in `sample`, so its draw is a trace
    site rather than an unscored function call."""
    model = _nospace(_model(_church_file(_FAMILIES / "matrixnormal.qvr")))
    assert "(sample(matrix-normalm_locm_row_covariancem_col_covariance))" in model


def test_matrix_normal_uses_row_major_kronecker() -> None:
    """The row-major flatten vec(X) has covariance ``U (x) V``; the ERP
    builds the multivariate-gaussian over `(mat-kron u v)`, not the
    transposed `(mat-kron v u)`."""
    assert "(define (matrix-normal mu u v)" in _RUNTIME_TEXT
    assert "(mat-kron u v)" in _RUNTIME_TEXT
    assert "(mat-kron v u)" not in _RUNTIME_TEXT


# ---------------------------------------------------------------------------
# Bernoulli flip coercion.
# ---------------------------------------------------------------------------


def test_flip_is_numeric_distribution() -> None:
    """`flip` draws and scores numeric 0/1 so an observed integer scores
    against the same coding rather than a boolean comparison."""
    assert "(define (flip p)" in _RUNTIME_TEXT
    assert "(if (< (uniform-random) p) 1 0)" in _RUNTIME_TEXT
    assert "(if (> x 0.5) (log p) (log (- 1.0 p)))" in _RUNTIME_TEXT


def test_bernoulli_observe_scores_numeric_data() -> None:
    """An observed Bernoulli indexes the numeric response and scores it
    against `flip`, whose numeric coding matches the 0/1 data."""
    src = (
        "object Obs : FinSet 4\n"
        "program bern_fixture : Obs -> Obs\n"
        "    observe y : Obs <- Bernoulli(probs=0.5)\n"
        "    return y\n"
        "export bern_fixture\n"
    )
    model = _nospace(_model(_church(src)))
    assert "(observe(flip0.5)(list-refym_Obs))" in model


# ---------------------------------------------------------------------------
# GP realisation as a sample site.
# ---------------------------------------------------------------------------


def test_gp_realisation_is_sampled() -> None:
    """The GP latent is routed through `sample`, making it a genuine
    trace site rather than a bare `(multivariate-gaussian ...)`
    binding."""
    model = _nospace(_model(_church_file(_FAMILIES / "gp.qvr")))
    assert "(definef(sample(multivariate-gaussian__gp_mean_f__gp_cov_f)))" in model


# ---------------------------------------------------------------------------
# Runtime graft + structural integrity.
# ---------------------------------------------------------------------------


def test_runtime_trace_primitives_grafted() -> None:
    """Every emit grafts the reference runtime so `sample`, `observe`,
    `factor`, and the `half` fold resolve through top-level lookup."""
    emitted = _church_file(_FAMILIES / "halfnormal.qvr")
    assert "(define (sample d)" in emitted
    assert "(define (observe d x)" in emitted
    assert "(define (factor s)" in emitted
    assert "(define (half base)" in emitted


def test_broadcasting_arithmetic_is_variadic() -> None:
    """Deterministic expressions may combine vectors and the runtime's
    own primitive draws call ``*`` / ``+`` at three or more arguments,
    so the runtime shadows the four operators with variadic,
    shape-polymorphic broadcasts over captured primitives."""
    for op in ("+", "*", "-", "/"):
        assert f"(define ({op} . args)" in _RUNTIME_TEXT, op
    for prim in ("%add +", "%sub -", "%mul *", "%div /"):
        assert f"(define {prim})" in _RUNTIME_TEXT, prim
    assert "(define (broadcast2 op a b)" in _RUNTIME_TEXT


def test_variadic_operators_execute_at_high_arity(tmp_path: pathlib.Path) -> None:
    """The shadowed operators accept the arities the runtime's own
    primitive draws use: `draw-standard-normal` multiplies three
    arguments and `draw-gamma-unit` cubes via `(* t t t)`, which a
    strictly binary shadow would reject with an arity error."""
    interp = _scheme_interpreter()
    program = _RUNTIME_TEXT + (
        "\n(display (* 2.0 3.0 4.0))(newline)"
        "\n(display (+ 1 2 3 4))(newline)"
        "\n(display (* (list 1.0 2.0) (list 3.0 4.0) 2.0))(newline)"
        "\n(display (+ 10.0 (list 1.0 2.0)))(newline)"
    )
    out = _run_scheme(interp, program, tmp_path)
    lines = out.strip().splitlines()
    assert lines[0] == "24.0", lines
    assert lines[1] == "10", lines
    assert lines[2] == "(6.0 16.0)", lines
    assert lines[3] == "(11.0 12.0)", lines


def test_emitted_church_reparses_to_a_fixed_point() -> None:
    """Every audited emit is stable under re-parse / re-emit (the Tier 2
    lens fixed point) and contains no parse-error nodes."""
    reg = parser_registry()
    fixtures = (
        _FAMILIES / "halfnormal.qvr",
        _FAMILIES / "halfcauchy.qvr",
        _AUDIT / "horseshoe_decomposed.qvr",
        _FAMILIES / "matrixnormal.qvr",
        _FAMILIES / "gp.qvr",
        _GALLERY / "bayesian_regression.qvr",
        _GALLERY / "factor_analysis.qvr",
        _GALLERY / "lda.qvr",
        _GALLERY / "negbin_regression.qvr",
        _GALLERY / "beta_regression.qvr",
    )
    for fixture in fixtures:
        emitted = _church_file(fixture).encode()
        schema = reg.parse_with_protocol("scheme", emitted, str(fixture))
        assert not any(v.kind in ("ERROR", "error") for v in schema.vertices), (
            fixture.name
        )
        reemitted = bytes(reg.emit_pretty("scheme", schema))
        assert reemitted == emitted, fixture.name


# ---------------------------------------------------------------------------
# Executed-joint numeric equivalence.
#
# The structural tests above assert on emitted text; these run the
# emitted module through a reachable Scheme interpreter and check the
# executed joint log-density against the QVR reference (constant-spread
# per Theorem 4.1 of the transpile-correctness note). A finite text
# assertion is not correctness: a program that passes the text tier but
# aborts in the interpreter, or scores the wrong density, fails here.
# ---------------------------------------------------------------------------


def _church_qvr_diffs(
    fixture: pathlib.Path,
    points: list[Point],
    scratch: pathlib.Path,
) -> tuple[list[float], list[float]]:
    """Return the (church, qvr) executed joint log-densities for a
    family / prior fixture with no covariate inputs."""
    source = fixture.read_text()
    emitted = transpile(parse(source), target="church")
    church = ChurchProbe().evaluate(
        emitted, fixture.stem, points, scratch=scratch / "church"
    )
    qvr = QvrProbe().evaluate(
        source.encode("utf-8"), fixture.stem, points, scratch=scratch / "qvr"
    )
    return church.log_densities, qvr.log_densities


def test_executed_halfnormal_matches_qvr(tmp_path: pathlib.Path) -> None:
    """The HalfNormal emit runs in Scheme and its executed joint tracks
    the QVR reference at every clamped positive point, which checks the
    variadic-arithmetic fix restored the Gaussian draw and the
    ``(half (gaussian 0 scale))`` fold scores correctly."""
    points = [Point(params={"theta": v}, data={}) for v in (0.2, 0.7, 1.5, 3.0)]
    church_lps, qvr_lps = _church_qvr_diffs(
        _FAMILIES / "halfnormal.qvr", points, tmp_path
    )
    _equivalence.assert_log_density_match(
        qvr_lps, church_lps, atol=1e-4, context="church@halfnormal"
    )


def test_executed_halfcauchy_matches_qvr(tmp_path: pathlib.Path) -> None:
    """The HalfCauchy emit folds a `cauchy`, not a `gaussian`; its
    executed joint tracks the QVR reference."""
    points = [Point(params={"theta": v}, data={}) for v in (0.2, 0.7, 1.5, 3.0)]
    church_lps, qvr_lps = _church_qvr_diffs(
        _FAMILIES / "halfcauchy.qvr", points, tmp_path
    )
    _equivalence.assert_log_density_match(
        qvr_lps, church_lps, atol=1e-4, context="church@halfcauchy"
    )


def test_executed_horseshoe_matches_qvr(tmp_path: pathlib.Path) -> None:
    """The horseshoe couples a HalfCauchy global scale, per-coordinate
    HalfCauchy local scales, and standard-Normal raw coefficients
    through a broadcast product; its executed joint tracks the QVR
    reference across a grid of clamped latents, exercising the
    element-wise `*` broadcast and the ordered per-coordinate draws."""
    grids = (
        {
            "tau": 0.5,
            "lambda_local": [0.2, 0.3, 0.4, 0.6],
            "z_raw": [1.0, -1.0, 0.5, -0.5],
        },
        {
            "tau": 1.2,
            "lambda_local": [0.8, 0.1, 1.5, 0.3],
            "z_raw": [-0.5, 0.7, -1.2, 0.9],
        },
        {
            "tau": 0.3,
            "lambda_local": [1.1, 0.9, 0.2, 0.7],
            "z_raw": [0.4, 0.4, -0.8, 1.3],
        },
    )
    points = [Point(params=g, data={}) for g in grids]
    church_lps, qvr_lps = _church_qvr_diffs(
        _AUDIT / "horseshoe_decomposed.qvr", points, tmp_path
    )
    _equivalence.assert_log_density_match(
        qvr_lps, church_lps, atol=1e-4, context="church@horseshoe"
    )


def _gallery_points(
    fixture: pathlib.Path,
) -> tuple[_gallery_data.GalleryDataset, list[Point]]:
    """Load a gallery dataset and build a small point grid: the ground-
    truth point plus two latent-scaled perturbations, so the
    constant-spread check has spread to measure."""
    dataset = _gallery_data.load_gallery_data(fixture)
    if dataset is None:
        raise AssertionError(f"no synthetic-data block for {fixture.name}")
    base = _gallery_data.point_from_dataset(dataset)

    def _scaled(scale: float) -> Point:
        params: dict[str, float | int | list[float] | list[int]] = {}
        for name, value in base.params.items():
            if isinstance(value, list):
                params[name] = [v * scale + 0.01 for v in value]
            else:
                params[name] = value * scale + 0.01
        return Point(params=params, data=base.data)

    return dataset, [base, _scaled(0.7), _scaled(1.3)]


def test_executed_beta_regression_matches_qvr(tmp_path: pathlib.Path) -> None:
    """The multi-output beta regression is the hierarchical witness: it
    threads per-output Normal / HalfCauchy priors through a
    ``via``-gathered ``out_idx`` index, a sigmoid link, and a Beta
    likelihood. Its executed joint tracks the QVR reference to a
    constant across the ground-truth point and two latent
    perturbations, checking the deterministic-let broadcast, the gather
    index, and the Beta parameterisation all score correctly under
    execution."""
    fixture = _GALLERY / "beta_regression.qvr"
    dataset, points = _gallery_points(fixture)
    emitted = transpile(parse(fixture.read_text()), target="church")
    church = ChurchProbe().evaluate(
        emitted, fixture.stem, points, scratch=tmp_path / "church"
    )
    qvr = QvrProbe().evaluate(
        fixture.read_text().encode("utf-8"),
        fixture.stem,
        points,
        scratch=tmp_path / "qvr",
        monadic=dataset.monadic,
        x_input=dataset.x_input,
        observations=dataset.observations,
    )
    _equivalence.assert_log_density_match(
        qvr.log_densities,
        church.log_densities,
        atol=1e-3,
        context="church@beta_regression",
    )
