"""Regression tests for audit-confirmed cross-backend shape / plate
defects in the QVR transpiler.

Each test transpiles a gallery model and asserts the emitted program
now carries the axis roles the QVR source declared. Assertions run on
the lowered IR where the defect is a lost axis role, and on the emitted
text where the defect is a missing broadcast or gather.

Covered defects:

* `marginalize-over-axis-dropped-numpyro-pymc`: flattening a
  `MarginalizeStep` into a `SampleStep` before lowering discarded the
  step's `[over=...]` grouping axes and promoted its `: T` support
  index to a batch axis, so the latent plated over the wrong object.
* `stan-continuous-marginalize-index-plate-dropped`: the let-plate
  propagation fixpoint never descended into a marginalize scope, so a
  `let` reading the plated latent stayed scalar and dragged every
  upstream `let` down with it.
* `numpyro-jags-scalar-dirichlet-concentration-not-broadcast`: a free
  scalar data input in a vector-expected argument slot was emitted
  verbatim, which is a rank error at run time.

A `marginalize` over an enumerable support lowers to one branch per
atom whose weighted `logsumexp` is the integrated density, so the LDA
emission assertions read the branch / reduction spelling rather than a
live `sample` site for `z`. The QVR joint for `lda` matches the NumPyro,
Pyro, and PyMC emissions to within 4e-4 across the gallery point set,
which is what fixes these expectations.
"""

from __future__ import annotations

from pathlib import Path

from quivers.dsl.parser import parse
from quivers.transpile import transpile
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile.ir import IRDeterministic, IRMarginalize, IRNode
from quivers.transpile.plan import Lower

_GALLERY = Path(__file__).resolve().parents[2] / "docs" / "examples" / "source"


def _emit(name: str, target: str) -> str:
    """Transpile the named gallery model to `target` and decode."""
    source = (_GALLERY / f"{name}.qvr").read_text()
    return transpile(parse(source), target=target).decode()


def _lower(name: str) -> tuple[IRNode, ...]:
    """Lower the named gallery model and return its IR body."""
    source = (_GALLERY / f"{name}.qvr").read_text()
    return Lower().forward(parse(source)).body


def _nospace(text: str) -> str:
    """Drop spaces so assertions ignore the formatter's comma spacing."""
    return text.replace(" ", "")


def _find_marginalize(body: tuple[IRNode, ...]) -> IRMarginalize:
    """Return the single `IRMarginalize` in `body`."""
    found = [n for n in body if isinstance(n, IRMarginalize)]
    assert len(found) == 1, f"expected one marginalize, got {len(found)}"
    return found[0]


def _find_let(body: tuple[IRNode, ...], name: str) -> IRDeterministic:
    """Return the `IRDeterministic` bound to `name` in `body`."""
    found = [n for n in body if isinstance(n, IRDeterministic) and n.name == name]
    assert len(found) == 1, f"expected one let named {name!r}"
    return found[0]


# ---------------------------------------------------------------------------
# marginalize-over-axis-dropped-numpyro-pymc
# ---------------------------------------------------------------------------


def test_marginalize_survives_composite_expansion_for_every_target() -> None:
    """`expand_composite_lets` must not rewrite a `MarginalizeStep`.

    A `SampleStep` has no slot for the marginalize distinction between
    the `[over=...]` grouping axes and the `: T` support index, so the
    rewrite could only preserve one of the two.
    """
    source = (_GALLERY / "lda.qvr").read_text()
    module = parse(source)
    for target in ("numpyro", "pymc", "pyro", "stan", "turing"):
        expanded = expand_composite_lets(module, target=target)
        marginalize = _find_marginalize(Lower().forward(expanded).body)
        assert marginalize.latent == "z"
        assert [d.name for d in marginalize.plate.batch_dims] == ["Doc"]
        assert marginalize.plate.event_dims == ()


def test_lda_marginalize_batch_axis_is_the_over_axis() -> None:
    """`marginalize z : Topic <- ... [over=Doc]` plates over `Doc`.

    `Topic` names the enumerated support the reduction sums over, not a
    replication axis, so it contributes no plate dim.
    """
    marginalize = _find_marginalize(_lower("lda"))
    batch = marginalize.plate.batch_dims
    assert [d.name for d in batch] == ["Doc"]
    assert [d.size for d in batch] == [20]


def test_lda_numpyro_integrates_the_topic_over_its_three_atoms() -> None:
    """NumPyro scores one branch per `Topic` atom and reduces them.

    `marginalize z : Topic` integrates the topic out, so no `sample("z")`
    site may survive: the emitted program carries a branch per atom, the
    log mixture weights, and a single `factor` holding the logsumexp.
    """
    emitted = _nospace(_emit("lda", "numpyro"))
    for atom in range(3):
        assert (
            f"__marg_z_{atom}=numpyro.distributions."
            f"Categorical(probs=phi[{atom}]).log_prob(w)" in emitted
        ), atom
    assert "__marg_z=jnp.stack([__marg_z_0,__marg_z_1,__marg_z_2],axis=-1)" in emitted
    assert (
        'numpyro.factor("z",jnp.sum(jsp.logsumexp(__marg_z_w+__marg_z,axis=-1)))'
        in emitted
    )
    assert 'numpyro.sample("z"' not in emitted
    # The `Topic`-sized plate belongs to `phi`, never to the integrated `z`.
    assert 'numpyro.plate("Topic",3)' in emitted
    assert 'numpyro.plate("Doc",20)' in emitted


def test_lda_numpyro_gathers_latent_through_the_via_fibration() -> None:
    """`observe w ... [via=word_idx]` gathers the mixture weights.

    `theta` carries the 20-document batch shape while `w` carries the
    200-word one, so the per-topic weights must be gathered to the word
    axis before they can be added to the per-branch likelihood.
    """
    emitted = _nospace(_emit("lda", "numpyro"))
    assert "__marg_z_w=jnp.log(theta[word_idx])" in emitted


def test_lda_pymc_mixes_over_topics_with_document_weights() -> None:
    """PyMC integrates the topic out as a `Mixture` over three atoms.

    The mixture weights are `theta` gathered through the `via=word_idx`
    fibration, so each word is scored under its own document's topic
    mixture.
    """
    emitted = _nospace(_emit("lda", "pymc"))
    for atom in range(3):
        assert f"__marg_z_{atom}=pymc.Categorical.dist(p=phi[{atom}])" in emitted, atom
    assert (
        'pymc.Mixture("w",w=theta[word_idx],'
        "comp_dists=[__marg_z_0,__marg_z_1,__marg_z_2],observed=w)" in emitted
    )
    assert 'pymc.Categorical("z"' not in emitted


def test_lda_pyro_integrates_the_topic_over_its_three_atoms() -> None:
    """Pyro scores one branch per topic and reduces them with a factor."""
    emitted = _nospace(_emit("lda", "pyro"))
    for atom in range(3):
        assert (
            f"__marg_z_{atom}=pyro.distributions."
            f"Categorical(phi[{atom}]).log_prob(w)" in emitted
        ), atom
    assert "__marg_z_w=torch.log(theta[word_idx])" in emitted
    assert (
        'pyro.factor("z",torch.sum(torch.logsumexp(__marg_z_w+__marg_z,dim=-1)))'
        in emitted
    )
    assert 'pyro.sample("z"' not in emitted
    assert 'pyro.plate("Doc",20)' in emitted


def test_lda_stan_enumerates_topics_per_word() -> None:
    """Stan keeps a three-topic accumulator per word and logsumexps it.

    The closed-form marginal `p(w_n) = sum_k theta[doc(n), k] *
    phi[k, w_n]` factorises per word, so the accumulator is sized by the
    200-position observation axis `object Token : FinSet 200`, whose
    loop variable is `n_Token`; `theta` reaches it through the
    `via=word_idx` gather that carries each token position to its
    document.
    """
    emitted = _nospace(_emit("lda", "stan"))
    assert "array[200]vector[3]lps_z=rep_array(rep_vector(0,3),200);" in emitted
    assert (
        "lps_z[n_Token,k]=categorical_lpmf(k|theta[word_idx[n_Token]+1]);"
    ) in emitted
    assert "lps_z[n_Token,k]+=categorical_lpmf(w[n_Token]+1|phi[k]);" in emitted
    assert "target+=log_sum_exp(lps_z[n_Token]);" in emitted


# ---------------------------------------------------------------------------
# stan-continuous-marginalize-index-plate-dropped
# ---------------------------------------------------------------------------


def test_zip_marginalize_latent_is_per_response() -> None:
    """A non-enumerable marginalize latent replicates over its index.

    `ContinuousBernoulli` has no finite support to sum over, so
    `marginalize z : Resp` means one latent value per response.
    """
    marginalize = _find_marginalize(_lower("zip_regression"))
    batch = marginalize.plate.batch_dims
    assert [d.name for d in batch] == ["Resp"]
    assert [d.size for d in batch] == [400]


def test_zip_scope_let_inherits_the_latent_plate() -> None:
    """`let gated_rate = z * rate` inside the scope is per-response."""
    marginalize = _find_marginalize(_lower("zip_regression"))
    gated = _find_let(marginalize.scope, "gated_rate")
    assert [d.name for d in gated.plate.batch_dims] == ["Resp"]


def test_zip_scope_let_propagates_the_plate_upstream() -> None:
    """The scope let drags every `let` it reads up to its own plate.

    `gated_rate` reads `rate`, which reads `ar` and `br`; all three sit
    outside the marginalize scope and all three are per-response.
    """
    body = _lower("zip_regression")
    for name in ("ar", "br", "rate"):
        node = _find_let(body, name)
        assert [d.name for d in node.plate.batch_dims] == ["Resp"], name


def test_zip_stan_declares_the_rate_as_a_response_array() -> None:
    """Stan declares every ZIP transformed parameter over 400 responses.

    `gated_rate` is not among them. It is the scope-local `let` of a
    `marginalize` block, so it exists once per atom rather than once
    per response, and the enumeration inlines it at each atom instead
    of declaring an array the block would have to index.
    """
    emitted = _nospace(_emit("zip_regression", "stan"))
    for name in ("ar", "br", "rate"):
        assert f"array[400]real{name};" in emitted, name
    assert "array[400]realgated_rate;" not in emitted


def test_zip_stan_indexes_the_rate_inside_the_response_loop() -> None:
    """The Poisson likelihood reads the per-response rate at each atom.

    The block enumerates the two atoms of the Bernoulli its
    `ContinuousBernoulli` relaxes, and the scope is unrolled so each
    atom is a literal. That is what lets the off atom's rate fold to a
    constant zero: written as `(k - 1) * rate[n]` the rate is one the
    parameters still reach, so its derivative where the count is
    positive is `0 * inf` and Stan rejects every initial value it
    tries, which no amount of re-initialising fixes.

    The likelihood is scored with an explicit `poisson_lpmf`
    increment; the `~` spelling drops the `- lgamma(y + 1)` term,
    which is data-dependent and part of the QVR measure.
    """
    emitted = _nospace(_emit("zip_regression", "stan"))
    assert "lps_z[n_Resp,1]+=poisson_lpmf(y[n_Resp]|0);" in emitted
    assert "lps_z[n_Resp,2]+=poisson_lpmf(y[n_Resp]|rate[n_Resp]);" in emitted
    assert "(k-1)*rate[n_Resp]" not in emitted
    assert "y[m_Resp]~poisson(" not in emitted
    assert "y[n_Resp]~poisson(" not in emitted


# ---------------------------------------------------------------------------
# numpyro-jags-scalar-dirichlet-concentration-not-broadcast
# ---------------------------------------------------------------------------


_SCALAR_DIRICHLET_SRC = """
object K : FinSet 4

program dirichlet_scalar_conc : K -> K
    sample p <- Dirichlet(concentration=a) [over=K]
    return p

export dirichlet_scalar_conc
"""


def test_numpyro_broadcasts_a_free_scalar_dirichlet_concentration() -> None:
    """A free scalar input in a vector slot is broadcast to the event axis.

    `Dirichlet(concentration=a)` with a scalar `a` is a rank error; the
    QVR `[over=K]` clause says the concentration is constant across the
    four-atom event axis.
    """
    emitted = _nospace(
        transpile(parse(_SCALAR_DIRICHLET_SRC), target="numpyro").decode()
    )
    assert "Dirichlet(concentration=jnp.full((4,),a))" in emitted


def test_numpyro_broadcasts_a_scalar_program_parameter() -> None:
    """The same broadcast fires for a declared `Real` program parameter.

    Each concentration is broadcast to the event axis its `[over=...]`
    clause names, so the two lengths are read off the two objects:
    `theta ~ Dirichlet(alpha) [over=Topic]` with `Topic : FinSet 3`
    gives `(3,)`, and `phi ~ Dirichlet(beta) [over=Vocab]` with
    `Vocab : FinSet 50` gives `(50,)`. `phi` is a distribution over the
    vocabulary, not over the 200 token positions those types are drawn
    at.
    """
    emitted = _nospace(_emit("lda", "numpyro"))
    assert "Dirichlet(concentration=jnp.full((3,),alpha))" in emitted
    assert "Dirichlet(concentration=jnp.full((50,),beta))" in emitted


def test_numpyro_leaves_an_already_vector_argument_alone() -> None:
    """A let-bound / sampled vector argument is emitted verbatim.

    Each `Categorical(probs=phi[k])` branch of the integrated topic
    reads a row of the simplex-valued `phi` sample, so no broadcast
    wrapper may be injected around it. The only two `jnp.full` calls in
    the program are the scalar Dirichlet concentrations.
    """
    emitted = _nospace(_emit("lda", "numpyro"))
    for atom in range(3):
        assert f"Categorical(probs=phi[{atom}])" in emitted, atom
    assert emitted.count("jnp.full(") == 2, emitted
