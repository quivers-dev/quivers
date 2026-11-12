"""How each dynamic target spells a QIEC distribution on host expressions.

A ``DistributionValue`` names a family of the semantic registry and
supplies its parameters by name. A host runtime needs that construction
as an expression of its own language, built from the host expressions
its arguments render to. This module owns those spellings: one function
per target that takes rendered argument expressions and returns the
target's constructor expression, applying the target's parameterization
conventions (rate against scale, complemented probabilities, reordered
or prepended arguments, leading matrix dimensions, folded half-line
families) and naming the runtime helpers the target must graft for
families it does not ship.

The conventions here are the ones the sample-step renderers apply to a
program's ``sample`` and ``observe`` steps; a family constructed as a
first-class value must score exactly as the same family sampled at a
site, so the two spellings agree parameter for parameter.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Mapping, Sequence

import didactic.api as dx

from quivers.qiec.families import FAMILIES, DistributionFamily
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.family_meta import FAMILY_META

_HERE = pathlib.Path(__file__).resolve().parent

DYNAMIC_TARGET_LANGUAGES: dict[str, str] = {
    "pyro": "python",
    "numpyro": "python",
    "pymc": "python",
    "edward2": "python",
    "turing": "julia",
    "gen": "julia",
    "webppl": "javascript",
    "church": "scheme",
}
"""The dynamic targets, each mapped to the host language it renders to."""

#: Runtime helper roots a target must graft for a family it does not
#: ship, keyed by target and family. Python targets graft one class per
#: root; the Julia, JavaScript, and Scheme targets graft their whole
#: helper file whenever any family of theirs is used, so the root names
#: there are the families themselves.
_HELPER_ROOTS: dict[str, dict[str, tuple[str, ...]]] = {
    "pyro": {
        "TruncatedNormal": ("TruncatedNormal",),
        "LogitNormal": ("LogitNormal",),
        "HalfStudentT": ("HalfStudentT",),
        "MatrixNormal": ("MatrixNormal",),
        "LogGamma": ("LogGamma",),
        "LogitBeta": ("LogitBeta",),
        "InverseWishart": ("InverseWishart",),
    },
    "numpyro": {
        "LogitNormal": ("LogitNormal",),
        "HalfStudentT": ("HalfStudentT",),
        "ContinuousBernoulli": ("ContinuousBernoulli",),
        "FisherSnedecor": ("FisherSnedecor",),
        "LogisticNormal": ("LogisticNormal",),
        "OneHotCategorical": ("OneHotCategorical",),
        "OrderedProbit": ("OrderedProbit",),
        "LogGamma": ("LogGamma",),
        "LogitBeta": ("LogitBeta",),
    },
    "pymc": {
        "ContinuousBernoulli": ("ContinuousBernoulli",),
        "LKJCholesky": ("LKJCholesky",),
    },
    "edward2": {},
    "turing": {
        "ContinuousBernoulli": ("ContinuousBernoulli",),
        "HalfStudentT": ("HalfStudentT",),
    },
    "gen": {
        name: (name,)
        for name in (
            "TruncatedNormal",
            "HalfNormal",
            "HalfCauchy",
            "Logistic",
            "BetaBinomial",
            "HalfStudentT",
            "Kumaraswamy",
            "ContinuousBernoulli",
            "LKJCholesky",
            "MatrixNormal",
            "LogNormal",
            "Weibull",
        )
    },
    "webppl": {
        name: (name,)
        for name in (
            "Logistic",
            "BetaBinomial",
            "HalfStudentT",
            "Kumaraswamy",
            "LKJCholesky",
            "ContinuousBernoulli",
            "MatrixNormal",
            "LogNormal",
            "StudentT",
            "Weibull",
            "NegativeBinomial",
            "Categorical",
        )
    },
    "church": {},
}

#: Families no dynamic target can construct as a first-class value:
#: ``GP`` is a structured lowering over a kernel of the program's domain
#: grid, ``Transformed`` takes a transform no QIEC value has, ``Horseshoe``
#: is a marginal over a local scale that no target's library carries, and
#: the remaining families have no spelling on the named target.
_UNSPELLABLE: dict[str, frozenset[str]] = {
    "pyro": frozenset({"GP", "Horseshoe", "Transformed"}),
    "numpyro": frozenset({"GP", "Horseshoe", "Transformed"}),
    "pymc": frozenset({"GP", "Horseshoe", "Transformed"}),
    "edward2": frozenset({"GP", "Horseshoe", "Transformed"}),
    "turing": frozenset({"GP", "Horseshoe", "Mixture", "Truncated"}),
    "gen": frozenset({"GP", "Horseshoe", "Mixture", "Truncated"}),
    "webppl": frozenset({"GP", "Horseshoe", "Mixture"}),
    "church": frozenset({"GP", "Horseshoe"}),
}


class DistributionSpelling(dx.Model):
    """One target's expression for a distribution construction.

    Parameters
    ----------
    expression
        The host expression constructing the distribution.
    helpers
        Runtime helper roots the target must graft for the expression
        to resolve.
    """

    expression: str
    helpers: tuple[str, ...] = ()


def bridge_source(target: str) -> str:
    """The target's QIEC distribution bridge.

    The bridge defines, in the target's host language, the array
    conversion tensor-valued arguments pass through, the log-density
    evaluation a ``LogDensity`` term renders to, and any wrapper a
    support convention needs.

    Parameters
    ----------
    target : str
        A dynamic target.

    Returns
    -------
    str
        The bridge's source text.

    Raises
    ------
    KeyError
        If ``target`` is not a dynamic target.
    """
    language = DYNAMIC_TARGET_LANGUAGES[target]
    extension = {"python": "py", "julia": "jl", "javascript": "js", "scheme": "scm"}
    return (_HERE / f"runtime_qiec_{target}.{extension[language]}").read_text()


def can_spell(target: str, family: str) -> bool:
    """Whether a target spells a family as a first-class distribution.

    Parameters
    ----------
    target : str
        A dynamic target.
    family : str
        A registry family name.

    Returns
    -------
    bool
        ``True`` when :func:`spell_distribution` has a spelling for the
        pair.
    """
    return (
        family in FAMILIES
        and family not in _UNSPELLABLE[target]
        and target in FAMILY_META[family].target_names
    )


#: Alternative parameterizations no speller here reads, each with the
#: parameter the spellers read in its place: a first-class construction
#: naming the alternative alone is refused rather than spelled
#: positionally as that parameter.
_UNSPELLED_ALTERNATIVES: dict[tuple[str, str], str] = {
    **{
        (family, "logits"): "probs"
        for family in ("Bernoulli", "Binomial", "Geometric", "NegativeBinomial")
    },
    ("MultivariateNormal", "precision_matrix"): "covariance_matrix",
    ("Wishart", "precision_matrix"): "scale_tril",
}


def spell_distribution(
    target: str,
    family: str,
    arguments: Mapping[str, str],
    event_shape: tuple[int, ...] | None,
) -> DistributionSpelling:
    """Spell a distribution construction for a dynamic target.

    Parameters
    ----------
    target : str
        The dynamic target.
    family : str
        The family's registry name.
    arguments : Mapping[str, str]
        The supplied parameters, each a host expression, keyed by the
        registry's parameter names.
    event_shape : tuple[int, ...] or None
        The literal event shape of the constructed ``Sampleable``, when
        its type spells one; families whose target takes the matrix
        dimension as an argument read it from here.

    Returns
    -------
    DistributionSpelling
        The expression and the helpers it needs.

    Raises
    ------
    UnsupportedConstruct
        If the target has no spelling for the family, or a parameter it
        needs is absent.
    """
    if not can_spell(target, family):
        raise UnsupportedConstruct(
            f"qvr-{target}", [f"qiec:distribution:{family}:no-{target}-spelling"]
        )
    alternative = next(
        (
            name
            for name in arguments
            if (family, name) in _UNSPELLED_ALTERNATIVES
            and _UNSPELLED_ALTERNATIVES[(family, name)] not in arguments
        ),
        None,
    )
    if alternative is not None:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [f"qiec:distribution:{family}:{alternative}:no-{target}-spelling"],
        )
    record = FAMILIES[family]
    speller = _SPELLERS[target]
    expression = speller(record, dict(arguments), event_shape)
    return DistributionSpelling(
        expression=expression, helpers=_HELPER_ROOTS[target].get(family, ())
    )


def helper_roots(target: str, families: frozenset[str]) -> frozenset[str]:
    """The runtime helper roots a target grafts for the given families.

    Parameters
    ----------
    target : str
        A dynamic target.
    families : frozenset[str]
        Family names constructed by the program.

    Returns
    -------
    frozenset[str]
        The roots (helper class or function names) those families need.
    """
    roots: set[str] = set()
    for family in families:
        roots.update(_HELPER_ROOTS[target].get(family, ()))
    return frozenset(roots)


def helper_families(target: str) -> frozenset[str]:
    """The families a target serves through grafted runtime helpers.

    Parameters
    ----------
    target : str
        A dynamic target.

    Returns
    -------
    frozenset[str]
        The family names.
    """
    return frozenset(_HELPER_ROOTS[target])


def _require(
    record: DistributionFamily,
    arguments: Mapping[str, str],
    target: str,
    *names: str,
) -> tuple[str, ...]:
    """Read parameters the spelling cannot do without.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    arguments : Mapping[str, str]
        The supplied parameters.
    target : str
        The target, for the diagnostic.
    *names : str
        The parameters needed.

    Returns
    -------
    tuple[str, ...]
        Their expressions, in the order asked.

    Raises
    ------
    UnsupportedConstruct
        If any is absent.
    """
    missing = [name for name in names if name not in arguments]
    if missing:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [f"qiec:distribution:{record.name}:missing:{','.join(missing)}"],
        )
    return tuple(arguments[name] for name in names)


def _dimension(
    record: DistributionFamily, event_shape: tuple[int, ...] | None, target: str
) -> int:
    """The matrix dimension of a correlation family's event.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    event_shape : tuple[int, ...] or None
        The event shape its type spells.
    target : str
        The target, for the diagnostic.

    Returns
    -------
    int
        The side of the square event.

    Raises
    ------
    UnsupportedConstruct
        If the type spells no square event shape.
    """
    if event_shape is None or len(event_shape) != 2 or event_shape[0] != event_shape[1]:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [f"qiec:distribution:{record.name}:no-square-event-shape"],
        )
    return event_shape[0]


def _arrays(
    record: DistributionFamily,
    arguments: Mapping[str, str],
    convert: Callable[[str], str],
) -> dict[str, str]:
    """Pass tensor-valued parameters through the target's array conversion.

    Parameters
    ----------
    record : DistributionFamily
        The family, whose parameter ranks say which are tensors.
    arguments : Mapping[str, str]
        The supplied parameters.
    convert : Callable[[str], str]
        Wraps one expression in the conversion.

    Returns
    -------
    dict[str, str]
        The parameters, tensors converted.
    """
    return {
        name: convert(value) if record.parameter(name).rank > 0 else value
        for name, value in arguments.items()
    }


def _keywords(
    callee: str,
    arguments: Mapping[str, str],
    aliases: Mapping[str, str] = {},
    leading: Sequence[str] = (),
) -> str:
    """A Python keyword call.

    Parameters
    ----------
    callee : str
        The constructor.
    arguments : Mapping[str, str]
        The parameters, in registry order.
    aliases : Mapping[str, str]
        Renames from registry names to the target's keywords.
    leading : Sequence[str]
        Positional expressions emitted before the keywords.

    Returns
    -------
    str
        ``callee(leading..., key=value, ...)``.
    """
    keywords = [
        f"{aliases.get(name, name)}={value}" for name, value in arguments.items()
    ]
    return f"{callee}({', '.join((*leading, *keywords))})"


def _positional(callee: str, values: Sequence[str], separator: str = ", ") -> str:
    """A positional call in a language with parenthesized arguments.

    Parameters
    ----------
    callee : str
        The constructor.
    values : Sequence[str]
        The arguments.
    separator : str
        What separates arguments.

    Returns
    -------
    str
        ``callee(values...)``.
    """
    return f"{callee}({separator.join(values)})"


def _ordered(record: DistributionFamily, arguments: Mapping[str, str]) -> list[str]:
    """The supplied parameters in registry order.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    arguments : Mapping[str, str]
        The supplied parameters.

    Returns
    -------
    list[str]
        Their expressions.
    """
    return [arguments[name] for name in record.parameter_names if name in arguments]


# ---------------------------------------------------------------------------
# Pyro
# ---------------------------------------------------------------------------


def _pyro_array(value: str) -> str:
    """Wrap a Python expression in the bridge's array conversion.

    Parameters
    ----------
    value : str
        A host expression evaluating to a QIEC tensor.

    Returns
    -------
    str
        The expression converting it to a torch tensor.
    """
    return f"_qvr_qiec_array({value})"


def _spell_pyro(
    record: DistributionFamily,
    arguments: dict[str, str],
    event_shape: tuple[int, ...] | None,
) -> str:
    """Pyro spells families as ``pyro.distributions`` classes with keywords.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    arguments : dict[str, str]
        The supplied parameters.
    event_shape : tuple[int, ...] or None
        The event shape of the constructed type.

    Returns
    -------
    str
        The constructor expression.
    """
    family = record.name
    meta = FAMILY_META[family]
    name = meta.target_names["pyro"]
    arguments = _arrays(record, arguments, _pyro_array)
    if family in _HELPER_ROOTS["pyro"]:
        callee = name
    else:
        callee = f"pyro.distributions.{name}"
    if family in ("LKJCholesky", "LKJCorrelationFactor"):
        dimension = _dimension(record, event_shape, "pyro")
        return _keywords(callee, arguments, leading=(str(dimension),))
    if family == "MixtureNormal":
        weights, loc, scale = _require(
            record, arguments, "pyro", "weights", "loc", "scale"
        )
        return (
            "pyro.distributions.MixtureSameFamily("
            f"pyro.distributions.Categorical(probs={weights}), "
            f"pyro.distributions.Normal(loc={_pyro_array(loc)}, scale={_pyro_array(scale)}))"
        )
    if family == "Mixture":
        weights, component = _require(record, arguments, "pyro", "weights", "component")
        return (
            "pyro.distributions.MixtureSameFamily("
            f"pyro.distributions.Categorical(probs={weights}), {component})"
        )
    if family == "Truncated":
        base, low, high = _require(record, arguments, "pyro", "base", "low", "high")
        return f"{callee}({base}, low={low}, high={high})"
    if family == "Independent":
        base, count = _require(
            record, arguments, "pyro", "base", "reinterpreted_batch_ndims"
        )
        return f"{callee}({base}, {count})"
    return _keywords(callee, arguments, meta.arg_aliases.get("pyro", {}))


# ---------------------------------------------------------------------------
# NumPyro
# ---------------------------------------------------------------------------


def _spell_numpyro(
    record: DistributionFamily,
    arguments: dict[str, str],
    event_shape: tuple[int, ...] | None,
) -> str:
    """NumPyro spells families as ``numpyro.distributions`` classes.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    arguments : dict[str, str]
        The supplied parameters.
    event_shape : tuple[int, ...] or None
        The event shape of the constructed type.

    Returns
    -------
    str
        The constructor expression.
    """
    family = record.name
    meta = FAMILY_META[family]
    name = meta.target_names["numpyro"]
    arguments = _arrays(record, arguments, _pyro_array)
    if family in _HELPER_ROOTS["numpyro"]:
        return _keywords(name, arguments)
    callee = f"numpyro.distributions.{name}"
    aliases = meta.arg_aliases.get("numpyro", {})
    if family == "NegativeBinomial":
        total, probs = _require(record, arguments, "numpyro", "total_count", "probs")
        return (
            f"{callee}(mean=({total} * {probs}) / (1 - {probs}), concentration={total})"
        )
    if family == "MatrixNormal":
        loc, rows, columns = _require(
            record, arguments, "numpyro", "loc", "row_covariance", "col_covariance"
        )
        return (
            f"{callee}(loc={loc}, scale_tril_row=jnp.linalg.cholesky({rows}), "
            f"scale_tril_column=jnp.linalg.cholesky({columns}))"
        )
    if family in ("LKJCholesky", "LKJCorrelationFactor"):
        dimension = _dimension(record, event_shape, "numpyro")
        return _keywords(callee, arguments, aliases, leading=(str(dimension),))
    if family == "MixtureNormal":
        weights, loc, scale = _require(
            record, arguments, "numpyro", "weights", "loc", "scale"
        )
        return (
            "numpyro.distributions.MixtureSameFamily("
            f"numpyro.distributions.Categorical(probs={weights}), "
            f"numpyro.distributions.Normal(loc={_pyro_array(loc)}, "
            f"scale={_pyro_array(scale)}))"
        )
    if family == "Mixture":
        weights, component = _require(
            record, arguments, "numpyro", "weights", "component"
        )
        return (
            "numpyro.distributions.MixtureSameFamily("
            f"numpyro.distributions.Categorical(probs={weights}), {component})"
        )
    if family == "Truncated":
        base, low, high = _require(record, arguments, "numpyro", "base", "low", "high")
        return f"{callee}({base}, low={low}, high={high})"
    if family == "Independent":
        base, count = _require(
            record, arguments, "numpyro", "base", "reinterpreted_batch_ndims"
        )
        return f"{callee}({base}, {count})"
    return _keywords(callee, arguments, aliases)


# ---------------------------------------------------------------------------
# PyMC
# ---------------------------------------------------------------------------


def _spell_pymc(
    record: DistributionFamily,
    arguments: dict[str, str],
    event_shape: tuple[int, ...] | None,
) -> str:
    """PyMC spells families as unregistered ``pymc.<Family>.dist`` calls.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    arguments : dict[str, str]
        The supplied parameters.
    event_shape : tuple[int, ...] or None
        The event shape of the constructed type.

    Returns
    -------
    str
        The constructor expression.
    """
    family = record.name
    meta = FAMILY_META[family]
    name = meta.target_names["pymc"]
    arguments = _arrays(record, arguments, _pyro_array)
    aliases = meta.arg_aliases.get("pymc", {})
    callee = f"pymc.{name}.dist"
    if family == "NegativeBinomial":
        if "probs" in arguments:
            arguments["probs"] = f"(1.0 - {arguments['probs']})"
        return _keywords(callee, arguments, aliases)
    if family == "Geometric":
        return f"_qvr_qiec_shifted({_keywords(callee, arguments, aliases)}, 1)"
    if family == "ContinuousBernoulli":
        (probs,) = _require(record, arguments, "pymc", "probs")
        return (
            f"pymc.CustomDist.dist({probs}, logp=_continuous_bernoulli_logp, "
            'random=_continuous_bernoulli_random, dtype="float64")'
        )
    if family == "LKJCholesky":
        dimension = _dimension(record, event_shape, "pymc")
        (eta,) = _require(record, arguments, "pymc", "concentration")
        return (
            f"pymc.CustomDist.dist({eta}, np.zeros({dimension}), "
            f"logp=lambda value, eta, marker: _lkj_cholesky_logp(value, {dimension}, eta), "
            f"random=lambda eta, marker, rng=None, size=None: "
            f"_lkj_cholesky_draw({dimension}, eta, rng, size), "
            'signature="(),(n)->(n,n)")'
        )
    if family == "MixtureNormal":
        weights, loc, scale = _require(
            record, arguments, "pymc", "weights", "loc", "scale"
        )
        return f"pymc.NormalMixture.dist(w={weights}, mu={loc}, sigma={scale})"
    if family == "Mixture":
        weights, component = _require(record, arguments, "pymc", "weights", "component")
        return f"pymc.Mixture.dist(w={weights}, comp_dists={component})"
    if family == "Truncated":
        base, low, high = _require(record, arguments, "pymc", "base", "low", "high")
        return f"pymc.Truncated.dist({base}, lower={low}, upper={high})"
    return _keywords(callee, arguments, aliases)


# ---------------------------------------------------------------------------
# Edward2
# ---------------------------------------------------------------------------


def _edward2_array(value: str) -> str:
    """Wrap a Python expression in the bridge's array conversion.

    Parameters
    ----------
    value : str
        A host expression evaluating to a QIEC tensor.

    Returns
    -------
    str
        The expression converting it to a TensorFlow tensor.
    """
    return f"_qvr_qiec_array({value})"


def _spell_edward2(
    record: DistributionFamily,
    arguments: dict[str, str],
    event_shape: tuple[int, ...] | None,
) -> str:
    """Edward2 spells families as ``tfp.distributions`` classes.

    A first-class distribution is a plain TensorFlow Probability
    distribution rather than an Edward2 random variable, so constructing
    it registers no site with a tracing interceptor.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    arguments : dict[str, str]
        The supplied parameters.
    event_shape : tuple[int, ...] or None
        The event shape of the constructed type.

    Returns
    -------
    str
        The constructor expression.
    """
    family = record.name
    meta = FAMILY_META[family]
    name = meta.target_names["edward2"]
    arguments = _arrays(record, arguments, _edward2_array)
    callee = f"tfp.distributions.{name}"
    aliases = {**meta.arg_aliases.get("edward2", {})}
    if family == "Pareto":
        aliases["alpha"] = "concentration"
    if family in ("Binomial", "BetaBinomial") and "total_count" in arguments:
        arguments["total_count"] = f"tf.cast({arguments['total_count']}, tf.float32)"
    if family == "HalfCauchy":
        (scale,) = _require(record, arguments, "edward2", "scale")
        return f"{callee}(loc=0.0, scale={scale})"
    if family == "HalfStudentT":
        df, scale = _require(record, arguments, "edward2", "df", "scale")
        return f"{callee}(df={df}, loc=0.0, scale={scale})"
    if family == "LKJCholesky":
        dimension = _dimension(record, event_shape, "edward2")
        (concentration,) = _require(record, arguments, "edward2", "concentration")
        return (
            f"{callee}(dimension={dimension}, concentration={concentration}, "
            "input_output_cholesky=True)"
        )
    if family == "MatrixNormal":
        loc, rows, columns = _require(
            record, arguments, "edward2", "loc", "row_covariance", "col_covariance"
        )
        return (
            f"{callee}(loc={loc}, "
            f"scale_row=tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky({rows})), "
            f"scale_column=tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky({columns})))"
        )
    if family == "MixtureNormal":
        weights, loc, scale = _require(
            record, arguments, "edward2", "weights", "loc", "scale"
        )
        return (
            "tfp.distributions.MixtureSameFamily("
            f"tfp.distributions.Categorical(probs={weights}), "
            f"tfp.distributions.Normal(loc={_edward2_array(loc)}, "
            f"scale={_edward2_array(scale)}))"
        )
    if family == "Mixture":
        weights, component = _require(
            record, arguments, "edward2", "weights", "component"
        )
        return (
            "tfp.distributions.MixtureSameFamily("
            f"tfp.distributions.Categorical(probs={weights}), {component})"
        )
    if family == "Independent":
        base, count = _require(
            record, arguments, "edward2", "base", "reinterpreted_batch_ndims"
        )
        return f"{callee}({base}, reinterpreted_batch_ndims={count})"
    return _keywords(callee, arguments, aliases)


# ---------------------------------------------------------------------------
# Turing
# ---------------------------------------------------------------------------


def _julia_array(value: str) -> str:
    """Wrap a Julia expression in the bridge's array conversion.

    Parameters
    ----------
    value : str
        A host expression evaluating to a QIEC tensor.

    Returns
    -------
    str
        The expression converting it to a Julia array.
    """
    return f"_qvr_qiec_array({value})"


def _spell_turing(
    record: DistributionFamily,
    arguments: dict[str, str],
    event_shape: tuple[int, ...] | None,
) -> str:
    """Turing spells families as ``Distributions.jl`` constructors.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    arguments : dict[str, str]
        The supplied parameters.
    event_shape : tuple[int, ...] or None
        The event shape of the constructed type.

    Returns
    -------
    str
        The constructor expression.
    """
    family = record.name
    name = FAMILY_META[family].target_names["turing"]
    arguments = _arrays(record, arguments, _julia_array)
    if family == "HalfNormal":
        (scale,) = _require(record, arguments, "turing", "scale")
        return f"truncated(Normal(0, {scale}), 0, Inf)"
    if family == "HalfCauchy":
        (scale,) = _require(record, arguments, "turing", "scale")
        return f"truncated(Cauchy(0, {scale}), 0, Inf)"
    if family == "HalfStudentT":
        df, scale = _require(record, arguments, "turing", "df", "scale")
        return f"HalfStudentT({df}, {scale})"
    if family == "TruncatedNormal":
        loc, scale, low, high = _require(
            record, arguments, "turing", "loc", "scale", "low", "high"
        )
        return f"truncated(Normal({loc}, {scale}), {low}, {high})"
    if family == "StudentT":
        df, loc, scale = _require(record, arguments, "turing", "df", "loc", "scale")
        return f"({loc} + {scale} * TDist({df}))"
    if family == "Exponential":
        (rate,) = _require(record, arguments, "turing", "rate")
        return f"Exponential(inv({rate}))"
    if family == "Gamma":
        concentration, rate = _require(
            record, arguments, "turing", "concentration", "rate"
        )
        return f"Gamma({concentration}, inv({rate}))"
    if family == "NegativeBinomial":
        total, probs = _require(record, arguments, "turing", "total_count", "probs")
        return f"NegativeBinomial({total}, 1 - {probs})"
    if family == "Weibull":
        scale, concentration = _require(
            record, arguments, "turing", "scale", "concentration"
        )
        return f"Weibull({concentration}, {scale})"
    if family == "Categorical":
        (probs,) = _require(record, arguments, "turing", "probs")
        return f"_qvr_qiec_shifted(Categorical({probs}), 1)"
    if family == "LKJCholesky":
        dimension = _dimension(record, event_shape, "turing")
        (concentration,) = _require(record, arguments, "turing", "concentration")
        return f"LKJCholesky({dimension}, {concentration})"
    if family == "InverseWishart":
        df, scale_tril = _require(record, arguments, "turing", "df", "scale_tril")
        return f"InverseWishart({df}, {scale_tril} * transpose({scale_tril}))"
    if family == "MixtureNormal":
        weights, loc, scale = _require(
            record, arguments, "turing", "weights", "loc", "scale"
        )
        return f"MixtureModel(Normal.({_julia_array(loc)}, {_julia_array(scale)}), {weights})"
    return _positional(name, _ordered(record, arguments))


# ---------------------------------------------------------------------------
# Gen
# ---------------------------------------------------------------------------


def _gen(distribution: str, values: Sequence[str]) -> str:
    """A Gen distribution paired with its arguments.

    Parameters
    ----------
    distribution : str
        The Gen distribution object.
    values : Sequence[str]
        The argument expressions.

    Returns
    -------
    str
        ``_qvr_qiec_distribution(distribution, values...)``.
    """
    return _positional("_qvr_qiec_distribution", (distribution, *values))


def _spell_gen(
    record: DistributionFamily,
    arguments: dict[str, str],
    event_shape: tuple[int, ...] | None,
) -> str:
    """Gen spells a distribution as its distribution object with arguments.

    A Gen distribution takes its parameters at each draw rather than at
    construction, so a first-class value pairs the object with the
    argument list the bridge unpacks when scoring.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    arguments : dict[str, str]
        The supplied parameters.
    event_shape : tuple[int, ...] or None
        The event shape of the constructed type.

    Returns
    -------
    str
        The constructor expression.
    """
    family = record.name
    name = FAMILY_META[family].target_names["gen"]
    arguments = _arrays(record, arguments, _julia_array)
    if family == "HalfNormal":
        (scale,) = _require(record, arguments, "gen", "scale")
        return _gen("truncated_normal", ("0", scale, "0", "Inf"))
    if family == "HalfCauchy":
        (scale,) = _require(record, arguments, "gen", "scale")
        return _gen("half_cauchy", (scale,))
    if family == "Gamma":
        concentration, rate = _require(
            record, arguments, "gen", "concentration", "rate"
        )
        return _gen("gamma", (concentration, f"inv({rate})"))
    if family == "NegativeBinomial":
        total, probs = _require(record, arguments, "gen", "total_count", "probs")
        return _gen("neg_binom", (total, f"1 - {probs}"))
    if family == "Categorical":
        (probs,) = _require(record, arguments, "gen", "probs")
        return f"_qvr_qiec_shifted({_gen('categorical', (probs,))}, 1)"
    if family == "LKJCholesky":
        dimension = _dimension(record, event_shape, "gen")
        (concentration,) = _require(record, arguments, "gen", "concentration")
        return _gen("lkj_cholesky", (str(dimension), concentration))
    if family == "MixtureNormal":
        weights, loc, scale = _require(
            record, arguments, "gen", "weights", "loc", "scale"
        )
        return _gen(
            "HomogeneousMixture(normal, [0, 0])",
            (weights, _julia_array(loc), _julia_array(scale)),
        )
    return _gen(name, _ordered(record, arguments))


# ---------------------------------------------------------------------------
# WebPPL
# ---------------------------------------------------------------------------


def _javascript_object(
    arguments: Mapping[str, str], aliases: Mapping[str, str] = {}
) -> str:
    """A JavaScript object literal of named parameters.

    Parameters
    ----------
    arguments : Mapping[str, str]
        The parameters.
    aliases : Mapping[str, str]
        Renames from registry names to the target's keys.

    Returns
    -------
    str
        ``{key: value, ...}``.
    """
    return (
        "{" + ", ".join(f"{aliases.get(k, k)}: {v}" for k, v in arguments.items()) + "}"
    )


def _webppl_array(value: str) -> str:
    """Wrap a JavaScript expression in the bridge's array conversion.

    Parameters
    ----------
    value : str
        A host expression evaluating to a QIEC tensor.

    Returns
    -------
    str
        The expression converting it to a plain array.
    """
    return f"_qvr_qiec_array({value})"


def _spell_webppl(
    record: DistributionFamily,
    arguments: dict[str, str],
    event_shape: tuple[int, ...] | None,
) -> str:
    """WebPPL spells families as constructors taking a parameter object.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    arguments : dict[str, str]
        The supplied parameters.
    event_shape : tuple[int, ...] or None
        The event shape of the constructed type.

    Returns
    -------
    str
        The constructor expression.
    """
    family = record.name
    meta = FAMILY_META[family]
    name = meta.target_names["webppl"]
    arguments = _arrays(record, arguments, _webppl_array)
    aliases = meta.arg_aliases.get("webppl", {})
    if family in _HELPER_ROOTS["webppl"] and family != "Categorical":
        if family == "LKJCholesky":
            dimension = _dimension(record, event_shape, "webppl")
            (concentration,) = _require(record, arguments, "webppl", "concentration")
            return f"LKJCholesky({{dim: {dimension}, concentration: {concentration}}})"
        return f"{name}({_javascript_object(arguments)})"
    if family == "HalfNormal":
        (scale,) = _require(record, arguments, "webppl", "scale")
        return f"_qvr_qiec_half(Gaussian({{mu: 0, sigma: {scale}}}))"
    if family == "HalfCauchy":
        (scale,) = _require(record, arguments, "webppl", "scale")
        return f"_qvr_qiec_half(Cauchy({{location: 0, scale: {scale}}}))"
    if family == "Gamma":
        concentration, rate = _require(
            record, arguments, "webppl", "concentration", "rate"
        )
        return f"Gamma({{shape: {concentration}, scale: 1 / {rate}}})"
    if family == "Categorical":
        (probs,) = _require(record, arguments, "webppl", "probs")
        return f"_qvr_qiec_categorical({probs})"
    if family == "Dirichlet":
        (concentration,) = _require(record, arguments, "webppl", "concentration")
        return f"Dirichlet({{alpha: Vector({concentration})}})"
    if family == "MultivariateNormal":
        loc, covariance = _require(
            record, arguments, "webppl", "loc", "covariance_matrix"
        )
        return f"MultivariateGaussian({{mu: Vector({loc}), cov: Matrix({covariance})}})"
    if family == "MixtureNormal":
        weights, loc, scale = _require(
            record, arguments, "webppl", "weights", "loc", "scale"
        )
        return f"_qvr_qiec_mixture_normal({weights}, {loc}, {scale})"
    return f"{name}({_javascript_object(arguments, aliases)})"


# ---------------------------------------------------------------------------
# Church
# ---------------------------------------------------------------------------


def _scheme_array(value: str) -> str:
    """Wrap a Scheme expression in the bridge's array conversion.

    Parameters
    ----------
    value : str
        A host expression evaluating to a QIEC tensor.

    Returns
    -------
    str
        The expression converting it to a nested list.
    """
    return f"(_qvr-qiec-array {value})"


def _scheme_call(callee: str, values: Sequence[str]) -> str:
    """A Scheme application.

    Parameters
    ----------
    callee : str
        The operator.
    values : Sequence[str]
        The operands.

    Returns
    -------
    str
        ``(callee values...)``.
    """
    return "(" + " ".join((callee, *values)) + ")"


def _spell_church(
    record: DistributionFamily,
    arguments: dict[str, str],
    event_shape: tuple[int, ...] | None,
) -> str:
    """Church spells families as the runtime's distribution constructors.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    arguments : dict[str, str]
        The supplied parameters.
    event_shape : tuple[int, ...] or None
        The event shape of the constructed type.

    Returns
    -------
    str
        The constructor expression.
    """
    family = record.name
    name = FAMILY_META[family].target_names["church"]
    arguments = _arrays(record, arguments, _scheme_array)
    if family == "HalfNormal":
        (scale,) = _require(record, arguments, "church", "scale")
        return _scheme_call("half", (_scheme_call("gaussian", ("0", scale)),))
    if family == "HalfCauchy":
        (scale,) = _require(record, arguments, "church", "scale")
        return _scheme_call("half", (_scheme_call("cauchy", ("0", scale)),))
    if family == "Pareto":
        alpha, scale = _require(record, arguments, "church", "alpha", "scale")
        return _scheme_call("pareto", (scale, alpha))
    return _scheme_call(name, _ordered(record, arguments))


_SPELLERS: dict[
    str,
    Callable[[DistributionFamily, dict[str, str], tuple[int, ...] | None], str],
] = {
    "pyro": _spell_pyro,
    "numpyro": _spell_numpyro,
    "pymc": _spell_pymc,
    "edward2": _spell_edward2,
    "turing": _spell_turing,
    "gen": _spell_gen,
    "webppl": _spell_webppl,
    "church": _spell_church,
}


__all__ = [
    "DYNAMIC_TARGET_LANGUAGES",
    "DistributionSpelling",
    "bridge_source",
    "can_spell",
    "helper_families",
    "helper_roots",
    "spell_distribution",
]
