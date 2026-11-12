"""Format structured transpilation refusals as actionable messages.

A refusal kind contains a dispatch head, structured segments, and an
optional explanation after the first colon-space. This module maps that
structure to user-facing prose while preserving the original kinds for
programmatic consumers.
"""

from __future__ import annotations

import didactic.api as dx


class RefusedDeclaration(dx.Model):
    """One top-level declaration a target's support tier rejected.

    `unsupported_for` reads these off the module's AST so the message
    can name the declaration the way the user wrote it (`composition
    pmf_scores`) and point at its line, rather than echoing the
    grammar rule's name (`composition_decl`).

    Parameters
    ----------
    kind
        The statement's didactic discriminator, such as
        ``"composition_decl"``.
    name
        The name the declaration binds, empty when the form binds none.
    line
        The one-based source line, 0 when the parser recorded none.
    """

    kind: str
    name: str = ""
    line: int = 0


# ---------------------------------------------------------------------------
# Target vocabulary
# ---------------------------------------------------------------------------


#: Pipeline stages that raise before any one backend is in play. A
#: refusal tagged with one of these is target-independent: it holds for
#: every backend, so the message says so instead of blaming a language.
_STAGE_TARGETS: frozenset[str] = frozenset({"transpile", "lower", "renderer", "expand"})


#: The name each backend's *language* goes by in prose, for the clauses
#: that talk about the language rather than about the quivers backend
#: (``"Stan has no method-dispatch syntax"``).
_LANGUAGE_NAME: dict[str, str] = {
    "stan": "Stan",
    "bugs": "BUGS",
    "jags": "JAGS",
    "pymc": "PyMC",
    "numpyro": "NumPyro",
    "pyro": "Pyro",
    "edward2": "Edward2",
    "turing": "Turing.jl",
    "gen": "Gen.jl",
    "church": "Church",
    "webppl": "WebPPL",
}


def _backend_display_name(target: str) -> str:
    """Strip the ``qvr-`` prefix the per-backend `target` carries
    internally and return the backend identifier the user typed
    (e.g. ``"qvr-stan"`` -> ``"stan"``)."""
    if target.startswith("qvr-"):
        return target[4:]
    return target


def _language(backend: str) -> str:
    """The target language's proper name, or a target-independent
    phrase when the refusal came from a shared pipeline stage."""
    if backend in _STAGE_TARGETS:
        return "no transpile target"
    return _LANGUAGE_NAME.get(backend, backend)


def _cannot(backend: str, verb_phrase: str) -> str:
    """``stan cannot X`` for a backend, ``no transpile target can X``
    for a shared pipeline stage, so a target-independent refusal never
    blames one language for a gap every language has."""
    if backend in _STAGE_TARGETS:
        return f"no transpile target can {verb_phrase}"
    return f"{backend} cannot {verb_phrase}"


def _has_no(backend: str, noun_phrase: str) -> str:
    """``stan has no X`` / ``no transpile target has X``."""
    if backend in _STAGE_TARGETS:
        return f"no transpile target has {noun_phrase}"
    return f"{backend} has no {noun_phrase}"


#: The vocabulary every PPL target does have, used to say what a
#: refused construct would have had to become.
_PPL_SURFACE = (
    "declaring data and parameters, drawing a variable from a "
    "distribution, and adding a term to the log density"
)


# ---------------------------------------------------------------------------
# IR vocabulary -> QVR surface vocabulary
# ---------------------------------------------------------------------------


#: How each IR argument variant reads back as QVR source. A renderer
#: that refuses an argument position knows only the IR class; the user
#: only ever saw the surface form, so the message names that.
_ARG_SURFACE: dict[str, str] = {
    "IRArgNumber": "a numeric literal",
    "IRArgRef": "a reference to a bound name",
    "IRArgBroadcast": "a scalar broadcast to a vector or matrix shape",
    "IRArgList": "a list literal `[a, b, c]`",
    "IRArgMatrix": "a matrix literal `[[a, b], [c, d]]`",
    "IRArgFamilyRef": (
        "a family-valued argument: the name of another morphism, "
        "read as the distribution that morphism declares"
    ),
    "IRArgKernel": "a Gaussian-process kernel argument",
}


#: How each IR axis variant reads back as QVR source.
_DIM_SURFACE: dict[str, str] = {
    "DimStatic": "an axis of statically known size",
    "DimDynamic": "an axis whose size is bound at run time from data",
}


#: How each `let` right-hand side reads back as QVR source.
_LET_EXPR_SURFACE: dict[str, str] = {
    "LetExprLiteral": "a numeric literal",
    "LetExprString": "a string literal",
    "LetExprRef": "a reference to a bound name",
    "LetExprIndex": "an indexed reference `name[i]`",
    "LetExprBinOp": "an arithmetic expression",
    "LetExprCall": "a function call",
    "LetExprMethodCall": "a method call",
    "LetExprLambda": "a lambda",
    "LetExprFactor": "a `factor` over an axis",
    "LetExprCases": "a `cases` expression",
    "LetExprAffineMap": "an affine map from bound values",
}


def _let_expr_surface(class_name: str) -> str:
    """QVR reading of a `let` right-hand-side class name."""
    return _LET_EXPR_SURFACE.get(
        class_name, f"a `let` body the IR carries as `{class_name}`"
    )


#: How each IR node variant reads back as a QVR program step.
_NODE_SURFACE: dict[str, str] = {
    "IRDataInput": "a program-domain data input",
    "IRSample": "a `sample` step",
    "IRObserve": "an `observe` step",
    "IRDeterministic": "a `let` binding",
    "IRScore": "a `score` step",
    "IRMarginalize": "a `marginalize` block",
    "IRReturn": "a `return` clause",
}


def _arg_surface(class_name: str) -> str:
    """QVR reading of an IR argument class name."""
    return _ARG_SURFACE.get(class_name, f"an argument the IR carries as `{class_name}`")


def _dim_surface(class_name: str) -> str:
    """QVR reading of an IR axis class name."""
    return _DIM_SURFACE.get(class_name, f"an axis the IR carries as `{class_name}`")


def _node_surface(class_name: str) -> str:
    """QVR reading of an IR node class name."""
    return _NODE_SURFACE.get(class_name, f"a step the IR carries as `{class_name}`")


# ---------------------------------------------------------------------------
# Top-level declarations
# ---------------------------------------------------------------------------


#: The keyword the user typed for each refused declaration kind.
_DECLARATION_KEYWORD: dict[str, str] = {
    "composition_decl": "composition",
    "contraction_decl": "contraction",
    "bundle_decl": "bundle",
    "schema_decl": "schema",
    "category_decl": "category",
    "rule_decl": "rule",
    "deduction_decl": "deduction",
    "signature_decl": "signature",
    "encoder_decl": "encoder",
    "decoder_decl": "decoder",
    "loss_decl": "loss",
    "object_decl": "object",
    "morphism_decl": "morphism",
    "define_decl": "define",
    "export_decl": "export",
    "program_decl": "program",
}


#: What each declaration *denotes*, stated in the user's own terms. The
#: first clause of every refusal: a user who wrote a `contraction` is
#: owed an account of what the transpiler read it as before being told
#: it cannot go.
_DECLARATION_MEANING: dict[str, str] = {
    "composition_decl": (
        "declares a composition rule: the algebra (its `tensor_op`, "
        "`join`, `unit` and `zero`) that says how morphism scores "
        "combine when morphisms are composed. It is structure over "
        "the whole module, not a random variable and not a step"
    ),
    "contraction_decl": (
        "declares a contraction: a morphism built by folding several "
        "input morphisms together over their shared axes, with the "
        "product and sum of the fold taken from the `rule=` "
        "composition rule it names. Its meaning lives entirely in "
        "that algebra"
    ),
    "bundle_decl": (
        "binds a name to a tuple of `schema` references, so a parser "
        "or a chart fold can splice the whole set in at once. It is a "
        "compile-time set of grammar rules"
    ),
    "schema_decl": (
        "declares a morphism schema: a family of morphisms quantified "
        "over type parameters, instantiated at concrete objects "
        "wherever it is used"
    ),
    "category_decl": (
        "names the ambient categories the module's objects and "
        "morphisms live in. It is metadata about the module's "
        "structure and denotes no measure"
    ),
    "rule_decl": (
        "declares an inference rule: a sequent taking premises to a "
        "conclusion over object patterns. It denotes a proof step"
    ),
    "deduction_decl": (
        "declares a deductive system: an alphabet, a rule set, a "
        "lexicon and a semiring, whose meaning is a chart over the "
        "proofs the rules license"
    ),
    "signature_decl": (
        "declares a term signature: the sorts, constructors and "
        "binders of an algebraic term language"
    ),
    "encoder_decl": (
        "declares a neural encoder over a `signature`. Its weights "
        "are model-internal: they appear in neither the wire form "
        "nor the sample sites"
    ),
    "decoder_decl": (
        "declares a neural decoder over a `signature`. Its weights "
        "are model-internal: they appear in neither the wire form "
        "nor the sample sites"
    ),
    "loss_decl": (
        "declares a training objective attached to a program, a "
        "deduction, an encoder or a decoder. A loss is something an "
        "optimiser minimises, not a term of the model's density"
    ),
    "object_decl": (
        "declares an object: the value space a variable of that type ranges over"
    ),
    "morphism_decl": (
        "declares a morphism between two objects, optionally with the "
        "family it draws from"
    ),
    "define_decl": "binds a name to a morphism expression",
    "export_decl": "names the program this module exports",
    "program_decl": (
        "declares a probabilistic program: the sample / observe / let "
        "/ return steps that make up the joint"
    ),
}


#: The surface each declaration would need in the target language, and
#: which no probabilistic-programming language provides.
_DECLARATION_GAP: dict[str, str] = {
    "composition_decl": "declaring the algebra a category composes in",
    "contraction_decl": (
        "a morphism defined by contracting other morphisms over an algebra's fold"
    ),
    "bundle_decl": "a first-class, nameable set of grammar rules",
    "schema_decl": (
        "a declaration quantified over type parameters: its variables "
        "and distributions are all at concrete shapes"
    ),
    "category_decl": "naming a category",
    "rule_decl": "an inference rule",
    "deduction_decl": ("a deductive system, a chart over proofs, or a lexicon"),
    "signature_decl": (
        "an algebraic term signature: sorts and constructors are not "
        "things its model block can declare"
    ),
    "encoder_decl": "a network whose weights are not themselves sites",
    "decoder_decl": "a network whose weights are not themselves sites",
    "loss_decl": ("an optimisation objective separate from the joint it scores"),
    "object_decl": "this object declaration",
    "morphism_decl": "this morphism declaration",
    "define_decl": "this define binding",
    "export_decl": "this export clause",
    "program_decl": "this program declaration",
}


#: What to write instead, per declaration kind. Where the construct has
#: no target-language counterpart at all, the entry says that plainly
#: rather than inventing an alternative.
_DECLARATION_REMEDY: dict[str, str] = {
    "composition_decl": (
        "A composition rule has no counterpart to emit. Add a "
        "`program ... :` block to the module and the rule is carried "
        "as module metadata while that program is transpiled; to run "
        "the rule itself, evaluate the module in quivers, which is "
        "where composition rules have meaning."
    ),
    "contraction_decl": (
        "A contraction has no counterpart to emit. Write the "
        "quantities you want scored as explicit `sample` / `observe` "
        "steps of a `program ... :` block, or evaluate the "
        "contraction in quivers and pass its result in as data."
    ),
    "bundle_decl": (
        "A rule bundle has no counterpart to emit. Parse in quivers "
        "and transpile a `program` over the resulting chart weights, "
        "or pass the parse in as observed data."
    ),
    "schema_decl": (
        "Instantiate the schema at the concrete objects you want and "
        "write the result as a `morphism` or a `program` step, which "
        "does have a target form."
    ),
    "category_decl": (
        "A category declaration has no counterpart to emit. Add a "
        "`program ... :` block: the category is then carried as "
        "module metadata and that program is transpiled."
    ),
    "rule_decl": (
        "An inference rule has no counterpart to emit. Run the "
        "deduction in quivers and transpile a `program` over its "
        "chart, or pass the derivation in as observed data."
    ),
    "deduction_decl": (
        "A deductive system has no counterpart to emit. Run the "
        "deduction in quivers and transpile a `program` over its "
        "chart weights, or pass the parse in as observed data."
    ),
    "signature_decl": (
        "A term signature has no counterpart to emit. Encode the "
        "terms you need as indices into a declared finite object and "
        "score those with an ordinary family."
    ),
    "encoder_decl": (
        "Express the network as explicit sampled weights and a "
        "deterministic forward pass, so every weight is a site the "
        "target can emit."
    ),
    "decoder_decl": (
        "Express the network as explicit sampled weights and a "
        "deterministic forward pass, so every weight is a site the "
        "target can emit."
    ),
    "loss_decl": (
        "If the term belongs in the density, write it as a `score` "
        "step inside the program; if it is an optimiser objective, "
        "keep it in quivers, which is where training happens."
    ),
    "object_decl": (
        "Declare the object inside a module that also declares the program using it."
    ),
    "morphism_decl": (
        "Declare the morphism inside a module that also declares the "
        "program drawing from it."
    ),
    "define_decl": (
        "Inline the bound expression at its use site, or write the "
        "step it stands for directly in a `program`."
    ),
    "export_decl": (
        "Name a `program` this module declares, so there is something "
        "for the export to point at."
    ),
    "program_decl": (
        "Check the backend's support tier: this target declares that "
        "it accepts no program declaration."
    ),
}


def _declaration_locator(entry: RefusedDeclaration) -> str:
    """```composition pmf_scores` (line 12)`` -- the declaration as
    the user wrote it, with its line when the parser recorded one."""
    keyword = _DECLARATION_KEYWORD.get(entry.kind, entry.kind)
    written = f"`{keyword} {entry.name}`" if entry.name else f"`{keyword}`"
    if entry.line:
        return f"{written} at line {entry.line}"
    return written


def _render_declaration_kind(
    backend: str,
    kind: str,
    entries: tuple[RefusedDeclaration, ...],
    module_has_program: bool,
) -> str:
    """Explain one refused top-level declaration: what the user wrote,
    where, what it denotes, why the target has no form for it, and what
    to write instead.

    `entries` are the declarations of this kind the module actually
    carries, so one `composition` is named singly and three are listed.
    """
    meaning = _DECLARATION_MEANING.get(kind)
    gap = _DECLARATION_GAP.get(kind)
    remedy = _DECLARATION_REMEDY.get(kind)
    if meaning is None or gap is None or remedy is None:
        return (
            f"{_cannot(backend, f'transpile a `{kind}` declaration')}, "
            f"and the refusal carries no explanation of that "
            f"declaration kind yet"
        )
    named = ", ".join(_declaration_locator(e) for e in entries)
    subject = named if named else f"the module's `{kind}` declaration"
    language = _language(backend)
    why = (
        f"A probabilistic-programming target has statements for "
        f"{_PPL_SURFACE}; {language} has none for {gap}."
        if backend not in _STAGE_TARGETS
        else (
            f"A probabilistic-programming target has statements for "
            f"{_PPL_SURFACE}, and none for {gap}."
        )
    )
    missing_program = (
        ""
        if module_has_program
        else (
            " This module also declares no `program`, so there is no "
            "probabilistic program here to transpile in its place."
        )
    )
    return f"{subject} {meaning}. {why}{missing_program} {remedy}"


# ---------------------------------------------------------------------------
# Per-head renderers
# ---------------------------------------------------------------------------


_NO_TARGET_HEADS: frozenset[str] = frozenset(
    {
        "no-stan-target",
        "no-bugs-target",
        "no-jags-target",
        "no-target-name",
        "no-webppl-target",
        "no-pymc-target",
        "no-edward2-target",
        "no-numpyro-target",
        "no-pyro-target",
        "no-gen-target",
        "no-turing-target",
        "no-church-target",
    }
)


def _render_family_kind(backend: str, tail: str, explained: bool) -> str:
    """`family:<F>[:<detail>]` -- the target has no form for the
    distribution family `F`, or the mapping exists but a downstream
    step (sentinel construction, shape derivation) could not complete.

    Two leading-token shapes are recognised:

    * ``family:<F>`` or ``family:<F>:<detail>`` -- the usual form,
      with `F` as the family name.
    * ``family:<no-*-target>:<F>`` -- the "no native mapping" form
      where the FAMILY_META lookup raised before reaching the
      target-specific handler; the family name lives in the detail
      slot.
    """
    family, _, detail = tail.partition(":")
    if family == "wrapper-inner-unknown":
        # `family:wrapper-inner-unknown:<inner>` names the INNER
        # family, so the wrapper is what the user wrote and the inner
        # name is what is missing.
        return (
            f"a draw wraps another distribution whose family "
            f"`{detail}` is not one quivers knows, so there is no "
            f"inner density for {_language(backend)} to wrap. Check "
            f"the spelling of the wrapped family, or register it "
            f"before wrapping it."
        )
    if family == "unknown":
        family, detail = detail, "unknown"
    if family in _NO_TARGET_HEADS:
        # Reshape from `family:no-stan-target:TruncatedNormal` to
        # `family:TruncatedNormal:no-stan-target` so the message
        # references the family the user wrote rather than the
        # absence sentinel.
        family, detail = detail, family
    if family == "Horseshoe":
        return (
            f"{_has_no(backend, 'horseshoe distribution')}: `Horseshoe(scale)` "
            f"denotes the marginal of a normal draw over a half-Cauchy local "
            f"scale, which has no closed form and no library distribution on "
            f"any target. Write the prior as its explicit decomposition, "
            f"`tau * lambda * z` with `lambda ~ HalfCauchy(1.0)` and "
            f"`z ~ Normal(0.0, 1.0)`, which every target expresses."
        )
    if explained:
        return _cannot(backend, f"score a draw from `{family}`")
    if not detail:
        return (
            f"{_has_no(backend, f'`{family}` distribution')}: the "
            f"family registry contains no matching target distribution. "
            f"Pick a family this target supports, "
            f"or write the density you want as an explicit `score` "
            f"step."
        )
    if detail in _NO_TARGET_HEADS or detail.startswith("no-"):
        return (
            f"{_has_no(backend, f'`{family}` distribution')}. Pick a "
            f"family this target supports, or write the density you "
            f"want as an explicit `score` step."
        )
    if detail.startswith("sentinel-failed"):
        sentinel_detail = detail.removeprefix("sentinel-failed:").removeprefix(
            "sentinel-failed"
        )
        return (
            f"the shape of the `{family}` draw could not be derived, "
            f"so no target can size the variable it binds. This "
            f"usually means an argument the family requires is "
            f"missing from the call site. Underlying cause: "
            f"{sentinel_detail.strip(': ') or 'unspecified'}. Supply "
            f"every required argument explicitly at the call site "
            f"rather than relying on inference."
        )
    if detail.startswith("unknown"):
        return (
            f"`{family}` is not a distribution family quivers knows. "
            f"Either the name is misspelled, or the family has to be "
            f"registered before any target can emit a draw from it."
        )
    if detail.startswith("arity-mismatch"):
        rest = detail.removeprefix("arity-mismatch:").strip()
        return (
            f"the call to `{family}` supplies the wrong number of "
            f"arguments. {rest}. Check the argument order the family "
            f"declares."
        )
    if detail.startswith("wrapper-inner-unknown"):
        inner = detail.removeprefix("wrapper-inner-unknown:").strip()
        return (
            f"the wrapper family `{family}` wraps an inner family "
            f"`{inner}` that {_language(backend)} does not have."
        )
    if detail.startswith("morphism-unknown"):
        morphism = detail.partition(":")[2] or "the referenced morphism"
        return (
            f"the `{family}` draw goes through morphism `{morphism}`, "
            f"which the module never declares, so its domain and "
            f"codomain (and with them the draw's shape) are unknown. "
            f"Declare `morphism {morphism} : Dom -> Cod`."
        )
    if detail.startswith("event_axis_source"):
        return (
            f"the event axes of the `{family}` draw are read from a "
            f"source the lowering does not recognise "
            f"({detail.rpartition(':')[2]}). Name the event axes "
            f"explicitly with `[over=...]` at the call site."
        )
    if detail.startswith("structured_arg"):
        rest = detail.removeprefix("structured_arg:").strip(": ")
        return (
            f"the structured argument `{rest}` of the `{family}` draw "
            f"does not line up with the axes the site declares. Name "
            f"the event axes explicitly with `[over=...]` at the call "
            f"site."
        )
    if detail in {"missing-event-dimension", "missing-dimension-axis"}:
        headline = _cannot(
            backend,
            f"size a `{family}` draw: the site declares no event axis "
            f"to read its dimension from",
        )
        if explained:
            return headline
        return (
            f"{headline}. `{family}` is matrix-valued, so its "
            f"dimension comes from the site's event axis. Give the "
            f"site one with `[over=Dim]` where `Dim` is an object of "
            f"fixed cardinality."
        )
    if detail in {
        "non-static-event-dimension",
        "dynamic-dimension-axis",
    }:
        headline = _cannot(
            backend,
            f"size a `{family}` draw from an event axis whose extent "
            f"is bound at run time",
        )
        if explained:
            return headline
        return (
            f"{headline}. `{family}` needs its dimension fixed when "
            f"the module compiles. Declare the event axis with an "
            f"explicit cardinality (`object Dim : FinSet 4`) rather "
            f"than binding its size from data."
        )
    if explained:
        return _cannot(backend, f"emit a `{family}` draw")
    return f"{_cannot(backend, f'emit a `{family}` draw')}: {detail}."


def _render_node_kind(backend: str, tail: str, explained: bool) -> str:
    """`node:<IRNodeClass>[:<detail>]` -- a program step the target's
    walker has no statement for."""
    node, _, detail = tail.partition(":")
    detail = detail.lstrip(": ").strip()
    surface = _node_surface(node)
    if node == "IRScore":
        reason = detail or (
            f"{_language(backend)} has no statement that adds a free "
            f"term to the log density"
        )
        return (
            f"a `score` step cannot be transpiled to {backend}: "
            f"{reason}. Express the term as a draw from a family with "
            f"that density, or move the scoring into the host-side "
            f"inference loop."
        )
    if not detail:
        return _cannot(backend, f"emit {surface}")
    if explained:
        return _cannot(backend, f"emit {surface}")
    return f"{_cannot(backend, f'emit {surface}')}: {detail}"


def _render_qiec_kind(backend: str, tail: str) -> str:
    """Render one selected-target QIEC capability mismatch."""
    reason, _, name = tail.partition(":")
    if reason == "capability":
        feature, _, subject = name.partition(":")
        where = (
            f"QIEC computation `{subject}`"
            if subject != "module"
            else "the QIEC module"
        )
        if feature == "search":
            return (
                f"{where} is a deduction's entry, which enumerates its "
                "derivations through a search handler that resumes once per "
                f"alternative, and {_language(backend)} has no runtime for that "
                "search. Run the deduction on the reference machine with "
                "`run_deduction`, or keep the program's chart out of the "
                "transpiled model."
            )
        return (
            f"{where} requires the `{feature}` QIEC capability, but "
            f"{_language(backend)} has no semantics-preserving lowering for it. "
            "Choose a target whose QIEC capability set includes this feature, "
            "or change the computation; silently erasing it would change the program."
        )
    if reason == "call":
        callee, _, program = name.partition(":")
        return (
            f"program `{program}` calls `{callee}`, which is not a "
            f"computation of the checked module, so the plan "
            f"{_language(backend)} is rendered from has nothing to call. "
            f"Declare the computation with `define` in the module, or call "
            f"one that is declared."
        )
    if reason == "stan":
        form, _, rest = name.partition(":")
        primitive, _, computation = rest.partition(":")
        return (
            f"QIEC computation `{computation}` applies the `{primitive}` "
            f"primitive, which has no spelling in a Stan user-defined "
            f"function: Stan's function library has no counterpart, or "
            f"(as for a conversion to an integer) admits only data. Write "
            f"the computation with the primitives Stan has, or keep it out "
            f"of the transpiled model."
        )
    if reason == "effect":
        effect, _, program = name.partition(":")
        return (
            f"program `{program}` performs on a `{effect}` instance, and "
            f"{_language(backend)} carries only the canonical `Random` and "
            f"`Score` requests of a program. Handle the instance inside a "
            f"computation the program calls, or keep it out of the "
            f"transpiled model."
        )
    if reason == "computation":
        form, _, program = name.partition(":")
        return (
            f"program `{program}` elaborates to a `{form}` computation form, "
            f"which the plan {_language(backend)} is rendered from has no "
            f"statement for. Write the program with sample, observe, let, "
            f"score, and marginalize steps, or keep the form out of the "
            f"transpiled model."
        )
    if reason == "argument":
        term, _, rest = name.partition(":")
        site, _, head = rest.partition(":")
        return (
            f"the `{head}` argument of the draw at `{site}` is a `{term}` "
            f"term of the checked module, which has no wire form in "
            f"{_language(backend)}: a family argument on the wire is a "
            f"literal, a bound name, an indexed name, a list, or a "
            f"broadcast. Bind the value with a `let` before the draw."
        )
    if reason == "expression":
        term, _, program = name.partition(":")
        return (
            f"a binding of program `{program}` is a `{term}` term of the "
            f"checked module, which has no expression form in "
            f"{_language(backend)}. Write the binding with the arithmetic, "
            f"calls, indexing, lists, and factors of a let expression."
        )
    if reason == "fibration":
        return (
            f"program `{name}` fibres an observation by an index that is "
            f"not a program input, and {_language(backend)} reads a "
            f"fibration only as a data input. Pass the fibration as data."
        )
    if reason == "sample":
        return (
            f"a `Random.sample` request of the program does not carry a "
            f"site and a family construction, so the plan {_language(backend)} "
            f"is rendered from cannot read it as a draw. This is a defect in "
            f"the elaboration rather than in the program; please report it "
            f"with the module that provoked it."
        )
    return f"{_cannot(backend, 'transpile this checked QIEC construct')}: {tail}"


def _render_call_kind(backend: str, tail: str) -> str:
    """`call:graph:<callee>` -- a graph language has no statement that
    runs a computation."""
    reason, _, callee = tail.partition(":")
    if reason == "graph":
        return (
            f"the program calls the computation `{callee}` at run time, and "
            f"{_language(backend)} is a graph language: a model is a set of "
            f"stochastic and deterministic relations between named nodes, "
            f"with no statement that runs a computation, draws under a "
            f"site name it computes, or scores a weight it accumulates. A "
            f"pure computation whose body is a chain of bindings is inlined "
            f"before the plan reaches this target, so `{callee}` is either "
            f"effectful or recursive. Write its draws and scores as steps of "
            f"the program, or transpile to a target with a host runtime."
        )
    return f"{_cannot(backend, 'transpile this call')}: {tail}"


def _render_declare_kind(backend: str, tail: str) -> str:
    """`declare:<type>:<dimension-issue>` -- the variable declaration
    the target needs more shape information than the program gives."""
    type_, _, rest = tail.partition(":")
    if "event-rank" in rest:
        return (
            f"the declared shape of a `{type_}` parameter has no "
            f"{_language(backend)} declaration form: {rest}. The "
            f"site's axes do not say which of them are event axes. "
            f"Name them with `[over=...]`, or draw from a family that "
            f"fixes its own event shape."
        )
    if type_ == "unsupported-support":
        return _has_no(backend, f"declaration form for a variable supported on {rest}")
    return f"{_cannot(backend, f'declare a `{type_}` of this shape')}: {rest}"


def _render_arg_kind(backend: str, tail: str, explained: bool) -> str:
    """`arg:<shape-detail>` -- an argument of a draw whose shape the
    target's argument position cannot take."""
    if tail.startswith("broadcast"):
        return (
            f"{_cannot(backend, 'broadcast a scalar literal into a vector or matrix argument')}. "
            f"Bind the value as a vector data input, or pass an "
            f"already-shaped argument."
        )
    if tail.startswith("family-ref"):
        rest = tail.removeprefix("family-ref:").strip(": ")
        name = rest.partition(":")[0]
        return (
            f"the argument `{name}` names a morphism that declares no "
            f"`~ Family(...)` init, so there is no distribution for "
            f"{backend} to put in that argument position. Give the "
            f"morphism a `~ Family(args)` init clause."
        )
    if tail.startswith("list-literal"):
        return (
            f"{_cannot(backend, 'take a list literal `[a, b, c]` in a distribution-argument position')}. "
            f"Bind the list as a data input, or draw from a "
            f"shape-aware family that takes the elements as one "
            f"vector."
        )
    if tail.startswith("matrix-literal"):
        return (
            f"{_cannot(backend, 'take a matrix literal `[[a, b], ...]` in a distribution-argument position')}. "
            f"Bind the matrix as a data input instead."
        )
    if tail.startswith("unknown:"):
        return _cannot(
            backend,
            f"take {_arg_surface(tail.partition(':')[2])} in a "
            f"distribution-argument position",
        )
    if tail.startswith("coerce:"):
        if explained:
            return _cannot(backend, "take this argument")
        return (
            f"a distribution argument reached the {backend} renderer "
            f"as a Python `{tail.partition(':')[2]}` rather than as a "
            f"lowered IR argument, so there is nothing to emit it "
            f"from. This is a defect in the transpiler rather than in "
            f"the program; please report it with the module that "
            f"provoked it."
        )
    if tail.startswith("family_ref:"):
        name = tail.partition(":")[2]
        if explained:
            return _cannot(backend, f"take the argument `{name}`")
        return (
            f"the argument `{name}` names a morphism whose "
            f"distribution is built by wrapping another one "
            f"(truncation, composition or rejection sampling), and "
            f"{_language(backend)} has no wrapper distributions to "
            f"build it with. Fold the wrapping into the outer "
            f"distribution call, or draw from a family that states "
            f"the wrapped density directly."
        )
    if tail and ":" not in tail:
        headline = _cannot(
            backend,
            f"take {_arg_surface(tail)} in a distribution-argument position",
        )
        if explained:
            return headline
        return (
            f"{headline}. Bind the value with a `let` or as a data "
            f"input, and pass the bound name."
        )
    if explained:
        return _cannot(backend, "take this argument")
    return f"{_cannot(backend, 'take this argument')}: {tail}"


def _render_let_expr_kind(backend: str, tail: str, explained: bool) -> str:
    """`let-expr:<shape>[:<detail>]` -- a `let` right-hand side with no
    expression form in the target."""
    kind, _, rest = tail.partition(":")
    if kind == "LetExprFactor" and rest.startswith("unresolved-binder-size"):
        binder = rest.partition(":")[2]
        return (
            f"the `factor` binder `{binder}` has no statically known "
            f"size, so {backend} cannot lay out the loop it would "
            f"become. Declare the binder's index type so its "
            f"cardinality is readable."
        )
    if kind == "LetExprFactor" and rest.startswith("multi-axis-body"):
        return (
            f"a `factor` whose body ranges over more than one axis "
            f"has no {_language(backend)} expression form: the "
            f"product would have to be built by nested loops over a "
            f"named array, and a `let` lowered to one expression has "
            f"nowhere to put it. Split the factor into one per axis, "
            f"or move the product into a plated `score` step."
        )
    if kind == "LetExprMethodCall":
        return (
            f"{_has_no(backend, 'method-dispatch syntax')}, so a "
            f"`receiver.method(...)` call in a `let` has no form to "
            f"take. Rewrite the call as a plain function of its "
            f"arguments, or compute it in quivers and pass the result "
            f"in as data."
        )
    if kind == "LetExprLambda":
        return (
            f"{_has_no(backend, 'anonymous-function syntax in a model-body expression')}, "
            f"so a `param -> body` lambda in a `let` has no form to "
            f"take. Inline the lambda's body at its use site."
        )
    if kind == "elementwise-axis-operator":
        return _cannot(backend, "lift an infix operator over an axis inside a `let`")
    if kind == "LetExprFactor" and rest == "cases-with-multi-axis-binders":
        return (
            "a `factor` written with `cases` enumerates one axis, so "
            "it may bind exactly one binder, and this one binds "
            "several. Split it into one `factor` per axis, or write "
            "the body as an expression over the binders instead of "
            "as cases."
        )
    if kind == "LetExprFactor" and rest == "cases-labels-not-dense":
        return (
            "a `factor` written with `cases` is emitted as a "
            "positional list, so its case labels have to run "
            "`0, 1, ... n-1` with none missing. Give the factor one "
            "case per value of the axis it ranges over."
        )
    if kind == "LetExprFactor" and rest == "no-body-no-cases":
        return (
            f"a `factor` has to say what it multiplies: either a body "
            f"expression over its binder or an explicit list of "
            f"`cases`. This one gives neither, so there is nothing "
            f"for {_language(backend)} to fold over the axis."
        )
    if kind == "factor" and rest.startswith("expected-factor:"):
        name = rest.partition(":")[2]
        return (
            f"the `let` binding `{name}` was routed to the "
            f"factor emitter but its body is not a `factor`. This is "
            f"a defect in the transpiler rather than in the program; "
            f"please report it with the module that provoked it."
        )
    if kind == "substitute":
        return (
            f"{_cannot(backend, f'substitute into {_let_expr_surface(rest)} while rewriting a `let` body')}. "
            f"Bind that expression to its own `let` and reference the "
            f"bound name."
        )
    if kind == "outer-kind":
        inner = rest.partition(":")[0] or rest
        headline = _cannot(
            backend,
            f"emit {_let_expr_surface(inner)} as the outermost form of a `let` body",
        )
        if explained:
            return headline
        return (
            f"{headline}. Bind the sub-expressions to their own "
            f"`let`s and combine the bound names."
        )
    headline = _cannot(backend, f"emit {_let_expr_surface(kind)} as a `let` body")
    if explained or not rest:
        return headline
    return f"{headline}: {rest}"


def _render_let_kind(backend: str, tail: str, explained: bool) -> str:
    """`let:<reason>` -- a `let` whose right-hand side the resolver
    could not reduce to something a target can emit."""
    if tail.startswith("composite_expression"):
        rest = tail.partition(":")[2]
        if explained:
            return _cannot(backend, "unfold this composite `let` into program steps")
        return (
            f"{_cannot(backend, 'unfold this composite `let` into program steps')} "
            f"(expression kind `{rest}`). Replace the composition "
            f"with a direct `~ Family(args)` declaration, or write "
            f"one `sample` step per stochastic link."
        )
    if tail.startswith("call:unknown:"):
        name = tail.partition("call:unknown:")[2]
        return (
            f"a `let` calls `{name}(...)`, which names no builtin of the "
            f"expression language, no lambda, and no computation of the "
            f"module, so nothing fixes what the call computes and "
            f"{_language(backend)} would be handed an unresolved name. "
            f"Write the computation as a `define` of the module, or as "
            f"arithmetic and builtins the expression language has."
        )
    if explained:
        return _cannot(backend, "resolve this `let` binding")
    return f"{_cannot(backend, 'resolve this `let` binding')}: {tail}"


def _render_define_kind(backend: str, tail: str, explained: bool) -> str:
    """`define:<reason>` -- a `define` binding the resolver could not
    reduce to a single distribution."""
    if explained:
        return _cannot(backend, "unfold this `define` into a single distribution")
    return (
        f"{_cannot(backend, 'unfold this `define` into a single distribution')}: {tail}"
    )


def _render_return_kind(backend: str, tail: str) -> str:
    """`return:undeclared:<name>` -- a `return` naming something the
    program never bound, so no shape can be derived for it."""
    if tail.startswith("undeclared"):
        name = tail.removeprefix("undeclared:").strip()
        return (
            f"`return {name}` names something the program never "
            f"binds: it is not a `sample`, an `observe`, or a `let` "
            f"earlier in the body, so {backend} has no shape to give "
            f"the returned quantity. Bind `{name}` in the program, or "
            f"return a name it does bind."
        )
    return f"{_cannot(backend, 'emit this `return`')}: {tail}"


def _render_broadcast_kind(backend: str, tail: str, explained: bool) -> str:
    """`broadcast:<reason>` -- a value the target cannot expand to the
    shape its use site needs."""
    if explained:
        return _cannot(backend, "broadcast this value to the shape it is used at")
    return (
        f"{_cannot(backend, 'broadcast this value to the shape it is used at')}: {tail}"
    )


def _render_step_kind(backend: str, tail: str, explained: bool) -> str:
    """`step:<step-kind>` -- a program step the lowering has no IR node
    for, so no target sees it at all."""
    step, _, rest = tail.partition(":")
    written = step.removesuffix("_step").replace("_", " ")
    headline = (
        f"the program contains a `{written}` step, which the transpile "
        f"pipeline does not lower, so {_language(backend) if backend not in _STAGE_TARGETS else 'no target'} "
        f"ever sees it"
    )
    if explained or not rest:
        return headline
    return f"{headline}: {rest}"


def _render_axes_kind(backend: str, tail: str, explained: bool) -> str:
    """`axes:<reason>` -- an `[over=...]` / `iid over` specification the
    target cannot honor."""
    if explained:
        return _cannot(backend, "honor this axis specification")
    return f"{_cannot(backend, 'honor this axis specification')}: {tail}"


def _render_axis_kind(backend: str, tail: str, explained: bool) -> str:
    """`axis:<reason>:<name>` -- a single axis whose size no target can
    read."""
    reason, _, rest = tail.partition(":")
    name = rest.partition(":")[0]
    if reason == "unknown-cardinality":
        headline = (
            f"the axis `{name}` has no statically known size, so no "
            f"target can size the plate it becomes"
        )
    else:
        headline = _cannot(backend, f"use the axis `{name}`")
    if explained:
        return headline
    return f"{headline}: {tail}"


def _render_option_kind(backend: str, tail: str) -> str:
    """`option:<key>` -- an option in a `[...]` block the target reads
    but has nothing to do with."""
    return (
        f"{_has_no(backend, f'form for the `{tail}` option')}, so the "
        f"program it would produce would not be the one written. "
        f"Remove the option, or pick a target that honors it."
    )


def _render_marginalize_kind(backend: str, tail: str, explained: bool) -> str:
    """`marginalize:<reason>` -- a `marginalize` block the target
    cannot integrate out."""
    if tail.startswith("non-finite-support"):
        family = tail.partition(":")[2]
        return (
            f"`marginalize` over a `{family}` latent asks for a sum "
            f"over a support that is not finite, so {backend} cannot "
            f"integrate it out at compile time. Reparameterise the "
            f"latent as a continuous relaxation, or do the "
            f"marginalisation in the host-side inference loop."
        )
    if tail.startswith("unknown-cardinality"):
        family = tail.partition(":")[2]
        return (
            f"`marginalize` over a `{family}` latent needs the size "
            f"of its support at compile time, and the declaration "
            f"does not give one. Declare the latent's index object "
            f"with an explicit cardinality (`object K : FinSet 3`)."
        )
    if tail.startswith("scope:"):
        node, _, rest = tail.removeprefix("scope:").partition(":")
        return (
            f"{_cannot(backend, f'emit {_node_surface(node)} inside a `marginalize` scope')}"
            f"{': ' + rest if rest and not explained else ''}. Hoist "
            f"it out of the `marginalize` block."
        )
    if tail.startswith("reduction"):
        reduction = tail.partition(":")[2]
        return (
            f"{_has_no(backend, f'form for a `{reduction}` reduction over a `marginalize` block')}; "
            f"the enumerated terms can be summed out, and nothing "
            f"else. Use the default sum reduction, or do the "
            f"reduction in the host-side inference loop."
        )
    if tail.startswith("arg:") or tail.startswith("expr:"):
        slot, _, cls = tail.partition(":")
        surface = _arg_surface(cls) if slot == "arg" else _let_expr_surface(cls)
        if explained:
            return _cannot(backend, f"rewrite {surface} inside a `marginalize` block")
        return (
            f"integrating the latent out means rewriting every "
            f"reference to it, and {_language(backend)} has no "
            f"rewrite for {surface}. Move that expression out of the "
            f"`marginalize` block, or bind it with a `let` before the "
            f"block and reference the bound name inside."
        )
    if tail.startswith("shape:"):
        helper = tail.partition(":")[2]
        return (
            f"the marginalization helper `{helper}` of the checked module "
            f"is not one enumerated draw over a collected scope, so the "
            f"plan cannot read it back as a `marginalize` block for "
            f"{backend}. This is a defect in the elaboration rather "
            f"than in the program; please report it with the module "
            f"that provoked it."
        )
    if tail == "no-enclosing-body":
        if explained:
            return _cannot(backend, "marginalize this latent out")
        return (
            f"a `marginalize` block has to emit its enumerated terms "
            f"into an enclosing program body, and the {backend} "
            f"renderer reached this block with no body open. This is "
            f"a defect in the transpiler rather than in the program; "
            f"please report it with the module that provoked it."
        )
    if tail.startswith("ungrouped-over-plate:"):
        latent = tail.partition(":")[2]
        return (
            f"`marginalize {latent}` carries no index and no `over =` "
            f"clause, so it declares one latent and every row of the "
            f"plated `observe` inside it is conditioned on that single "
            f"draw. Its density thus accumulates the body's rows "
            f"and reduces over the latent once, and "
            f"{_language(backend)} scores the rows the other way "
            f"round, giving each its own draw. That is a different "
            f"measure, not a different base measure, so it is refused "
            f"rather than emitted. Give the latent the plate its rows "
            f"share (`marginalize {latent} : A`) or a grouping "
            f"`over =` clause, either of which this target does emit "
            f"correctly."
        )
    if tail.startswith("weight-family:"):
        family = tail.partition(":")[2]
        if explained:
            return _cannot(backend, f"build a probability vector from `{family}`")
        return (
            f"integrating the latent out means summing its atoms "
            f"against their probabilities, and the weights here come "
            f"from `{family}`, which {_language(backend)} cannot turn "
            f"into an explicit probability vector. Draw the mixing "
            f"weights from a family that states them directly "
            f"(`Categorical`, `Dirichlet`), or do the marginalisation "
            f"in the host-side inference loop."
        )
    if explained:
        return _cannot(backend, "marginalize this latent out")
    return f"{_cannot(backend, 'marginalize this latent out')}: {tail}"


def _render_dim_kind(backend: str, tail: str, explained: bool) -> str:
    """`dim:<DimClass>` -- an axis whose extent the target cannot size.

    An axis reaches a target in one of two forms: a cardinality fixed
    when the module compiles, or a size read from the data at run
    time. A third form has no extent a target could emit a loop bound
    from, so the kind names the variant and the message names the two
    that work.
    """
    variant = tail.removeprefix("unknown:").partition(":")[0].strip() or tail
    if variant:
        headline = _cannot(backend, f"size {_dim_surface(variant)}")
        if explained:
            return headline
        return (
            f"{headline}. An axis reaches a target either as a "
            f"cardinality fixed at compile time (`object K : FinSet "
            f"3`) or as an extent read from the data at run time; "
            f"declare the axis one of those two ways."
        )
    if explained:
        return _cannot(backend, "resolve this axis")
    return f"{_cannot(backend, 'resolve this axis')}: {tail}"


def _render_dim_class_kind(backend: str, tail: str) -> str:
    """`dim-kind:<DimClass>` / `plate:dim-kind:<DimClass>` -- an axis
    variant the target's plate emitter has no case for."""
    return (
        f"{_cannot(backend, f'emit a plate over {_dim_surface(tail)}')}. "
        f"Declare the axis as an object with a fixed cardinality, or "
        f"bind its size from data with a declared size name."
    )


def _render_plate_kind(backend: str, tail: str, explained: bool) -> str:
    """`plate:<reason>` -- a plate the target's loop / plate form
    cannot carry."""
    if tail.startswith("dim-kind:"):
        return _render_dim_class_kind(backend, tail.partition(":")[2])
    if explained:
        return _cannot(backend, "emit this plate")
    return f"{_cannot(backend, 'emit this plate')}: {tail}"


def _linear_width_mismatch_headline(detail: str) -> str:
    """The headline for
    `param-source:linear:width-mismatch:<program>:<site>:<family>:<arg>:<ref>:<site>`.

    A morphism declared between objects of different width carries a
    linear parameter map from its domain to the family's parameter
    heads on its codomain. Its weights are drawn when the module
    compiles, so they appear in neither the wire form nor the sample
    sites. The user needs the site and the two widths to find the
    declaration that disagrees, so the kind carries them.
    """
    parts = detail.split(":")
    if len(parts) != 6:
        return (
            "a morphism between objects of different width carries a "
            "compiled linear parameter map, whose weights are not "
            "sites the program declares"
        )
    program, site, family, arg, ref_width, site_width = parts
    return (
        f"program `{program}` scores site `{site}` with "
        f"`{family}`, passing `{arg}` a value of width {ref_width} at "
        f"a site of width {site_width}. A morphism declared between "
        f"objects of different width carries a linear parameter map "
        f"from its domain to the family's parameter heads on its "
        f"codomain, and that map's weights are drawn when the module "
        f"compiles rather than declared as sites"
    )


def _render_param_source_kind(backend: str, tail: str, explained: bool) -> str:
    """`param-source:<kind>[:...]` -- a morphism whose parameters come
    from a network or a compiled linear map rather than from sites the
    program declares.

    The raise sites carry the full account, since only they know which
    of the three consuming positions fired; the headline names the
    morphism and leaves the reasoning to the explanation.
    """
    kind, _, rest = tail.partition(":")
    name = rest.rpartition(":")[2]
    if kind == "linear" and rest.startswith("width-mismatch:"):
        headline = _linear_width_mismatch_headline(rest.removeprefix("width-mismatch:"))
    elif kind == "linear":
        headline = (
            "a morphism between objects of different width carries a "
            "compiled linear parameter map, whose weights are not "
            "sites the program declares"
        )
    elif kind == "unnamed":
        return (
            f"the morphism `{name}` carries a `param_source` option "
            f"that names no architecture, so there is nothing for any "
            f"target to read it as. Write `[param_source=<kind>]` "
            f"with an architecture name."
        )
    elif kind == "table":
        return (
            f"the morphism `{name}` is a kernel over a finite domain, whose "
            f"parameters are read from a learned table with one row per "
            f"element, and {_language(backend)} has no wire form for a "
            f"table-indexed parameter map. Draw the row's parameters as "
            f"sites indexed by the element, or write the step against a "
            f"closed-form family."
        )
    else:
        headline = (
            f"a morphism draws its parameters from a `{kind}` network, "
            f"whose weights are not sites the program declares"
        )
    if explained:
        return headline
    unable = (
        "no target can reconstruct"
        if backend in _STAGE_TARGETS
        else f"{_language(backend)} cannot reconstruct"
    )
    return (
        f"{headline}, so {unable} the parameter it computes. Express "
        f"the network as explicit sampled weights and a deterministic "
        f"forward pass, or write the step as a `sample` / `observe` "
        f"against a closed-form family."
    )


def _render_embed_kind(backend: str, tail: str, explained: bool) -> str:
    """`embed:<name>` -- a draw through an embedding morphism, whose
    centres and scales are learned tables rather than sites the program
    declares.

    Parameters
    ----------
    backend : str
        The target the refusal is reported for.
    tail : str
        The morphism's name.
    explained : bool
        Whether the raise site's own account follows the headline.

    Returns
    -------
    str
        The rendered explanation.
    """
    headline = (
        f"the morphism `{tail}` is an embedding, a Normal kernel placed at "
        f"each element's learned centre, whose centres and scales are not "
        f"sites the program declares"
    )
    if explained:
        return headline
    unable = (
        "no target can reconstruct"
        if backend in _STAGE_TARGETS
        else f"{_language(backend)} cannot reconstruct"
    )
    return (
        f"{headline}, so {unable} the kernel it draws from. Sample the "
        f"centres and scales as explicit sites indexed by the element, or "
        f"write the step as a `sample` / `observe` against a closed-form "
        f"family."
    )


def _render_program_kind(backend: str, tail: str, explained: bool) -> str:
    """`program:<reason>[:<name>]` -- something about the program as a
    whole rather than about one step."""
    reason, _, rest = tail.partition(":")
    if reason == "absent":
        return (
            f"this module declares no `program`, so there is no "
            f"program_decl to lower and nothing for {backend} to "
            f"emit. Add a `program NAME : Dom -> Cod` block with the "
            f"`sample` / `observe` / `return` steps you want scored."
        )
    if reason == "undeclared-arg":
        name, _, site = rest.partition(":")
        return (
            f"an argument in program `{name}` names `{site}`, which "
            f"the program never binds and never takes as an input, so "
            f"{backend} would emit a reference to a free variable and "
            f"score a different measure. Bind it with a `sample`, an "
            f"`observe`, or a `let`, or take it as a program input."
        )
    if reason == "undeclared-kernel-input":
        name, _, site = rest.partition(":")
        return (
            f"the Gaussian-process kernel in program `{name}` reads "
            f"its input from `{site}`, which the program never binds "
            f"and never takes as an input. Bind it, or take it as a "
            f"program input."
        )
    if reason == "elaboration":
        return (
            f"the elaboration of the program into the checked module refused "
            f"it under `{rest}`, so {_language(backend)} has no computation "
            f"to derive its plan from"
        )
    if explained:
        return _cannot(backend, f"emit program `{rest or reason}`")
    if reason == "unelaborated":
        return (
            f"program `{rest}` has no computation in the checked module, so "
            f"the plan {_language(backend)} is rendered from cannot be "
            f"derived. This is a defect in the elaboration rather than in "
            f"the program; please report it with the module that provoked it."
        )
    return f"{_cannot(backend, 'emit this program')}: {tail}"


def _render_program_domain_kind(backend: str, tail: str, explained: bool) -> str:
    """`program-domain:<reason>:<name>` -- the program's domain does
    not lower to distinct data inputs."""
    if explained:
        return "the program's domain does not lower to a distinct data input per factor"
    return f"{_cannot(backend, 'lower this program domain to data inputs')}: {tail}"


def _render_object_kind(backend: str, tail: str, explained: bool) -> str:
    """`object:<name>:<detail>` -- an `object` declaration whose shape
    no target can read."""
    if explained:
        return "an `object` declaration does not give a shape the target can read"
    return f"{_cannot(backend, 'read this object declaration')}: {tail}"


def _render_object_expr_kind(backend: str, tail: str, explained: bool) -> str:
    """`object-expr:<form>[:<detail>]` -- a type expression in an axis
    position whose size is not statically known."""
    form, _, rest = tail.partition(":")
    headline = (
        f"the type expression in this axis position (a `{form}`) has "
        f"no statically known size, so no target can size the plate "
        f"it becomes"
    )
    if explained or not rest:
        return (
            f"{headline}. Declare the axis as an object with a fixed "
            f"cardinality, such as `object A : FinSet 3` or an enum "
            f"set."
        )
    return f"{headline}: {rest}"


def _render_object_bounds_kind(backend: str, tail: str, explained: bool) -> str:
    """`object-bounds:<reason>:<name>[:...]` -- a `{low=..., high=...}`
    box that does not reach the target as one declared support."""
    reason, _, rest = tail.partition(":")
    name = rest.partition(":")[0]
    headline = (
        f"the bounds on `{name}` do not reach {backend} as a single "
        f"declared support ({reason.replace('-', ' ')})"
    )
    if explained:
        return headline
    return f"{headline}: {tail}"


def _render_class_index_kind(backend: str, tail: str, explained: bool) -> str:
    """`class-index:<name>:<reason>:...` -- a categorical draw whose
    alphabet width the program states two ways."""
    name, _, rest = tail.partition(":")
    reason = rest.partition(":")[0]
    headline = (
        f"the categorical draw `{name}` does not name one alphabet "
        f"({reason.replace('-', ' ')}), so no target can size the "
        f"probability vector it scores against"
    )
    if explained:
        return headline
    return f"{headline}: {rest}"


def _render_sample_kind(backend: str, tail: str, explained: bool) -> str:
    """`sample:<reason>[:...]` -- a `sample` step the target's site
    form cannot carry."""
    if "destructuring-tuple" in tail:
        return (
            "a `sample` step binds more than one name, and a target "
            "site is one variable. Split it into one `sample` per "
            "bound name."
        )
    if tail.startswith("no-enclosing-body"):
        return (
            f"a `sample` step appears where {backend} has no model "
            f"body to attach it to. Move the draw inside the "
            f"program's body."
        )
    if explained:
        return _cannot(backend, "emit this `sample` step")
    return f"{_cannot(backend, 'emit this `sample` step')}: {tail}"


def _render_observe_kind(backend: str, tail: str, explained: bool) -> str:
    """`observe:<reason>` -- an `observe` step the target cannot emit
    as one observed variable."""
    if tail.startswith("multiple-vars"):
        return (
            "an `observe` step names more than one observed variable, "
            "and a target statement observes one. Split it into one "
            "`observe` per variable."
        )
    if explained:
        return _cannot(backend, "emit this `observe` step")
    return f"{_cannot(backend, 'emit this `observe` step')}: {tail}"


def _render_family_ref_kind(backend: str, tail: str, explained: bool) -> str:
    """`family_ref:<name>` / `family-ref:<reason>:<name>` -- an
    argument naming a morphism that resolves to no distribution."""
    first, _, rest = tail.partition(":")
    if first in {"unresolved", "unknown-family"}:
        name = rest.partition(":")[0]
        return (
            f"the argument names `{name}`, which resolves to no "
            f"distribution {backend} has, so there is nothing to put "
            f"in that argument position. Give the referenced morphism "
            f"a `~ Family(args)` init clause naming a family this "
            f"target supports."
        )
    if explained:
        return (
            f"the argument names the morphism `{first}`, which "
            f"resolves to no distribution to put in that position"
        )
    return (
        f"the argument names the morphism `{first}`, which resolves to "
        f"no distribution {backend} can put in that position: {rest}. "
        f"Give the morphism a `~ Family(args)` init clause."
    )


def _render_morphism_kind(backend: str, tail: str, explained: bool) -> str:
    """`morphism:<reason>:<name>` -- a morphism the resolver cannot
    turn into a distribution."""
    reason, _, rest = tail.partition(":")
    name = rest.partition(":")[0]
    if reason == "cycle":
        headline = (
            f"resolving morphism `{name}` runs in a cycle, so it names no distribution"
        )
    elif reason == "no-init":
        headline = (
            f"morphism `{name}` declares neither a `~ Family(...)` nor "
            f"a `~ <expr>` init clause, so it names no distribution to "
            f"draw from"
        )
    elif reason == "duplicate":
        headline = f"morphism `{name}` is declared twice"
    else:
        headline = _cannot(backend, f"resolve morphism `{name}`")
    if explained:
        return headline
    return f"{headline}: {tail}"


def _render_nested_distribution_kind(backend: str, tail: str) -> str:
    """`nested-distribution-arg:<F>` -- one draw is passed a whole
    distribution as an argument."""
    family = tail.partition(":")[0]
    return (
        f"a draw takes the distribution `{family}` itself as an "
        f"argument. A target argument position holds a number, a "
        f"vector, or a name bound to one; it does not hold a "
        f"distribution. Draw from `{family}` in a `sample` step of its "
        f"own and pass the drawn name, or use a wrapper family that "
        f"names its base distribution."
    )


def _render_index_kind(backend: str, tail: str, explained: bool) -> str:
    """`index:<IRArgClass>` / `user-index:<IRArgClass>` /
    `subscript-arg:<IRArgClass>` -- a subscript expression the target's
    index position cannot take."""
    if explained:
        return _cannot(backend, f"take {_arg_surface(tail)} as a subscript")
    return (
        f"{_cannot(backend, f'take {_arg_surface(tail)} as a subscript')}. "
        f"A target subscript is an integer literal or a name bound to "
        f"one; bind the expression with a `let` and subscript by that "
        f"name."
    )


def _render_arg_class_kind(backend: str, tail: str, where: str) -> str:
    """`arg-kind:` / `inner-arg-kind:` / `arg-atom:` / `raw-arg:` --
    an argument variant the target's argument emitter has no case for.
    `where` names the position so the message says which one."""
    return (
        f"{_cannot(backend, f'take {_arg_surface(tail)} {where}')}. "
        f"Bind the value with a `let` or as a data input and pass that "
        f"name instead."
    )


def _render_support_kind(backend: str, tail: str, explained: bool) -> str:
    """`support:<ConstraintClass>` -- a variable whose value space the
    target has no storage type for."""
    constraint = tail.partition(":")[0]
    headline = _has_no(
        backend,
        f"storage type for a variable supported on `{constraint}`",
    )
    if explained:
        return headline
    return (
        f"{headline}. Declare the object with a support the target "
        f"has: a real scalar, a real vector or matrix, or a bounded "
        f"integer."
    )


def _render_truncated_kind(backend: str, tail: str, explained: bool) -> str:
    """`truncated:<reason>` / `truncation:<reason>` -- a truncated draw
    the target's `T(lo, hi)` form cannot carry."""
    if tail.startswith("base:"):
        rest = tail.removeprefix("base:")
        name = rest.partition(":")[0]
        return (
            f"the truncated draw is built over `{name}`, which names "
            f"no `~ Family(args)` declaration, so there is no base "
            f"distribution to truncate. Declare `morphism {name} : "
            f"Dom -> Cod ~ Family(args)`."
        )
    if tail.startswith("missing-bounds"):
        return (
            f"the truncated draw carries no bounds, and "
            f"{_language(backend)}'s truncation form needs at least "
            f"one. Give the draw a `low=` or a `high=`."
        )
    if tail.startswith("expected"):
        return (
            f"the truncated draw does not have the shape "
            f"{_language(backend)}'s truncation form takes ({tail}). "
            f"Write the truncation as `Truncated(Base, low, high)` "
            f"over a morphism with a `~ Family(args)` init."
        )
    if explained:
        return _cannot(backend, "emit this truncated draw")
    return f"{_cannot(backend, 'emit this truncated draw')}: {tail}"


def _render_wrapper_family_kind(backend: str, tail: str) -> str:
    """`wrapper-family:<F>` -- a wrapper distribution the target has no
    construction for."""
    family = tail.partition(":")[0]
    return (
        f"{_has_no(backend, f'construction for the wrapper family `{family}`')}. "
        f"Write the wrapped draw directly against a family this target "
        f"supports."
    )


def _render_batch_rank_kind(backend: str, tail: str, explained: bool) -> str:
    """`batch-rank:<n>` -- more plate axes than the target's site form
    carries."""
    rank = tail.partition(":")[0]
    headline = (
        f"the site is plated over {rank} axes, and a {_language(backend)} "
        f"draw carries one"
    )
    if explained:
        return headline
    return (
        f"{headline}. Flatten the axes into one product object, or "
        f"draw once per outer axis."
    )


def _render_type_expr_kind(backend: str, tail: str, explained: bool) -> str:
    """`type-expr:<spec>` -- a storage-type spelling the target's type
    grammar cannot parse."""
    spec = tail.partition(":")[0]
    headline = f"{_cannot(backend, f'read `{spec}` as a {_language(backend)} type')}"
    if explained:
        return headline
    return f"{headline}: {tail}"


def _render_internal_kind(backend: str, kind: str, explained: bool) -> str:
    """Heads that mark a broken invariant inside the pipeline rather
    than something the user wrote. The message says so, because telling
    a user to change their program would be wrong."""
    if explained:
        return (
            f"the {backend} renderer hit a broken internal invariant "
            f"while emitting this program"
        )
    return (
        f"the {backend} renderer hit a broken internal invariant while "
        f"emitting this program: {kind}. The program is not at fault; "
        f"please report it."
    )


def _render_target_kind(backend: str, tail: str, explained: bool) -> str:
    """`target:unknown:<name>` -- no backend registered under that
    name."""
    if tail.startswith("unknown"):
        rest = tail.removeprefix("unknown:")
        name, _, available = rest.partition(":")
        headline = f"there is no transpile target named `{name}`"
        if explained:
            return headline
        offered = ", ".join(
            f"`{t}`" for t in sorted(filter(None, available.split(",")))
        )
        if not offered:
            return headline
        return f"{headline}. The targets that exist are {offered}."
    return f"{_cannot(backend, 'select a target')}: {tail}"


def _render_scan_kind(backend: str, tail: str) -> str:
    """`scan:no-lowering:<cell>` -- a `scan(...)` consumed at a draw;
    `scan:marginal:<reason>:<name>` -- a marginalized chain."""
    cell = tail.rpartition(":")[2] or "the cell"
    if tail.startswith("marginal:no-lowering:"):
        return (
            f"`scan({cell}, init=p)` sums the states of a discrete Markov "
            f"chain out: `{cell}` returns the latent it marginalizes, so the "
            f"chain's likelihood is the forward algorithm's sum over every "
            f"path of states, computed from a table of the cell's weights at "
            f"each position and each pair of states. {_has_no(backend, 'lowering of that recursion')}: "
            f"its positions arrive with the data, and the cell's weights "
            f"would have to be rendered as target code at every pair of "
            f"states. Fit the model with the compiled runtime, or score it on "
            f"the reference machine with `qvr run`."
        )
    if tail.startswith("marginal:"):
        reason = tail.removeprefix("marginal:").partition(":")[0]
        return (
            f"the marginalized chain over `{cell}` is refused before "
            f"lowering ({reason}): its cell must take an entry and the "
            f"previous state, bind lets, then marginalize the next state "
            f"from a `Categorical` and observe what it emits, and return it; "
            f"the chain is drawn whole with `sample` from "
            f"`scan({cell}, init=p)` and its value is not read."
        )
    return (
        f"`scan({cell})` threads `{cell}` across the positions of a "
        f"sequence, so it denotes one draw per position over "
        f"intermediate states the program never names. Writing that "
        f"out needs a loop whose bound is the sequence length and one "
        f"sample site per position, and the sequence axis is not an "
        f"object this module declares: it arrives with the data, so "
        f"there is no extent to size the loop from and no name to "
        f"bind the per-position states to. Write the "
        f"recurrence as a program over a declared axis, giving each "
        f"position its own `sample` step, or unroll the chain into one "
        f"`sample` per step."
    )


def _render_transform_kind(backend: str, tail: str) -> str:
    """`transform:<name>` -- an argument transform the target has no
    expression for."""
    name = tail.partition(":")[0]
    return (
        f"{_has_no(backend, f'expression for the `{name}` argument transform')}; "
        f"the transforms it can emit are `inv_square`, `inv`, `neg`, "
        f"`log` and `exp`. Bind the transformed value with a `let` "
        f"whose body uses those, and pass the bound name."
    )


def _render_draw_arg_kind(backend: str, tail: str) -> str:
    """`draw-arg:<form>` -- a surface argument form the lowering has no
    IR argument for."""
    return (
        f"{_cannot(backend, f'take a `{tail}` in a distribution-argument position')}. "
        f"Bind the value with a `let` or as a data input and pass that "
        f"name instead."
    )


def _render_data_kind(backend: str, tail: str) -> str:
    """`data:per-row-vector-parameter:<name>: ...`: a data input that
    gives a vector-valued family parameter one vector per plate row.

    Parameters
    ----------
    backend : str
        The target.
    tail : str
        The kind after `data:`.

    Returns
    -------
    str
        The explanation.
    """
    form, _, rest = tail.partition(":")
    name = rest.partition(":")[0] or "the input"
    if form == "per-row-vector-parameter":
        return (
            f"The data input `{name}` fills a vector-valued family "
            f"parameter (a mean vector, a concentration, a probability "
            f"vector) with one vector per row of the draw's plate, so it "
            f"is a matrix whose rows the plated draw reads one at a time. "
            f"{_cannot(backend, 'hand a plated draw a different parameter vector at each row')}: "
            f"its plated draw takes one parameter vector shared by every "
            f"row. Pass one vector shared by every row, or transpile to "
            f"edward2, numpyro or pyro, which take one vector per row."
        )
    return (
        f"{_cannot(backend, 'transpile this program')}. The refusal is "
        f"tagged `data:{tail}`, which has no explanation registered "
        f"yet; please report it."
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


#: Heads whose refusal is an internal-invariant break rather than
#: anything the user's program chose.
_INTERNAL_HEADS: frozenset[str] = frozenset(
    {
        "ctx",
        "arg-names-mismatch",
        "prob-arg-complement",
        "rate-arg-invert",
    }
)


def _split_explanation(kind: str) -> tuple[str, str]:
    """Split a kind into its structured part and its raise-site prose.

    Structured segments never contain a space, so the first
    colon-*space* in the kind is the boundary. A kind with no
    colon-space is all structure.
    """
    structured, separator, explanation = kind.partition(": ")
    if not separator:
        return kind, ""
    return structured, explanation.strip()


def _render_kind(
    backend: str,
    kind: str,
    declarations: tuple[RefusedDeclaration, ...],
    module_has_program: bool,
) -> str:
    """One bullet of a refusal: what was refused, where, why this
    target cannot take it, and what to write instead."""
    structured, explanation = _split_explanation(kind)
    head, _, tail = structured.partition(":")
    explained = bool(explanation)
    headline = _render_head(
        backend,
        head,
        tail,
        structured,
        explained,
        declarations,
        module_has_program,
    )
    if explanation:
        return f"{headline}: {explanation}"
    return headline


def _render_head(
    backend: str,
    head: str,
    tail: str,
    structured: str,
    explained: bool,
    declarations: tuple[RefusedDeclaration, ...],
    module_has_program: bool,
) -> str:
    """Dispatch one kind's head to the renderer that owns it."""
    if head in _DECLARATION_MEANING:
        entries = tuple(e for e in declarations if e.kind == head)
        return _render_declaration_kind(backend, head, entries, module_has_program)
    if head in _INTERNAL_HEADS:
        return _render_internal_kind(backend, structured, explained)
    if head == "family":
        return _render_family_kind(backend, tail, explained)
    if head == "node":
        return _render_node_kind(backend, tail, explained)
    if head == "qiec":
        return _render_qiec_kind(backend, tail)
    if head == "call":
        return _render_call_kind(backend, tail)
    if head == "declare":
        return _render_declare_kind(backend, tail)
    if head == "arg":
        return _render_arg_kind(backend, tail, explained)
    if head == "let-expr":
        return _render_let_expr_kind(backend, tail, explained)
    if head == "let":
        return _render_let_kind(backend, tail, explained)
    if head == "define":
        return _render_define_kind(backend, tail, explained)
    if head == "return":
        return _render_return_kind(backend, tail)
    if head == "broadcast":
        return _render_broadcast_kind(backend, tail, explained)
    if head == "step":
        return _render_step_kind(backend, tail, explained)
    if head == "axes":
        return _render_axes_kind(backend, tail, explained)
    if head == "axis":
        return _render_axis_kind(backend, tail, explained)
    if head == "option":
        return _render_option_kind(backend, tail)
    if head == "marginalize":
        return _render_marginalize_kind(backend, tail, explained)
    if head == "dim":
        return _render_dim_kind(backend, tail, explained)
    if head == "dim-kind":
        return _render_dim_class_kind(backend, tail)
    if head == "plate":
        return _render_plate_kind(backend, tail, explained)
    if head == "param-source":
        return _render_param_source_kind(backend, tail, explained)
    if head == "embed":
        return _render_embed_kind(backend, tail, explained)
    if head == "program":
        return _render_program_kind(backend, tail, explained)
    if head == "program-domain":
        return _render_program_domain_kind(backend, tail, explained)
    if head == "object":
        return _render_object_kind(backend, tail, explained)
    if head == "object-expr":
        return _render_object_expr_kind(backend, tail, explained)
    if head == "object-bounds":
        return _render_object_bounds_kind(backend, tail, explained)
    if head == "class-index":
        return _render_class_index_kind(backend, tail, explained)
    if head == "sample":
        return _render_sample_kind(backend, tail, explained)
    if head == "observe":
        return _render_observe_kind(backend, tail, explained)
    if head in {"family_ref", "family-ref"}:
        return _render_family_ref_kind(backend, tail, explained)
    if head == "morphism":
        return _render_morphism_kind(backend, tail, explained)
    if head == "nested-distribution-arg":
        return _render_nested_distribution_kind(backend, tail)
    if head in {"index", "user-index", "subscript-arg"}:
        return _render_index_kind(backend, tail, explained)
    if head == "arg-kind":
        return _render_arg_class_kind(
            backend, tail, "in a distribution-argument position"
        )
    if head == "inner-arg-kind":
        return _render_arg_class_kind(
            backend, tail, "inside a list, matrix, or broadcast argument"
        )
    if head == "arg-atom":
        return _render_arg_class_kind(
            backend, tail, "as an element of a list or matrix argument"
        )
    if head == "raw-arg":
        return _render_arg_class_kind(
            backend, tail, "as an argument of a referenced morphism"
        )
    if head == "reciprocal":
        return _render_arg_class_kind(
            backend,
            tail.removeprefix("arg-kind:"),
            "in an argument position it has to invert",
        )
    if head == "support":
        return _render_support_kind(backend, tail, explained)
    if head in {"truncated", "truncation"}:
        return _render_truncated_kind(backend, tail, explained)
    if head == "wrapper-family":
        return _render_wrapper_family_kind(backend, tail)
    if head == "batch-rank":
        return _render_batch_rank_kind(backend, tail, explained)
    if head == "type-expr":
        return _render_type_expr_kind(backend, tail, explained)
    if head == "target":
        return _render_target_kind(backend, tail, explained)
    if head == "scan":
        return _render_scan_kind(backend, tail)
    if head == "transform":
        return _render_transform_kind(backend, tail)
    if head == "draw-arg":
        return _render_draw_arg_kind(backend, tail)
    if head == "data":
        return _render_data_kind(backend, tail)
    return (
        f"{_cannot(backend, 'transpile this program')}. The refusal is "
        f"tagged `{structured}`, which has no explanation registered "
        f"yet; please report it."
    )


def user_facing_message(
    target: str,
    kinds: tuple[str, ...],
    declarations: tuple[RefusedDeclaration, ...] = (),
    module_has_program: bool = False,
) -> str:
    """Render a refusal's `kinds` as the message the user reads.

    One kind renders as one sentence-or-more; several render as a
    headed bullet list, one bullet per independent problem.
    """
    backend = _backend_display_name(target)
    lines = [_render_kind(backend, k, declarations, module_has_program) for k in kinds]
    if len(lines) == 1:
        return lines[0]
    bullet = "\n  - "
    return f"{_cannot(backend, 'transpile this program')}:{bullet}{bullet.join(lines)}"
