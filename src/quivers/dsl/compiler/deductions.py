"""Compiler mixin: deduction systems and lexicon loading."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from pathlib import Path

import torch
import torch.nn as nn

from quivers.core.algebras import BOOLEAN
from quivers.dsl.ast_nodes import (
    DeductionDecl,
    LetExprNode,
    LetExprVar,
    LetStep,
    LexiconCategoryFixed,
    LexiconCategoryRestricted,
    LexiconCategoryWildcard,
    ObjectEffectApply,
    ObjectProduct,
    ObjectSlash,
    ProgramDecl,
    TypeName,
)
from quivers.dsl.ast_nodes import (
    LetExprBinOp as _LetBinOp,
)
from quivers.dsl.ast_nodes import (
    LetExprCall as _LetCall,
)
from quivers.dsl.ast_nodes import (
    LetExprIndex as _LetIndex,
)
from quivers.dsl.ast_nodes import (
    LetExprLambda as _LetLambda,
)
from quivers.dsl.ast_nodes import (
    LetExprList as _LetList,
)
from quivers.dsl.ast_nodes import (
    LetExprVar as _LetVar,
)
from quivers.dsl.ast_nodes import (
    LexiconEntry as _LexiconEntryAst,
)
from quivers.dsl.ast_nodes._shared import OptionName as _OptionName
from quivers.dsl.compiler._deduction_positions import (
    is_pattern_variable,
    misplaced_positions,
    position_slots,
    rule_position_variables,
)
from quivers.dsl.compiler._deduction_productions import (
    NORMALIZE_OPTION,
    normalization_mode,
    production_groups,
)
from quivers.dsl.compiler._deduction_terms import (
    LET_BOUND_VARIABLES_KEY,
    LET_CONSTRUCTORS_KEY,
    RESERVED_TERM_TAGS,
    TERM_ATOM_TAG,
    TERM_LEAF_TAGS,
    atom_term,
    variable_name,
    variable_term,
)
from quivers.dsl.compiler._options import (
    check_option_keys,
    find_option,
    get_option_flag,
    get_option_float,
    get_option_int,
    get_option_name,
)
from quivers.dsl.compiler._prelude import CompileError
from quivers.dsl.compiler.programs import _ProgramsMixin
from quivers.dsl.parser import parse as _parse_qvr
from quivers.stochastic.agenda import (
    Bindings,
    DeductionSystem,
    InferenceRule,
    Wildcard,
    cky_agenda,
    depth_first_agenda,
    semi_naive_agenda,
)
from quivers.stochastic.agenda import instantiate as _instantiate
from quivers.stochastic.productions import (
    Production,
    ProductionKey,
    ProductionWeights,
)
from quivers.stochastic.semiring import (
    BOOLEAN as SEMIRING_BOOLEAN,
)
from quivers.stochastic.semiring import (
    COUNTING as SEMIRING_COUNTING,
)
from quivers.stochastic.semiring import (
    LOG_PROB as SEMIRING_LOG_PROB,
)
from quivers.stochastic.semiring import (
    VITERBI as SEMIRING_VITERBI,
)

# Category-side pattern carried by a lexicon entry: either a
# wildcard variable (any identifier not declared as an atom) or a
# nested structural pattern over atoms (the result of
# ``_convert_pattern`` walking a ObjectExpr).
type LexiconPattern = Wildcard | tuple[str | int | "LexiconPattern", ...]

# Logical-form payload attached to a lexicon entry: the result of
# evaluating the entry's ``lf`` template under the empty
# environment.  The let-expression evaluator can return any value
# the program theory admits at the LF position; the concrete
# union below enumerates the possibilities.
type LexiconLF = (
    torch.Tensor
    | int
    | float
    | bool
    | str
    | tuple["LexiconLF", ...]
    | Callable[[dict[str, "LexiconLF"]], "LexiconLF"]
)

# A lexicon entry quadruple ``(word, category_pattern, lf,
# learnable_flag)``.
type LexiconEntry = tuple[str, LexiconPattern, LexiconLF, bool]

# Closed option-key sets for the deduction surface. Each set lives
# next to the code that consumes it; `check_option_keys`
# rejects anything outside the set with a did-you-mean diagnostic.
#
# * Deduction blocks read the ``semiring``, the ``axioms`` source,
#   the ``start`` goal symbol, the ``depth`` bound, the Kleene-star
#   ``tolerance``, the agenda ``max_iterations``, and the attached
#   item ``signature`` / ``encoder``, and whether the weights are
#   normalized per left-hand side (``normalize``).
# * Sequent-rule pragmas read the ``learnable`` / ``bounded`` flags
#   and the ``parent`` composition reference.
# * Lexicon-entry pragmas (inline and ``from "file"``) read only the
#   ``learnable`` flag.
_DEDUCTION_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "semiring",
        "axioms",
        "start",
        "depth",
        "tolerance",
        "max_iterations",
        "signature",
        "encoder",
        NORMALIZE_OPTION,
    }
)
_SEQUENT_RULE_OPTION_KEYS: frozenset[str] = frozenset(
    {"learnable", "parent", "bounded"}
)
_LEXICON_ENTRY_OPTION_KEYS: frozenset[str] = frozenset({"learnable"})


def _words_display(entry: _LexiconEntryAst) -> str:
    """Human-readable form of a lexicon entry's word tuple."""
    return ", ".join(repr(w) for w in entry.words)


def _candidate_atoms(
    entry: _LexiconEntryAst,
    decl: DeductionDecl,
    atoms_set: set[str],
) -> tuple[str, ...]:
    """Return the atoms over which a lexicon entry's latent category
    distribution ranges.

    * `LexiconCategoryFixed`     -> empty tuple (caller emits
      a single axiom with the fixed category instead of expanding).
    * `LexiconCategoryWildcard`  -> every atom on the
      enclosing deduction.
    * `LexiconCategoryRestricted`-> the listed atoms; each
      atom must appear in the deduction's atom set or compilation
      raises.
    """
    if isinstance(entry.category, LexiconCategoryFixed):
        return ()
    if isinstance(entry.category, LexiconCategoryWildcard):
        return tuple(decl.atoms)
    if isinstance(entry.category, LexiconCategoryRestricted):
        unknown = [a for a in entry.category.atoms if a not in atoms_set]
        if unknown:
            raise CompileError(
                f"deduction {decl.name!r}: lexicon entry for "
                f"{_words_display(entry)} restricts category to "
                f"{list(entry.category.atoms)!r}, but the atom(s) "
                f"{unknown!r} are not declared on the deduction",
                entry.line,
                entry.col,
            )
        return tuple(entry.category.atoms)
    raise CompileError(
        f"deduction {decl.name!r}: lexicon entry for {_words_display(entry)} "
        f"has an unknown category kind {type(entry.category).__name__!r}",
        entry.line,
        entry.col,
    )


def _category_depth(value: Hashable) -> int:
    """Constructor-tree depth of a chart category or item.

    Atoms, bound variables, integer literals, and positions count as
    depth 0; the tagged pairs ``("atom", "S")``, ``("var", "#v1")``,
    and ``("literal", 3)`` are leaves. A ``span`` item has the depth of
    its category. A product ``("product", (c_1, ..., c_n))`` and any
    other wrapping tuple ``(<ctor>, <args>...)`` count as
    :math:`1 + \\max_i \\text{depth}(\\text{args}_i)`. Used to gate
    rules that would otherwise rewrite ``A`` into ``Dia(A)`` or
    ``Cont(A)`` ad infinitum; the checked QIEC search measures its
    conclusions the same way.

    Parameters
    ----------
    value : Hashable
        A chart item, category, or field of one.

    Returns
    -------
    int
        The depth.
    """
    if not isinstance(value, tuple):
        return 0
    if not value:
        return 0
    head = value[0]
    if head in TERM_LEAF_TAGS or head == "literal":
        return 0
    if head == "product" and len(value) == 2 and isinstance(value[1], tuple):
        return 1 + max((_category_depth(v) for v in value[1]), default=0)
    if head == "span":
        # span(I, J, C, ...): depth of the surrounding span is the
        # depth of its category subterm, ignoring positional
        # indices.
        sub = value[3] if len(value) > 3 else None
        return _category_depth(sub) if sub is not None else 0
    # Generic structural constructor: count one and recurse.
    return 1 + max(
        (_category_depth(v) for v in value[1:]),
        default=0,
    )


def _with_depth_guard(rule: InferenceRule, depth_n: int) -> InferenceRule:
    """Wrap ``rule.side_condition`` with a category-depth guard.

    The guard rejects firings whose *conclusion category* would
    have constructor depth strictly greater than ``depth_n`` after
    pattern instantiation. Composes with any pre-existing side
    condition the rule already carries (``AND``).

    Parameters
    ----------
    rule
        The inference rule whose conclusion depth is bounded.
    depth_n
        The largest accepted constructor depth.

    Returns
    -------
    InferenceRule
        A rule carrying the conjunction of both conditions.
    """

    existing = rule.side_condition
    conclusion_pattern = rule.conclusion

    def _depth_ok(bindings: Bindings) -> bool:
        """Check the rule's condition and the instantiated conclusion's depth.

        Parameters
        ----------
        bindings
            The variable bindings from matching the rule's premises.

        Returns
        -------
        bool
            Whether the existing condition and depth bound accept the match.
        """
        if existing is not None and not existing(bindings):
            return False
        try:
            instantiated = _instantiate(conclusion_pattern, bindings)
        except KeyError:
            return True  # let the engine handle the missing binding downstream
        return _category_depth(instantiated) <= depth_n

    return rule.with_(side_condition=_depth_ok)


def load_lexicon_tsv(
    path: str, decl: DeductionDecl
) -> list[tuple[str, TypeName, LetExprNode]]:
    """Read a lexicon file into words, categories, and logical forms.

    Each row has three tab-separated columns: the word, the category,
    an atom name, and the logical form, a let expression; blank lines
    and lines starting with ``#`` are skipped. The logical form is read
    by the live grammar, as the let step of a synthetic program, so the
    file's syntax is the module's.

    Parameters
    ----------
    path : str
        The file, absolute or relative to the working directory.
    decl : DeductionDecl
        The declaring deduction, for diagnostics.

    Returns
    -------
    list[tuple[str, TypeName, LetExprNode]]
        One row per entry: the word, the category as a type name, and
        the logical form's expression.

    Raises
    ------
    CompileError
        If the file is missing, a row has fewer than three columns, or
        a logical form does not parse.
    """
    file = Path(path)
    if not file.exists():
        raise CompileError(
            f"deduction {decl.name!r}: lexicon file {path!r} not found",
            decl.line,
            decl.col,
        )
    rows: list[tuple[str, TypeName, LetExprNode]] = []
    with file.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise CompileError(
                    f"deduction {decl.name!r}: lexicon file "
                    f"{path!r}:{lineno}: expected 3 tab-separated "
                    f"columns (word, category, lf), got {len(parts)}",
                    decl.line,
                    decl.col,
                )
            word, cat_text, lf_text = parts[0], parts[1], parts[2]
            category = TypeName(name=cat_text)
            if "(" not in lf_text:
                rows.append((word, category, LetExprVar(name=lf_text)))
                continue
            syn_src = (
                "object LexTerm : FinSet 1\n"
                "morphism lex_source : LexTerm -> LexTerm "
                "[role=parameter]\n"
                "program lex_program : LexTerm -> LexTerm\n"
                "    sample lex_value : LexTerm <- lex_source\n"
                f"    let lex_lf = {lf_text}\n"
                "    return lex_value\n"
            )
            syn_mod = _parse_qvr(syn_src.encode(), "<lex-lf>")
            prog = next(
                (
                    s
                    for s in syn_mod.statements
                    if isinstance(s, ProgramDecl) and s.name == "lex_program"
                ),
                None,
            )
            if prog is None:
                raise CompileError(
                    f"deduction {decl.name!r}: lexicon file "
                    f"{path!r}:{lineno}: LF template {lf_text!r} "
                    f"did not parse to the synthetic program",
                    decl.line,
                    decl.col,
                )
            let_step = prog.draws[1]
            if not isinstance(let_step, LetStep):
                raise CompileError(
                    f"deduction {decl.name!r}: lexicon file "
                    f"{path!r}:{lineno}: LF template {lf_text!r} "
                    f"did not parse to a let binding",
                    decl.line,
                    decl.col,
                )
            rows.append((word, category, let_step.value))
    return rows


#: The weight key of a rule binding no non-position variable.
BASE_WEIGHT_KEY = "BASE"


def _bindings_key(bindings: dict) -> str:
    """Stable string key over a binding map.

    `nn.ParameterDict` keys must be valid Python identifiers (no
    dots, no spaces) under PyTorch's parameter-name validator; we
    thus canonicalise the binding map to a sorted, JSON-flavoured
    string and replace structural punctuation with underscore
    sequences. The function is total over any hashable, JSON-stable
    binding values the runtime emits (atoms as `("atom", "S")`,
    slashed categories as nested tuples, integers, …). The caller
    passes only the non-position bindings; the empty map keys as
    `BASE_WEIGHT_KEY`.

    Parameters
    ----------
    bindings : dict
        The variables to key by and their ground values.

    Returns
    -------
    str
        The key.
    """

    def encode(v) -> str:
        if isinstance(v, tuple):
            return "T" + "_".join(encode(x) for x in v) + "E"
        if isinstance(v, str):
            return "S" + v.replace("_", "__")
        if isinstance(v, int):
            return f"I{v}"
        if isinstance(v, float):
            return f"F{v}"
        return "Q" + repr(v).replace(" ", "")

    items = sorted(bindings.items(), key=lambda kv: kv[0])
    return "_".join(f"{k}_{encode(v)}" for k, v in items) or BASE_WEIGHT_KEY


def _pattern_wildcards(pattern: LexiconPattern | str | int) -> set[str]:
    """The wildcard names an agenda pattern mentions.

    Parameters
    ----------
    pattern : LexiconPattern | str | int
        A converted rule pattern: a `Wildcard`, a ground leaf, or a
        tuple of patterns.

    Returns
    -------
    set[str]
        The names.
    """
    if isinstance(pattern, Wildcard):
        return {pattern.name}
    if isinstance(pattern, tuple):
        return {name for part in pattern for name in _pattern_wildcards(part)}
    return set()


def _make_rule_weight_fn(
    *,
    rule_name: str,
    positions: frozenset[str],
    rule_parent: dict[str, str],
    rule_param_dicts: dict[str, nn.ParameterDict],
    rule_learnable: dict[str, bool],
    rule_bounded: dict[str, bool],
    n_bounded: int = 1,
) -> Callable:
    """Build the `weight_fn` for a learnable / parented rule.

    Semantics:

    * The rule's conclusion weight is
      ``semiring.times(*premise_weights, rule_log_weight)`` where
      ``rule_log_weight`` is one ``nn.Parameter`` per distinct binding
      of the rule's non-position variables (categories and logical
      forms), keyed by `_bindings_key`. Chart positions never enter
      the key, so the rule has one weight at every position and every
      sentence length; a rule with no non-position variable has
      exactly one weight, under `BASE_WEIGHT_KEY`.
    * If the rule declares ``parent=other``, the parent's
      ``rule_log_weight`` under the same key is added in the log
      semirings (LogProb / Viterbi), and so on up the parent chain.
    * A rule in the chain that is not learnable contributes no weight
      of its own; a bounded rule's raw parameter is mapped below zero
      by its own ``bounded`` flag.

    Parameters
    ----------
    rule_name : str
        The rule.
    positions : frozenset[str]
        The rule's position variables, which the key leaves out.
    rule_parent : dict[str, str]
        Each parented rule's parent.
    rule_param_dicts : dict[str, nn.ParameterDict]
        Each learnable or parented rule's parameters, by key.
    rule_learnable : dict[str, bool]
        Whether each rule declares ``learnable``.
    rule_bounded : dict[str, bool]
        Whether each rule declares ``bounded``.
    n_bounded : int
        The number of bounded rules, whose log divides each bounded
        rule's per-firing factor.

    Returns
    -------
    Callable
        The weight function the agenda calls with a firing's bindings,
        its premise weights, and the semiring.

    Raises
    ------
    CompileError
        At firing time, if the parent chain cycles.
    """

    # ``bounded=True`` parameterizes the rule weight as
    # ``-softplus(raw_param) - log(n_bounded)`` where ``n_bounded``
    # counts the deduction's bounded rules. The per-firing factor is
    # then strictly below ``1 / n_bounded``, so the total mass any
    # chart item can push through the deduction's bounded rules is
    # strictly below 1. For cycles built from unary bounded rules
    # (each such rule contributes at most one out-edge per item),
    # this caps the delta-propagation operator's row sums below 1,
    # and the LogProb chart's Kleene-star series converges for every
    # parameter value. A per-rule cap of 1 alone would not suffice:
    # interlocking cycles (e.g. an introduction / elimination pair
    # over nested constructors) can diverge even when every
    # individual cycle's log-weight is negative.
    bounded_log_cap = math.log(n_bounded) if n_bounded > 1 else 0.0

    def _bounded_weight(raw: torch.Tensor) -> torch.Tensor:
        return -torch.nn.functional.softplus(raw) - bounded_log_cap

    def _rule_log_weight(key: str, rule_nm: str) -> torch.Tensor:
        own_dict = rule_param_dicts.get(rule_nm)
        if own_dict is None or not rule_learnable.get(rule_nm, False):
            # No own parameter to add; identity in the additive log
            # semiring (and in LogProb specifically).
            return torch.tensor(0.0)
        existing = own_dict.get(key)
        if existing is None:
            # Initialise ``raw_param`` to zero: for bounded rules the
            # initial log-weight is ``-log(2) - log(n_bounded)``, a
            # mildly sub-stochastic per-firing rate.
            existing = nn.Parameter(torch.zeros(()))
            own_dict[key] = existing
        return _bounded_weight(existing) if rule_bounded[rule_nm] else existing

    def weight_fn(bindings, premise_weights, semiring):
        # 1. Premise product (the default semiring-parsing aggregation).
        if not premise_weights:
            base = torch.tensor(
                float(semiring.one) if hasattr(semiring.one, "__float__") else 0.0,
                dtype=torch.get_default_dtype(),
            )
        else:
            base = premise_weights[0]
            for w in premise_weights[1:]:
                base = semiring.times(base, w)
        # 2. Own contribution. A learned weight is a log weight, so only
        # the log semirings read one; the others multiply by their one.
        # The key covers the non-position bindings only.
        key = _bindings_key(
            {name: value for name, value in bindings.items() if name not in positions}
        )
        if semiring is SEMIRING_LOG_PROB or semiring is SEMIRING_VITERBI:
            own_w = _rule_log_weight(key, rule_name)
        else:
            own_w = torch.tensor(float(semiring.one), dtype=torch.get_default_dtype())
        total = semiring.times(base, own_w)
        # 3. Parent chain (additive composition in the LogProb / Viterbi
        # log semirings; semiring-product for general K).
        cur = rule_name
        seen = {cur}
        while cur in rule_parent and (
            semiring is SEMIRING_LOG_PROB or semiring is SEMIRING_VITERBI
        ):
            parent = rule_parent[cur]
            if parent in seen:
                raise CompileError(
                    f"rule {rule_name!r}: parent chain cycles at {parent!r}",
                )
            seen.add(parent)
            # A learnable parent contributes its weight under the same
            # key; a parent that only forwards to its own parent
            # contributes nothing of its own.
            total = semiring.times(total, _rule_log_weight(key, parent))
            cur = parent
        return total

    return weight_fn


def _make_production_weight_fn(
    key: ProductionKey, productions: ProductionWeights
) -> Callable:
    """Build the `weight_fn` of a rule whose weight is a production weight.

    The conclusion weight is the semiring product of the premise weights
    and the rule's entry in the production table the run has made
    active: under ``normalize=lhs``, the rule's log-probability among
    the productions of its left-hand side.

    Parameters
    ----------
    key : ProductionKey
        The rule's production address.
    productions : ProductionWeights
        The deduction's production weights.

    Returns
    -------
    Callable
        The weight function the agenda calls with a firing's bindings,
        its premise weights, and the semiring.
    """

    def weight_fn(bindings, premise_weights, semiring):
        total = productions.weight(key)
        for weight in premise_weights:
            total = semiring.times(weight, total)
        return total

    return weight_fn


class _DeductionsMixin:
    """Mixin: deduction-block compilation methods.

    The compiler base supplies every environment slot below; the
    annotations let the type checker verify each access from a
    mixin method.
    """

    _morphisms: dict
    _deductions: dict
    _signatures: dict
    _encoders: dict

    def _lex_globals_for_structural(self) -> dict:
        """Provided by `_StructuralMixin`."""
        raise NotImplementedError

    def _compile_deduction(self, decl: DeductionDecl) -> None:
        """Compile a ``deduction { … }`` block into an agenda-engine
        `DeductionSystem` and register it under ``decl.name``.

        Translates the declarative sequent-style rules into the
        runtime's `InferenceRule` form (an identifier that is neither a
        declared atom nor a numeral is a wildcard variable; an atom's
        name is the atom, and is rejected in a chart-position slot).
        Resolves the
        semiring by name. Wires the axiom source — one of:

        * a ``lexicon { ... }`` block, compiled into a learnable
          dispatch table keyed on the input token at each position;
        * a ``lexicon from "path"`` declaration, loaded from a TSV
          file at compile time and treated identically to the
          inline form;
        * an ``axioms = source_name`` declaration, naming a
          previously-defined morphism whose callable returns a
          list of `(item, weight)` pairs given an input;
        * none of the above — the user supplies axioms directly at
          call time (identity axiom-injector).

        The result is callable as ``parse(NAME, input)`` from
        program bodies, producing a `ChartView`.

        Parameters
        ----------
        decl : DeductionDecl
            The declaration.

        Raises
        ------
        CompileError
            If the deduction is declared twice, an option is unknown or
            malformed, a rule fills a chart-position slot with an atom
            or a structured term, a ``parent`` names no rule, or the
            lexicon or an attached signature or encoder does not
            resolve.
        """
        if not hasattr(self, "_deductions"):
            self._deductions = {}

        globals_ = dict(getattr(self, "_deductions", {}))
        # The deduction's declared atomic + complex constructor
        # symbols form the user-controlled free term algebra used
        # by lexicon LF expressions, rule weights, and any other
        # let-expressions evaluated inside this deduction's scope.
        # ``atom`` and ``var`` are the reserved leaf tags; every other
        # constructor symbol remains under the user's control.
        globals_[LET_CONSTRUCTORS_KEY] = frozenset(decl.atoms)

        if decl.name in self._deductions or decl.name in self._morphisms:
            raise CompileError(
                f"deduction {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        check_option_keys(
            decl.options,
            _DEDUCTION_OPTION_KEYS,
            owner=f"deduction {decl.name!r}",
            line=decl.line,
            col=decl.col,
        )

        # Pattern-conversion: ObjectExpr -> agenda-engine Pattern.
        # The conversion is fully general; users may use any
        # type-expression shape and the runtime pattern-matcher
        # walks it structurally. Identifiers that match a declared
        # atom name become ground atoms; identifiers NOT in the
        # atoms set are treated as wildcard variables (the standard
        # Prolog / Datalog convention: declared constants are
        # ground; undeclared identifiers in patterns are variables).
        atoms_set = set(decl.atoms)

        reserved_clash = sorted((atoms_set | set(decl.binders)) & RESERVED_TERM_TAGS)
        if reserved_clash:
            names = ", ".join(repr(name) for name in reserved_clash)
            noun = "tag" if len(reserved_clash) == 1 else "tags"
            verb = "is" if len(reserved_clash) == 1 else "are"
            raise CompileError(
                f"deduction {decl.name!r}: {names} {verb} reserved "
                f"as term-algebra {noun}; nullary constants use "
                f"{atom_term('<name>')!r} and bound variables use "
                f"{variable_term('<name>')!r}",
                decl.line,
                decl.col,
            )

        frozen_atoms = frozenset(atoms_set)

        def _convert_pattern(texpr):
            if isinstance(texpr, TypeName):
                name = texpr.name
                # Variable convention, shared with the QIEC elaboration:
                # an identifier that is neither a declared atom nor a
                # numeral is a wildcard, whatever its spelling; a
                # declared atom's name is always that atom.
                if is_pattern_variable(texpr, frozen_atoms):
                    return Wildcard(name=name)
                if name.isdigit():
                    return ("literal", int(name))
                return atom_term(name)
            if isinstance(texpr, ObjectProduct):
                return (
                    "product",
                    tuple(_convert_pattern(c) for c in texpr.components),
                )
            if isinstance(texpr, ObjectSlash):
                # Categorial-grammar slash types: X/Y, X\Y.
                return (
                    texpr.direction,
                    _convert_pattern(texpr.result),
                    _convert_pattern(texpr.argument),
                )
            if isinstance(texpr, ObjectEffectApply):
                # T(X) = ("effect_apply", T_name, *X_args). The
                # constructor's args are recursively converted; this
                # encodes any structured term — proof witnesses, LF
                # constructors, dependent-type applications.
                args = tuple(_convert_pattern(a) for a in texpr.args)
                return (texpr.effect, *args)
            # Fallback: a structural-equality probe.
            return atom_term(repr(texpr))

        # Detect whether this deduction's item algebra carries an
        # LF slot. The signal: any rule pattern of the form
        # ``span(I, J, C, F)`` (i.e. a 4-arity ``span(...)`` type
        # expression) or any non-empty ``binders`` block. When
        # ``_uses_lf`` is True, span items are 5-tuples and the
        # short ``span(I, J, C)`` rule pattern is auto-extended
        # with a fresh LF wildcard; when False, span items are
        # 4-tuples and the surface is untouched.
        def _span_arity(texpr) -> int:
            if isinstance(texpr, ObjectEffectApply) and texpr.effect == "span":
                return len(texpr.args)
            return 0

        _uses_lf = (
            bool(decl.binders)
            or any(_span_arity(p) >= 4 for sr in decl.rules for p in sr.premises)
            or any(_span_arity(sr.conclusion) >= 4 for sr in decl.rules)
        )

        _lf_wildcard_counter = {"n": 0}

        def _normalise_span(pat):
            if not _uses_lf:
                return pat
            if isinstance(pat, tuple) and pat and pat[0] == "span":
                children = tuple(_normalise_span(c) for c in pat[1:])
                if len(children) == 3:
                    _lf_wildcard_counter["n"] += 1
                    fresh = Wildcard(name=f"_lf_{_lf_wildcard_counter['n']}")
                    return ("span", *children, fresh)
                return ("span", *children)
            if isinstance(pat, tuple):
                return tuple(_normalise_span(c) for c in pat)
            return pat

        semiring_registry = {
            "LogProb": SEMIRING_LOG_PROB,
            "Boolean": SEMIRING_BOOLEAN,
            "Viterbi": SEMIRING_VITERBI,
            "Counting": SEMIRING_COUNTING,
        }
        semiring_name = get_option_name(
            decl.options,
            "semiring",
            line=decl.line,
            col=decl.col,
        )
        if semiring_name is None:
            semiring = SEMIRING_LOG_PROB
        elif semiring_name in semiring_registry:
            semiring = semiring_registry[semiring_name]
        else:
            raise CompileError(
                f"deduction {decl.name!r} names the semiring {semiring_name!r}; "
                f"the semirings are {', '.join(semiring_registry)}",
                decl.line,
                decl.col,
            )

        inference_rules: list = []
        # Each rule may carry a ``#[learnable]`` pragma. A learnable
        # rule has one log-weight per distinct binding of its
        # non-position variables, owned by a per-deduction
        # ``_rule_module`` so the parameters appear in
        # ``model.parameters()`` automatically. Chart positions are
        # inferred from how the rules use each slot (see
        # `position_slots`) and never enter the key. The conclusion
        # weight is the semiring product of the premise weights and
        # the keyed rule parameter; with no learnable pragma a rule
        # reduces to the standard semiring-parsing default (semiring
        # product of premise weights only).
        slots = position_slots(decl)
        for misplaced in misplaced_positions(decl, slots):
            raise CompileError(
                misplaced.message(decl.name), misplaced.line, misplaced.col
            )
        rule_positions = {
            sr.name: rule_position_variables(decl, slots, sr) for sr in decl.rules
        }
        # The productions sharing a left-hand side, when the grouping is
        # fixed at compile time; ``normalize=lhs`` requires it.
        normalize_mode, mode_refusal = normalization_mode(decl)
        if mode_refusal is not None:
            raise CompileError(
                mode_refusal.message, mode_refusal.line, mode_refusal.col
            )
        file_rows: tuple[tuple[str, TypeName, LetExprNode], ...] = (
            tuple(load_lexicon_tsv(decl.lexicon_from_file, decl))
            if decl.lexicon_from_file is not None
            else ()
        )
        analysis = production_groups(
            decl, slots, tuple((word, category) for word, category, _ in file_rows)
        )
        if normalize_mode == "lhs" and analysis.refusals:
            first = analysis.refusals[0]
            raise CompileError(
                f"{first.message} (required by normalize=lhs)", first.line, first.col
            )
        lexicon_weights: dict[int, torch.Tensor] = {}

        def _free_weight(production: Production) -> torch.Tensor:
            if not production.learnable:
                return torch.zeros(())
            if production.kind == "rule":
                name = decl.rules[production.index].name
                return rule_param_dicts[name][BASE_WEIGHT_KEY]
            return lexicon_weights[production.index]

        productions = (
            None
            if analysis.refusals
            else ProductionWeights(
                analysis.groups, _free_weight, normalize=normalize_mode == "lhs"
            )
        )
        rule_learnable = {
            sr.name: get_option_flag(sr.options, "learnable") for sr in decl.rules
        }
        rule_bounded = {
            sr.name: get_option_flag(sr.options, "bounded") for sr in decl.rules
        }
        rule_module = nn.Module()
        rule_param_dicts: dict[str, nn.ParameterDict] = {}
        # ``parent`` pragmas form a directed graph from a
        # specialisation to its parent rule; we resolve weight
        # composition by chaining lookups at run time.
        rule_parent: dict[str, str] = {}
        rule_names_declared = {sr.name for sr in decl.rules}
        # Count the rules whose weight is bounded-reparameterised: the
        # joint cap divides each bounded rule's per-firing factor by
        # this count so their total out-flow from any item stays
        # below 1 (see `_make_rule_weight_fn`).
        n_bounded_rules = sum(
            1
            for sr in decl.rules
            if get_option_flag(sr.options, "bounded")
            and (
                get_option_flag(sr.options, "learnable")
                or find_option(sr.options, "parent") is not None
            )
        )
        for rule_index, sr in enumerate(decl.rules):
            check_option_keys(
                sr.options,
                _SEQUENT_RULE_OPTION_KEYS,
                owner=f"deduction {decl.name!r}: rule {sr.name!r}",
                line=sr.line,
                col=sr.col,
            )
            premises = tuple(_normalise_span(_convert_pattern(p)) for p in sr.premises)
            conclusion = _normalise_span(_convert_pattern(sr.conclusion))
            is_learnable = get_option_flag(sr.options, "learnable")
            parent_entry = find_option(sr.options, "parent")
            parent_name: str | None = None
            if parent_entry is not None:
                if not isinstance(parent_entry.value, _OptionName):
                    raise CompileError(
                        f"deduction {decl.name!r}: rule {sr.name!r}: "
                        f"`parent=` must name another rule "
                        f"(identifier), got {parent_entry.value!r}",
                        sr.line,
                        sr.col,
                    )
                parent_name = parent_entry.value.value
                if parent_name not in rule_names_declared:
                    raise CompileError(
                        f"deduction {decl.name!r}: rule {sr.name!r}: "
                        f"`parent={parent_name}` references unknown rule",
                        sr.line,
                        sr.col,
                    )
                rule_parent[sr.name] = parent_name
            weight_fn: Callable | None = None
            if is_learnable or parent_name is not None:
                param_dict = nn.ParameterDict()
                rule_param_dicts[sr.name] = param_dict
                rule_module.add_module(
                    f"rule_{sr.name}",
                    param_dict,
                )
                weight_fn = _make_rule_weight_fn(
                    rule_name=sr.name,
                    positions=rule_positions[sr.name],
                    rule_parent=rule_parent,
                    rule_param_dicts=rule_param_dicts,
                    rule_learnable=rule_learnable,
                    rule_bounded=rule_bounded,
                    n_bounded=max(n_bounded_rules, 1),
                )
            if productions is not None and productions.normalize:
                weight_fn = _make_production_weight_fn(
                    ("rule", rule_index), productions
                )
            inference_rules.append(
                InferenceRule(
                    name=sr.name,
                    premises=premises,
                    conclusion=conclusion,
                    weight_fn=weight_fn,
                )
            )
        # A rule binding no non-position variable has exactly one weight
        # key, so its parameters (its own and its learnable ancestors')
        # exist before any input is parsed and an optimizer built on
        # ``parameters()`` sees them.
        for rule in inference_rules:
            if rule.name not in rule_param_dicts:
                continue
            variables = {
                name
                for pattern in (*rule.premises, rule.conclusion)
                for name in _pattern_wildcards(pattern)
            }
            if variables - rule_positions[rule.name]:
                continue
            current: str | None = rule.name
            visited: set[str] = set()
            while current is not None and current not in visited:
                visited.add(current)
                owned = rule_param_dicts.get(current)
                if (
                    owned is not None
                    and rule_learnable[current]
                    and BASE_WEIGHT_KEY not in owned
                ):
                    owned[BASE_WEIGHT_KEY] = nn.Parameter(torch.zeros(()))
                current = rule_parent.get(current)

        # ---- Axiom source ----
        #
        # Resolve in priority order:
        #   1. `axioms = some_morphism` (most general).
        #   2. `lexicon { ... }` or `lexicon from "..."` (sugar
        #      for the label-indexed-lookup case).
        #   3. Identity — input itself is the axiom list.

        axioms_source = get_option_name(
            decl.options,
            "axioms",
            line=decl.line,
            col=decl.col,
        )
        if axioms_source is not None:
            # General axiom source — look up the named morphism and
            # invoke it on the input at call time. The morphism may
            # be any callable.
            src_name = axioms_source
            if src_name not in self._morphisms:
                raise CompileError(
                    f"deduction {decl.name!r}: axioms source "
                    f"{src_name!r} is not a declared morphism",
                    decl.line,
                    decl.col,
                )
            morph = self._morphisms[src_name]

            def _morphism_axioms(input_value, _morph=morph):
                # The morphism is expected to be a callable that,
                # given the input, returns a list of (item, weight)
                # pairs.
                return list(_morph(input_value))

            axiom_injector: Callable = _morphism_axioms

            axiom_module = morph if isinstance(morph, nn.Module) else None
        elif decl.lexicon or decl.lexicon_from_file is not None:
            # Lexicon-based axiom source. Build a learnable lookup
            # table keyed on the literal word string; emit one
            # axiom per matching (entry, candidate-category) pair
            # per input position. Wildcard ``*`` and restricted
            # ``{A, B, C}`` category positions expand to one axiom
            # per atom in the candidate set, each carrying its own
            # learnable weight; the Categorical distribution over
            # the entry's possible categories is the softmax over
            # these per-atom weights at fit time.
            entries: list[LexiconEntry] = []
            binders_set = frozenset(decl.binders)
            if decl.lexicon_from_file is not None:
                check_option_keys(
                    decl.lexicon_from_file_options,
                    _LEXICON_ENTRY_OPTION_KEYS,
                    owner=f"deduction {decl.name!r}: lexicon from file",
                    line=decl.line,
                    col=decl.col,
                )

            # Pre-pass: walk every lexicon entry's LF AST and collect
            # the names of variables bound by any declared binder
            # constructor. These names are then treated as
            # constructor-valued during let-expression compilation
            # (so ``Lam(x, App(bark, Var(x)))`` accepts ``x`` as a
            # bound name without requiring the user to list it in
            # ``atoms``). The subsequent alpha-renaming pass
            # (``_normalise_binders``) replaces every such name with
            # a fresh canonical symbol.

            def _collect_bound_vars(node, acc: set[str]) -> None:
                if isinstance(node, _LetCall):
                    if node.func in binders_set and node.args:
                        first = node.args[0]
                        if isinstance(first, _LetVar):
                            acc.add(first.name)
                    for a in node.args:
                        _collect_bound_vars(a, acc)
                elif isinstance(node, _LetBinOp):
                    _collect_bound_vars(node.left, acc)
                    _collect_bound_vars(node.right, acc)
                elif isinstance(node, _LetList):
                    for it in node.items:
                        _collect_bound_vars(it, acc)
                elif isinstance(node, _LetIndex):
                    _collect_bound_vars(node.array, acc)
                    for idx in node.indices:
                        _collect_bound_vars(idx, acc)
                elif isinstance(node, _LetLambda):
                    _collect_bound_vars(node.body, acc)

            bound_var_names: set[str] = set()
            for entry in decl.lexicon:
                _collect_bound_vars(entry.lf, bound_var_names)
            for _word, _category, form in file_rows:
                _collect_bound_vars(form, bound_var_names)
            # A file-backed lexicon historically allowed a bare LF symbol
            # that was not repeated in the deduction's ``atoms`` list. Keep
            # accepting that spelling, but give it the same tagged constant
            # representation as every declared atom.
            file_constant_names = {
                form.name
                for _word, _category, form in file_rows
                if isinstance(form, _LetVar)
            }
            lf_atom_names = atoms_set | file_constant_names
            # Extend the constructor set so the LF compiler accepts
            # binder constructors AND references to bound variables.
            # Binder constructors are first-class data constructors
            # (they appear as the head of LF subterms); the bound
            # variable names are constructor-like only inside their
            # scope, but at compile time we expand the set
            # uniformly and rely on the alpha-renaming pass to
            # disambiguate scopes.
            globals_[LET_CONSTRUCTORS_KEY] = (
                frozenset(decl.atoms)
                | frozenset(decl.binders)
                | frozenset(bound_var_names)
                | frozenset(file_constant_names)
            )
            globals_[LET_BOUND_VARIABLES_KEY] = frozenset(bound_var_names)
            # The LF compiler treats binder-applied terms specially:
            # the first argument of any constructor listed in
            # ``binders`` is the bound variable. We alpha-rename it
            # to a fresh canonical symbol per term construction so
            # the chart's structural identity collapses
            # alpha-equivalent terms. Implementation lives in
            # `_normalise_binders` below; the lexicon LF value
            # is the post-normalised tree.
            fresh_counter = {"n": 0}

            def _fresh() -> str:
                fresh_counter["n"] += 1
                return f"#v{fresh_counter['n']}"

            def _normalise_binders(term, env: dict):
                # Apply alpha-renaming of bound variables to fresh
                # canonical symbols. ``env`` maps original variable
                # names to their canonical replacements; tagged
                # variable references are substituted. Tagged
                # constants are never rewritten.
                if isinstance(term, tuple) and term:
                    head = term[0]
                    if head in binders_set and len(term) >= 3:
                        var_term = term[1]
                        var_name = variable_name(var_term)
                        if var_name is None:
                            # Malformed binder; pass through.
                            return tuple(_normalise_binders(x, env) for x in term)
                        canonical = _fresh()
                        new_env = dict(env)
                        new_env[var_name] = canonical
                        body_norm = tuple(
                            _normalise_binders(x, new_env) for x in term[2:]
                        )
                        return (head, variable_term(canonical), *body_norm)
                    name = variable_name(term)
                    if name is not None:
                        if name in env:
                            return variable_term(env[name])
                        if name in lf_atom_names:
                            return atom_term(name)
                        raise CompileError(
                            f"logical form has unbound variable {name!r}"
                        )
                    return tuple(_normalise_binders(x, env) for x in term)
                return term

            for entry in decl.lexicon:
                check_option_keys(
                    entry.options,
                    _LEXICON_ENTRY_OPTION_KEYS,
                    owner=(
                        f"deduction {decl.name!r}: lexicon entry for "
                        f"{_words_display(entry)}"
                    ),
                    line=entry.line,
                    col=entry.col,
                )
                lf_fn = _ProgramsMixin._compile_let_expr(entry.lf, globals_=globals_)
                # Evaluate the LF eagerly under an empty environment;
                # LF templates in lexicons must be closed expressions.
                try:
                    raw_lf_value = lf_fn({})
                    lf_value = _normalise_binders(raw_lf_value, {})
                    # Keep the original symbol for compatibility with
                    # consumers that didn't opt in to binders; the
                    # rewriter is a no-op when ``binders`` is empty.
                except CompileError as e:
                    raise CompileError(
                        f"deduction {decl.name!r}: lexicon entry for "
                        f"{_words_display(entry)} has unresolved variable: {e}",
                        entry.line,
                        entry.col,
                    ) from e
                # Determine the candidate atom set for this entry.
                # Fixed categories produce a single entry; wildcard
                # expands to every atom declared on the deduction;
                # restricted expands to the listed atoms (after
                # validating each appears in the deduction's atom set).
                candidate_atoms = _candidate_atoms(entry, decl, atoms_set)
                fixed_category = (
                    _convert_pattern(entry.category.category)
                    if isinstance(entry.category, LexiconCategoryFixed)
                    else None
                )
                learnable_flag = get_option_flag(entry.options, "learnable")
                # Plural word entries expand here: each word maps to
                # the same category pattern and logical form, with
                # its own axiom rows (and, for latent categories, its
                # own per-atom learnable weights).
                for word in entry.words:
                    if fixed_category is not None:
                        entries.append(
                            (
                                word,
                                fixed_category,
                                lf_value,
                                learnable_flag,
                            )
                        )
                    else:
                        # Wildcard / restricted: expand to one axiom
                        # per candidate atom. Force learnable=True
                        # since the point of the wildcard is to learn
                        # the distribution; the user's [learnable]
                        # flag additionally controls the LF body's
                        # parameters.
                        for atom in candidate_atoms:
                            entries.append(
                                (
                                    word,
                                    atom_term(atom),
                                    lf_value,
                                    True,
                                )
                            )
            # File-loaded lexicon: TSV with `word\tcategory\tlf` rows.
            if file_rows:
                file_globals = self._lex_globals_for_structural()
                file_globals.update(globals_)
                file_learnable = get_option_flag(
                    decl.lexicon_from_file_options,
                    "learnable",
                )
                for word, category, form in file_rows:
                    try:
                        raw_lf_value = _ProgramsMixin._compile_let_expr(
                            form, globals_=file_globals
                        )({})
                        lf_value = _normalise_binders(raw_lf_value, {})
                    except CompileError as error:
                        raise CompileError(
                            f"deduction {decl.name!r}: lexicon file "
                            f"{decl.lexicon_from_file!r} entry for {word!r} "
                            f"has unresolved variable: {error}",
                            decl.line,
                            decl.col,
                        ) from error
                    entries.append(
                        (word, atom_term(category.name), lf_value, file_learnable)
                    )
            # Allocate one learnable Parameter per learnable entry.
            # We keep the Parameter list on a small nn.Module so it
            # participates in `.parameters()` of any Program that
            # owns the deduction.
            axiom_module = nn.Module()
            param_list: list = []
            for idx, (_w, _cat, _lf, is_learnable) in enumerate(entries):
                if is_learnable:
                    p = nn.Parameter(torch.zeros(()))
                    axiom_module.register_parameter(f"lex_weight_{idx}", p)
                    param_list.append(p)
                    lexicon_weights[idx] = p
                else:
                    param_list.append(None)
            # Capture the axiom-injector as a closure over the
            # entries + parameter list. The LF side-table is
            # keyed by emitted item; consumers can recover the
            # logical form for any chart span by looking it up.
            entries_local = tuple(entries)
            params_local = tuple(param_list)
            _lf_table: dict = {}

            def _lexicon_axioms(
                input_value,
                _entries=entries_local,
                _params=params_local,
                _lf_table=_lf_table,
                _one=semiring.one,
                _productions=productions,
            ):
                # `input_value` may be a list/tuple of token strings,
                # OR a list of `(token, position)` pairs. We accept
                # bare-string lists for the common case.
                tokens = list(input_value)
                out: list = []
                for pos, tok in enumerate(tokens):
                    if isinstance(tok, tuple) and len(tok) == 2:
                        tok = tok[0]
                    for idx, (word, cat_pat, lf_val, _learn) in enumerate(_entries):
                        if word != tok:
                            continue
                        weight_param = _params[idx]
                        if _productions is not None and _productions.normalize:
                            # A normalized grammar reads the entry's
                            # log-probability among its category's
                            # productions.
                            weight_tensor = _productions.weight(("lexicon", idx))
                        elif weight_param is not None:
                            weight_tensor = weight_param
                        else:
                            # An entry with no learned weight contributes
                            # the semiring's multiplicative identity.
                            weight_tensor = torch.tensor(float(_one))
                        # Span items: 4-tuple ``("span", i, j,
                        # cat)`` when no rule references an LF
                        # slot; 5-tuple ``("span", i, j, cat, lf)``
                        # when at least one rule or the
                        # ``binders`` block makes the LF visible.
                        # The shape is fixed per deduction and
                        # detected once at compile time below.
                        if _uses_lf:
                            item = ("span", pos, pos + 1, cat_pat, lf_val)
                        else:
                            item = ("span", pos, pos + 1, cat_pat)
                        out.append((item, weight_tensor))
                        _lf_table[item] = lf_val
                return out

            axiom_injector = _lexicon_axioms
        else:
            # Identity injector — input is already a list of axioms.
            def _given_axioms(input_value):
                if isinstance(input_value, list):
                    return input_value
                return list(input_value)

            axiom_injector = _given_axioms

            axiom_module = None

        # Goal: items matching the start symbol's atom form for
        # top-level spans. Users override by composing the parse
        # result with their own predicate.
        start = get_option_name(
            decl.options,
            "start",
            line=decl.line,
            col=decl.col,
        )

        def _goal(item) -> bool:
            if start is None:
                return True
            if not (isinstance(item, tuple) and len(item) > 0):
                return False
            # Three goal-item shapes the framework recognizes by
            # default; users can override via a custom goal
            # predicate (the `axioms = source_kernel` escape hatch
            # composes with an arbitrary `goal` field on the
            # underlying DeductionSystem).
            head = item[0]
            # 1. Bare atom: ("atom", "S").
            if head == TERM_ATOM_TAG and len(item) == 2 and item[1] == start:
                return True
            # 2. Head-keyed (Datalog-style): ("reach", ...).
            if isinstance(head, str) and head == start:
                return True
            # 3. CKY-shaped span: ("span", i, j, ("atom", "S"), lf).
            if head == "span" and len(item) >= 4:
                cat = item[3]
                if (
                    isinstance(cat, tuple)
                    and len(cat) == 2
                    and cat[0] == TERM_ATOM_TAG
                    and cat[1] == start
                ):
                    return True
            return False

        # Choose an agenda strategy. Default to CKY for
        # context-free-shaped systems; depth-first for proof
        # search (Boolean semiring with rule-arity-2-or-less);
        # semi-naive for Datalog-shaped (no aggregation needed).
        if semiring is BOOLEAN and any(len(r.premises) == 1 for r in inference_rules):
            agenda_factory = depth_first_agenda
        elif semiring is BOOLEAN:
            agenda_factory = semi_naive_agenda
        else:
            agenda_factory = cky_agenda

        # Optional category-depth bound on the conclusion of every
        # rule. Rules like ``dia_intro : span(I, J, A) |- span(I,
        # J, Dia(A))`` add a constructor layer per firing and would
        # otherwise grow chart categories without bound. We attach
        # a uniform `InferenceRule.side_condition` to every
        # rule that gates firings on the *conclusion category's*
        # constructor-tree depth: items whose depth would exceed
        # the user-supplied ``depth=N`` option are dropped. The
        # measure counts every wrapping ``(<ctor>, ...)`` tuple
        # whose head is a category constructor as one unit of depth.
        depth_bound = get_option_int(
            decl.options,
            "depth",
            line=decl.line,
            col=decl.col,
        )
        if depth_bound is not None:
            depth_n = int(depth_bound)
            inference_rules = [_with_depth_guard(r, depth_n) for r in inference_rules]

        # ``tolerance=epsilon`` option exposes Kleene-star
        # convergence semantics for non-idempotent semirings with
        # cyclic rule graphs: when set, the chart's
        # `insert_or_aggregate` terminates re-firings on an
        # item once successive weight updates fall below ``epsilon``.
        # Convergent cycles (cycle log-weight :math:`< 0`) reach a
        # finite fixed point under this rule; divergent cycles
        # (cycle log-weight :math:`\\ge 0`) fail the agenda's
        # ``max_iterations`` safety net, which is the correct
        # rejection of an ill-posed model.
        tolerance_opt = get_option_float(
            decl.options,
            "tolerance",
            line=decl.line,
            col=decl.col,
        )
        tol = float(tolerance_opt) if tolerance_opt is not None else 0.0
        # Compile-time analysis: a deduction whose rule graph
        # contains a strict cycle on a non-idempotent semiring
        # (LogProb / Counting / Inside) needs a positive tolerance
        # to terminate (or it diverges). We do not infer this
        # automatically — that would mask divergent models. Instead
        # the user opts in via ``tolerance=...``; without it the
        # chart insists on exact equality and either terminates
        # immediately (acyclic) or fails the safety net (cyclic
        # without tolerance), preserving the standard
        # semiring-parsing semantics.
        max_iters_opt = get_option_int(
            decl.options,
            "max_iterations",
            line=decl.line,
            col=decl.col,
        )
        system = DeductionSystem(
            name=decl.name,
            rules=tuple(inference_rules),
            semiring=semiring,
            axiom_injector=axiom_injector,
            goal=_goal,
            agenda_factory=agenda_factory,
            max_iterations=(
                int(max_iters_opt) if max_iters_opt is not None else 100_000
            ),
            tolerance=tol,
            productions=productions,
            production_refusal=(
                analysis.refusals[0].message if analysis.refusals else None
            ),
        )
        # Stash the axiom-module on the system so Programs that
        # reach it (via `parse(NAME, …)`) can include its
        # parameters in their optimizer.
        if axiom_module is not None:
            system.modules["_axiom_module"] = axiom_module
        # Stash the rule-weight module so its parameters appear in
        # ``model.parameters()`` of any Program that references
        # this deduction. Empty if no rule declared `#[learnable]`.
        if rule_param_dicts:
            system.modules["_rule_module"] = rule_module
        # Attach a signature / encoder pairing, if declared. The
        # chart-query operations (`chart.embedding(pattern)`) consult
        # this attached encoder to compute on-demand item
        # embeddings.
        item_signature = get_option_name(
            decl.options,
            "signature",
            line=decl.line,
            col=decl.col,
        )
        if item_signature is not None:
            sigs = getattr(self, "_signatures", {})
            if item_signature not in sigs:
                raise CompileError(
                    f"deduction {decl.name!r}: unknown item signature "
                    f"{item_signature!r}",
                    decl.line,
                    decl.col,
                )
            system = system.with_(item_signature=sigs[item_signature])
        item_encoder = get_option_name(
            decl.options,
            "encoder",
            line=decl.line,
            col=decl.col,
        )
        if item_encoder is not None:
            comps = getattr(self, "_encoders", {})
            if item_encoder not in comps:
                raise CompileError(
                    f"deduction {decl.name!r}: unknown item encoder {item_encoder!r}",
                    decl.line,
                    decl.col,
                )
            system = system.with_(item_encoder=comps[item_encoder])
        self._deductions[decl.name] = system


__all__ = []
