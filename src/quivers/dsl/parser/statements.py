"""Top-level statement dispatcher and per-declaration walkers.

Sixteen `Statement` variants, one walker each. Every walker
takes a tree-sitter vertex id, reads its fields/positional children
via the `_Tree` view, and emits a fully constructed AST node.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    BinderArg,
    BinderDecl,
    BinderVar,
    BundleDecl,
    CategoryDecl,
    CompositionDecl,
    CompositionLevel,
    CompositionRuleEntry,
    ConstructorDecl,
    ContractionDecl,
    ContractionInput,
    DecoderDecl,
    DeductionDecl,
    DefineDecl,
    EdgeKindDecl,
    EncoderDecl,
    EncoderInitRule,
    EncoderMessageRule,
    EncoderRule,
    EncoderUpdateRule,
    EncoderVarInit,
    ExportDecl,
    LetExprNode,
    OptionEntry,
    OptionFlag,
    OptionName,
    OptionNumber,
    OptionString,
    OptionValue,
    LexiconCategory,
    LexiconCategoryFixed,
    LexiconCategoryRestricted,
    LexiconCategoryWildcard,
    LexiconEntry,
    LossDecl,
    MorphismDecl,
    MorphismInitFamily,
    ProgramDecl,
    ProgramParam,
    ProgramStep,
    RuleDecl,
    SchemaDecl,
    SchemaParameter,
    SequentRule,
    SignatureDecl,
    SortDecl,
    SortDim,
    Statement,
    ObjectDecl,
    TypeEnumSet,
    TypeFreeMonoid,
    TypeFreeResiduated,
    TypeFromExpr,
    TypeInitializer,
    VertexKindDecl,
)
from quivers.dsl.parser._helpers import (
    _field_text,
    _required_field,
    _required_text,
    _walk_draw_arg,
)
from quivers.dsl.parser._registry import ParseError, _Tree
from quivers.dsl.parser.expressions import _walk_expr, _walk_let_arith, _walk_type
from quivers.dsl.parser.options import _walk_option_block, _walk_option_value
from quivers.dsl.parser.program_steps import (
    _walk_program_param,
    _walk_program_step,
    _walk_return_pattern,
)
from quivers.dsl.parser.qiec import _QIEC_STATEMENT_KINDS, _walk_qiec_statement

# ---------------------------------------------------------------------------
# top-level dispatcher
# ---------------------------------------------------------------------------


def _decl_docs(t: _Tree, vid: str) -> tuple[str, ...]:
    """Collect a declaration's ``#!`` doc-comment lines.

    Doc comments ride the declaration's ``docs`` field as a
    ``doc_comment_group`` vertex whose children are the individual
    ``doc_comment`` tokens; each line's ``#!`` opener and surrounding
    whitespace are stripped.
    """
    group_vid = t.field(vid, "docs")
    if group_vid is None:
        return ()
    out: list[str] = []
    for line_vid in t.positional(group_vid):
        if t.kind(line_vid) != "doc_comment":
            continue
        text = t.text(line_vid)
        if text.startswith("#!"):
            text = text[2:]
        out.append(text.strip())
    return tuple(out)


def _walk_statement(t: _Tree, vid: str) -> Statement | list[Statement]:
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k in _QIEC_STATEMENT_KINDS:
        return _walk_qiec_statement(t, vid)
    if k == "composition_decl":
        return _walk_composition_decl(t, vid, line, col)
    if k == "category_decl":
        docs = _decl_docs(t, vid)
        out: list[Statement] = []
        for nv in t.fields(vid, "names"):
            ln, cl = t.line_col(nv)
            out.append(CategoryDecl(names=(t.text(nv),), docs=docs, line=ln, col=cl))
        if not out:
            return CategoryDecl(names=(), docs=docs, line=line, col=col)
        return out if len(out) > 1 else out[0]
    if k == "rule_decl":
        return _walk_rule_decl(t, vid, line, col)
    if k == "schema_decl":
        return _walk_schema_decl(t, vid, line, col)
    if k == "object_decl":
        return _walk_type_decl(t, vid, line, col)
    if k == "morphism_decl":
        return _walk_morphism_decl(t, vid, line, col)
    if k == "bundle_decl":
        return _walk_bundle_decl(t, vid, line, col)
    if k == "program_decl":
        return _walk_program_decl(t, vid, line, col)
    if k == "contraction_decl":
        return _walk_contraction_decl(t, vid, line, col)
    if k == "define_decl":
        return _walk_define_decl(t, vid, line, col)
    if k == "export_decl":
        return _walk_export_decl(t, vid, line, col)
    if k == "deduction_decl":
        return _walk_deduction_decl(t, vid, line, col)
    if k == "signature_decl":
        return _walk_signature_decl(t, vid, line, col)
    if k == "encoder_decl":
        return _walk_encoder_decl(t, vid, line, col)
    if k == "decoder_decl":
        return _walk_decoder_decl(t, vid, line, col)
    if k == "loss_decl":
        return _walk_loss_decl(t, vid, line, col)
    if k == "ERROR":
        cs = t.consts(vid)
        snippet = t.source[int(cs["start-byte"]) : int(cs["end-byte"])].decode("utf-8")
        raise ParseError(f"syntax error at line {line}, col {col}: {snippet!r}")
    raise ParseError(f"unexpected statement kind: {k}")


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


_COMPOSITION_LEVELS = ("algebra", "semigroupoid", "magmoid", "rule")


def _walk_composition_decl(t: _Tree, vid: str, line: int, col: int) -> CompositionDecl:
    name_vid = t.field(vid, "name")
    name = _required_text(t, name_vid, vid, "name")
    opt_vid = t.field(vid, "options")
    level: CompositionLevel | None = None
    if opt_vid is not None:
        for entry in _walk_option_block(t, opt_vid):
            if entry.key != "level":
                raise ParseError(
                    f"unexpected option {entry.key!r} on composition {name!r} "
                    f"at line {entry.line}, col {entry.col}; a composition "
                    "option block carries only [level=...]"
                )
            lt = _option_value_text(entry.value)
            if lt not in _COMPOSITION_LEVELS:
                raise ParseError(
                    f"unknown composition level {lt!r} at line {entry.line}, "
                    f"col {entry.col}; valid levels: " + ", ".join(_COMPOSITION_LEVELS)
                )
            level = lt  # type: ignore[assignment]
    body: list[CompositionRuleEntry] = []
    for child in t.positional(vid):
        if t.kind(child) == "composition_rule_entry":
            body.append(_walk_composition_rule_entry(t, child))
    return CompositionDecl(
        name=name,
        level=level,
        body=tuple(body),
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


def _option_value_text(value: OptionValue) -> str:
    """Render an option value for membership checks and error messages."""
    if isinstance(value, (OptionName, OptionString)):
        return value.value
    if isinstance(value, OptionNumber):
        return str(value.value)
    return type(value).__name__


def _walk_composition_rule_entry(t: _Tree, vid: str) -> CompositionRuleEntry:
    line, col = t.line_col(vid)
    key_vid = t.field(vid, "key")
    body_vid = t.field(vid, "body")
    if key_vid is None or body_vid is None:
        raise ParseError(f"composition_rule_entry missing key/body at {vid}")
    params = tuple(t.text(p) for p in t.fields(vid, "params"))
    return CompositionRuleEntry(
        key=t.text(key_vid),
        params=params,
        body=_walk_let_arith(t, body_vid),
        line=line,
        col=col,
    )


# ---------------------------------------------------------------------------
# rule (top-level CCG/Lambek)
# ---------------------------------------------------------------------------


def _walk_rule_decl(t: _Tree, vid: str, line: int, col: int) -> RuleDecl:
    name_vid = t.field(vid, "name")
    concl_vid = t.field(vid, "conclusion")
    if concl_vid is None:
        raise ParseError(f"rule_decl missing conclusion at {vid}")
    return RuleDecl(
        name=_required_text(t, name_vid, vid, "name"),
        variables=tuple(t.text(v) for v in t.fields(vid, "variables")),
        premises=tuple(_walk_type(t, p) for p in t.fields(vid, "premises")),
        conclusion=_walk_type(t, concl_vid),
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def _walk_schema_decl(t: _Tree, vid: str, line: int, col: int) -> SchemaDecl:
    name_vid = t.field(vid, "name")
    dom_vid = t.field(vid, "domain")
    cod_vid = t.field(vid, "codomain")
    if dom_vid is None or cod_vid is None:
        raise ParseError(f"schema_decl missing domain/codomain at {vid}")
    params: list[SchemaParameter] = []
    for pv in t.fields(vid, "parameters"):
        type_vid = t.field(pv, "type")
        if type_vid is None:
            raise ParseError(f"schema_parameter missing type at {pv}")
        pl, pc = t.line_col(pv)
        params.append(
            SchemaParameter(
                names=tuple(t.text(n) for n in t.fields(pv, "names")),
                type_expr=_walk_type(t, type_vid),
                line=pl,
                col=pc,
            )
        )
    return SchemaDecl(
        name=_required_text(t, name_vid, vid, "name"),
        parameters=tuple(params),
        domain=_walk_type(t, dom_vid),
        codomain=_walk_type(t, cod_vid),
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


# ---------------------------------------------------------------------------
# type
# ---------------------------------------------------------------------------


def _walk_type_decl(t: _Tree, vid: str, line: int, col: int) -> ObjectDecl:
    names = tuple(t.text(nv) for nv in t.fields(vid, "names"))
    value_vid = t.field(vid, "value")
    if not names or value_vid is None:
        raise ParseError(f"object_decl missing names/value at {vid}")
    return ObjectDecl(
        names=names,
        init=_walk_type_value(t, value_vid),
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


def _walk_type_value(t: _Tree, vid: str) -> TypeInitializer:
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k == "enum_set_literal":
        return TypeEnumSet(
            elements=tuple(t.text(e) for e in t.fields(vid, "elements")),
            line=line,
            col=col,
        )
    if k == "free_residuated_expr":
        gen_vid = t.field(vid, "generators")
        if gen_vid is None:
            raise ParseError(f"free_residuated_expr missing generators at {vid}")
        depth = 1
        ops: list[str] = []
        for arg_vid in t.positional(vid):
            if t.kind(arg_vid) != "free_residuated_arg":
                continue
            d = t.field(arg_vid, "depth")
            if d is not None:
                depth = int(t.text(d))
                continue
            for op_vid in t.fields(arg_vid, "op"):
                ops.append(t.text(op_vid))
        if not ops:
            ops = ["slash"]
        return TypeFreeResiduated(
            generators=t.text(gen_vid),
            depth=depth,
            ops=tuple(ops),
            line=line,
            col=col,
        )
    if k == "free_monoid_expr":
        gen_vid = t.field(vid, "generators")
        ml_vid = t.field(vid, "max_length")
        if gen_vid is None or ml_vid is None:
            raise ParseError(f"free_monoid_expr missing generators/max_length at {vid}")
        return TypeFreeMonoid(
            generators=t.text(gen_vid),
            max_length=int(t.text(ml_vid)),
            line=line,
            col=col,
        )
    # Anything else is a plain type expression (FinSet, Real, etc.).
    return TypeFromExpr(expr=_walk_type(t, vid))


# ---------------------------------------------------------------------------
# morphism
# ---------------------------------------------------------------------------


def _walk_morphism_decl(t: _Tree, vid: str, line: int, col: int) -> MorphismDecl:
    names = tuple(t.text(nv) for nv in t.fields(vid, "names"))
    dom_vid = t.field(vid, "domain")
    cod_vid = t.field(vid, "codomain")
    if not names or dom_vid is None or cod_vid is None:
        raise ParseError(f"morphism_decl missing names/domain/codomain at {vid}")
    opt_vid = t.field(vid, "options")
    options = _walk_option_block(t, opt_vid) if opt_vid else ()
    init_vid = t.field(vid, "init")
    init_family: MorphismInitFamily | None = None
    init_expr = None
    if init_vid is not None:
        if t.kind(init_vid) == "morphism_init_family":
            init_family = _walk_morphism_init_family(t, init_vid)
        else:
            init_expr = _walk_expr(t, init_vid)
    return MorphismDecl(
        names=names,
        domain=_walk_type(t, dom_vid),
        codomain=_walk_type(t, cod_vid),
        options=options,
        init_family=init_family,
        init_expr=init_expr,
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


def _walk_morphism_init_family(t: _Tree, vid: str) -> MorphismInitFamily:
    line, col = t.line_col(vid)
    family_vid = t.field(vid, "family")
    if family_vid is None:
        raise ParseError(f"morphism_init_family missing family at {vid}")
    return MorphismInitFamily(
        family=t.text(family_vid),
        args=tuple(_walk_draw_arg(t, a) for a in t.fields(vid, "args")),
        line=line,
        col=col,
    )


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------


def _walk_bundle_decl(t: _Tree, vid: str, line: int, col: int) -> BundleDecl:
    name_vid = t.field(vid, "name")
    return BundleDecl(
        name=_required_text(t, name_vid, vid, "name"),
        rules=tuple(t.text(r) for r in t.fields(vid, "rules")),
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


# ---------------------------------------------------------------------------
# contraction
# ---------------------------------------------------------------------------


def _walk_contraction_decl(t: _Tree, vid: str, line: int, col: int) -> ContractionDecl:
    name_vid = t.field(vid, "name")
    dom_vid = t.field(vid, "domain")
    cod_vid = t.field(vid, "codomain")
    if name_vid is None or dom_vid is None or cod_vid is None:
        raise ParseError(f"contraction_decl missing required field at {vid}")
    inputs: list[ContractionInput] = []
    for inp_vid in t.fields(vid, "inputs"):
        in_name_vid = t.field(inp_vid, "name")
        in_dom_vid = t.field(inp_vid, "input_domain")
        in_cod_vid = t.field(inp_vid, "input_codomain")
        if in_name_vid is None or in_dom_vid is None or in_cod_vid is None:
            raise ParseError(f"contraction_input missing field at {inp_vid}")
        il, ic = t.line_col(inp_vid)
        inputs.append(
            ContractionInput(
                name=t.text(in_name_vid),
                input_domain=_walk_type(t, in_dom_vid),
                input_codomain=_walk_type(t, in_cod_vid),
                line=il,
                col=ic,
            )
        )
    opt_vid = t.field(vid, "options")
    options = _walk_option_block(t, opt_vid) if opt_vid else ()
    return ContractionDecl(
        name=t.text(name_vid),
        inputs=tuple(inputs),
        domain=_walk_type(t, dom_vid),
        codomain=_walk_type(t, cod_vid),
        options=options,
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


# ---------------------------------------------------------------------------
# define / export
# ---------------------------------------------------------------------------


def _walk_define_decl(t: _Tree, vid: str, line: int, col: int) -> DefineDecl:
    name_vid = t.field(vid, "name")
    value_vid = t.field(vid, "value")
    if value_vid is None:
        raise ParseError(f"define_decl missing value at {vid}")
    # `where`-block entries nest as positional `define_decl` children.
    where: list[Statement] = []
    for child in t.positional(vid):
        if t.kind(child) == "define_decl":
            result = _walk_statement(t, child)
            if isinstance(result, list):
                where.extend(result)
            else:
                where.append(result)
    return DefineDecl(
        name=_required_text(t, name_vid, vid, "name"),
        expr=_walk_expr(t, value_vid),
        where=tuple(where),
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


def _walk_export_decl(t: _Tree, vid: str, line: int, col: int) -> ExportDecl:
    value_vid = t.field(vid, "value")
    if value_vid is None:
        raise ParseError(f"export_decl missing value at {vid}")
    return ExportDecl(
        expr=_walk_expr(t, value_vid),
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


# ---------------------------------------------------------------------------
# deduction
# ---------------------------------------------------------------------------


def _walk_deduction_decl(t: _Tree, vid: str, line: int, col: int) -> DeductionDecl:
    name_vid = t.field(vid, "name")
    dom_vid = t.field(vid, "domain")
    cod_vid = t.field(vid, "codomain")
    if name_vid is None or dom_vid is None or cod_vid is None:
        raise ParseError(f"deduction_decl missing required field at {vid}")
    opt_vid = t.field(vid, "options")
    options = _walk_option_block(t, opt_vid) if opt_vid else ()
    atoms: list[str] = []
    rules: list[SequentRule] = []
    lex_entries: list[LexiconEntry] = []
    lex_from_file: str | None = None
    lex_from_file_options: tuple = ()
    binders: list[str] = []
    for child in t.positional(vid):
        kk = t.kind(child)
        if kk == "deduction_atoms":
            for av in t.fields(child, "atoms"):
                atoms.append(t.text(av))
        elif kk == "deduction_binders":
            for bv in t.fields(child, "binders"):
                binders.append(t.text(bv))
        elif kk == "deduction_rule":
            rname = _required_text(t, t.field(child, "name"), child, "name")
            premises = tuple(_walk_type(t, p) for p in t.fields(child, "premises"))
            conc_vid = t.field(child, "conclusion")
            if conc_vid is None:
                raise ParseError(f"deduction_rule missing conclusion at {child}")
            rl, rc = t.line_col(child)
            rule_options = _walk_lexicon_pragma(t, t.field(child, "pragma"))
            rules.append(
                SequentRule(
                    name=rname,
                    premises=premises,
                    conclusion=_walk_type(t, conc_vid),
                    options=rule_options,
                    line=rl,
                    col=rc,
                )
            )
        elif kk == "deduction_lexicon":
            for entry_vid in t.positional(child):
                if t.kind(entry_vid) == "lexicon_entry":
                    lex_entries.append(_walk_lexicon_entry(t, entry_vid))
        elif kk == "deduction_lexicon_from_file":
            path_raw = _required_text(t, t.field(child, "path"), child, "path")
            if path_raw.startswith('"') and path_raw.endswith('"'):
                path_raw = path_raw[1:-1]
            lex_from_file = path_raw
            sub_opt_vid = t.field(child, "options")
            lex_from_file_options = (
                _walk_option_block(t, sub_opt_vid) if sub_opt_vid else ()
            )
    return DeductionDecl(
        name=t.text(name_vid),
        domain=_walk_type(t, dom_vid),
        codomain=_walk_type(t, cod_vid),
        options=options,
        atoms=tuple(atoms),
        binders=tuple(binders),
        rules=tuple(rules),
        lexicon=tuple(lex_entries),
        lexicon_from_file=lex_from_file,
        lexicon_from_file_options=lex_from_file_options,
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


def _walk_lexicon_entry(t: _Tree, vid: str) -> LexiconEntry:
    cat_vid = t.field(vid, "category")
    lf_vid = t.field(vid, "lf")
    words_raw = [t.text(wv) for wv in t.fields(vid, "words")]
    if not words_raw or lf_vid is None:
        raise ParseError(f"lexicon_entry malformed at {vid}")
    words = tuple(
        w[1:-1] if w.startswith('"') and w.endswith('"') else w for w in words_raw
    )
    # Lexicon-entry attributes ride a dedicated ``#[…]`` pragma
    # (``lexicon_pragma``) rather than an option block: a bracketed
    # tail after a let-arith expression would otherwise be greedily
    # absorbed as ``let_index``. The pragma's ``#[`` opener cannot
    # extend any let expression, so the surface is unambiguous.
    options = _walk_lexicon_pragma(t, t.field(vid, "pragma"))
    line, col = t.line_col(vid)

    # Three category shapes:
    #
    # *  The ``*`` wildcard is captured as ``field:category = *``
    #    on the lexicon_entry vertex (literal-string field, no
    #    child vertex). The grammar's ``_lexicon_category`` choice
    #    puts ``*`` as a bare alternative.
    # *  ``{A, B, C}`` parses to a ``category`` edge whose target
    #    is an ``enum_set_literal`` vertex carrying the candidate
    #    atoms as positional children.
    # *  Anything else is a fixed ``_object_expr`` and walks
    #    through `_walk_type` as before.
    category: LexiconCategory
    field_cat = t.consts(vid).get("field:category")
    if field_cat == "*":
        category = LexiconCategoryWildcard()
    elif cat_vid is not None and t.kind(cat_vid) == "enum_set_literal":
        atoms = tuple(t.text(av) for av in t.fields(cat_vid, "elements"))
        category = LexiconCategoryRestricted(atoms=atoms)
    elif cat_vid is not None:
        category = LexiconCategoryFixed(category=_walk_type(t, cat_vid))
    else:
        raise ParseError(
            f"lexicon_entry malformed at {vid}: missing category",
        )

    return LexiconEntry(
        words=words,
        category=category,
        lf=_walk_let_arith(t, lf_vid),
        options=options,
        line=line,
        col=col,
    )


def _walk_lexicon_pragma(
    t: _Tree,
    vid: str | None,
) -> tuple[OptionEntry, ...]:
    """Translate a ``lexicon_pragma`` vertex's ``pragma_entry`` children
    into the option-entry list the compiler already consumes.

    ``#[learnable]`` becomes ``[OptionEntry(key='learnable',
    value=OptionFlag())]``; ``#[learnable, frozen]`` returns both.
    """
    if vid is None:
        return ()
    out: list[OptionEntry] = []
    for entry_vid in t.fields(vid, "entries"):
        key = _field_text(t, entry_vid, "key")
        line, col = t.line_col(entry_vid)
        val_vid = t.field(entry_vid, "value")
        value: OptionValue = (
            OptionFlag() if val_vid is None else _walk_option_value(t, val_vid)
        )
        out.append(OptionEntry(key=key, value=value, line=line, col=col))
    return tuple(out)


# ---------------------------------------------------------------------------
# signature
# ---------------------------------------------------------------------------


def _walk_signature_decl(t: _Tree, vid: str, line: int, col: int) -> SignatureDecl:
    name_vid = t.field(vid, "name")
    if name_vid is None:
        raise ParseError(f"signature_decl missing name at {vid}")
    name = t.text(name_vid)
    params = tuple(t.text(c) for c in t.fields(vid, "params"))
    sorts: list[SortDecl] = []
    constructors: list[ConstructorDecl] = []
    binders: list[BinderDecl] = []
    vertex_kinds: list[VertexKindDecl] = []
    edge_kinds: list[EdgeKindDecl] = []
    for child in t.positional(vid):
        ck = t.kind(child)
        if ck == "signature_sorts":
            for s in t.positional(child):
                if t.kind(s) == "sort_decl":
                    sorts.append(_walk_sort_decl(t, s))
        elif ck == "signature_constructors":
            for c in t.positional(child):
                if t.kind(c) == "constructor_decl":
                    constructors.append(_walk_constructor_decl(t, c))
        elif ck == "signature_binders":
            for b in t.positional(child):
                if t.kind(b) == "binder_decl":
                    binders.append(_walk_binder_decl(t, b))
        elif ck == "signature_vertex_kinds":
            for v in t.positional(child):
                if t.kind(v) == "vertex_kind_decl":
                    vertex_kinds.append(_walk_vertex_kind_decl(t, v))
        elif ck == "signature_edge_kinds":
            for e in t.positional(child):
                if t.kind(e) == "edge_kind_decl":
                    edge_kinds.append(_walk_edge_kind_decl(t, e))
    return SignatureDecl(
        name=name,
        params=params,
        sorts=tuple(sorts),
        constructors=tuple(constructors),
        binders=tuple(binders),
        vertex_kinds=tuple(vertex_kinds),
        edge_kinds=tuple(edge_kinds),
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


def _walk_sort_decl(t: _Tree, vid: str) -> SortDecl:
    name = _field_text(t, vid, "name")
    kind_txt = _field_text(t, vid, "kind")
    opt_vid = t.field(vid, "options")
    options = _walk_option_block(t, opt_vid) if opt_vid else ()
    ln, cl = t.line_col(vid)
    return SortDecl(
        name=name,
        kind=kind_txt,  # type: ignore[arg-type]
        options=options,
        line=ln,
        col=cl,
    )


def _walk_constructor_decl(t: _Tree, vid: str) -> ConstructorDecl:
    name = _field_text(t, vid, "name")
    domain = tuple(t.text(d) for d in t.fields(vid, "domain"))
    codomain = _field_text(t, vid, "codomain")
    ln, cl = t.line_col(vid)
    return ConstructorDecl(
        name=name,
        domain=domain,
        codomain=codomain,
        line=ln,
        col=cl,
    )


def _walk_binder_decl(t: _Tree, vid: str) -> BinderDecl:
    name = _field_text(t, vid, "name")
    binds_list: list[BinderVar] = []
    for b in t.fields(vid, "binds"):
        annot_vid = t.field(b, "annot")
        annot_sort_vid = t.field(b, "annot_sort")
        binds_list.append(
            BinderVar(
                var=_field_text(t, b, "var"),
                sort=_field_text(t, b, "sort"),
                annot=t.text(annot_vid) if annot_vid is not None else None,
                annot_sort=(
                    t.text(annot_sort_vid) if annot_sort_vid is not None else None
                ),
            )
        )
    scoped = tuple(
        BinderArg(
            arg=_field_text(t, a, "arg"),
            sort=_field_text(t, a, "sort"),
        )
        for a in t.fields(vid, "scoped")
    )
    codomain = _field_text(t, vid, "codomain")
    ln, cl = t.line_col(vid)
    return BinderDecl(
        name=name,
        binds=tuple(binds_list),
        scoped=scoped,
        codomain=codomain,
        line=ln,
        col=cl,
    )


def _walk_vertex_kind_decl(t: _Tree, vid: str) -> VertexKindDecl:
    name = _field_text(t, vid, "name")
    kind_txt = _field_text(t, vid, "kind")
    opt_vid = t.field(vid, "options")
    options = _walk_option_block(t, opt_vid) if opt_vid else ()
    ln, cl = t.line_col(vid)
    return VertexKindDecl(
        name=name,
        kind=kind_txt,  # type: ignore[arg-type]
        options=options,
        line=ln,
        col=cl,
    )


def _walk_edge_kind_decl(t: _Tree, vid: str) -> EdgeKindDecl:
    name = _field_text(t, vid, "name")
    src = _field_text(t, vid, "src")
    tgt = _field_text(t, vid, "tgt")
    arrow_txt = _field_text(t, vid, "arrow")
    if arrow_txt == "->":
        directed = True
    elif arrow_txt == "--":
        directed = False
    else:
        raise ParseError(f"edge_kind_decl at {vid}: unknown arrow {arrow_txt!r}")
    ln, cl = t.line_col(vid)
    return EdgeKindDecl(
        name=name,
        src=src,
        tgt=tgt,
        directed=directed,
        line=ln,
        col=cl,
    )


# ---------------------------------------------------------------------------
# encoder / decoder / loss
# ---------------------------------------------------------------------------


def _walk_encoder_decl(t: _Tree, vid: str, line: int, col: int) -> EncoderDecl:
    name = _field_text(t, vid, "name")
    signature = _field_text(t, vid, "signature")
    sig_args = tuple(t.text(c) for c in t.fields(vid, "sig_args"))
    opt_vid = t.field(vid, "options")
    options = _walk_option_block(t, opt_vid) if opt_vid else ()
    dims: list[SortDim] = []
    op_rules: list[EncoderRule] = []
    init_rules: list[EncoderInitRule] = []
    message_rules: list[EncoderMessageRule] = []
    update_rules: list[EncoderUpdateRule] = []
    iterations: int | None = None
    readout: LetExprNode | None = None
    var_inits: list[EncoderVarInit] = []
    for child in t.positional(vid):
        ck = t.kind(child)
        if ck == "encoder_dim":
            dims.append(
                SortDim(
                    sort=_field_text(t, child, "sort"),
                    dim=int(_field_text(t, child, "dim")),
                )
            )
        elif ck == "encoder_iterations":
            iterations = int(_field_text(t, child, "iterations"))
        elif ck == "encoder_readout":
            readout = _walk_let_arith(t, _required_field(t, child, "body"))
        elif ck == "encoder_op_rule":
            op = _field_text(t, child, "op")
            args = tuple(t.text(a) for a in t.fields(child, "args"))
            body = _walk_let_arith(t, _required_field(t, child, "body"))
            state_v = t.field(child, "state")
            prefix_v = t.field(child, "prefix")
            mode: str = "plain"
            state_var = None
            prefix_var = None
            if state_v is not None:
                mode = "recurrent"
                state_var = t.text(state_v)
            elif prefix_v is not None:
                mode = "attention"
                prefix_var = t.text(prefix_v)
            cln, ccl = t.line_col(child)
            op_rules.append(
                EncoderRule(
                    op=op,
                    args=args,
                    body=body,
                    mode=mode,  # type: ignore[arg-type]
                    state_var=state_var,
                    prefix_var=prefix_var,
                    line=cln,
                    col=ccl,
                )
            )
        elif ck == "encoder_init_rule":
            init_rules.append(
                EncoderInitRule(
                    kind=_field_text(t, child, "kind"),
                    arg=_field_text(t, child, "arg"),
                    body=_walk_let_arith(t, _required_field(t, child, "body")),
                )
            )
        elif ck == "encoder_message_rule":
            message_rules.append(
                EncoderMessageRule(
                    edge_kind=_field_text(t, child, "edge_kind"),
                    src=_field_text(t, child, "src"),
                    tgt=_field_text(t, child, "tgt"),
                    body=_walk_let_arith(t, _required_field(t, child, "body")),
                )
            )
        elif ck == "encoder_update_rule":
            update_rules.append(
                EncoderUpdateRule(
                    vertex_kind=_field_text(t, child, "vertex_kind"),
                    self_var=_field_text(t, child, "self"),
                    msgs_var=_field_text(t, child, "msgs"),
                    body=_walk_let_arith(t, _required_field(t, child, "body")),
                )
            )
        elif ck == "encoder_var_init":
            annot_vid = t.field(child, "annot_sort")
            ty_vid = t.field(child, "ty")
            ln, cl = t.line_col(child)
            var_inits.append(
                EncoderVarInit(
                    var_sort=_field_text(t, child, "var_sort"),
                    annot_sort=(t.text(annot_vid) if annot_vid is not None else None),
                    ty=t.text(ty_vid) if ty_vid is not None else None,
                    body=_walk_let_arith(t, _required_field(t, child, "body")),
                    line=ln,
                    col=cl,
                )
            )
    return EncoderDecl(
        name=name,
        signature=signature,
        sig_args=sig_args,
        options=options,
        dims=tuple(dims),
        op_rules=tuple(op_rules),
        init_rules=tuple(init_rules),
        message_rules=tuple(message_rules),
        update_rules=tuple(update_rules),
        iterations=iterations,
        readout=readout,
        var_inits=tuple(var_inits),
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


def _walk_decoder_decl(t: _Tree, vid: str, line: int, col: int) -> DecoderDecl:
    name = _field_text(t, vid, "name")
    signature = _field_text(t, vid, "signature")
    sig_args = tuple(t.text(c) for c in t.fields(vid, "sig_args"))
    opt_vid = t.field(vid, "options")
    options = _walk_option_block(t, opt_vid) if opt_vid else ()
    dims: list[SortDim] = []
    structure_body: LetExprNode | None = None
    structure_arg: str | None = None
    primitive_body: LetExprNode | None = None
    primitive_arg: str | None = None
    factor_body: LetExprNode | None = None
    factor_arg: str | None = None
    binder_body: LetExprNode | None = None
    binder_arg: str | None = None
    recursive_default = False
    for child in t.positional(vid):
        ck = t.kind(child)
        if ck == "decoder_dim":
            dims.append(
                SortDim(
                    sort=_field_text(t, child, "sort"),
                    dim=int(_field_text(t, child, "dim")),
                )
            )
        elif ck == "decoder_structure":
            structure_arg = _field_text(t, child, "arg")
            structure_body = _walk_let_arith(t, _required_field(t, child, "body"))
        elif ck == "decoder_primitive":
            primitive_arg = _field_text(t, child, "arg")
            primitive_body = _walk_let_arith(t, _required_field(t, child, "body"))
        elif ck == "decoder_factor":
            factor_arg = _field_text(t, child, "arg")
            factor_body = _walk_let_arith(t, _required_field(t, child, "body"))
        elif ck == "decoder_binder_select":
            binder_arg = _field_text(t, child, "arg")
            binder_body = _walk_let_arith(t, _required_field(t, child, "body"))
        elif ck == "decoder_body_default":
            recursive_default = True
    return DecoderDecl(
        name=name,
        signature=signature,
        sig_args=sig_args,
        options=options,
        dims=tuple(dims),
        structure=structure_body,
        structure_arg=structure_arg,
        primitive=primitive_body,
        primitive_arg=primitive_arg,
        factor=factor_body,
        factor_arg=factor_arg,
        binder_select=binder_body,
        binder_select_arg=binder_arg,
        recursive_default=recursive_default,
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


def _walk_loss_decl(t: _Tree, vid: str, line: int, col: int) -> LossDecl:
    name = _field_text(t, vid, "name")
    opt_vid = t.field(vid, "options")
    options = _walk_option_block(t, opt_vid) if opt_vid else ()
    body_vid = t.field(vid, "body")
    if body_vid is None:
        raise ParseError(f"loss_decl missing body at {vid}")
    return LossDecl(
        name=name,
        options=options,
        body=_walk_let_arith(t, body_vid),
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


# ---------------------------------------------------------------------------
# program
# ---------------------------------------------------------------------------


def _walk_program_decl(t: _Tree, vid: str, line: int, col: int) -> ProgramDecl:
    name_vid = t.field(vid, "name")
    dom_vid = t.field(vid, "domain")
    cod_vid = t.field(vid, "codomain")
    if dom_vid is None or cod_vid is None:
        raise ParseError(f"program_decl missing domain/codomain at {vid}")
    opt_vid = t.field(vid, "options")
    options = _walk_option_block(t, opt_vid) if opt_vid else ()
    data_params: list[str] = []
    type_params_list: list[ProgramParam] = []
    for pv in t.fields(vid, "params"):
        pk = t.kind(pv)
        if pk == "typed_program_param":
            type_params_list.append(_walk_program_param(t, pv))
        elif pk == "identifier":
            data_params.append(t.text(pv))
        else:
            raise ParseError(f"unexpected program param kind {pk} at {pv}")
    if data_params and type_params_list:
        raise ParseError(
            f"program {_required_text(t, name_vid, vid, 'name')!r} "
            "mixes bare data parameters and typed template parameters"
        )
    params: tuple[str, ...] | None = tuple(data_params) if data_params else None
    type_params: tuple[ProgramParam, ...] | None = (
        tuple(type_params_list) if type_params_list else None
    )
    steps: list[ProgramStep] = []
    for sv in t.fields(vid, "steps"):
        steps.append(_walk_program_step(t, sv))
    ret_vars: tuple[str, ...] = ()
    ret_labels: tuple[str, ...] | None = None
    # The terminal return_step is a positional child (the grammar
    # spells it out after the `repeat($._program_step)` and before
    # the dedent).
    for child in t.positional(vid):
        if t.kind(child) == "return_step":
            ret_vid = t.field(child, "return")
            if ret_vid is None:
                raise ParseError(f"return_step missing return at {child}")
            ret_vars, ret_labels = _walk_return_pattern(t, ret_vid)
            break
    return ProgramDecl(
        name=_required_text(t, name_vid, vid, "name"),
        params=params,
        type_params=type_params,
        domain=_walk_type(t, dom_vid),
        codomain=_walk_type(t, cod_vid),
        options=options,
        draws=tuple(steps),
        return_vars=ret_vars,
        return_labels=ret_labels,
        docs=_decl_docs(t, vid),
        line=line,
        col=col,
    )


__all__ = []
