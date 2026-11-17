"""Walkers for program-block steps and program parameters."""

from __future__ import annotations

from typing import Literal, cast

from quivers.dsl.ast_nodes import (
    AxisSpec,
    CallStep,
    QiecCallComputation,
    LetStep,
    MarginalizeStep,
    MorphismParam,
    ObjectParam,
    ObserveStep,
    OptionList,
    OptionName,
    ProgramParam,
    ProgramStep,
    ReturnStep,
    SampleStep,
    ScalarParam,
    ScoreStep,
)
from quivers.dsl.parser._helpers import _walk_draw_arg
from quivers.dsl.parser._registry import ParseError, _Tree
from quivers.dsl.parser.expressions import _walk_let_arith, _walk_type
from quivers.dsl.parser.options import _walk_option_block
from quivers.dsl.parser.qiec import _walk_computation

# ---------------------------------------------------------------------------
# program step dispatch
# ---------------------------------------------------------------------------


def _walk_program_step(t: _Tree, vid: str) -> ProgramStep:
    k = t.kind(vid)
    if k == "sample_step":
        return _walk_sample_step(t, vid)
    if k == "observe_step":
        return _walk_observe_step(t, vid)
    if k == "marginalize_step":
        return _walk_marginalize_step(t, vid)
    if k == "let_step":
        return _walk_let_step(t, vid)
    if k == "call_step":
        return _walk_call_step(t, vid)
    if k == "score_step":
        return _walk_score_step(t, vid)
    if k == "return_step":
        return _walk_return_step(t, vid)
    raise ParseError(f"unexpected program step kind: {k}")


def _walk_sample_step(t: _Tree, vid: str) -> SampleStep:
    line, col = t.line_col(vid)
    vars_vid = t.field(vid, "vars")
    if vars_vid is None:
        raise ParseError(f"sample_step missing vars at {vid}")
    vars_t = _walk_var_pattern(t, vars_vid)
    morph_vid = t.field(vid, "morphism")
    if morph_vid is None:
        raise ParseError(f"sample_step missing morphism at {vid}")
    index_vid = t.field(vid, "index")
    index = _walk_type(t, index_vid) if index_vid else None
    args_vids = t.fields(vid, "args")
    args = tuple(_walk_draw_arg(t, av) for av in args_vids) if args_vids else None
    options_vid = t.field(vid, "options")
    options = _walk_option_block(t, options_vid) if options_vid else ()
    axes = _extract_axes(options)
    return SampleStep(
        vars=vars_t,
        morphism=t.text(morph_vid),
        args=args,
        static_args=tuple(
            _walk_draw_arg(t, child) for child in t.fields(vid, "static_args")
        ),
        index=index,
        axes=axes,
        options=options,
        line=line,
        col=col,
    )


def _extract_axes(options: tuple):
    """Lift ``over=...`` / ``iid_over=...`` options into an `AxisSpec`.

    ``over`` and ``iid_over`` may each be a single identifier
    (``over=Topic``) or a list of identifiers (``over=[Doc, Topic]``).
    """
    over_names: tuple[str, ...] = ()
    iid_names: tuple[str, ...] = ()
    line = col = 0
    for entry in options:
        v = entry.value
        if entry.key == "over":
            if isinstance(v, OptionName):
                over_names = (v.value,)
            elif isinstance(v, OptionList):
                over_names = tuple(
                    it.value for it in v.items if isinstance(it, OptionName)
                )
            line, col = entry.line, entry.col
        elif entry.key == "iid_over":
            if isinstance(v, OptionName):
                iid_names = (v.value,)
            elif isinstance(v, OptionList):
                iid_names = tuple(
                    it.value for it in v.items if isinstance(it, OptionName)
                )
    if not over_names and not iid_names:
        return None
    return AxisSpec(over=over_names, iid_over=iid_names, line=line, col=col)


def _walk_observe_step(t: _Tree, vid: str) -> ObserveStep:
    line, col = t.line_col(vid)
    vars_vid = t.field(vid, "vars")
    morph_vid = t.field(vid, "morphism")
    if vars_vid is None or morph_vid is None:
        raise ParseError(f"observe_step missing vars/morphism at {vid}")
    vars_t = _walk_var_pattern(t, vars_vid)
    index_vid = t.field(vid, "index")
    index = _walk_type(t, index_vid) if index_vid else None
    args_vids = t.fields(vid, "args")
    args = tuple(_walk_draw_arg(t, av) for av in args_vids) if args_vids else None
    options_vid = t.field(vid, "options")
    options = _walk_option_block(t, options_vid) if options_vid else ()
    via, via_axes = _extract_via(options)
    return ObserveStep(
        vars=vars_t,
        morphism=t.text(morph_vid),
        args=args,
        static_args=tuple(
            _walk_draw_arg(t, child) for child in t.fields(vid, "static_args")
        ),
        index=index,
        via=via,
        via_axes=via_axes,
        options=options,
        line=line,
        col=col,
    )


def _extract_via(
    options: tuple,
) -> tuple[str | None, tuple[str, ...] | None]:
    """Lift the ``via=`` option key into dedicated ``via`` / ``via_axes``
    AST fields. ``via=idx`` -> ``via='idx'``; ``via=[a, b]`` ->
    ``via_axes=('a', 'b')``."""
    for entry in options:
        if entry.key != "via":
            continue
        v = entry.value
        if isinstance(v, OptionName):
            return v.value, None
        if isinstance(v, OptionList):
            axes = []
            for item in v.items:
                if isinstance(item, OptionName):
                    axes.append(item.value)
            if axes:
                return None, tuple(axes)
    return None, None


def _walk_marginalize_step(t: _Tree, vid: str) -> MarginalizeStep:
    line, col = t.line_col(vid)
    var_vid = t.field(vid, "var")
    morph_vid = t.field(vid, "morphism")
    if var_vid is None or morph_vid is None:
        raise ParseError(f"marginalize_step missing var/morphism at {vid}")
    index_vid = t.field(vid, "index")
    index = _walk_type(t, index_vid) if index_vid else None
    args_vids = t.fields(vid, "args")
    args = tuple(_walk_draw_arg(t, av) for av in args_vids) if args_vids else None
    options_vid = t.field(vid, "options")
    options = _walk_option_block(t, options_vid) if options_vid else ()
    over, over_objs, reduction = _extract_marginalize_options(options)
    scope = tuple(_walk_program_step(t, sv) for sv in t.fields(vid, "scope"))
    return MarginalizeStep(
        var=t.text(var_vid),
        morphism=t.text(morph_vid),
        args=args,
        static_args=tuple(
            _walk_draw_arg(t, child) for child in t.fields(vid, "static_args")
        ),
        index=index,
        over=over,
        over_objs=over_objs,
        reduction=reduction,
        options=options,
        scope=scope,
        line=line,
        col=col,
    )


def _extract_marginalize_options(
    options: tuple,
) -> tuple[str | None, tuple[str, ...] | None, str | None]:
    """Pull ``over=...`` / ``over=[...]`` / ``reduction=...`` keys out of
    a parsed option block into the dedicated AST fields. ``over=A``
    becomes ``over='A'``; ``over=[A, B]`` becomes
    ``over_objs=('A','B')``; ``reduction=logsumexp`` becomes
    ``reduction='logsumexp'``."""
    over: str | None = None
    over_objs: tuple[str, ...] | None = None
    reduction: str | None = None
    for entry in options:
        v = entry.value
        if entry.key == "over":
            if isinstance(v, OptionName):
                over = v.value
            elif isinstance(v, OptionList):
                names: list[str] = []
                for item in v.items:
                    if isinstance(item, OptionName):
                        names.append(item.value)
                if names:
                    over_objs = tuple(names)
        elif entry.key == "reduction":
            if isinstance(v, OptionName):
                reduction = v.value
    return over, over_objs, reduction


def _walk_let_step(t: _Tree, vid: str) -> LetStep:
    line, col = t.line_col(vid)
    name_vid = t.field(vid, "name")
    value_vid = t.field(vid, "value")
    if name_vid is None or value_vid is None:
        raise ParseError(f"let_step missing name/value at {vid}")
    return LetStep(
        name=t.text(name_vid),
        value=_walk_let_arith(t, value_vid),
        line=line,
        col=col,
    )


def _walk_call_step(t: _Tree, vid: str) -> CallStep:
    line, col = t.line_col(vid)
    name_vid = t.field(vid, "name")
    call_vid = t.field(vid, "call")
    if name_vid is None or call_vid is None:
        raise ParseError(f"call_step missing name/call at {vid}")
    call = _walk_computation(t, call_vid)
    if not isinstance(call, QiecCallComputation):
        raise ParseError(f"call_step binds a non-call at {vid}")
    return CallStep(name=t.text(name_vid), call=call, line=line, col=col)


def _walk_score_step(t: _Tree, vid: str) -> ScoreStep:
    line, col = t.line_col(vid)
    name_vid = t.field(vid, "name")
    value_vid = t.field(vid, "value")
    if name_vid is None or value_vid is None:
        raise ParseError(f"score_step missing name/value at {vid}")
    return ScoreStep(
        name=t.text(name_vid),
        value=_walk_let_arith(t, value_vid),
        line=line,
        col=col,
    )


def _walk_return_step(t: _Tree, vid: str) -> ReturnStep:
    line, col = t.line_col(vid)
    return_vid = t.field(vid, "return")
    if return_vid is None:
        raise ParseError(f"return_step missing return at {vid}")
    vars_t, labels = _walk_return_pattern(t, return_vid)
    return ReturnStep(vars=vars_t, labels=labels, line=line, col=col)


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------


def _walk_var_pattern(t: _Tree, vid: str) -> tuple[str, ...]:
    k = t.kind(vid)
    if k == "identifier":
        return (t.text(vid),)
    if k == "var_tuple":
        return tuple(t.text(c) for c in t.positional(vid) if t.kind(c) == "identifier")
    raise ParseError(f"unexpected var pattern kind: {k}")


def _walk_return_pattern(
    t: _Tree, vid: str
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    k = t.kind(vid)
    if k == "identifier":
        return ((t.text(vid),), None)
    if k == "return_tuple":
        names = tuple(t.text(c) for c in t.positional(vid) if t.kind(c) == "identifier")
        return (names, None)
    if k == "return_labeled_tuple":
        entries = [e for e in t.positional(vid) if t.kind(e) == "return_label_entry"]
        entry_names: list[str] = []
        labels: list[str] = []
        for entry in entries:
            label_vid = t.field(entry, "label")
            var_vid = t.field(entry, "var")
            if label_vid is None or var_vid is None:
                raise ParseError(f"return_label_entry missing field at {entry}")
            labels.append(t.text(label_vid))
            entry_names.append(t.text(var_vid))
        return (tuple(entry_names), tuple(labels))
    raise ParseError(f"unexpected return pattern kind: {k}")


# ---------------------------------------------------------------------------
# program params (parametric programs)
# ---------------------------------------------------------------------------


def _walk_program_param(t: _Tree, vid: str) -> ProgramParam:
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k == "identifier":
        # Concrete program: bare-identifier parameter naming a domain
        # factor. Treated as an ObjectParam over the FinSet/Space/Object
        # universe; the compiler refines based on context.
        return ObjectParam(
            name=t.text(vid),
            universe="Object",
            line=line,
            col=col,
        )
    if k == "typed_program_param":
        name_vid = t.field(vid, "name")
        kind_vid = t.field(vid, "kind")
        if name_vid is None or kind_vid is None:
            raise ParseError(f"typed_program_param missing name/kind at {vid}")
        name = t.text(name_vid)
        kk = t.kind(kind_vid)
        if kk == "object_kind":
            kids = t.positional(kind_vid)
            universe_text = t.text(kids[0]) if kids else t.text(kind_vid)
            return ObjectParam(
                name=name,
                universe=cast('Literal["FinSet", "Space", "Object"]', universe_text),
                line=line,
                col=col,
            )
        if kk == "scalar_kind":
            kids = t.positional(kind_vid)
            scalar_text = t.text(kids[0]) if kids else t.text(kind_vid)
            return ScalarParam(
                name=name,
                scalar_kind=cast('Literal["Real", "Nat"]', scalar_text),
                line=line,
                col=col,
            )
        if kk == "morphism_kind":
            dom_vid = t.field(kind_vid, "domain")
            cod_vid = t.field(kind_vid, "codomain")
            if dom_vid is None or cod_vid is None:
                raise ParseError(f"morphism_kind missing domain/codomain at {kind_vid}")
            return MorphismParam(
                name=name,
                domain=_walk_type(t, dom_vid),
                codomain=_walk_type(t, cod_vid),
                line=line,
                col=col,
            )
        raise ParseError(f"unexpected param kind: {kk}")
    raise ParseError(f"unexpected _program_param kind: {k}")


__all__ = []
