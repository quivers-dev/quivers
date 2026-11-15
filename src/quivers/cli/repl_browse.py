"""Browsable children of the environment's bindings.

The REPL's ``:browse`` and the TUI's environment tree both render each
binding as a labelled node with its children: a program's steps, a
deduction's rules, a signature's constructors. The renderers here work
on the compiler's objects and know nothing of either front end.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import ObjectDecl
from quivers.dsl.emit import (
    _emit_program_param,
    _emit_qiec_binder,
    _emit_qiec_type,
    draw_call_to_source,
    module_to_source,
)

# Each ``_children_for_*`` returns ``(head_label, children)`` where
# ``children`` is a possibly nested list of either ``str`` leaves or
# ``(label, sub_children)`` tuples. ``_populate_children`` walks that
# structure onto a Textual ``Tree`` node. Builders consult only public
# accessors on the runtime objects so they never look into compiler
# internals.


Children = list  # list[str | tuple[str, "Children"]]


def _pretty(obj):  # type: ignore[no-untyped-def]
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        return name
    return repr(obj)


def _children_for_object(name, obj):  # type: ignore[no-untyped-def]
    if isinstance(obj, ObjectDecl):
        from quivers.dsl.ast_nodes import Module

        return module_to_source(Module(statements=(obj,))).strip(), []
    card = getattr(obj, "cardinality", None)
    if card is not None:
        return f"{name} : FinSet {card}", []
    return f"{name} : {_pretty(obj)}", []


def _children_for_space(name, sp):  # type: ignore[no-untyped-def]
    if isinstance(sp, ObjectDecl):
        return _children_for_object(name, sp)
    dim = getattr(sp, "dim", None)
    if dim is not None:
        return f"{name} : Real {dim}", []
    return f"{name} : {_pretty(sp)}", []


def _children_for_morphism(name, morph):  # type: ignore[no-untyped-def]
    dom = _pretty(getattr(morph, "domain", None))
    cod = _pretty(getattr(morph, "codomain", None))
    return f"{name} : {dom} -> {cod}", []


def _children_for_rule(name, rule):  # type: ignore[no-untyped-def]
    return name, []


def _program_parameter_groups(tmpl):  # type: ignore[no-untyped-def]
    """Render static and runtime telescopes without duplicating scalar binders."""
    from quivers.dsl.program_templates import (
        static_program_parameters,
        runtime_program_parameters,
    )
    from quivers.dsl.ast_nodes import QiecBinder

    static = [
        _emit_qiec_binder(parameter)
        if isinstance(parameter, QiecBinder)
        else _emit_program_param(parameter)
        for parameter in static_program_parameters(tmpl)
    ]
    runtime = [str(name) for name in tmpl.params or ()]
    runtime.extend(
        f"{parameter.name} : {_emit_qiec_type(parameter.type_expr)}"
        if hasattr(parameter, "type_expr")
        else _emit_program_param(parameter)
        for parameter in runtime_program_parameters(tmpl)
    )
    return static, runtime


def _children_for_program(name, tmpl):  # type: ignore[no-untyped-def]
    static, runtime = _program_parameter_groups(tmpl)
    dom = _pretty(getattr(tmpl, "domain", None))
    cod = _pretty(getattr(tmpl, "codomain", None))
    head = f"{name}"
    if static:
        head += f"[{', '.join(static)}]"
    if runtime:
        head += f"({', '.join(runtime)})"
    head += f" : {dom} -> {cod}"
    steps = getattr(tmpl, "draws", ()) or ()
    children = [_step_node(step) for step in steps]
    ret_vars = getattr(tmpl, "return_vars", ()) or ()
    if ret_vars:
        children.append((f"return {', '.join(ret_vars)}", []))
    return head, children


def _step_node(step):  # type: ignore[no-untyped-def]
    cls = type(step).__name__
    if cls == "SampleStep":
        vars_ = getattr(step, "vars", ()) or ()
        var = vars_[0] if vars_ else "?"
        idx = _index_suffix(getattr(step, "index", None))
        return f"sample {var}{idx} <- {_call_str(step)}", []
    if cls == "ObserveStep":
        vars_ = getattr(step, "vars", ()) or ()
        var = ", ".join(vars_) if vars_ else "?"
        idx = _index_suffix(getattr(step, "index", None))
        return f"observe {var}{idx} <- {_call_str(step)}", []
    if cls == "LetStep":
        return f"let {getattr(step, 'name', '?')} = ...", []
    if cls == "ScoreStep":
        return f"score {getattr(step, 'name', '?')} = ...", []
    if cls == "MarginalizeStep":
        var = getattr(step, "var", "?")
        idx = _index_suffix(getattr(step, "index", None))
        head = f"marginalize {var}{idx} <- {_call_str(step)}"
        body = [_step_node(s) for s in getattr(step, "scope", ()) or ()]
        return head, body
    if cls == "ReturnStep":
        vars_ = getattr(step, "vars", ()) or ()
        return f"return {', '.join(vars_)}", []
    return cls, []


def _index_suffix(idx):  # type: ignore[no-untyped-def]
    if idx is None:
        return ""
    return f" : {_pretty(idx)}"


def _call_str(step):  # type: ignore[no-untyped-def]
    head = getattr(step, "morphism", "?") or "?"
    return draw_call_to_source(
        str(head),
        getattr(step, "args", None),
        static_args=getattr(step, "static_args", ()),
    )


def _children_for_deduction(name, system):  # type: ignore[no-untyped-def]
    head = name
    semiring = type(getattr(system, "semiring", system)).__name__
    children = []
    rules = getattr(system, "rules", ()) or ()
    if rules:
        rule_kids = [
            (f"{getattr(r, 'name', '?')} : {_rule_line(r)}", []) for r in rules
        ]
        children.append(("rules", rule_kids))
    children.append((f"semiring: {semiring}", []))
    tol = getattr(system, "tolerance", None)
    if tol is not None and tol != 0:
        children.append((f"tolerance: {tol}", []))
    return head, children


def _rule_line(rule):  # type: ignore[no-untyped-def]
    premises = getattr(rule, "premises", ()) or ()
    conclusion = getattr(rule, "conclusion", None)
    prem_str = ", ".join(_pat_str(p) for p in premises)
    return f"{prem_str} |- {_pat_str(conclusion)}"


def _pat_str(pat):  # type: ignore[no-untyped-def]
    if pat is None:
        return "?"
    if isinstance(pat, tuple):
        return "(" + ", ".join(_pat_str(p) for p in pat) + ")"
    return str(pat)


def _children_for_signature(name, sig):  # type: ignore[no-untyped-def]
    children = []
    sorts = getattr(sig, "sorts_t", ()) or ()
    if sorts:
        children.append(
            (
                "sorts",
                [
                    (
                        f"{s.name} : {getattr(s, 'kind', '?')}"
                        + (f" [dim={s.dim}]" if getattr(s, "dim", None) else ""),
                        [],
                    )
                    for s in sorts
                ],
            )
        )
    ctors = getattr(sig, "constructors_t", ()) or ()
    if ctors:
        children.append(
            (
                "constructors",
                [(f"{c.name} : {_ctor_line(c)}", []) for c in ctors],
            )
        )
    binders = getattr(sig, "binders_t", ()) or ()
    if binders:
        children.append(("binders", [(b.name, []) for b in binders]))
    vkinds = getattr(sig, "vertex_kinds_t", ()) or ()
    if vkinds:
        children.append(("vertex_kinds", [(v.name, []) for v in vkinds]))
    ekinds = getattr(sig, "edge_kinds_t", ()) or ()
    if ekinds:
        children.append(("edge_kinds", [(e.name, []) for e in ekinds]))
    return name, children


def _ctor_line(ctor):  # type: ignore[no-untyped-def]
    args = getattr(ctor, "args", ()) or ()
    ret = getattr(ctor, "return_sort", None) or getattr(ctor, "result", "?")
    return f"{', '.join(str(a) for a in args)} -> {ret}"


def _children_for_encoder(name, enc):  # type: ignore[no-untyped-def]
    sig_name = getattr(enc, "signature_name", None) or getattr(enc, "signature", "?")
    return f"{name} : {sig_name}", []


def _children_for_decoder(name, dec):  # type: ignore[no-untyped-def]
    sig_name = getattr(dec, "signature_name", None) or getattr(dec, "signature", "?")
    return f"{name} : {sig_name}", []


def _children_for_loss(name, entry):  # type: ignore[no-untyped-def]
    kind = getattr(entry, "attachment_kind", "global")
    target = getattr(entry, "target", None)
    if target:
        return f"{name} [on={kind}({target})]", []
    return f"{name} [on={kind}]", []


def _children_for_bundle(name, members):  # type: ignore[no-untyped-def]
    return f"{name}", [(m, []) for m in members]


def _children_for_contraction(name, contr):  # type: ignore[no-untyped-def]
    return name, []


__all__ = []
