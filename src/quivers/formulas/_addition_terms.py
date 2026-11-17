"""Edit the response addition terms of a formula: brms ``update_adterms``."""

from __future__ import annotations

import ast
from typing import Literal, overload

from quivers.formulas._multivariate import ResponseModel


def _terms(text: str) -> list[ast.Call]:
    """Split a sum of addition-term calls.

    Parameters
    ----------
    text : str
        Calls joined by ``+``, such as ``trials(n) + weights(w)``.

    Returns
    -------
    list[ast.Call]
        The calls in written order.

    Raises
    ------
    ValueError
        If a summand is not a named call.
    """
    try:
        expression = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as error:
        raise ValueError(
            f"addition terms {text!r} are not calls joined by '+'"
        ) from error
    pending = [expression]
    calls: list[ast.Call] = []
    while pending:
        node = pending.pop()
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            pending.extend((node.right, node.left))
            continue
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError(f"addition term {ast.unparse(node)!r} is not a named call")
        calls.append(node)
    return calls


@overload
def update_adterms(
    formula: str, adform: str, action: Literal["update", "replace"] = "update"
) -> str: ...


@overload
def update_adterms(
    formula: ResponseModel,
    adform: str,
    action: Literal["update", "replace"] = "update",
) -> ResponseModel: ...


def update_adterms(
    formula: str | ResponseModel,
    adform: str,
    action: Literal["update", "replace"] = "update",
) -> str | ResponseModel:
    """Update or replace the response addition terms of a formula.

    brms ``update_adterms``: ``"update"`` replaces terms of the same kind
    and adds new kinds; ``"replace"`` drops every existing term first.

    Parameters
    ----------
    formula : str or ResponseModel
        A formula such as ``"y | weights(w) ~ x"``, or a `bf` response.
    adform : str
        The new terms as a one-sided formula, ``"~ trials(10) + cens(c)"``.
    action : {"update", "replace"}
        Keep unrelated existing terms, or drop them all.

    Returns
    -------
    str or ResponseModel
        The edited formula, of the same kind as ``formula``.

    Raises
    ------
    ValueError
        If ``action`` is unknown, ``adform`` is not one-sided, or a term is
        not a named call.
    """
    if action not in ("update", "replace"):
        raise ValueError("action must be 'update' or 'replace'")
    if isinstance(formula, ResponseModel):
        return formula.with_(formula=update_adterms(formula.formula, adform, action))
    left, separator, rhs = adform.partition("~")
    if not separator or left.strip():
        raise ValueError("adform must be a one-sided formula such as '~ trials(10)'")
    new = _terms(rhs)
    lhs, separator, right = formula.partition("~")
    if not separator:
        raise ValueError("formula requires '~'")
    response, bar, existing = lhs.partition("|")
    current = _terms(existing) if bar and action == "update" else []
    names = [call.func.id for call in new if isinstance(call.func, ast.Name)]
    kept = [
        call
        for call in current
        if isinstance(call.func, ast.Name) and call.func.id not in names
    ]
    terms = " + ".join(ast.unparse(call) for call in (*kept, *new))
    return (
        f"{response.strip()} | {terms} ~{right}"
        if terms
        else f"{response.strip()} ~{right}"
    )


__all__ = ["update_adterms"]
