"""Canonical single-expression parsing shared by formula and retained-draw APIs."""

from quivers.dsl.ast_nodes import LetExprNode, LetStep, ProgramDecl


def _parse_value_expression(text: str) -> LetExprNode:
    """Parse one deterministic value through the ordinary QVR parser.

    Parameters
    ----------
    text : str
        One authored expression, without statement boundaries.

    Returns
    -------
    LetExprNode
        The existing QVR expression AST.

    Raises
    ------
    TypeError, ValueError
        If the input is not one nonempty expression.
    """
    from quivers.dsl import parse

    if not isinstance(text, str):
        raise TypeError("a value expression must be text")
    if not text.strip() or any(mark in text for mark in ("\n", "\r", ";")):
        raise ValueError("value expressions require one nonempty QVR expression")
    module = parse(
        "object GraphRow : FinSet 1\n"
        "program graph_expression : GraphRow -> GraphRow\n"
        f"    let graph_value = {text}\n"
        "    return graph_value\n"
    )
    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    if len(program.draws) != 1 or not isinstance(program.draws[0], LetStep):
        raise ValueError("input must contain exactly one value expression")
    return program.draws[0].value


__all__: list[str] = []
