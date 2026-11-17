"""brms formula-level syntax: the ``Intercept`` term and dot-notation updates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import io
import itertools
import re
import tokenize

BRMS_INTERCEPT = "qvrBrmsInterceptOnes"
"""Reserved column holding brms's ``Intercept`` term, a constant of one."""


def _tokens(formula: str) -> list[tokenize.TokenInfo]:
    """Tokenize a formula as Python source, dropping layout tokens.

    Parameters
    ----------
    formula : str
        Formula text.

    Returns
    -------
    list[tokenize.TokenInfo]
        Significant tokens.
    """
    return [
        token
        for token in tokenize.generate_tokens(io.StringIO(formula).readline)
        if token.type
        not in {
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.COMMENT,
            tokenize.ENDMARKER,
        }
    ]


def _offsets(formula: str) -> list[int]:
    """Return the character offset of each line start.

    Parameters
    ----------
    formula : str
        Formula text.

    Returns
    -------
    list[int]
        Offsets, one per line plus the end.
    """
    starts = [0]
    for line in formula.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def normalize_intercept_term(formula: str) -> str:
    """Replace brms's ``Intercept`` variable by a reserved constant column.

    brms treats a right-hand-side variable named ``Intercept`` as a column of
    ones in the population design, so ``y ~ 0 + Intercept + x`` estimates the
    intercept as an ordinary coefficient of class ``b`` without centering.

    Parameters
    ----------
    formula : str
        Formula text.

    Returns
    -------
    str
        The formula with each right-hand-side ``Intercept`` name (not a call)
        replaced by `BRMS_INTERCEPT`; unchanged when there is none.
    """
    tilde = formula.find("~")
    if tilde < 0:
        return formula
    tokens = _tokens(formula)
    starts = _offsets(formula)
    spans = []
    for index, token in enumerate(tokens):
        position = starts[token.start[0] - 1] + token.start[1]
        following = tokens[index + 1].string if index + 1 < len(tokens) else ""
        if (
            token.type == tokenize.NAME
            and token.string == "Intercept"
            and position > tilde
            and following != "("
        ):
            spans.append((position, position + len("Intercept")))
    for start, end in reversed(spans):
        formula = formula[:start] + BRMS_INTERCEPT + formula[end:]
    return formula


def display_label(name: str) -> str:
    """Restore the authored ``Intercept`` label of a reserved column name.

    Parameters
    ----------
    name : str
        Evaluated term or column label.

    Returns
    -------
    str
        The label with the reserved column spelled ``Intercept``.
    """
    return name.replace(BRMS_INTERCEPT, "Intercept").replace(R_SCALE, "scale")


def _split_formula(formula: str) -> tuple[str, str]:
    """Split a formula at its top-level ``~``.

    Parameters
    ----------
    formula : str
        Two-sided or one-sided formula.

    Returns
    -------
    tuple[str, str]
        Left-hand side (empty for one-sided) and right-hand side.

    Raises
    ------
    ValueError
        If the formula has no ``~``.
    """
    if "~" not in formula:
        raise ValueError(f"formula {formula!r} has no '~'")
    left, right = formula.split("~", 1)
    return left.strip(), right.strip()


def expand_dot_formula(old: str, new: str) -> str:
    """Expand R's dot notation of ``update.formula`` against a fitted formula.

    In ``new``, a ``.`` on the left stands for the old response side and a
    ``.`` on the right for the old right-hand side, grouped in parentheses
    as R's ``update.formula`` substitutes it; ``- term`` then removes that
    term through ordinary formula algebra.

    Parameters
    ----------
    old : str
        The fitted formula.
    new : str
        A formula that may contain ``.`` on either side, such as
        ``". ~ . + z"`` or ``"log(.) ~ . - x"``.

    Returns
    -------
    str
        The expanded formula; ``new`` itself when it contains no dot.

    Raises
    ------
    ValueError
        If either formula lacks ``~``, or ``new`` uses a dot on the left of a
        one-sided formula.
    """
    old_left, old_right = _split_formula(old)
    if "~" not in new:
        raise ValueError(f"formula {new!r} has no '~'")
    new_left, new_right = _split_formula(new)
    dot = re.compile(r"(?<![\w.])\.(?![\w.])")
    if new_left and dot.search(new_left):
        if not old_left:
            raise ValueError("a dot on the left needs a fitted two-sided formula")
        new_left = dot.sub(lambda _: old_left, new_left)
    elif not new_left:
        new_left = old_left
    if dot.search(new_right):
        new_right = dot.sub(lambda _: f"({old_right})", new_right)
    return f"{new_left} ~ {new_right}" if new_left else f"~ {new_right}"


R_SCALE = "qvrRScale"
"""Reserved call name for R's ``scale``, which divides by the n - 1 deviation."""


def _close(text: str, start: int) -> int:
    """Return the index of the parenthesis closing the one at ``start``.

    Parameters
    ----------
    text : str
        Formula text.
    start : int
        Index of an opening parenthesis.

    Returns
    -------
    int
        Index of its closing parenthesis.

    Raises
    ------
    ValueError
        If the parentheses are unbalanced.
    """
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unbalanced parentheses in formula {text!r}")


def _split_top(text: str, separator: str) -> list[str]:
    """Split text at a separator outside parentheses.

    Parameters
    ----------
    text : str
        Expression text.
    separator : str
        One-character separator.

    Returns
    -------
    list[str]
        Stripped pieces.
    """
    pieces, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == separator and depth == 0:
            pieces.append(text[start:index].strip())
            start = index + 1
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]


def _power_inside_identity(text: str) -> str:
    """Spell R's ``^`` as ``**`` inside every ``I(...)`` call.

    Parameters
    ----------
    text : str
        Formula text.

    Returns
    -------
    str
        The text with arithmetic powers in ``I()`` in Python spelling.
    """
    out, index = [], 0
    for match in re.finditer(r"(?<![\w.])I\s*\(", text):
        if match.start() < index:
            continue
        opening = match.end() - 1
        closing = _close(text, opening)
        out.append(text[index:opening])
        out.append(text[opening : closing + 1].replace("^", "**"))
        index = closing + 1
    out.append(text[index:])
    return "".join(out)


def _expand_crossing(text: str) -> str:
    """Expand R's crossing operator ``(a + b + c)^k`` into its interactions.

    Parameters
    ----------
    text : str
        Formula right-hand side.

    Returns
    -------
    str
        Every interaction of at most ``k`` distinct terms of the group, in
        R's order (all main effects, then all pairs, and so on).

    Raises
    ------
    ValueError
        If the exponent is not a positive integer.
    """
    pattern = re.compile(r"\)\s*\^\s*(\d+)")
    while True:
        match = pattern.search(text)
        if match is None:
            return text
        closing = match.start()
        depth, opening = 0, closing
        for index in range(closing, -1, -1):
            if text[index] == ")":
                depth += 1
            elif text[index] == "(":
                depth -= 1
                if depth == 0:
                    opening = index
                    break
        order = int(match.group(1))
        if order < 1:
            raise ValueError("a crossing exponent must be a positive integer")
        terms = _split_top(text[opening + 1 : closing], "+")
        expanded = [
            ":".join(combination)
            for size in range(1, min(order, len(terms)) + 1)
            for combination in itertools.combinations(terms, size)
        ]
        text = text[:opening] + "(" + " + ".join(expanded) + ")" + text[match.end() :]


def normalize_r_syntax(formula: str) -> str:
    """Translate R formula operators the formula parser spells differently.

    R and brms read ``^`` inside ``I()`` as a power, ``(a + b)^k`` as all
    interactions up to order ``k``, ``a %in% b`` as the nested term
    ``a:b``, and ``scale(x)`` as ``(x - mean(x)) / sd(x)`` with the
    ``n - 1`` standard deviation.

    Parameters
    ----------
    formula : str
        Formula text.

    Returns
    -------
    str
        Equivalent text for the formula parser; unchanged when no operator
        applies.
    """
    if "~" not in formula:
        return formula
    left, right = formula.split("~", 1)
    right = _power_inside_identity(right)
    right = _expand_crossing(right)
    right = re.sub(r"\s*%in%\s*", ":", right)
    right = re.sub(r"(?<![\w.])scale\s*\(", R_SCALE + "(", right)
    return left + "~" + right


def without_zero_intercept(formula: str) -> str:
    """Remove ``0 +``, ``+ 0`` and ``- 1`` from a formula's right-hand side.

    Parameters
    ----------
    formula : str
        Formula text.

    Returns
    -------
    str
        The formula with its intercept restored.
    """
    left, right = formula.split("~", 1)
    right = re.sub(r"(^|\+)\s*0\s*(?=\+|$)", r"\1", right.strip())
    right = re.sub(r"-\s*1(?![\w.])", "", right)
    right = re.sub(r"^\s*\+\s*|\s*\+\s*$", "", right.strip())
    right = re.sub(r"\+\s*\+", "+", right)
    return f"{left}~ {right or '1'}"


def sparse_center(center: bool | None, decomp: str, *, sparse: bool) -> bool | None:
    """Apply brms's rule that sparse designs are not centered.

    Parameters
    ----------
    center : bool or None
        Requested centering.
    decomp : str
        Requested decomposition.
    sparse : bool
        brms ``sparse``.

    Returns
    -------
    bool or None
        False for a sparse design, ``center`` otherwise.

    Raises
    ------
    ValueError
        If a sparse design requests centering or the QR decomposition, which
        brms refuses for sparse matrices.
    """
    if not sparse:
        return center
    if decomp != "none":
        raise ValueError("decomp='QR' cannot be used with sparse=True, as in brms")
    if center is True:
        raise ValueError("a sparse design is not centered; omit center=True")
    return False


def checked_unused(
    unused: Sequence[str], columns: Iterable[str], *, caller: str
) -> tuple[str, ...]:
    """Validate brms ``unused`` column names against the data.

    Parameters
    ----------
    unused : Sequence[str]
        Column names, or a one-sided formula ``"~ a + b"``.
    columns : Iterable[str]
        Available data columns.
    caller : str
        Entry point named in errors.

    Returns
    -------
    tuple[str, ...]
        The column names.

    Raises
    ------
    ValueError
        If a name is not a data column.
    """
    if isinstance(unused, str):
        names = tuple(
            name.strip() for name in unused.replace("~", " ").split("+") if name.strip()
        )
    else:
        names = tuple(unused)
    available = set(columns)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"{caller}: unused names absent columns {missing}")
    return names


__all__ = [
    "BRMS_INTERCEPT",
    "R_SCALE",
    "checked_unused",
    "expand_dot_formula",
    "normalize_intercept_term",
    "normalize_r_syntax",
    "sparse_center",
    "without_zero_intercept",
]
