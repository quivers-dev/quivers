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
    return (
        name.replace(BRMS_INTERCEPT, "Intercept")
        .replace(R_SCALE, "scale")
        .replace(R_BS, "bs")
    )


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


_DOT = re.compile(r"(?<![\w.])\.(?![\w.])")
"""A lone ``.``: not part of a number or of a dotted name."""

_BINARY_BEFORE_SIGN = frozenset("*:/^|%,(=<>!&~")
"""Characters after which ``+`` or ``-`` is a unary sign inside a term."""


def _signed_summands(text: str) -> list[tuple[bool, str]]:
    """Split an expression into its top-level signed summands.

    Parameters
    ----------
    text : str
        Right-hand side of a formula, or the inside of a parenthesis.

    Returns
    -------
    list[tuple[bool, str]]
        ``(added, summand)`` pairs in order; ``added`` is False for a
        summand preceded by ``-``. A sign directly after another operator,
        or in the exponent of a number such as ``1e-3``, belongs to its term.

    Raises
    ------
    ValueError
        If the parentheses are unbalanced or the expression ends on a sign.
    """
    pieces: list[tuple[bool, str]] = []
    depth, start, added = 0, 0, True
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced parentheses in formula {text!r}")
        elif char in "+-" and depth == 0:
            before = text[start:index].strip()
            unary = bool(before) and (
                before[-1] in _BINARY_BEFORE_SIGN
                or re.search(r"(?<![\w.])(\d+\.?\d*|\.\d+)[eE]$", before) is not None
            )
            if unary:
                continue
            if before:
                pieces.append((added, before))
                added = char == "+"
            else:
                # Consecutive signs multiply, as in ``x + -1``.
                added = added == (char == "+")
            start = index + 1
    if depth:
        raise ValueError(f"unbalanced parentheses in formula {text!r}")
    last = text[start:].strip()
    if not last:
        raise ValueError(f"formula {text!r} ends without a term")
    pieces.append((added, last))
    return pieces


def _is_group_term(summand: str) -> bool:
    """Whether a summand is a parenthesized group-level term such as ``(1 | g)``.

    Parameters
    ----------
    summand : str
        One top-level summand.

    Returns
    -------
    bool
        True when the summand is one parenthesis holding a ``|`` at its
        own level.
    """
    if not summand.startswith("(") or _close(summand, 0) != len(summand) - 1:
        return False
    depth = 0
    for char in summand[1:-1]:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "|" and depth == 0:
            return True
    return False


def _flat_summands(text: str) -> list[tuple[bool, str]]:
    """Signed summands with parenthesized sums opened, as R's ``terms`` does.

    Parameters
    ----------
    text : str
        Right-hand side of a formula.

    Returns
    -------
    list[tuple[bool, str]]
        Signed summands in order. A summand that is one parenthesis around a
        sum is replaced by that sum's summands; under a ``-`` each of its
        added summands is removed and its removed summands are dropped, as
        ``a - (b - c)`` removes only ``b`` in R. A group-level term stays
        whole.

    Raises
    ------
    ValueError
        If the expression is malformed.
    """
    flat: list[tuple[bool, str]] = []
    for added, summand in _signed_summands(text):
        whole = summand.startswith("(") and _close(summand, 0) == len(summand) - 1
        if not whole or _is_group_term(summand):
            flat.append((added, summand))
            continue
        for inner_added, inner in _flat_summands(summand[1:-1]):
            if added:
                flat.append((inner_added, inner))
            elif inner_added:
                flat.append((False, inner))
    return flat


def _term_key(summand: str) -> str:
    """Identity of a summand for R's term matching: its text without spaces.

    Parameters
    ----------
    summand : str
        One summand.

    Returns
    -------
    str
        The comparison key.
    """
    return re.sub(r"\s+", "", summand)


def expand_dot_formula(old: str, new: str) -> str:
    """Update a fitted formula as R's ``update.formula`` does.

    In ``new``, a ``.`` on the left stands for the old response side and a
    ``.`` on the right for the old right-hand side. The right-hand side is
    then simplified term by term in the order written: ``+ term`` adds a
    term that is not yet present, ``- term`` removes a present term,
    parenthesized sums are opened, and a group-level term such as
    ``(1 | g)`` is one term that is kept, added, or removed whole. Removing
    the intercept is written ``- 1`` or ``+ 0`` and is placed last as
    ``- 1``; ``+ 1`` restores it. An emptied right-hand side is ``1``.

    A term is matched by its text without spaces, so ``- x:z`` after
    ``x * z`` is kept as a removal for the formula algebra instead of being
    cancelled here; R prints the same model with ``x * z`` expanded.

    Parameters
    ----------
    old : str
        The fitted formula.
    new : str
        A formula that may contain ``.`` on either side, such as
        ``". ~ . + z"``, ``"~ . - x"`` or ``"log(.) ~ . - (1 | g)"``.

    Returns
    -------
    str
        The updated formula; ``new`` itself, stripped, when its right-hand
        side has no dot.

    Raises
    ------
    ValueError
        If either formula lacks ``~``, ``new`` uses a dot on the left of a
        one-sided fitted formula, a side is malformed, or a dot inside a
        larger term, such as ``.:z``, meets a fitted formula with
        group-level terms.
    """
    old_left, old_right = _split_formula(old)
    new_left, new_right = _split_formula(new)
    if new_left and _DOT.search(new_left):
        if not old_left:
            raise ValueError("a dot on the left needs a fitted two-sided formula")
        new_left = _DOT.sub(lambda _: old_left, new_left)
    elif not new_left:
        new_left = old_left
    if _DOT.search(new_right):
        new_right = _updated_right_side(old_right, new_right)
    return f"{new_left} ~ {new_right}" if new_left else f"~ {new_right}"


def _updated_right_side(old_right: str, new_right: str) -> str:
    """Substitute the old right-hand side for ``.`` and simplify the terms.

    Parameters
    ----------
    old_right : str
        Right-hand side of the fitted formula.
    new_right : str
        Right-hand side of the update, containing at least one ``.``.

    Returns
    -------
    str
        The simplified right-hand side.

    Raises
    ------
    ValueError
        If a side is malformed, or a dot inside a larger term meets
        group-level terms.
    """
    old_terms = _flat_summands(old_right)
    has_group = any(_is_group_term(term) for _, term in old_terms)
    directives: list[tuple[bool, str]] = []
    for added, summand in _flat_summands(new_right):
        if _term_key(summand) == ".":
            directives.extend(
                (inner_added, inner) if added else (False, inner)
                for inner_added, inner in old_terms
                if added or inner_added
            )
        elif _DOT.search(summand):
            if has_group:
                raise ValueError(
                    f"update formula term {summand!r} combines '.' with another "
                    f"operator, but the fitted right-hand side {old_right!r} has "
                    "group-level terms; write the combined terms out"
                )
            directives.append((added, _DOT.sub(lambda _: f"({old_right})", summand)))
        else:
            directives.append((added, summand))

    kept: dict[str, str] = {}
    removed: dict[str, str] = {}
    intercept = True
    for added, summand in directives:
        key = _term_key(summand)
        if key in {"0", "1"}:
            intercept = added == (key == "1")
            continue
        if added:
            removed.pop(key, None)
            kept.setdefault(key, summand)
        elif key in kept:
            del kept[key]
        else:
            removed.setdefault(key, summand)
    text = " + ".join(kept.values())
    if not text:
        text = "1" if intercept else "0"
        intercept = True
    for summand in removed.values():
        text += f" - {summand}"
    return text if intercept else f"{text} - 1"


R_SCALE = "qvrRScale"
"""Reserved call name for R's ``scale``, which divides by the n - 1 deviation."""

R_BS = "qvrRBs"
"""Reserved call name for R's ``splines::bs``, which drops the first basis column."""


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
    ``a:b``, ``scale(x)`` as ``(x - mean(x)) / sd(x)`` with the
    ``n - 1`` standard deviation, and ``as.factor(x)`` as ``factor(x)``.

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
    right = re.sub(r"(?<![\w.])bs\s*\(", R_BS + "(", right)
    right = re.sub(r"(?<![\w.])as\.factor\s*\(", "factor(", right)
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
    "R_BS",
    "R_SCALE",
    "checked_unused",
    "expand_dot_formula",
    "normalize_intercept_term",
    "normalize_r_syntax",
    "sparse_center",
    "without_zero_intercept",
]
