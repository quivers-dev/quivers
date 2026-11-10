"""A binder may not be spelled as a word a let expression reads otherwise.

A let expression reads ``unit``, ``true``, ``false``, ``not``, and
``factor`` as a literal or a keyword, so a binder with one of those names
could be bound but never read. The source is refused at the binder, on
every route that reads it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from quivers.dsl import loads
from quivers.dsl.parser import RESERVED_LET_WORDS, ParseError, parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec

_GRAMMAR = Path(__file__).resolve().parents[2] / "grammars/qvr/src/grammar.json"

_HEADER = (
    "object Cell : FinSet 2\nobject Target : Real 1\n\nprogram total : Cell -> Target\n"
)

#: Each binder kind: its program body, with ``NAME`` where the binder is
#: and its first reference, and the binder's 1-based line and 0-based
#: column once the header precedes it.
BINDERS: dict[str, tuple[str, int, int]] = {
    "sample site": (
        "    sample NAME : Cell <- Normal(loc=0.0, scale=1.0)\n"
        "    let t = NAME * 2.0\n    return t\n",
        5,
        11,
    ),
    "observed site": (
        "    observe NAME : Cell <- Normal(loc=0.0, scale=1.0)\n"
        "    let t = NAME * 2.0\n    return t\n",
        5,
        12,
    ),
    "sample site in a tuple": (
        "    sample (a, NAME) <- pair\n    let t = NAME * 2.0\n    return t\n",
        5,
        15,
    ),
    "marginalized variable": (
        "    marginalize NAME : Cell <- Categorical(probs=[0.5, 0.5])\n"
        "        observe y <- Normal(loc=0.0, scale=1.0)\n    return y\n",
        5,
        16,
    ),
    "let binding": (
        "    sample v : Cell <- Normal(loc=0.0, scale=1.0)\n"
        "    let NAME = v * 2.0\n    let t = NAME * 2.0\n    return t\n",
        6,
        8,
    ),
    "call binding": (
        "    let NAME <- other(1.0)\n    return NAME\n",
        5,
        8,
    ),
    "score": (
        "    sample v : Cell <- Normal(loc=0.0, scale=1.0)\n"
        "    score NAME = sum(v)\n    return v\n",
        6,
        10,
    ),
    "factor binder": (
        "    sample v : Cell <- Normal(loc=0.0, scale=1.0)\n"
        "    let t = factor NAME : Cell in v[NAME] * 2.0\n    return t\n",
        6,
        19,
    ),
    "lambda parameter": (
        "    sample v : Cell <- Normal(loc=0.0, scale=1.0)\n"
        "    let t = map(v, NAME -> NAME * 2.0)\n    return t\n",
        6,
        19,
    ),
    "curried lambda parameter": (
        "    sample v : Cell <- Normal(loc=0.0, scale=1.0)\n"
        "    let t = fold(v, 0.0, acc -> NAME -> acc + NAME)\n    return t\n",
        6,
        32,
    ),
}


def _source(kind: str, word: str) -> str:
    """A program binding ``word`` in the given binder position.

    Parameters
    ----------
    kind : str
        The key of the binder kind in `BINDERS`.
    word : str
        The binder's name.

    Returns
    -------
    str
        The QVR source.
    """
    body, _, _ = BINDERS[kind]
    return _HEADER + body.replace("NAME", word) + "\nexport total\n"


def _role(kind: str) -> str:
    """The binder's role as the diagnostic names it.

    Parameters
    ----------
    kind : str
        The key of the binder kind in `BINDERS`.

    Returns
    -------
    str
        The role.
    """
    return {
        "sample site in a tuple": "sample site",
        "curried lambda parameter": "lambda parameter",
    }.get(kind, kind)


@pytest.mark.parametrize("word", tuple(RESERVED_LET_WORDS))
@pytest.mark.parametrize("kind", tuple(BINDERS))
def test_a_reserved_binder_is_refused_where_it_is_bound(kind: str, word: str) -> None:
    source = _source(kind, word)
    _, line, col = BINDERS[kind]
    message = (
        rf"<source>: reserved word at line {line}, col {col}: {word!r} names "
        rf"{re.escape(RESERVED_LET_WORDS[word])} in expressions, so it cannot "
        rf"name a {_role(kind)}"
    )
    # The compiled route and the checked route both read the source first.
    with pytest.raises(ParseError, match=message):
        loads(source)
    with pytest.raises(ParseError, match=message):
        lower_qvr_to_qiec(parse(source))


@pytest.mark.parametrize("word", tuple(RESERVED_LET_WORDS))
def test_a_reserved_program_parameter_is_refused(word: str) -> None:
    source = (
        "object Cell : FinSet 2\n\n"
        f"program total({word}) : Cell -> Cell\n"
        "    sample v <- Normal(loc=0.0, scale=1.0)\n    return v\n\nexport total\n"
    )
    with pytest.raises(ParseError, match=rf"at line 3, col 14: {word!r} names"):
        parse(source)


def test_the_reported_binder_is_the_first_in_the_source() -> None:
    source = _HEADER + (
        "    sample true : Cell <- Normal(loc=0.0, scale=1.0)\n"
        "    let unit = true * 2.0\n    return unit\n\nexport total\n"
    )
    with pytest.raises(ParseError, match=r"line 5, col 11: 'true'"):
        parse(source)


def test_the_name_unit_in_a_comment_or_another_word_is_not_a_binder() -> None:
    source = _HEADER + (
        "    # map(xs, unit -> unit) is how a reserved lambda would read\n"
        "    sample units : Cell <- Normal(loc=0.0, scale=1.0)\n"
        "    let t = fold(units, 0.0, acc -> u -> acc + u)\n    return t\n\n"
        "export total\n"
    )
    loads(source)


@pytest.mark.parametrize("word", ("in", "Unit", "units", "Real", "FinSet", "Simplex"))
def test_a_word_expressions_read_as_a_name_binds(word: str) -> None:
    source = _source("sample site", word)
    loads(source)


def _let_expression_keywords() -> frozenset[str]:
    """The word-shaped tokens a let expression's grammar contains.

    Returns
    -------
    frozenset[str]
        Every keyword reachable from ``_let_arith``, apart from those only
        an object expression (a factor binder's index) reaches.
    """
    rules = json.loads(_GRAMMAR.read_text())["rules"]

    def keywords(start: str, skip: frozenset[str]) -> set[str]:
        seen: set[str] = set()
        found: set[str] = set()

        def walk(node: dict[str, object]) -> None:
            kind = node["type"]
            if kind == "STRING":
                value = str(node["value"])
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
                    found.add(value)
            elif kind == "SYMBOL":
                name = str(node["name"])
                if name in rules and name not in seen and name not in skip:
                    seen.add(name)
                    walk(rules[name])
            elif "members" in node:
                for member in node["members"]:  # type: ignore[union-attr]
                    walk(member)
            elif "content" in node:
                walk(node["content"])  # type: ignore[arg-type]

        walk(rules[start])
        return found

    return frozenset(keywords("_let_arith", frozenset({"_object_expr"})))


def test_the_reserved_words_are_the_let_expression_keywords() -> None:
    # `in` separates a factor's binders from its body; nowhere else in an
    # expression is it a keyword, so a name `in` is read as a name.
    assert _let_expression_keywords() - {"in"} == frozenset(RESERVED_LET_WORDS)
