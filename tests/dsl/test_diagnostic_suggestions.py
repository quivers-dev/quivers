"""Every code fix a compile diagnostic suggests parses.

Each case triggers one diagnostic that offers a fix written as QVR
source, checks that the message carries the suggestion verbatim, and
parses the suggestion inside a module where it belongs. A suggestion
that names syntax the grammar does not have (``algebra X``, ``Term :
object dim 64``, ``observe r : N via idx <- ...``) fails here.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from quivers.dsl import CompileError, loads, parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError

_AUTOENCODER = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "examples"
    / "source"
    / "term_autoencoder.qvr"
).read_text()


def _message(source: str) -> str:
    """The message of the diagnostic compiling ``source`` raises.

    Parameters
    ----------
    source : str
        QVR source that fails to compile.

    Returns
    -------
    str
        The diagnostic's message.
    """
    with pytest.raises((CompileError, QiecDiagnosticError)) as caught:
        loads(textwrap.dedent(source))
    return str(caught.value)


_GROUPED = """\
object Doc : FinSet 2
object Topic : FinSet 2
object Vocab : FinSet 3
object Token : FinSet 6
object Mix : Real 2

program lda : Token -> Mix
    sample theta : Doc <- Dirichlet(concentration=1.0) [over=Topic, iid_over=Doc]
    sample phi : Topic <- Dirichlet(concentration=0.5) [over=Vocab, iid_over=Topic]
    let theta_tok = theta[doc_idx]
    marginalize z : Topic <- Categorical(probs=theta_tok) [over=Token]
        observe w : Token <- Categorical(probs=phi[z]) [via=token_idx]
    return theta

export lda
"""

_GROUPED_TEMPLATE = (
    "object K : FinSet 2\nobject G : FinSet 3\nobject N : FinSet 6\n"
    "object M : Real 2\nprogram p : N -> M\n"
    "    sample mu : K <- Normal(loc=0.0, scale=1.0)\n"
    "    {marginalize}\n        {observe}\n    return mu\n"
)


_CASES: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "identity under a magmoid",
        """
        object A : FinSet 2
        composition material_impl [level=magmoid]
        define i = identity(A)
        export i
        """,
        [
            (
                "composition product_fuzzy [level=algebra]",
                "object A : FinSet 2\n{s}\ndefine i = identity(A)\nexport i\n",
            )
        ],
    ),
    (
        "sort without a dim",
        """
        signature S
            sorts
                Term : object
            constructors
                App : Term, Term -> Term
        encoder E : S
            op App(fun, arg) |-> relu(fun + arg)
        """,
        [
            (
                "Term : object [dim=64]",
                "signature S\n    sorts\n        {s}\n    constructors\n"
                "        App : Term, Term -> Term\n",
            ),
            (
                "dim Term = 64",
                "signature S\n    sorts\n        Term : object\n    constructors\n"
                "        App : Term, Term -> Term\nencoder E : S\n    {s}\n",
            ),
        ],
    ),
    (
        "observed morphism without an initializer",
        """
        object A : FinSet 2
        object B : FinSet 3
        morphism f : A -> B [role=observed]
        export f
        """,
        [
            (
                'morphism f : A -> B [role=observed] ~ from_data("f")',
                "object A : FinSet 2\nobject B : FinSet 3\n{s}\n",
            ),
            (
                "~ identity(A)",
                "object A : FinSet 2\nmorphism g : A -> A [role=observed] {s}\n",
            ),
        ],
    ),
    (
        "draw from a learned function",
        """
        object X : Real 2
        object Y : Real 1
        morphism f : X -> Y [role=function]
        program p : X -> Y
            sample s <- f(x)
            return s
        export p
        """,
        [
            (
                "let v = f(x)",
                "object X : Real 2\nprogram p : X -> X\n    {s}\n    return v\n",
            ),
            (
                "sample s <- Normal(loc=v, scale=1.0)",
                "object X : Real 2\nprogram p : X -> X\n    {s}\n    return s\n",
            ),
        ],
    ),
    (
        "draw from a decoder",
        _AUTOENCODER
        + """
object Code : FinSet 24
program p : Code -> Code
    sample h : Code <- Normal(loc=0.0, scale=1.0)
    sample t <- Dec(h)
    return h
export p
""",
        [
            (
                "loss l [on=decoder(Dec)]",
                _AUTOENCODER + "\n{s}\n    -Dec.log_prob(term, code)\n",
            ),
        ],
    ),
    (
        "rule attachment missing its deduction",
        """
        deduction D : Item -> Item [semiring=LogProb]
            atoms NP, S
            rule app : NP, NP |- S
        loss per_app [weight=1.0, on=rule(app)]
            0.5
        """,
        [
            (
                "on=rule(RULE, DEDUCTION)",
                "loss l [weight=1.0, {s}]\n    0.5\n",
            ),
        ],
    ),
    (
        "grouped marginalize without a class annotation",
        _GROUPED.replace("marginalize z : Topic <-", "marginalize z <-"),
        [
            (
                "marginalize c : K <- Categorical(probs=weights) [over=G]",
                _GROUPED_TEMPLATE.replace("{marginalize}", "{s}").replace(
                    "{observe}",
                    "observe r : N <- Normal(loc=mu[c], scale=1.0) [via=idx]",
                ),
            ),
        ],
    ),
    (
        "grouped marginalize observe without via",
        _GROUPED.replace(" [via=token_idx]", ""),
        [
            (
                "observe r : N <- Normal(loc=mu[c], scale=1.0) [via=idx]",
                _GROUPED_TEMPLATE.replace("{observe}", "{s}").replace(
                    "{marginalize}",
                    "marginalize c : K <- Categorical(probs=weights) [over=G]",
                ),
            ),
        ],
    ),
]


@pytest.mark.parametrize(
    ("trigger", "suggestions"),
    [(trigger, suggestions) for _, trigger, suggestions in _CASES],
    ids=[name for name, _, _ in _CASES],
)
def test_suggested_fix_parses(trigger: str, suggestions: list[tuple[str, str]]) -> None:
    """The diagnostic prints each suggestion, and each parses in context."""
    message = _message(trigger)
    for suggestion, template in suggestions:
        assert suggestion in message, (suggestion, message)
        parse(template.replace("{s}", suggestion))


def test_the_old_identity_suggestion_did_not_parse() -> None:
    """``algebra X``, which the identity diagnostic used to suggest, is refused."""
    with pytest.raises(Exception):  # noqa: B017 - any parse failure
        parse("algebra X\n")
