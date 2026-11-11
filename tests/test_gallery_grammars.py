"""Grammar-level properties of the gallery's weighted-deduction examples.

Each gallery grammar claims a construction; these tests check the claim
against the compiled deduction rather than against its weights:

* every CCG and custom-rules combinator is needed: a showcase sentence
  parses with the full rule set and fails once that rule is removed;
* the weighted MCFG's productions are linear (each premise interval is
  used exactly once, in order, in the conclusion), its rank-2 items are
  genuinely discontinuous, it generates the cross-serial order that the
  nested context-free approximation rejects and rejects the nested order
  that approximation accepts, and the page's direct description of its
  language agrees with brute-force enumeration;
* the multimodal type-logical and quantifier-scope grammars score the
  same goal weight on the agenda and on the checked search, at learnable
  rule and lexicon weights read under the same keys.
"""

from __future__ import annotations

import dataclasses
import itertools
import re

import pytest
import torch

from quivers.dsl import load, loads
from quivers.qiec.program_runtime import run_deduction
from quivers.stochastic.agenda import DeductionSystem, Wildcard

_SOURCE = "docs/examples/source/{stem}.qvr"


def _deduction(stem: str, name: str) -> DeductionSystem:
    """The named deduction of a gallery source.

    Parameters
    ----------
    stem : str
        The source file's stem.
    name : str
        The deduction's name.

    Returns
    -------
    DeductionSystem
        The compiled deduction.
    """
    return load(_SOURCE.format(stem=stem)).deductions[name]


def _parses(system: DeductionSystem, sentence: str) -> bool:
    """Whether the system derives its goal over the whole sentence.

    Parameters
    ----------
    system : DeductionSystem
        The deduction.
    sentence : str
        Space-separated tokens.

    Returns
    -------
    bool
        True when the goal weight is finite.
    """
    return bool(torch.isfinite(system(sentence.split()).goal_weight()))


def _without(system: DeductionSystem, rule_name: str) -> DeductionSystem:
    """The system with one rule removed.

    Parameters
    ----------
    system : DeductionSystem
        The deduction.
    rule_name : str
        The rule to remove.

    Returns
    -------
    DeductionSystem
        A copy of the system lacking that rule.

    Raises
    ------
    AssertionError
        If the system has no rule of that name.
    """
    rules = tuple(rule for rule in system.rules if rule.name != rule_name)
    assert len(rules) == len(system.rules) - 1, rule_name
    return dataclasses.replace(system, rules=rules)


_NEEDED = [
    ("ccg", "CCG", "fwd_app", "the cat sleeps"),
    ("ccg", "CCG", "bwd_app", "the cat sleeps"),
    ("ccg", "CCG", "fwd_comp", "the dog whom Mary saw sleeps"),
    ("ccg", "CCG", "bwd_comp", "the cat that sleeps today purrs"),
    ("ccg", "CCG", "fwd_xcomp", "Mary probably sleeps"),
    ("ccg", "CCG", "bwd_xcomp", "Mary saw yesterday the cat"),
    ("custom_rules", "AB", "fwd_app", "the dog runs"),
    ("custom_rules", "AB", "bwd_app", "the dog runs"),
    ("custom_rules", "AB", "fwd_comp", "the dog whom Mary saw runs"),
    ("custom_rules", "AB", "bwd_comp", "the dog that runs today barks"),
]


@pytest.mark.parametrize(("stem", "name", "rule", "sentence"), _NEEDED)
def test_each_combinator_is_needed(
    stem: str, name: str, rule: str, sentence: str
) -> None:
    """The sentence parses, and stops parsing once the rule is removed."""
    system = _deduction(stem, name)
    assert _parses(system, sentence)
    assert not _parses(_without(system, rule), sentence)


@pytest.mark.parametrize(("stem", "name"), [("ccg", "CCG"), ("custom_rules", "AB")])
def test_every_rule_has_a_showcase_sentence(stem: str, name: str) -> None:
    """No rule of the grammar goes unexercised by the showcase list."""
    covered = {rule for s, _, rule, _ in _NEEDED if s == stem}
    assert covered == {rule.name for rule in _deduction(stem, name).rules}


def _components(pattern: tuple) -> list[tuple[str, str]]:
    """The position intervals of a span or rank-2 item pattern.

    Parameters
    ----------
    pattern : tuple
        A rule premise or conclusion.

    Returns
    -------
    list[tuple[str, str]]
        The ``(start, end)`` variable names of each component.

    Raises
    ------
    AssertionError
        If the pattern is neither a span nor a ``vc`` item, or a position
        is not a variable.
    """
    head = pattern[0]
    if head == "span":
        positions = pattern[1:3]
    else:
        assert head == "vc", pattern
        positions = pattern[1:5]
    assert all(isinstance(p, Wildcard) for p in positions), pattern
    names = [p.name for p in positions]
    return list(zip(names[0::2], names[1::2]))


def test_mcfg_productions_are_linear() -> None:
    """Each conclusion component is tiled, left to right, by premise
    intervals, and every premise interval tiles exactly one of them."""
    system = _deduction("pmcfg", "PMCFG")
    for rule in system.rules:
        premise_intervals = [c for p in rule.premises for c in _components(p)]
        unused = list(premise_intervals)
        for start, end in _components(rule.conclusion):
            cursor = start
            while cursor != end:
                step = [iv for iv in unused if iv[0] == cursor]
                assert len(step) == 1, (rule.name, cursor, unused)
                unused.remove(step[0])
                cursor = step[0][1]
        assert not unused, (rule.name, unused)


def test_mcfg_rank2_items_are_discontinuous() -> None:
    """The example sentence's chart holds a vc item whose two non-empty
    components are separated by other material."""
    system = _deduction("pmcfg", "PMCFG")
    sentence = "mer d'chind em Hans es huus lönd hälfe aastriiche".split()
    items = [item for item, _ in system(sentence).chart.items() if item[0] == "vc"]
    assert ("vc", 2, 4, 7, 8) in items
    gapped = [(i1, j1, i2, j2) for _, i1, j1, i2, j2 in items if i1 < j1 < i2 < j2]
    assert gapped


# The nested (center-embedding) context-free approximation of the same
# clause: each object pairs with a verb by nesting, so the verbs mirror
# the objects and the finite verb comes last, as in Standard German.
_NESTED_CFG = """
object Term : FinSet 16

deduction Nested : Term -> Term [semiring=LogProb, start=S, depth=2]
    atoms S, VC, Inner, Nom, Dat, Acc, DetDat, DetAcc, Noun, VDat, VAcc, VDatFin, VAccFin, span, mer, chind, em, es, hans, huus, loend, laa, haelfed, haelfe, aastriiche
    rule dat_np : span(I, K, DetDat), span(K, J, Noun) |- span(I, J, Dat)
    rule acc_np : span(I, K, DetAcc), span(K, J, Noun) |- span(I, J, Acc)
    rule inner_acc_base : span(I, K, Acc), span(K, J, VAcc) |- span(I, J, Inner)
    rule inner_dat_base : span(I, K, Dat), span(K, J, VDat) |- span(I, J, Inner)
    rule inner_acc : span(I, K, Acc), span(K, L, Inner), span(L, J, VAcc) |- span(I, J, Inner)
    rule inner_dat : span(I, K, Dat), span(K, L, Inner), span(L, J, VDat) |- span(I, J, Inner)
    rule fin_acc : span(I, K, Acc), span(K, L, Inner), span(L, J, VAccFin) |- span(I, J, VC)
    rule fin_dat : span(I, K, Dat), span(K, L, Inner), span(L, J, VDatFin) |- span(I, J, VC)
    rule clause : span(I, J, Nom), span(J, K, VC) |- span(I, K, S)
    lexicon
        "mer" : Nom = mer
        "d'chind" : Acc = chind
        "em" : DetDat = em
        "es" : DetAcc = es
        "Hans" : Noun = hans
        "huus" : Noun = huus
        "lönd" : VAccFin = loend
        "laa" : VAcc = laa
        "hälfed" : VDatFin = haelfed
        "hälfe" : VDat = haelfe
        "aastriiche" : VAcc = aastriiche
"""


@pytest.mark.parametrize(
    ("sentence", "cross_serial"),
    [
        ("mer em Hans es huus hälfed aastriiche", True),
        ("mer d'chind em Hans es huus lönd hälfe aastriiche", True),
        ("mer em Hans es huus aastriiche hälfed", False),
        ("mer d'chind em Hans es huus aastriiche hälfe lönd", False),
    ],
)
def test_mcfg_and_nested_cfg_disagree_on_crossing(
    sentence: str, cross_serial: bool
) -> None:
    """The MCFG accepts exactly the cross-serial order; the nested CFG
    accepts exactly the mirrored one."""
    mcfg = _deduction("pmcfg", "PMCFG")
    nested = loads(_NESTED_CFG).deductions["Nested"]
    assert _parses(mcfg, sentence) is cross_serial
    assert _parses(nested, sentence) is not cross_serial


_OBJECTS = {
    "acc": [["d'chind"], ["es", "Hans"], ["es", "huus"]],
    "dat": [["em", "Hans"], ["em", "huus"]],
}
_FINITE = {"acc": ["lönd"], "dat": ["hälfed"]}
_INFINITIVE = {"acc": ["laa", "aastriiche"], "dat": ["hälfe"]}
_VOCABULARY = [
    "mer",
    "d'chind",
    "em",
    "es",
    "Hans",
    "huus",
    "lönd",
    "laa",
    "hälfed",
    "hälfe",
    "aastriiche",
]


def _cross_serial(length: int) -> set[tuple[str, ...]]:
    """The page's direct description of the language, at one length.

    Parameters
    ----------
    length : int
        The number of words.

    Returns
    -------
    set[tuple[str, ...]]
        Every sentence of that length.
    """
    out: set[tuple[str, ...]] = set()
    for k in range(2, length):
        for cases in itertools.product(["acc", "dat"], repeat=k):
            for objects in itertools.product(*(_OBJECTS[c] for c in cases)):
                if 1 + sum(map(len, objects)) + k != length:
                    continue
                verb_choices = [_FINITE[cases[0]]] + [_INFINITIVE[c] for c in cases[1:]]
                for verbs in itertools.product(*verb_choices):
                    words = [w for obj in objects for w in obj]
                    out.add(("mer", *words, *verbs))
    return out


@pytest.mark.slow
@pytest.mark.parametrize("length", [4, 5])
def test_cross_serial_description_matches_brute_force(length: int) -> None:
    """Every string over the vocabulary that parses is described, and
    every described string parses."""
    system = _deduction("pmcfg", "PMCFG")
    parsed = {
        words
        for words in itertools.product(_VOCABULARY, repeat=length)
        if bool(torch.isfinite(system(list(words)).goal_weight()))
    }
    assert parsed == _cross_serial(length)


def test_cross_serial_description_parses_at_page_length() -> None:
    """The 28 seven-word sentences the page normalizes over all parse."""
    system = _deduction("pmcfg", "PMCFG")
    sentences = _cross_serial(7)
    assert len(sentences) == 28
    assert all(bool(torch.isfinite(system(list(s)).goal_weight())) for s in sentences)


def _show(encoded: str) -> str:
    """Render an agenda parameter's encoded category as the checked search shows it.

    The agenda names a rule weight after its bindings, each category
    spelled ``TS<head>_<argument>..._E`` with a string atom spelled
    ``S<text>E`` (``_`` doubled inside it); the checked search keys the
    same weight by ``show`` of each category, ``Head(argument, ...)``,
    with an atom shown as its text.

    Parameters
    ----------
    encoded : str
        One binding's encoded category.

    Returns
    -------
    str
        The category as ``show`` renders it.
    """

    def parse(index: int) -> tuple[str, int]:
        assert encoded.startswith("TS", index)
        index += 2
        head = re.match(r"[A-Za-z]+", encoded[index:])
        assert head is not None
        index += len(head.group(0))
        arguments: list[str] = []
        while encoded[index] == "_":
            index += 1
            if encoded.startswith("TS", index):
                argument, index = parse(index)
            else:
                assert encoded[index] == "S"
                end = encoded.index("E", index)
                argument, index = encoded[index + 1 : end].replace("__", "_"), end
            arguments.append(argument)
        assert encoded[index] == "E"
        if head.group(0) == "atom":
            return arguments[0], index + 1
        return f"{head.group(0)}({','.join(arguments)})", index + 1

    shown, end = parse(0)
    assert end == len(encoded)
    return shown


def _checked_key(deduction: str, parameter: str) -> str:
    """The checked search's key for one agenda parameter.

    Parameters
    ----------
    deduction : str
        The deduction's name.
    parameter : str
        The agenda's parameter name.

    Returns
    -------
    str
        ``<deduction>.lex.<i>`` for a lexicon weight, and
        ``<deduction>.rule.<rule>:<bindings>`` for a rule weight, the
        non-position bindings sorted by variable and joined by ``|``.
    """
    if parameter.startswith("_axiom_module."):
        index = parameter.removeprefix("_axiom_module.lex_weight_")
        return f"{deduction}.lex.{index}"
    _, rule, bindings = parameter.split(".")
    name = rule.removeprefix("rule_")
    if bindings == "BASE":
        return f"{deduction}.rule.{name}:"
    pairs = re.findall(r"([A-Z])_(TS.*?E)(?=_[A-Z]_TS|$)", bindings)
    return f"{deduction}.rule.{name}:" + "|".join(
        _show(encoded) for _, encoded in sorted(pairs)
    )


@pytest.mark.parametrize(
    ("stem", "name", "sentence"),
    [
        ("multimodal_tlg", "MMTLG", "every linguist that admires Kim sleeps"),
        ("multimodal_tlg", "MMTLG", "Kim admires every linguist"),
        ("quantifier_scope", "QScope", "every linguist admires some model"),
        ("quantifier_scope", "QScope", "every linguist sleeps"),
    ],
)
def test_the_agenda_and_the_checked_search_agree(
    stem: str, name: str, sentence: str
) -> None:
    """Every learnable weight the sentence's chart reads is set to a
    distinct value on the agenda and passed under its checked key; the
    two evaluators' goal weights agree. A key that did not match its
    agenda weight would read as zero and break the agreement, which the
    perturbed control checks."""
    program = load(_SOURCE.format(stem=stem))
    system = program.deductions[name]
    tokens = sentence.split()
    system(tokens)
    generator = torch.Generator().manual_seed(7)
    values: dict[str, float] = {}
    with torch.no_grad():
        for parameter_name, parameter in system.named_parameters():
            value = float(torch.randn((), generator=generator)) * 0.7
            parameter.fill_(value)
            values[_checked_key(name, parameter_name)] = value
    assert any(".rule." in key for key in values)
    agenda = float(system(tokens).goal_weight().detach())
    checked = run_deduction(program.qiec, name, tokens=tokens, parameters=values).weight
    assert checked == pytest.approx(agenda, rel=1e-6, abs=1e-9)
    shifted = {key: value + 0.5 for key, value in values.items()}
    moved = run_deduction(program.qiec, name, tokens=tokens, parameters=shifted).weight
    assert moved != pytest.approx(agenda, rel=1e-6, abs=1e-9)
