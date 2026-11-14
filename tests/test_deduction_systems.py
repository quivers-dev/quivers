"""End-to-end tests for the deduction-system pipeline.

Covers:
* The stdlib registry of pre-built deductions.
* The `deduction { ... }` DSL block.
* The `extract_deduction_schema` panproto integration.
* MLTT-style type-checking against a small library of well-typed
  and ill-typed terms.
"""

from __future__ import annotations

import textwrap

import torch

from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse
from quivers.dsl.program_theory import (
    extract_deduction_schema,
)
from quivers.stochastic.agenda import Wildcard
from quivers.stochastic.stdlib import (
    AB,
    CCG,
    STDLIB_DEDUCTIONS,
    Datalog,
    MLTT,
)
from quivers.stochastic.semiring import COUNTING


# ---------------------------------------------------------------------------
# Stdlib smoke tests
# ---------------------------------------------------------------------------


class TestStdlibRegistry:
    def test_registry_contains_canonical_systems(self):
        expected = {
            "CCG",
            "AB",
            "STLC",
            "MLTT",
            "Datalog",
            "Dijkstra",
            "HMM",
            "ViterbiHMM",
            "EditDistance",
        }
        assert expected <= set(STDLIB_DEDUCTIONS)

    def test_datalog_transitive_closure(self):
        """reach is the transitive closure of edge."""
        axioms = [
            (("edge", 1, 2), torch.tensor(1.0)),
            (("edge", 2, 3), torch.tensor(1.0)),
            (("edge", 3, 4), torch.tensor(1.0)),
        ]
        view = Datalog(axioms)
        reaches = view.enumerate(("reach", 1, Wildcard(name="Y")))
        targets = sorted(item[2] for item, _ in reaches)
        assert targets == [2, 3, 4]

    def test_datalog_no_spurious_reach(self):
        """Disconnected graph: 1 -> 2,   3 -> 4."""
        axioms = [
            (("edge", 1, 2), torch.tensor(1.0)),
            (("edge", 3, 4), torch.tensor(1.0)),
        ]
        view = Datalog(axioms)
        reaches = view.enumerate(("reach", 1, Wildcard(name="Y")))
        targets = sorted(item[2] for item, _ in reaches)
        # 1 can only reach 2; not 3, 4.
        assert targets == [2]


def _fwd(result, argument):
    return ("slash", "/", result, argument)


def _bwd(result, argument):
    return ("slash", "\\", argument, result)


def _lexicon(categories):
    """One axiom per token: ``("span", i, i + 1, category)`` of log weight 0."""
    return [
        (("span", i, i + 1, category), torch.tensor(0.0))
        for i, category in enumerate(categories)
    ]


class TestCategorialStdlib:
    """CCG and AB recognize ``S`` over the whole input from adjacent spans."""

    TRANSITIVE = _fwd(_bwd("S", "NP"), "NP")  # (S\\NP)/NP

    def test_application_derives_a_transitive_clause(self):
        """NP (S\\NP)/NP NP has exactly one AB derivation of S."""
        for system in (AB, CCG):
            view = system(_lexicon(["NP", self.TRANSITIVE, "NP"]))
            assert len(view.goal_items) == 1
            assert view.goal_items[0][0] == ("span", 0, 3, "S")

    def test_adjacency_is_required(self):
        """Categories that combine but are not adjacent derive nothing."""
        axioms = [
            (("span", 0, 1, _fwd("S", "NP")), torch.tensor(0.0)),
            (("span", 2, 3, "NP"), torch.tensor(0.0)),
            (("span", 1, 2, "PP"), torch.tensor(0.0)),
        ]
        assert AB(axioms).goal_items == []

    def test_composition_is_ccg_only(self):
        """S/VP VP/NP NP: CCG adds the composed bracketing; AB has one parse."""
        categories = [_fwd("S", "VP"), _fwd("VP", "NP"), "NP"]
        ab = AB(_lexicon(categories))
        ccg = CCG(_lexicon(categories))
        assert ("span", 0, 2, _fwd("S", "NP")) not in {i for i, _ in ab.chart.items()}
        assert ("span", 0, 2, _fwd("S", "NP")) in {i for i, _ in ccg.chart.items()}

    def test_ccg_counts_both_bracketings(self):
        """Under Counting, S/VP VP/NP NP has 2 CCG derivations and 1 AB derivation."""
        categories = [_fwd("S", "VP"), _fwd("VP", "NP"), "NP"]
        axioms = [(item, torch.tensor(1.0)) for item, _ in _lexicon(categories)]
        counts = {}
        for name, system in (("AB", AB), ("CCG", CCG)):
            counted = type(system)(
                rules=system.rules,
                semiring=COUNTING,
                axiom_injector=system.axiom_injector,
                goal=system.goal,
                agenda_factory=system.agenda_factory,
            )
            counts[name] = float(counted(axioms).goal_weight())
        assert counts == {"AB": 1.0, "CCG": 2.0}


# ---------------------------------------------------------------------------
# Deduction-block DSL surface
# ---------------------------------------------------------------------------


class TestDeductionDSL:
    def _compile(self, source: str) -> Compiler:
        m = parse(source)
        c = Compiler(m)
        c.compile()
        return c

    def test_deduction_block_compiles(self):
        src = textwrap.dedent("""
        object Atom : FinSet 4

        deduction MyD : Atom -> Atom [semiring=LogProb, start=S, depth=4]
            atoms NP, S, VP
            rule combine : NP, VP |- S
        """)
        c = self._compile(src)
        assert "MyD" in c._deductions
        sys = c._deductions["MyD"]
        assert len(sys.rules) == 1
        assert sys.rules[0].name == "combine"

    def test_multi_rule_block(self):
        src = textwrap.dedent("""
        object Atom : FinSet 4

        deduction CG2 : Atom -> Atom [semiring=Boolean, start=S]
            atoms NP, S, VP
            rule fwd : NP, VP |- S
            rule bwd : VP, NP |- S
        """)
        c = self._compile(src)
        sys = c._deductions["CG2"]
        names = {r.name for r in sys.rules}
        assert names == {"fwd", "bwd"}


# ---------------------------------------------------------------------------
# Panproto integration
# ---------------------------------------------------------------------------


class TestPanprotoIntegration:
    def test_extract_deduction_schema(self):
        src = textwrap.dedent("""
        object Atom : FinSet 4

        deduction D1 : Atom -> Atom [semiring=LogProb, start=S]
            atoms NP, S
            rule r1 : NP, NP |- S

        deduction D2 : Atom -> Atom [semiring=Boolean]
            atoms VP
            rule r2 : VP |- VP
        """)
        m = parse(textwrap.dedent(src))
        c = Compiler(m)
        c.compile()
        schema = extract_deduction_schema(c)
        # 2 systems × (1 system + 1 rule + (premises + 1 conclusion))
        # D1: 1 sys + 1 rule + 2 premise + 1 conclusion = 5
        # D2: 1 sys + 1 rule + 1 premise + 1 conclusion = 4
        # Total: 9.
        assert len(schema.vertices) == 9


# ---------------------------------------------------------------------------
# MLTT type-checking
# ---------------------------------------------------------------------------


class TestMLTT:
    """Exercise the MLTT deduction system on a small library of terms.

    Items are ``('judges', Γ, t, A)`` for a typing judgment and
    ``('in_ctx', Γ, x, A)`` for context lookup; rules derive
    typing judgments by application and variable lookup.
    """

    def test_identity_type_application(self):
        """Given (id : A → A) and (x : A) in Γ, derive id x : A."""
        gamma = "Gamma"
        axioms = [
            (("judges", gamma, "id", ("pi", "A", "A")), torch.tensor(1.0)),
            (("in_ctx", gamma, "x", "A"), torch.tensor(1.0)),
            (("judges", gamma, "x", "A"), torch.tensor(1.0)),
        ]
        view = MLTT(axioms)
        target = ("judges", gamma, ("app", "id", "x"), "A")
        result = view.try_weight(target)
        # Boolean semiring; derived facts have weight 1.
        assert float(result) == 1.0

    def test_no_spurious_typing(self):
        """Without an `x : A` context entry, `id x` should not type-check."""
        gamma = "Gamma"
        axioms = [
            (("judges", gamma, "id", ("pi", "A", "A")), torch.tensor(1.0)),
            # NO `x : A` premise.
        ]
        view = MLTT(axioms)
        target = ("judges", gamma, ("app", "id", "x"), "A")
        # The item shouldn't appear in the chart.
        result = view.try_weight(target, default=torch.tensor(0.0))
        assert float(result) == 0.0
