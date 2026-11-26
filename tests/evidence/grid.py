"""The grid of executors by problem classes.

A row is an executor and a column is a class of problems. A cell is
``pass``, which names at least one check that is run; ``refuses``, whose
refusal is executed and has to be exactly the recorded one; or
``ledger``, which names known defects of the executor on problems of the
class (a cell of this kind may also have passing checks on other members
of the class). An empty cell fails. A cell moves to ``pass`` only.

A problem an executor has no input for is handed to the executor's entry
point as the plain data of the oracles. The entry point has to raise, so
no executor answers a question it was not asked.
"""

from __future__ import annotations

import json
import re
import traceback
from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path

import didactic.api as dx
import torch

from quivers._values import HeldValue
from quivers.categorical.traced import KleeneStarTrace
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.plate import marginalize_categorical, marginalize_grouped
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.scan import ScanMorphism
from quivers.continuous.spaces import Euclidean, ProductSpace
from quivers.core.algebras import MarkovAlgebra, RealAlgebra
from quivers.core.objects import FinSet
from quivers.core.elimination import plan_contraction, run_contraction
from quivers.core.wiring import EinsumWiring
from quivers.effects import collapse
from quivers.inference import trace
from quivers.continuous.tensor_backend import TensorBackend
from quivers.qiec.program_runtime import run_deduction, run_program
from quivers.stochastic._kleene import kleene_star
from quivers.stochastic.agenda import InferenceRule, run_agenda
from quivers.stochastic.recursion import solve_polynomial
from quivers.stochastic.span import span_grammar
from quivers.stochastic.semiring import LOG_PROB
from tests.evidence.checks import deduction, machine, mixture, span, terms
from tests.evidence.checks.deduction import weighted
from tests.evidence.oracles.paths import Arc, Machine, Mark
from tests.evidence.problems import DEDUCTIONS, logarithm
from tests.evidence.runs import CONSUMERS, RUNS, consume
from tests.evidence.sources import program

COLUMNS: dict[str, str] = {
    "Q1": "weight of an item in an acyclic system",
    "Q1g": "gradient of a weight",
    "Q2": "recognition by a grammar",
    "Q3": "partition value of a grammar",
    "Q4": "best derivation of a grammar",
    "Q5": "full join of a rule body, or its provenance",
    "Q6": "aggregate of a rule body, or a contraction network",
    "Q7": "dense closure",
    "Q8": "path sum of a cyclic graph",
    "Q9": "nonlinear recursive weight",
    "Q10": "path sum, best path, or count of an acyclic graph",
    "Q11": "expected features",
    "Q12": "composition of two machines",
    "Q13": "alignment lattice of two strings",
    "Q14": "conditional normalizer over outputs",
    "Q15": "best string with alignments summed out",
    "Q16": "minimization of a machine",
    "Q17": "failure transitions",
    "Q18": "join of several tapes",
    "Q19": "directed spanning trees",
    "Q20": "graph of string factors",
    "Q21": "dual decomposition",
    "Q22": "expectation propagation",
    "Q23": "pushforward to outcomes",
    "Q24": "presentation, repeated calls, and updates",
}
"""The problem classes of the executors."""

CONSUMER_COLUMNS: dict[str, str] = {
    "R1": "constrained scalar latent with a closed-form posterior",
    "R2": "latent weights of a summed-out draw",
    "R3": "latent rows of a summed-out chain",
    "R4": "latent productions of a grammar that scores sentences",
    "R5": "discrete latent that is not summed out",
    "R6": "program that only the reference machine runs",
    "R7": "program that draws a structured term",
}
"""The problem classes of the inference consumers."""

#: The source a consumer column is asked with.
CONSUMER_SOURCES = {
    "R1": "beta",
    "R2": "ungrouped",
    "R3": "hmm",
    "R4": "parsed",
    "R5": "discrete",
    "R6": "recursive",
    "R7": "structured",
}


class Cell(dx.Model):
    """One cell of the grid.

    Parameters
    ----------
    status : str
        ``"pass"``, ``"refuses"``, or ``"ledger"``.
    entries : tuple of str
        The ledger entries of a ``ledger`` cell.
    """

    status: str
    entries: tuple[str, ...] = ()


LEDGERED: dict[str, dict[str, tuple[str, ...]]] = {
    **{
        f"agenda-{discipline}": {
            "Q1": (
                "nullary-rule-never-fires",
                "repeated-propagation",
                "parameters-created-by-runs",
                "chart-view-promises",
            ),
            "Q5": ("provenance-always-kept",),
            "Q8": ("residual-is-the-value",),
            "Q10": ("repeated-propagation", "corpus-enumerates-strings"),
            "Q24": (
                "composition-has-no-map",
                "lifts-ignore-the-rule",
            ),
        }
        for discipline in deduction.AGENDAS
    },
    "chart-parser": {
        "Q2": ("linear-semiring-reads-logarithms",),
        "Q10": ("linear-semiring-reads-logarithms",),
    },
    "machine-lexical": {"Q24": ("module-traversed-per-run",)},
    "machine-nonlexical": {"Q24": ("module-traversed-per-run",)},
    "machine-chained": {"Q24": ("module-traversed-per-run",)},
    "machine-program": {"Q24": ("module-traversed-per-run",)},
    "einsum-wiring": {
        "Q6": (
            "certain-entry-clamped",
            "dense-is-the-contract",
            "kan-extension-detaches",
        ),
    },
    "marginalize-ungrouped": {"Q6": ("nested-marginalize-asserts",)},
    "decoder": {"Q14": ("depth-bound-shifts-choices",)},
    "svi-laplace": {"R1": ("laplace-adds-jitter",)},
    "trace": {"R6": ("checked-program-runs-per-row",)},
}
"""The cells with known defects."""

ROWS: tuple[str, ...] = tuple(RUNS)
"""The executors."""


def cells(passing: Mapping[tuple[str, str], int]) -> dict[tuple[str, str], Cell]:
    """Every cell of both blocks of the grid.

    Parameters
    ----------
    passing : Mapping of (str, str) to int
        The number of registered checks of an executor in a column.

    Returns
    -------
    dict of (str, str) to Cell
        The cells.
    """
    grid: dict[tuple[str, str], Cell] = {}
    for rows, columns in ((ROWS, COLUMNS), (CONSUMERS, CONSUMER_COLUMNS)):
        for row in rows:
            for column in columns:
                entries = LEDGERED.get(row, {}).get(column, ())
                if entries:
                    grid[(row, column)] = Cell(status="ledger", entries=entries)
                elif passing.get((row, column), 0):
                    grid[(row, column)] = Cell(status="pass")
                else:
                    grid[(row, column)] = Cell(status="refuses")
    return grid


# -- the plain data of every column --------------------------------------

_LINE = Machine(
    arcs=(Arc(name="a", source=0, target=1, labels=("a", "b"), weight="w"),),
    initial=(Mark(state=0, weight="i"),),
    final=(Mark(state=1, weight="f"),),
)
_LOOP = _LINE.with_(
    arcs=(
        *_LINE.arcs,
        Arc(name="b", source=1, target=0, labels=(None, "b"), weight="w"),
    )
)

SAMPLES: dict[str, HeldValue] = {
    "Q1": DEDUCTIONS["diamond"],
    "Q1g": (DEDUCTIONS["diamond"], "a"),
    "Q2": (span.BINARY, (0, 1), "support"),
    "Q3": (span.BINARY, (0, 1), "weight"),
    "Q4": (span.BINARY, (0, 1), "best"),
    "Q5": DEDUCTIONS["high_arity"],
    "Q6": ("ij,jk->ik", {"i": 2, "j": 2, "k": 2}),
    "Q7": (("1/2", "1/3"), ("1/4", "1/5")),
    "Q8": _LOOP,
    "Q9": (("x", ("1/5", ("x", "x"))), ("x", ("4/5", ()))),
    "Q10": _LINE,
    "Q11": (_LINE, {"a": (Fraction(1), Fraction(2))}),
    "Q12": (_LINE, _LINE),
    "Q13": (("a", "b"), ("a", "b", "a")),
    "Q14": (_LINE, ("a",)),
    "Q15": (_LINE, "string"),
    "Q16": (_LINE, "minimal"),
    "Q17": (_LINE, {0: 1}),
    "Q18": (_LINE, _LOOP, 1),
    "Q19": {(0, 1): "1/2", (0, 2): "1/3", (1, 2): "1/5", (2, 1): "1/7"},
    "Q20": ((("x", "y"), _LINE), (("y",), _LOOP)),
    "Q21": ((_LINE, _LOOP), Fraction(1, 2)),
    "Q22": ((_LINE, _LOOP), 2),
    "Q23": (("low", "1/4"), ("high", "1/5")),
    "Q24": {"presentation": "rules"},
}
"""A member of every column as the plain data of the oracles."""

type Entry = Callable[[HeldValue], torch.Tensor | HeldValue]
"""An executor's entry point, called with one value in every data argument."""


def _scan(value: HeldValue) -> torch.Tensor:
    """Score a value with a scan.

    Parameters
    ----------
    value : HeldValue
        The value.

    Returns
    -------
    torch.Tensor
        The score.
    """
    inputs = Euclidean(name="Input", dim=1)
    hidden = Euclidean(name="Hidden", dim=1)
    cell = ConditionalNormal(ProductSpace(components=(inputs, hidden)), hidden)
    return ScanMorphism(cell).log_prob(value, value)  # type: ignore[arg-type]


def _collapse(value: HeldValue) -> HeldValue:
    """Collapse the pairs a value lists.

    Parameters
    ----------
    value : HeldValue
        The value.

    Returns
    -------
    HeldValue
        The trace.
    """
    model = program("conjugate").morphism
    assert isinstance(model, MonadicProgram)
    with collapse(value):  # type: ignore[arg-type]
        return trace(
            model,
            torch.zeros(1, 1),
            observations=value,  # type: ignore[arg-type]
        )


def _trace(value: HeldValue) -> HeldValue:
    """Trace a value over one feedback wire.

    Parameters
    ----------
    value : HeldValue
        The value.

    Returns
    -------
    HeldValue
        The trace.
    """
    wire = FinSet(name="U", cardinality=1)
    return KleeneStarTrace(MarkovAlgebra()).trace(
        value,  # type: ignore[arg-type]
        wire,
        wire,
        wire,
    )


ENTRIES: dict[str, Entry] = {
    **{
        f"agenda-{name}": lambda value, name=name: run_agenda(
            value,  # type: ignore[arg-type]
            value,  # type: ignore[arg-type]
            semiring=LOG_PROB,
            agenda=deduction.AGENDAS[name](),
        )
        for name in deduction.AGENDAS
    },
    "machine-lexical": lambda value: run_deduction(
        machine.module_of("lexical", SEMIRING="LogProb"),
        "G",
        tokens=value,  # type: ignore[arg-type]
        fuel=5000,
    ),
    "machine-nonlexical": lambda value: run_deduction(
        machine.module_of("nonlexical", SEMIRING="LogProb"),
        "W",
        axioms=value,  # type: ignore[arg-type]
        axiom_weights=value,  # type: ignore[arg-type]
        fuel=5000,
    ),
    "machine-chained": lambda value: run_deduction(
        machine.module_of("chained", SEMIRING="Boolean", TOLERANCE=""),
        "M",
        tokens=value,  # type: ignore[arg-type]
        fuel=5000,
    ),
    "machine-program": lambda value: run_program(
        machine.module_of("recursive"),
        "recursive",
        data={"y": value},
        sites={"t": value},
        distributions=TensorBackend(),
    ),
    "kleene-star": lambda value: kleene_star(value, LOG_PROB),  # type: ignore[arg-type]
    "kleene-star-trace": _trace,
    "einsum-wiring": lambda value: EinsumWiring(RealAlgebra(), "ij,jk->ik").apply(
        value,  # type: ignore[arg-type]
        value,  # type: ignore[arg-type]
    ),
    "contraction-plan": lambda value: run_contraction(
        plan_contraction(
            RealAlgebra(),
            value,  # type: ignore[arg-type]
            "",
            {},
        ),
        RealAlgebra(),
        (value,),  # type: ignore[arg-type]
    )[0],
    "span-plan": lambda value: span_grammar(value),  # type: ignore[arg-type]
    "newton-plan": lambda value: solve_polynomial(
        value,  # type: ignore[arg-type]
        MarkovAlgebra(),
        {},
        like=torch.zeros((), dtype=torch.float64),
    ),
    "chart-parser": lambda value: span.chart_parser(LOG_PROB, {})(value),
    "inside": lambda value: span.inside_algorithm({})[0](value),
    "marginalize-grouped": lambda value: marginalize_grouped(
        value,  # type: ignore[arg-type]
        value,  # type: ignore[arg-type]
        value,  # type: ignore[arg-type]
        2,
    ),
    "marginalize-ungrouped": lambda value: marginalize_categorical(
        value  # type: ignore[arg-type]
    ),
    "marginalize-chain": lambda value: _model("hmm").log_joint(
        value,  # type: ignore[arg-type]
        mixture.chain_sites(),
    ),
    "scan": _scan,
    "collapse": _collapse,
    "encoder": lambda value: terms.encoder(terms.parameters())(
        value  # type: ignore[arg-type]
    ),
    "decoder": lambda value: terms.decoder(torch.zeros(2)).log_prob(
        value,  # type: ignore[arg-type]
        torch.zeros(1),
    ),
}
"""The entry point of every executor."""


def _model(name: str) -> MonadicProgram:
    """A compiled program.

    Parameters
    ----------
    name : str
        The source's name.

    Returns
    -------
    MonadicProgram
        The program.
    """
    model = program(name).morphism
    assert isinstance(model, MonadicProgram)
    return model


def _branching(discipline: str) -> HeldValue:
    """Run a nonlinear recursion within a small budget.

    Parameters
    ----------
    discipline : str
        The agenda.

    Returns
    -------
    HeldValue
        The run.
    """
    return run_agenda(
        [(("X",), torch.tensor(logarithm(Fraction(4, 5)), dtype=torch.float64))],
        [
            InferenceRule(
                name="split",
                premises=(("X",), ("X",)),
                conclusion=("X",),
                weight_fn=weighted(
                    torch.tensor(logarithm(Fraction(1, 5)), dtype=torch.float64)
                ),
            )
        ],
        semiring=LOG_PROB,
        agenda=deduction.AGENDAS[discipline](),
        max_iterations=200,
    )


def ask(row: str, column: str) -> HeldValue:
    """Ask an executor a question of a column it does not answer.

    Parameters
    ----------
    row : str
        The executor.
    column : str
        The problem class.

    Returns
    -------
    HeldValue
        The answer, which no refusing cell gives.
    """
    if row in CONSUMERS:
        return consume(row, CONSUMER_SOURCES[column])
    if column == "Q9" and row in ("agenda-lifo", "agenda-priority"):
        return _branching(row.removeprefix("agenda-"))
    return ENTRIES[row](SAMPLES[column])


def refusal(row: str, column: str) -> str | None:
    """The diagnostic of a refused question.

    Parameters
    ----------
    row : str
        The executor.
    column : str
        The problem class.

    Returns
    -------
    str or None
        The exception's class and the place it was raised, and the message
        of an exception raised by a ``raise`` statement, on one line
        without memory addresses; ``None`` when the question is answered.
        An exception that a compiled function raised while called from a
        line is named by that line, because the wording of such messages
        is not part of any interface.
    """
    try:
        ask(row, column)
    except Exception as failure:
        last = traceback.extract_tb(failure.__traceback__)[-1]
        place = f"{Path(last.filename).name}:{last.name}"
        line = (last.line or "").strip()
        if line.startswith("raise"):
            text = f"{type(failure).__name__} at {place}: {failure}"
        else:
            text = f"{type(failure).__name__} at {place} in `{line}`"
        return re.sub(r"0x[0-9a-f]+", "0x", " ".join(text.split()))[:160]
    return None


REFUSALS_FILE = Path(__file__).with_name("refusals.json")
"""The recorded diagnostics."""


def recorded_refusals() -> dict[str, str]:
    """The recorded diagnostic of every refusing cell.

    Returns
    -------
    dict of str to str
        The diagnostics by ``"row/column"``.
    """
    return json.loads(REFUSALS_FILE.read_text())
