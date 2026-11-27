"""Deliberately wrong executors.

A mutation rewrites the source of one library function with literal
replacements, each with the exact number of times its text occurs, and
installs the result for the duration of a check. The unmutated check has
to pass, the mutated run has to execute, the named check has to reject
it, and the deviation it reports has to exceed the recorded minimum. A
replacement whose text has moved fails, so a mutation is never silently
absent. A mutation of a program source replaces the program by one that
denotes another law.
"""

from __future__ import annotations

import importlib
import inspect
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager, ExitStack
from types import FunctionType
from unittest.mock import patch

import didactic.api as dx

from tests.evidence import sources


class Rewrite(dx.Model):
    """One literal replacement.

    Parameters
    ----------
    old : str
        The text.
    new : str
        Its replacement.
    occurrences : int
        How often the text occurs.
    """

    old: str
    new: str
    occurrences: int = 1


class Mutation(dx.Model):
    """One wrong executor and the checks that reject it.

    Parameters
    ----------
    name : str
        The mutation.
    defect_class : str
        The kind of error.
    provenance : str
        Why the error is plausible.
    target : str
        ``"module:qualified.name"`` of a function, or ``"source:name"``.
    rewrites : tuple of Rewrite
        The replacements.
    also_in : tuple of str
        Further modules that hold the function under its own name.
    rejected_by : tuple of str
        The checks that have to reject the mutation.
    min_effect : float
        A lower bound on the deviation each check reports, about half of
        the smallest one measured; one for a deviation that is infinite
        (a missing gradient, a refusal that is not raised, a wrong type).
    """

    name: str
    defect_class: str
    provenance: str
    target: str
    rewrites: tuple[Rewrite, ...]
    also_in: tuple[str, ...] = ()
    rejected_by: tuple[str, ...]
    min_effect: float


class MovedAnchor(AssertionError):
    """A replacement's text does not occur as often as recorded."""


def rewrite(text: str, rewrites: tuple[Rewrite, ...], what: str) -> str:
    """Apply replacements with their exact counts.

    Parameters
    ----------
    text : str
        The text.
    rewrites : tuple of Rewrite
        The replacements.
    what : str
        A description for the report.

    Returns
    -------
    str
        The changed text.


    Raises
    ------
    MovedAnchor
        If a replacement's text occurs another number of times.
    """
    for change in rewrites:
        found = text.count(change.old)
        if found != change.occurrences:
            raise MovedAnchor(
                f"{what}: {change.old!r} occurs {found} times, recorded "
                f"{change.occurrences}; find the operation again and do not "
                "relax the count"
            )
        text = text.replace(change.old, change.new)
    return text


def _mutated_function(target: str, rewrites: tuple[Rewrite, ...]) -> FunctionType:
    """A library function with its source rewritten.

    Parameters
    ----------
    target : str
        ``"module:qualified.name"``.
    rewrites : tuple of Rewrite
        The replacements.

    Returns
    -------
    FunctionType
        The function, compiled in its own module's namespace.
    """
    module_name, path = target.split(":")
    module = importlib.import_module(module_name)
    holder = module
    for part in path.split(".")[:-1]:
        holder = getattr(holder, part)
    function = inspect.getattr_static(holder, path.split(".")[-1])
    assert isinstance(function, FunctionType)
    source = textwrap.dedent(inspect.getsource(function))
    changed = rewrite(source, rewrites, target)
    namespace = dict(vars(module))
    compiled = compile(
        "from __future__ import annotations\n" + changed, f"<{target}>", "exec"
    )
    exec(compiled, namespace)
    mutant = namespace[function.__name__]
    assert isinstance(mutant, FunctionType)
    # The module's own globals stay live, so a later patch is seen.
    return FunctionType(
        mutant.__code__,
        vars(module),
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )


@contextmanager
def installed(mutation: Mutation) -> Iterator[None]:
    """Install a mutation.

    Parameters
    ----------
    mutation : Mutation
        The mutation.

    Yields
    ------
    None
        Inside, the library is wrong.
    """
    kind, name = mutation.target.split(":")
    if kind == "source":
        text = rewrite(sources.SOURCES[name], mutation.rewrites, mutation.target)
        with sources.replaced(name, text):
            yield
        return
    original_module = importlib.import_module(kind)
    mutant = _mutated_function(mutation.target, mutation.rewrites)
    mutant.__kwdefaults__ = _keyword_defaults(mutation.target)
    holder = original_module
    for part in name.split(".")[:-1]:
        holder = getattr(holder, part)
    with ExitStack() as stack:
        stack.enter_context(patch.object(holder, name.split(".")[-1], mutant))
        for module in mutation.also_in:
            stack.enter_context(
                patch.object(importlib.import_module(module), name, mutant)
            )
        yield


def _keyword_defaults(target: str) -> dict[str, str | int | float | bool | None]:
    """The keyword-only defaults of a function.

    Parameters
    ----------
    target : str
        ``"module:qualified.name"``.

    Returns
    -------
    dict of str to (str or int or float or bool or None)
        The defaults.
    """
    module_name, path = target.split(":")
    holder = importlib.import_module(module_name)
    for part in path.split(".")[:-1]:
        holder = getattr(holder, part)
    function = inspect.getattr_static(holder, path.split(".")[-1])
    return dict(function.__kwdefaults__ or {})


AGENDA = "quivers.stochastic.agenda"
ALGEBRAS = "quivers.core.algebras"
SPAN = "quivers.stochastic.span"
INSIDE = "quivers.stochastic.inside"
ELIMINATION = "quivers.core.elimination"
PLATE = "quivers.continuous.plate"
CHART_FIFO = "deduction/value/fifo"
BUILTINS = "quivers.qiec.builtins"
EXECUTION = "quivers.qiec.execution"
PRESENTATION = "quivers.dsl.compiler._deduction_presentation"
CHART = "quivers.dsl._deduction_chart"

MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="repeated-premise-uses-the-update-everywhere",
        defect_class="repeated premise",
        provenance=(
            "an item that fills several premises contributes its new weight at "
            "every position, so terms with several increments are counted again"
        ),
        target=f"{AGENDA}:_fire",
        rewrites=(
            Rewrite(
                old="if i < fixed_idx and premise_item == update.item:",
                new="if False:",
            ),
        ),
        rejected_by=(
            "deduction/value/fifo/logprob/repeated_premise",
            "deduction/count/fifo/repeated_premise",
            "deduction/value/fifo/logprob/three_equal_premises",
            "deduction/value/lifo/counting/three_equal_premises",
        ),
        min_effect=0.16,
    ),
    Mutation(
        name="repeated-premise-uses-the-old-weight-everywhere",
        defect_class="repeated premise",
        provenance="every other position of the popped item reads its old weight",
        target=f"{AGENDA}:_fire",
        rewrites=(
            Rewrite(
                old="if i < fixed_idx and premise_item == update.item:",
                new="if i != fixed_idx and premise_item == update.item:",
            ),
        ),
        rejected_by=(
            "deduction/count/fifo/repeated_premise",
            "deduction/count/fifo/three_equal_premises",
        ),
        min_effect=0.5,
    ),
    Mutation(
        name="input-side-normalizer",
        defect_class="normalizer",
        provenance="a decoder that does not normalize its choices",
        target="quivers.structural.decoder:Decoder._logp_object",
        rewrites=(
            Rewrite(
                old="structure_lp = torch.log_softmax(\n"
                "        self._structure_logits(vec, sort, cands),\n"
                "        dim=-1,\n"
                "    )[cands.index(term.op)]",
                new="structure_lp = self._structure_logits(vec, sort, cands)"
                "[cands.index(term.op)]",
            ),
        ),
        rejected_by=("terms/decoder", "terms/decoder_gradient", "terms/decoder_replay"),
        min_effect=0.79,
    ),
    Mutation(
        name="productions-not-normalized",
        defect_class="normalizer",
        provenance="the weights of a left-hand side are used as they are",
        target="quivers.stochastic.productions:ProductionWeights.table",
        rewrites=(
            Rewrite(
                old="vector = torch.log_softmax(vector, dim=0)", new="vector = vector"
            ),
        ),
        rejected_by=("machine/normalized",),
        min_effect=1.0,
    ),
    Mutation(
        name="lexicon-normalized-over-words",
        defect_class="normalizer",
        provenance="the normalizer runs over the wrong axis of the lexicon",
        target=f"{SPAN}:LexicalAxiom.forward",
        rewrites=(
            Rewrite(
                old="log_lex = self.log_lexicon[tokens]",
                new="log_lex = torch.log_softmax(self.lexicon_logits, dim=0)[tokens]",
            ),
        ),
        rejected_by=(
            "span/chart-parser/logprob/none",
            "span/chart-parser/viterbi/none",
        ),
        min_effect=0.49,
    ),
    Mutation(
        name="sum-replaced-by-maximum",
        defect_class="aggregation",
        provenance="a best-path reduction in a partition computation",
        target=f"{ALGEBRAS}:LogProbAlgebra.join",
        rewrites=(
            Rewrite(
                old="total = torch.logsumexp(safe, dim=axes)",
                new="total = safe.amax(dim=axes)",
            ),
        ),
        rejected_by=(
            "deduction/value/fifo/logprob/diamond",
            "deduction/value/fifo/logprob/equal_proofs",
            "deduction/outcomes/fifo",
            "span/chart-parser/logprob/none",
        ),
        min_effect=0.016,
    ),
    Mutation(
        name="maximum-replaced-by-sum",
        defect_class="aggregation",
        provenance="a partition reduction in a best-path computation",
        target=f"{ALGEBRAS}:Selective.join",
        rewrites=(
            Rewrite(
                old="return self.select(t, axes).value",
                new="return torch.logsumexp(t, dim=axes)",
            ),
        ),
        rejected_by=(
            "deduction/value/fifo/viterbi/diamond",
            "span/chart-parser/viterbi/none",
        ),
        min_effect=0.016,
    ),
    Mutation(
        name="contraction-keeps-the-largest-term",
        defect_class="aggregation",
        provenance="a maximum over the contracted axes of a sum",
        target=f"{ALGEBRAS}:RealAlgebra.join",
        rewrites=(
            Rewrite(
                old="result = result.sum(dim=d)", new="result = result.amax(dim=d)"
            ),
        ),
        rejected_by=(
            "contraction/value/real/product/positive",
            "contraction/value/real/triangle/zeros",
        ),
        min_effect=0.29,
    ),
    Mutation(
        name="mixture-keeps-the-best-class",
        defect_class="aggregation",
        provenance="the most probable class instead of the sum over classes",
        target=f"{PLATE}:marginalize_categorical",
        rewrites=(
            Rewrite(
                old="torch.logsumexp(log_probs_per_class, dim=-1)",
                new="log_probs_per_class.max(dim=-1).values",
            ),
        ),
        also_in=("tests.evidence.checks.mixture",),
        rejected_by=("mixture/primitives",),
        min_effect=0.0031,
    ),
    Mutation(
        name="gradient-cut-at-one-premise",
        defect_class="gradient path",
        provenance="a lookup returns a detached value",
        target=f"{AGENDA}:_fire",
        rewrites=(
            Rewrite(
                old="w = chart.get(premise_item)",
                new="w = chart.get(premise_item)\n"
                "            w = None if w is None else w.detach()",
            ),
        ),
        rejected_by=(
            "deduction/gradient/fifo/logprob/reach",
            "deduction/gradient/fifo/counting/ambiguous",
            "deduction/gradient/lifo/logprob/high_arity",
        ),
        min_effect=1.0,
    ),
    Mutation(
        name="closure-gradient-cut",
        defect_class="gradient path",
        provenance="a closure computed without its gradient",
        target=f"{ALGEBRAS}:LogProbAlgebra.star_with_work",
        rewrites=(
            Rewrite(
                old="star, work = _eliminate(self, matrix, repeat)",
                new="star, work = _eliminate(self, matrix.detach(), repeat)",
            ),
        ),
        rejected_by=("span/chart-parser/gradient",),
        min_effect=1.0,
    ),
    Mutation(
        name="inside-unary-uses-dropped",
        defect_class="gradient path",
        provenance="the outside pass leaves out the unary rules' uses",
        target="quivers.stochastic.inside:_outside",
        rewrites=(
            Rewrite(
                old="_UNARY_USES.apply(closed[width], before)",
                new="_UNARY_USES.apply(torch.full_like(closed[width], -torch.inf), before)",
            ),
        ),
        rejected_by=("span/inside/gradient",),
        min_effect=0.93,
    ),
    Mutation(
        name="wiring-gradient-cut",
        defect_class="gradient path",
        provenance="one factor of a contraction is detached",
        target=f"{ELIMINATION}:_product",
        rewrites=(
            Rewrite(
                old="result = algebra.tensor_op(total, _aligned(factor, target, rank))",
                new=(
                    "result = algebra.tensor_op("
                    "total, _aligned(factor, target, rank).detach())"
                ),
            ),
        ),
        rejected_by=(
            "contraction/gradient/real/chain",
            "contraction/gradient/markov/triangle",
        ),
        min_effect=1.0,
    ),
    Mutation(
        name="groups-share-one-draw",
        defect_class="effect sharing",
        provenance="two draws of two groups are replaced by one draw",
        target="source:grouped",
        rewrites=(
            Rewrite(old="[over=G]\n", new="[reduction=logsumexp]\n"),
            Rewrite(old=" [via=idx]", new=""),
        ),
        rejected_by=("mixture/compiled/grouped",),
        min_effect=0.068,
    ),
    Mutation(
        name="one-draw-duplicated",
        defect_class="effect sharing",
        provenance="one draw that several rows read is drawn once per group",
        target="source:ungrouped",
        rewrites=(
            Rewrite(
                old="object Obs : FinSet 4\n",
                new="object G : FinSet 4\nobject Obs : FinSet 4\n",
            ),
            Rewrite(old="[reduction=logsumexp]\n", new="[over=G]\n"),
            Rewrite(
                old="Bernoulli(probs=rate)\n", new="Bernoulli(probs=rate) [via=rows]\n"
            ),
        ),
        rejected_by=("mixture/compiled/ungrouped",),
        min_effect=0.05,
    ),
    Mutation(
        name="closure-stops-on-a-residual",
        defect_class="stopping rule",
        provenance="a truncated series whose last term is small",
        target=f"{ALGEBRAS}:LogProbAlgebra.star_with_work",
        rewrites=(
            Rewrite(
                old="return -torch.log(-torch.expm1(loop))",
                new="return torch.logsumexp("
                "loop * torch.arange(64, dtype=loop.dtype), 0)",
            ),
        ),
        rejected_by=(
            "closure/near_boundary",
            "closure/star/logprob/near_one",
        ),
        min_effect=1.4,
    ),
    Mutation(
        name="divergence-test-removed",
        defect_class="stopping rule",
        provenance="a divergent cycle returns a number",
        target=f"{ALGEBRAS}:_linear_star",
        rewrites=(
            Rewrite(
                old="loop * (1 - relative_error) >= 1,",
                new='loop * (1 - relative_error) >= float("inf"),',
            ),
            Rewrite(
                old="loop * (1 + relative_error) < 1,",
                new='loop * (1 + relative_error) < float("inf"),',
            ),
        ),
        rejected_by=("closure/divergence/counting/above_one",),
        min_effect=1.0,
    ),
    Mutation(
        name="budget-returns-the-partial-chart",
        defect_class="budget",
        provenance="a spent budget ends the loop instead of refusing",
        target=f"{AGENDA}:_drain",
        rewrites=(
            Rewrite(
                old="raise RuntimeError(\n"
                '                f"agenda exceeded max_iterations={max_iterations}; "\n'
                '                f"likely a non-terminating deduction"\n'
                "            )",
                new="break",
            ),
        ),
        rejected_by=("deduction/budget/fifo", "deduction/budget/lifo"),
        min_effect=1.0,
    ),
    Mutation(
        name="tolerance-stop-claims-completion",
        defect_class="stopping rule",
        provenance=(
            "two iterates within the tolerance are read as a finished method, "
            "so the last iterate is returned as the exact weight"
        ),
        target=f"{BUILTINS}:settlement_report",
        rewrites=(
            Rewrite(
                old="        Completed()\n        if certified\n",
                new="        Completed()\n        if True\n",
            ),
        ),
        rejected_by=("machine/residual/cycle", "machine/residual/chained"),
        min_effect=0.003,
    ),
    Mutation(
        name="spent-budget-returns-zero",
        defect_class="budget",
        provenance="an exhausted step budget answers the zero instead of refusing",
        target=f"{EXECUTION}:run_named",
        rewrites=(
            Rewrite(
                old="        _fail(\n"
                '            "qiec-run-fuel",\n'
                '            f"computation {selected.name!r} did not finish within '
                '{error.steps} steps",\n'
                "            selected.name,\n"
                "            selected.origin,\n"
                "        )",
                new="        value = 0",
            ),
        ),
        also_in=("quivers.qiec.program_runtime",),
        rejected_by=("machine/budget/cycle", "machine/budget/chained"),
        min_effect=1.0,
    ),
    Mutation(
        name="rule-parameters-share-one-name",
        defect_class="parameter identity",
        provenance=(
            "the parameter of a rule is named by its deduction alone, so the "
            "rules' weights collapse into one when the presentation is stored"
        ),
        target=f"{PRESENTATION}:deduction_presentation",
        rewrites=(
            Rewrite(
                old='ParameterId.derive(decl.name, "rule", rule.name)',
                new='ParameterId.derive(decl.name, "rule")',
            ),
        ),
        also_in=("tests.evidence.checks.presented",),
        rejected_by=("presented/codec",),
        min_effect=1.0,
    ),
    Mutation(
        name="bare-item-takes-the-first-form",
        defect_class="omitted alternative",
        provenance=(
            "a rule whose item is a bare variable is instantiated at the first "
            "judgment form only, the term form, as if every item were a term"
        ),
        target=f"{PRESENTATION}:_instances",
        rewrites=(
            Rewrite(
                old="[TERM_ITEM] if bare.name in variables else [TERM_ITEM, *positioned]",
                new="[TERM_ITEM]",
            ),
        ),
        rejected_by=("presented/bare_items",),
        min_effect=1.0,
    ),
    Mutation(
        name="goal-need-not-cover-the-sentence",
        defect_class="omitted alternative",
        provenance=(
            "a chart over a sentence accepts a goal span over part of it, so "
            "derivations of a fragment count as parses"
        ),
        target=f"{CHART}:_Chart.covers",
        rewrites=(Rewrite(old="if not self.lexical:", new="if True:"),),
        rejected_by=("machine/judgments/compiled",),
        min_effect=0.5,
    ),
    Mutation(
        name="rounded-agreement-before-a-cycle-is-shown",
        defect_class="stopping rule",
        provenance=(
            "a chart in rounded arithmetic stops when its weights agree, before "
            "its counts settle, although no cycle is shown"
        ),
        target=f"{CHART}:_Chart.derive",
        rewrites=(
            Rewrite(
                old='c.primitive("gt_int", round_, Var(local=size))',
                new="self.boolean(True)",
            ),
        ),
        rejected_by=("machine/judgments/compiled",),
        min_effect=1.0,
    ),
    Mutation(
        name="first-item-omitted",
        defect_class="omitted alternative",
        provenance="the join of a rule's other premises skips its first candidate",
        target="quivers.stochastic.tries:intersect",
        also_in=(AGENDA,),
        rewrites=(
            Rewrite(
                old="for value in list(sets[smallest]):",
                new="for value in list(sets[smallest])[1:]:",
            ),
        ),
        rejected_by=(
            "deduction/count/fifo/ambiguous",
            "deduction/count/fifo/reach",
            "deduction/forest/fifo/reach",
        ),
        min_effect=1.5,
    ),
    Mutation(
        name="last-split-omitted",
        defect_class="omitted alternative",
        provenance="a span's last split point is skipped",
        target=f"{SPAN}:BinarySpanDeduction.forward",
        rewrites=(
            Rewrite(
                old="for k in range(i + 1, j):",
                new="for k in range(i + 1, max(i + 2, j - 1)):",
            ),
        ),
        rejected_by=("span/chart-parser/logprob/none",),
        min_effect=1.1,
    ),
    Mutation(
        name="support-from-current-numbers",
        defect_class="support",
        provenance="a weight that is zero now is dropped from the chart",
        target=f"{AGENDA}:HashChart.insert_or_aggregate",
        rewrites=(
            Rewrite(
                old="if item not in self._store:\n        self._store[item] = weight",
                new="if float(weight) == float(semiring.zero):\n"
                "        return False\n"
                "    if item not in self._store:\n"
                "        self._store[item] = weight",
            ),
        ),
        rejected_by=(
            "deduction/gradient/fifo/counting/exact_zero",
            "deduction/gradient/lifo/counting/exact_zero",
        ),
        min_effect=1.0,
    ),
    Mutation(
        name="contraction-masks-zeros",
        defect_class="support",
        provenance="products with a zero factor are masked out",
        target=f"{ELIMINATION}:_product",
        rewrites=(
            Rewrite(
                old="result = algebra.tensor_op(total, _aligned(factor, target, rank))",
                new=(
                    "result = algebra.tensor_op(total, _aligned(factor, target, rank))"
                    "\n        result = result.where(result.detach() != 0, "
                    "result.detach())"
                ),
            ),
        ),
        rejected_by=("contraction/gradient/real/product",),
        min_effect=0.57,
    ),
    Mutation(
        name="duplicate-contributions-dropped",
        defect_class="multiplicity",
        provenance="a contribution equal to the stored weight counts as seen",
        target=f"{AGENDA}:HashChart.insert_or_aggregate",
        rewrites=(
            Rewrite(
                old="existing = self._store[item]\n",
                new="existing = self._store[item]\n"
                "    if bool((existing == weight).all()):\n"
                "        return False\n",
            ),
        ),
        rejected_by=(
            "deduction/value/lifo/logprob/equal_proofs",
            "deduction/outcomes/lifo",
            "deduction/count/lifo/equal_proofs",
        ),
        min_effect=0.34,
    ),
    Mutation(
        name="contraction-sum-replaced-by-max",
        defect_class="aggregation",
        provenance="the sum that eliminates a letter keeps only the largest term",
        target=f"{ELIMINATION}:_sum",
        rewrites=(
            Rewrite(
                old="result = algebra.join(terms, axes)",
                new="result = terms.amax(dim=axes)",
            ),
        ),
        rejected_by=(
            "contraction/value/real/chain/positive",
            "contraction/value/markov/random_8/positive",
        ),
        min_effect=0.44,
    ),
    Mutation(
        name="contraction-eliminated-without-its-laws",
        defect_class="aggregation",
        provenance="an algebra without distributivity is contracted by elimination",
        target=f"{ELIMINATION}:plan_contraction",
        also_in=("quivers.core.wiring", "quivers.core.algebras"),
        rewrites=(
            Rewrite(
                old="algebra.establishes(*ELIMINATION_LAWS),",
                new="True,",
            ),
        ),
        rejected_by=("contraction/value/product_fuzzy/chain/positive",),
        min_effect=0.065,
    ),
    Mutation(
        name="inside-clamps-zero-probabilities",
        defect_class="support",
        provenance="probabilities are clamped below by a small number before the log",
        target="quivers.stochastic.inside:_logarithms",
        rewrites=(
            Rewrite(
                old="return torch.log(binary), torch.log(lexical), star",
                new=(
                    "return torch.log(binary.clamp(min=1e-30)), "
                    "torch.log(lexical.clamp(min=1e-30)), star"
                ),
            ),
        ),
        rejected_by=("span/inside/impossible_sentence",),
        min_effect=1.0,
    ),
    Mutation(
        name="inside-derivative-not-normalized",
        defect_class="gradient path",
        provenance="the outside derivative of P is returned as that of log P",
        target="quivers.stochastic.inside:_cotangent",
        rewrites=(
            Rewrite(
                old="ratio = torch.exp(uses - level.reshape(shape))",
                new="ratio = torch.exp(uses)",
            ),
        ),
        rejected_by=("span/inside/gradient", "span/inside/zero_rule_derivative"),
        min_effect=0.98,
    ),
    Mutation(
        name="premise-order-permuted",
        defect_class="order",
        provenance="a contraction folds its inputs from the right",
        target=f"{ELIMINATION}:_product",
        rewrites=(
            Rewrite(
                old="result = algebra.tensor_op(total, _aligned(factor, target, rank))",
                new="result = algebra.tensor_op(_aligned(factor, target, rank), total)",
            ),
        ),
        rejected_by=("contraction/order",),
        min_effect=3.5,
    ),
    Mutation(
        name="applications-merged-by-conclusion",
        defect_class="provenance",
        provenance="applications of one rule with one conclusion are one record",
        target=f"{AGENDA}:_fire",
        rewrites=(
            Rewrite(
                old="key = (rule_index, tuple(antecedents))",
                new="key = (rule_index, (conclusion_item,))",
            ),
        ),
        rejected_by=("deduction/forest/fifo/ambiguous", "deduction/forest/fifo/reach"),
        min_effect=1.0,
    ),
    Mutation(
        name="observer-applied-to-an-aggregate",
        defect_class="observer",
        provenance="the square of a sum instead of the sum of squares",
        target="tests.evidence.checks.deduction:square_of_each",
        rewrites=(
            Rewrite(
                old="torch.stack(weights).square().sum()",
                new="torch.stack(weights).sum().square()",
            ),
        ),
        rejected_by=("deduction/nonlinear/fifo", "deduction/nonlinear/lifo"),
        min_effect=1.0,
    ),
    Mutation(
        name="empty-reduction-loses-its-type",
        defect_class="empty reduction",
        provenance="the zero of an empty reduction is made in the default type",
        target=f"{ALGEBRAS}:LogProbAlgebra.join",
        rewrites=(
            Rewrite(
                old="return self.empty_join(t, axes)",
                new="return torch.full(reduced_shape(t, axes), self.zero)",
            ),
        ),
        rejected_by=("contraction/empty/logprob",),
        min_effect=1.0,
    ),
    Mutation(
        name="empty-reduction-loses-an-axis",
        defect_class="empty reduction",
        provenance="an empty reduction returns a scalar",
        target=f"{ALGEBRAS}:LogProbAlgebra.join",
        rewrites=(
            Rewrite(
                old="return self.empty_join(t, axes)",
                new="return torch.tensor(self.zero, dtype=t.dtype)",
            ),
        ),
        rejected_by=("contraction/empty/logprob",),
        min_effect=1.0,
    ),
    Mutation(
        name="impossible-sum-without-a-mask",
        defect_class="empty reduction",
        provenance="an impossible sum takes the logarithm of zero",
        target=f"{ALGEBRAS}:LogProbAlgebra.join",
        rewrites=(
            Rewrite(
                old="safe = torch.where(impossible, torch.zeros_like(t), t)",
                new="safe = t",
            ),
        ),
        rejected_by=("contraction/impossible_row",),
        min_effect=1.0,
    ),
    Mutation(
        name="log-weights-exponentiated",
        defect_class="scaling",
        provenance="a sum in the linear domain",
        target=f"{ALGEBRAS}:LogProbAlgebra.join",
        rewrites=(
            Rewrite(
                old="total = torch.logsumexp(safe, dim=axes)",
                new="total = safe.exp().sum(dim=axes).log()",
            ),
        ),
        rejected_by=("deduction/value/fifo/logprob/tiny",),
        min_effect=1.0,
    ),
    Mutation(
        name="closure-in-the-linear-domain",
        defect_class="scaling",
        provenance="a closure computed from exponentials of the weights",
        target=f"{ALGEBRAS}:LogProbAlgebra.star_with_work",
        rewrites=(
            Rewrite(
                old="largest = _largest_finite(matrix)\n",
                new="return torch.linalg.inv(torch.eye(matrix.shape[0], "
                "dtype=matrix.dtype) - matrix.exp()).log(), ClosureWork("
                "pivots=0, sums=0, products=0)\n",
            ),
        ),
        rejected_by=("closure/tiny_edge",),
        min_effect=1.0,
    ),
    Mutation(
        name="stale-closure-across-updates",
        defect_class="cache",
        provenance="a closure is kept from an earlier parameter value",
        target=f"{SPAN}:UnarySpanDeduction.closure",
        rewrites=(
            Rewrite(
                old="return kleene_star(self.rule_matrix(n_categories, semiring), semiring)",
                new="if not hasattr(self, '_kept'):\n"
                "        object.__setattr__(self, '_kept', kleene_star("
                "self.rule_matrix(n_categories, semiring), semiring))\n"
                "    return self._kept",
            ),
        ),
        rejected_by=("span/chart-parser/update",),
        min_effect=0.33,
    ),
    Mutation(
        name="rules-copied-per-run",
        defect_class="graph work",
        provenance="an immutable rule set is copied at every call",
        target=f"{AGENDA}:run_agenda",
        rewrites=(
            Rewrite(
                old="rules = tuple(rules)",
                new="rules = tuple(rule.with_(name=rule.name) for rule in rules)",
            ),
        ),
        also_in=("tests.evidence.scaling",),
        rejected_by=("work/repeated-runs-of-one-rule-set",),
        min_effect=32.0,
    ),
    Mutation(
        name="signature-copied-per-node",
        defect_class="graph work",
        provenance=(
            "an encoder copies its immutable signature at every argument, so "
            "each node builds the signature's tables again"
        ),
        target="quivers.structural.encoder:Encoder._compress_arg",
        rewrites=(
            Rewrite(old="sig = self.signature", new="sig = self.signature.with_()"),
        ),
        rejected_by=("terms/encoder_tables",),
        min_effect=8.0,
    ),
    Mutation(
        name="weight-read-as-a-number",
        defect_class="scalar coercion",
        provenance="a premise weight is turned into a Python number and back",
        target=f"{AGENDA}:_conclusion_weight",
        rewrites=(
            Rewrite(
                old="acc = semiring.times(acc, w)",
                new="acc = semiring.times(acc, torch.tensor(float(w), dtype=w.dtype))",
            ),
        ),
        rejected_by=(
            "deduction/gradient/fifo/logprob/three_equal_premises",
            "deduction/gradient/lifo/counting/side_condition",
        ),
        min_effect=1.0,
    ),
    Mutation(
        name="machine-sum-read-as-a-number",
        defect_class="scalar coercion",
        provenance="a reduction of the machine turns its tensor into a Python number",
        target="quivers.qiec.evaluator:_reduce",
        rewrites=(
            Rewrite(
                old="        return flat.sum()\n",
                new="        return torch.tensor(float(flat.sum()), dtype=flat.dtype)\n",
            ),
        ),
        rejected_by=("evaluator/gradient/structures",),
        min_effect=1.0,
    ),
    Mutation(
        name="machine-product-read-as-a-number",
        defect_class="scalar coercion",
        provenance="a collector's product reads one factor as a Python number",
        target="quivers.qiec.foreign:KernelSemiring.times",
        rewrites=(
            Rewrite(
                old="self.member.tensor_op(self.tensor(left), self.tensor(right))",
                new=(
                    "self.member.tensor_op(torch.tensor(float(self.tensor(left)), "
                    "dtype=torch.float64), self.tensor(right))"
                ),
            ),
        ),
        rejected_by=(
            "machine/gradient/lexical/logprob",
            "machine/gradient/nonlexical/logprob",
        ),
        min_effect=1.0,
    ),
    Mutation(
        name="machine-shot-weight-detached",
        defect_class="gradient path",
        provenance="a search sums the weights of its shots without their gradients",
        target=f"{BUILTINS}:search_handler",
        rewrites=(
            Rewrite(
                old="weights.append(shot[0])",
                new="weights.append(shot[0].detach())",
            ),
        ),
        also_in=(EXECUTION,),
        rejected_by=(
            "machine/gradient/lexical/logprob",
            "machine/gradient/nonlexical/logprob",
        ),
        min_effect=1.0,
    ),
    Mutation(
        name="machine-occurrences-share-one-draw",
        defect_class="effect sharing",
        provenance="a drawing handler keys its draws by label, so every "
        "occurrence of a site in a recursion reuses the first draw",
        target=f"{BUILTINS}:draw_scoring_handler",
        rewrites=(
            Rewrite(
                old='    definition = _handler_def(\n        "Random.draw-scoring",',
                new='    shared = {}\n    definition = _handler_def(\n        "Random.draw-scoring",',
            ),
            Rewrite(
                old="            else _default_draw(sampleable)\n",
                new="            else shared.setdefault(site, _default_draw(sampleable))\n",
            ),
        ),
        also_in=("quivers.effects.checked_program",),
        rejected_by=("checked/forward/flips",),
        min_effect=0.036,
    ),
    Mutation(
        name="machine-draw-scored-twice",
        defect_class="effect sharing",
        provenance="a drawing handler sends one draw's density twice, so the "
        "joint holds the draw as two",
        target=f"{BUILTINS}:draw_scoring_handler",
        rewrites=(
            Rewrite(
                old='            role="draw-score",\n        )\n',
                new=(
                    '            role="draw-score",\n        )\n'
                    "        _emit_score(context, request, score_instance, "
                    "_log_density(sampleable, value), weight_validator, "
                    'role="draw-score-repeated")\n'
                ),
            ),
        ),
        also_in=("quivers.effects.checked_program",),
        rejected_by=("checked/forward/flips",),
        min_effect=0.17,
    ),
    Mutation(
        name="machine-occurrence-address-reused",
        defect_class="site address",
        provenance="a trace names every occurrence of a site by its label, so a "
        "later occurrence takes the earlier one's address",
        target="quivers.effects.trace_handler:_occurrences",
        rewrites=(
            Rewrite(
                old='named.append(site.with_(name=f"{site.name}@{occurrence}"))',
                new="named.append(site)",
            ),
        ),
        rejected_by=("checked/sites/flips",),
        min_effect=1.0,
    ),
    Mutation(
        name="machine-density-detached",
        defect_class="gradient path",
        provenance="a tensor backend scores a draw without the gradient of its "
        "arguments and its point",
        target="quivers.continuous.tensor_backend:TensorBackend.log_prob",
        rewrites=(
            Rewrite(
                old="return density.to(distribution.numbers.real_dtype)",
                new="return density.detach().to(distribution.numbers.real_dtype)",
            ),
        ),
        rejected_by=("evaluator/gradient/program",),
        min_effect=1.0,
    ),
    Mutation(
        name="replayed-site-dropped",
        defect_class="site address",
        provenance="a replay that loses a recorded latent site draws it again",
        target="quivers.effects.replay:ReplayHandler.install",
        rewrites=(
            Rewrite(
                old="    kernel = run.kernel\n",
                new="    kernel = run.kernel\n"
                "    dropped = next(name for name, site in self._trace.sites.items()\n"
                "                   if not site.is_deterministic and not site.is_observed)\n",
            ),
            Rewrite(
                old="if name not in run.observations",
                new="if name != dropped and name not in run.observations",
            ),
        ),
        rejected_by=("consumer/replay/beta", "consumer/replay/hmm"),
        min_effect=1.0,
    ),
    Mutation(
        name="guide-scores-first-row",
        defect_class="run batching",
        provenance=(
            "a guide that reads one point for every run scores the first run "
            "and repeats its density for the others"
        ),
        target="quivers.inference.registry:LatentSite.unconstrain_runs",
        rewrites=(
            Rewrite(
                old="self.unconstrain(self.align_runs(value, runs))",
                new="self.unconstrain(self.align_runs(value[:1].expand_as(value), runs))",
            ),
        ),
        rejected_by=(
            "consumer/rows-alone/mvn",
            "consumer/rows-alone/low-rank",
            "consumer/rows-alone/structured",
        ),
        min_effect=0.05,
    ),
)
"""The catalogue."""
