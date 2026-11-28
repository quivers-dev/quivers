"""Check normalized string scores at the binary64 overflow boundary."""

from fractions import Fraction
from math import inf, nextafter
from sys import float_info

import pytest
import torch
from torch import nn

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedRelationRequest,
    FixedRelationRuntimeProvider,
    QiecModule,
    RuntimeConfiguration,
    TreeStringGraphRequest,
    TreeStringGraphRuntimeProvider,
    dumps,
    loads,
    run_named,
)

_SOURCE = """object Letters : FinSet 2

relation channel [algebra=logprob, states=2]
    input Letters
    output Letters
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0, 0), weight=rare]
    arc 0 -> 1 [labels=(0, 1), weight=common]
    final 1 [weight=unit]

string_graph model
    variable input_word : Letters [observed=(0)]
    variable output_word : Letters
    factor channel(input_word, output_word)

relation_score conditional : LogWeight [operation=conditional_log_prob, relation=channel]
string_graph_score posterior : LogWeight [operation=tree_log_prob, graph=model]
"""


def _scores(common_value: float):
    compiler = Compiler(parse(_SOURCE), module_name="authored_string_log_range")
    compiler.compile()
    module = loads(dumps(compiler.qiec_module))
    assert isinstance(module, QiecModule)
    spec = compiler.relation_specs["channel"]
    graph = compiler.string_graph_specs["model"]
    rare = nn.Parameter(torch.tensor(-float_info.max, dtype=torch.float64))
    common = nn.Parameter(rare.new_tensor(common_value))
    values = {"unit": rare.new_tensor(0.0), "rare": rare, "common": common}
    parameters = {
        id_: values[name]
        for name, id_ in zip(spec.weight_names, spec.weight_parameters, strict=True)
    }
    limits = {"max_states": 20, "max_work": 40}
    relation = spec.bind(parameters)
    conditional = relation.conditional(((0,),), **limits)
    marginal = graph.bind(parameters).tree_marginal("output_word", **limits)
    assert (
        conditional.normalizer().item() == marginal.partition().item() == common_value
    )
    assert relation.weight(((0,), (0,))).item() == rare.item()
    fixed_request = FixedRelationRequest(
        relation=spec, operation="conditional_log_prob", words=((0,), (0,)), **limits
    )
    tree_request = TreeStringGraphRequest(
        graph=graph, variable="output_word", word=(0,), **limits
    )

    def conditional_native(word):
        attachment = FixedRelationRuntimeProvider.for_score(
            "conditional", fixed_request.with_(words=((0,), word)), parameters
        )
        runtime = RuntimeConfiguration(
            selections=(), providers=(CoreRuntimeProvider(), attachment)
        )
        return run_named(module, "conditional", runtime=runtime).value

    def tree_native(word):
        attachment = TreeStringGraphRuntimeProvider.for_score(
            "posterior", tree_request.with_(word=word), parameters
        )
        runtime = RuntimeConfiguration(
            selections=(), providers=(CoreRuntimeProvider(), attachment)
        )
        return run_named(module, "posterior", runtime=runtime).value

    def conditional_python(word):
        return conditional.log_prob((word,))

    return (rare, common), {
        "conditional_python": conditional_python,
        "tree_python": marginal.log_prob,
        "conditional_native": conditional_native,
        "tree_native": tree_native,
    }


@pytest.mark.parametrize(
    "entry", ("conditional_python", "tree_python", "conditional_native", "tree_native")
)
def test_positive_mass_is_distinct_from_overflow_and_impossible_support(entry):
    largest = Fraction(float_info.max)
    predecessor = Fraction(nextafter(float_info.max, 0))
    overflow_threshold = largest + (largest - predecessor) / 2
    halfway = float(overflow_threshold - largest)
    # The true correction to the larger raw log mass is
    # log(1 + exp(-largest - common)), which is less than one.
    # Its binary64 rounding at either common input is thus common.
    below = nextafter(halfway, 0)
    owners, scores = _scores(below)
    # Even with the log-normalizer correction, this value still rounds
    # to the most negative finite binary64 number.
    assert largest < largest + Fraction(below) < overflow_threshold - 1
    score = scores[entry]((0,))
    assert score.item() == -float_info.max
    assert tuple(value.item() for value in torch.autograd.grad(score, owners)) == (
        1,
        -1,
    )
    # No accepted paths emit the empty output. It retains log zero.
    assert scores[entry](()).item() == -inf

    _, scores = _scores(halfway)
    assert largest + Fraction(halfway) == overflow_threshold
    native = entry.endswith("_native")
    error = ExecutionFailure if native else OverflowError
    with pytest.raises(error, match="outside the evaluation precision") as rejected:
        scores[entry]((0,))
    if isinstance(rejected.value, ExecutionFailure):
        assert rejected.value.diagnostic.code == "qiec-run-evaluation"
    assert scores[entry](()).item() == -inf
