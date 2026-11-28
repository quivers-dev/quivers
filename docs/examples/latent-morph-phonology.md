# Latent morphs and phonology

One underlying form can occur in several observed words. In this model,
each stem and suffix has one latent underlying form, with a uniform
prior on two segments. A three-tape relation concatenates the stem
and suffix. A phonological transducer then generates the surface form.
This follows the generative structure of
[Cotterell, Peng, and Eisner (2015)](https://aclanthology.org/Q15-1031/).

Let $x,y$ be the underlying segments of a word and let $f(z\mid x)$
copy its input with probability $1-\eta$ and flip it with probability
$\eta$. The transducer copies the first segment noisily. For the
second segment, it first chooses the stem with probability $p$ or the
suffix with probability $1-p$, then applies the same noise. Thus

$$
P(Z_1=z_1,Z_2=z_2\mid X=x,Y=y)=f(z_1\mid x)
\big[(1-p)f(z_2\mid y)+p f(z_2\mid x)\big].
$$

The transducer sums over the two choices. When the segments agree,
the second segment has the original copy and flip probabilities. When
they differ, the surface agrees with the stem with probability
$q=\eta+(1-2\eta)p$. The remaining probability is $1-q$. Both
weights read the same trainable parameter $q$.

## Try it

<!-- python: run -->
```python
from itertools import product

import torch

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ParameterId,
    PrunedStringBPConfig,
    PrunedStringGraphRequest,
    PrunedStringGraphRuntimeProvider,
    RelationWeightComplement,
    RuntimeConfiguration,
    TreeStringGraphRequest,
    TreeStringGraphRuntimeProvider,
    run_named,
)

relations = """
object Segments : FinSet 2

relation prior [algebra=markov, states=2]
    output Segments
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(0), weight=uniform]
    arc 0 -> 1 [labels=(1), weight=uniform]
    final 1 [weight=one]

relation concat [algebra=markov, states=3]
    input Segments
    input Segments
    output Segments
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(0, eps, 0), weight=one]
    arc 0 -> 1 [labels=(1, eps, 1), weight=one]
    arc 1 -> 2 [labels=(eps, 0, 0), weight=one]
    arc 1 -> 2 [labels=(eps, 1, 1), weight=one]
    final 2 [weight=one]

relation phonology [algebra=markov, states=4]
    input Segments
    output Segments
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(0, 0), weight=copy]
    arc 0 -> 1 [labels=(0, 1), weight=flip]
    arc 0 -> 2 [labels=(1, 0), weight=flip]
    arc 0 -> 2 [labels=(1, 1), weight=copy]
    arc 1 -> 3 [labels=(0, 0), weight=copy]
    arc 1 -> 3 [labels=(0, 1), weight=flip]
    arc 1 -> 3 [labels=(1, 0), weight=match_stem]
    arc 1 -> 3 [labels=(1, 1), weight=match_suffix]
    arc 2 -> 3 [labels=(0, 0), weight=match_suffix]
    arc 2 -> 3 [labels=(0, 1), weight=match_stem]
    arc 2 -> 3 [labels=(1, 0), weight=flip]
    arc 2 -> 3 [labels=(1, 1), weight=copy]
    final 3 [weight=one]
"""

lexicon = ("stem_a", "stem_b", "suffix_a", "suffix_b")
training = (
    ("stem_a", "suffix_a", (0, 0)),
    ("stem_b", "suffix_a", (1, 0)),
    ("stem_b", "suffix_b", (1, 1)),
)
loopy_data = (*training, ("stem_a", "suffix_b", (0, 1)))
one_segment = ((0,), (1,))
two_segments = tuple(product(range(2), repeat=2))


def graph_source(name, data):
    lines = [f"\nstring_graph {name}"]
    lines.extend(f"    variable {morph} : Segments" for morph in lexicon)
    lines.extend(f"    factor prior({morph})" for morph in lexicon)
    supports = dict.fromkeys(lexicon, one_segment)
    for i, (stem, suffix, surface) in enumerate(data):
        underlying, observed = f"underlying_{i}", f"surface_{i}"
        lines.extend((
            f"    variable {underlying} : Segments",
            f"    variable {observed} : Segments [observed={surface}]",
            f"    factor concat({stem}, {suffix}, {underlying})",
            f"    factor phonology({underlying}, {observed})",
        ))
        supports[underlying] = two_segments
        supports[observed] = (surface,)
    return "\n".join(lines) + "\n", tuple(supports.items())


def belief_source(export, request):
    name = request.handler_name
    candidates = len(request.candidate_words)
    rows = sum(len(scope) for _, scope in request.graph.factors)
    result = (
        "(String * String * Int * Real * Bool * "
        f"Tensor[LogWeight]([{candidates}]) * Bool * Tensor[LogWeight]([{rows}, 3]))"
    )
    instance = f"{export}_solver"
    return f"""
instance {instance} : Compute[Unit, {result}]
handler {name} for Compute[Unit, {result}] : {result} -> {result} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define {export} : {result} !{{}} =
    handle {instance} with {name} in
        let report <- perform {instance}.apply(unit)
        return report
"""


forest_source, forest_support = graph_source("forest", training)
cycle_source, cycle_support = graph_source("cycle", loopy_data)
source = relations + forest_source + cycle_source + """
string_graph_score partition : Real [operation=tree_partition, graph=forest]
string_graph_score suffix : LogWeight [operation=tree_log_prob, graph=forest]

define log_evidence : Real !{} =
    let mass <- partition()
    return log(mass)
"""
compiler = Compiler(parse(source), module_name="latent_morph")
forest = compiler.string_graph_specs["forest"]
cycle = compiler.string_graph_specs["cycle"]
limits = dict(max_states=4096, max_work=32768)


def belief_request(graph, supports):
    return PrunedStringGraphRequest(
        graph=graph,
        config=PrunedStringBPConfig(
            supports=supports, beam=4, max_sweeps=80, max_entries=192,
            damping=0.2, tolerance=1e-9, **limits,
        ),
        variable="suffix_a",
    )


forest_belief = belief_request(forest, forest_support)
cycle_belief = belief_request(cycle, cycle_support)
source += belief_source("forest_belief", forest_belief)
source += belief_source("cycle_belief", cycle_belief)
module = Compiler(parse(source), module_name="latent_morph").qiec_module
assert module is not None

noise = 0.1
q = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))


def owner(name):
    return ParameterId.derive(module.module, "relation_weights", name)


weights = {
    owner("match_stem"): q,
    owner("match_suffix"): RelationWeightComplement(owner=owner("match_stem")),
    owner("copy"): q.new_tensor(1 - noise),
    owner("flip"): q.new_tensor(noise),
    owner("uniform"): q.new_tensor(0.5),
    owner("one"): q.new_ones(()),
}


def exact_runtime(score, **query):
    request = TreeStringGraphRequest(graph=forest, variable="suffix_a", **limits, **query)
    provider = TreeStringGraphRuntimeProvider.for_score(score, request, weights)
    return RuntimeConfiguration(providers=(provider,))


def approximate(export, request):
    provider = PrunedStringGraphRuntimeProvider(request, weights)
    runtime = RuntimeConfiguration(providers=(provider,))
    return run_named(module, export, runtime=runtime).value


def enumerate_lexicons(data, stem_match):
    probability = (stem_match - noise) / (1 - 2 * noise)
    masses = []
    selected = [[], []]
    for assignment in product(range(2), repeat=len(lexicon)):
        morph = dict(zip(lexicon, assignment, strict=True))
        mass = probability.new_tensor(0.5 ** len(lexicon))
        for stem, suffix, surface in data:
            x, y = morph[stem], morph[suffix]
            first = 1 - noise if x == surface[0] else noise
            from_suffix = 1 - noise if y == surface[1] else noise
            from_stem = 1 - noise if x == surface[1] else noise
            mass = mass * first * ((1 - probability) * from_suffix + probability * from_stem)
        masses.append(mass)
        selected[morph["suffix_a"]].append(mass)
    total = torch.stack(masses).sum()
    posterior = torch.stack([torch.stack(group).sum() / total for group in selected])
    return total.log(), posterior


runtime = exact_runtime("partition")
suffix_runtimes = [exact_runtime("suffix", word=word) for word in one_segment]
reference_q = torch.nn.Parameter(q.detach().clone())
optimizer = torch.optim.SGD([q], lr=0.08)
reference_optimizer = torch.optim.SGD([reference_q], lr=0.08)
before = run_named(module, "log_evidence", runtime=runtime).value.detach()
for _ in range(12):
    optimizer.zero_grad()
    reference_optimizer.zero_grad()
    actual = run_named(module, "log_evidence", runtime=runtime).value
    expected, posterior = enumerate_lexicons(training, reference_q)
    actual_posterior = torch.stack([
        run_named(module, "suffix", runtime=item).value.exp() for item in suffix_runtimes
    ])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_posterior, posterior)
    (-actual).backward()
    (-expected).backward()
    torch.testing.assert_close(q.grad, reference_q.grad)
    optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(q, reference_q)
    assert noise < q.item() < 1 - noise
assert run_named(module, "log_evidence", runtime=runtime).value > before

with torch.no_grad():
    tree_report = approximate("forest_belief", forest_belief)
    short = approximate("cycle_belief", cycle_belief.with_(
        config=cycle_belief.config.with_(max_sweeps=1),
    ))
    fitted = approximate("cycle_belief", cycle_belief)
    _, tree_truth = enumerate_lexicons(training, q)
    _, cycle_truth = enumerate_lexicons(loopy_data, q)
    torch.testing.assert_close(tree_report[5].exp(), tree_truth)
    marginal_error = (fitted[5].exp() - cycle_truth).abs().max().item()
assert tree_report[0] == "converged" and not tree_report[4].item()
assert short[0] == "limit" and short[3].item() > 1e-9
assert fitted[0] == "converged" and fitted[3].item() <= 1e-9
assert 0 < marginal_error < 0.01
```

Three observations connect the four latent morphemes in a tree. The
exact tree request retains message scale for the evidence. Its partition,
suffix marginal, and training gradient agree with enumeration over all
sixteen lexicons. The finite-support request uses the same tree graph,
and its suffix marginal also agrees with enumeration. The code optimizes
$q$ directly and checks that the updates stay inside $(\eta, 1-\eta)$.

The fourth observation closes a cycle through both stems and suffixes.
The same relation declarations supply this graph's factors. The one-sweep
request reports its iteration limit. The longer request reports convergence
when the largest total variation across an undamped message update falls
within tolerance. Enumeration separately measures the suffix marginal
error for this instance.
