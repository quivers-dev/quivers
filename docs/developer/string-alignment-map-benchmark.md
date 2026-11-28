# Joint alignment MAP benchmark

`python -m tests.benchmarks.string_alignment_map` compares
`dual_alignment_map` with `intersect` followed by an independent
longest-path decoder on the materialized DAG. Both use the
joint maximum over words and original paths. This is the objective
in §2.5 of Peng, Cotterell, and Eisner. Paul and Eisner motivate
avoiding the materialized intersection. [Peng, Cotterell, and Eisner (2015)](https://www.cs.jhu.edu/~jason/papers/peng%2Bal.emnlp15.pdf), [Paul and Eisner (2012)](https://www.cs.jhu.edu/~jason/papers/paul%2Beisner.naacl12.pdf).

## Exact checks and costs

The two unary factors have $n$ distinct accepting paths for each of
the words $01$ and $10$. The left paths begin with epsilon and the
right paths end with epsilon. Parallel paths can tie. In the agreement
family both factors initially choose $01$. In the adaptive family they
initially choose different words with the same unigram counts. The
solver activates a boundary bigram and certifies $01$ on its second
iteration.

The benchmark enumerates the original compatible path pairs and
checks equality with the materialized multiset of word and score.
It separately checks each local maximization, including original
path identities, and verifies that the multiplier for each feature
sums to zero across copies on every dual update. This yields an exact
upper bound because the multiplier terms cancel on every feasible
assignment. Full-word agreement attains the bound. The oracle checks
the reported minimum bound and the best feasible score. A finite
iteration budget can end with disagreement. Termination on broader
domains requires separate arguments.

Each local graph here has $6n+1$ states. In the materialized
product the root and left-leading epsilon records account for $1+2n$
states. For each of the two symbols in a compatible pair, the
$2n^2$ matches allocate one product state and one intermediate
state to retain the two primitive weights. The right-trailing
epsilon accounts for another $2n^2$ states. The builder retains all
$10n^2+2n+1$ states to completion. The largest implicit decode
reaches $6n+1$ configurations because each original branch has a
unique prefix. Its arc counts include the additional fixed-word
decodes used to check primal feasibility. They count examinations
rather than elapsed time or memory. The JSON report also records
the explicit decoder's peak pending score entries and arc visits. The
native solver then verifies the selected local maxima with an independent
scalar suffix pass and validates the original walks. Each certified row
therefore examines another $24n+6$ arcs, including the two walks' three
original arcs each. Its largest verification graph also has $6n+1$
states. These counts exclude the benchmark's separate enumeration oracle.

| Family | Paths per word and factor | Peak product states | Largest local decode | Local calls | Local arc examinations | Verification arc examinations | Product label probes | Product arcs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Agreement | 2 | 45 | 13 | 2 | 24 | 54 | 13 | 44 |
| Agreement | 8 | 657 | 49 | 2 | 96 | 198 | 145 | 656 |
| Agreement | 24 | 5,809 | 145 | 2 | 288 | 582 | 1,201 | 5,808 |
| Agreement | 48 | 23,137 | 289 | 2 | 576 | 1,158 | 4,705 | 23,136 |
| Adaptive | 2 | 45 | 13 | 6 | 66 | 54 | 13 | 44 |
| Adaptive | 8 | 657 | 49 | 6 | 264 | 198 | 145 | 656 |
| Adaptive | 24 | 5,809 | 145 | 6 | 792 | 582 | 1,201 | 5,808 |
| Adaptive | 48 | 23,137 | 289 | 6 | 1,584 | 1,158 | 4,705 | 23,136 |

All timed rows use eight dual iterations at most, n-gram order two, a
four-feature limit and starting step $3/2$ divided by the iteration.
Both builders receive limits of 100,000 states and 1,000,000 work units.
The dual limits apply to each local decode. The product work includes
label probes and constructed arcs. The families retain zero and one
active features respectively.

An untimed control changes only the allowed n-gram order to one and
returns a feasible score of $1$, an upper bound of $3/2$, and no
certificate. A separate control updates one original tensor in place,
using the same graph, limits, and previously materialized product.
The optimum changes from $01$ at score $1$ to $10$ at score
$2$. Suppressing both word weights in the right factor makes its
support empty. The benchmark also checks resource, positive-cycle,
and algebra refusals. The update control reruns discrete MAP on each live parameter
snapshot.

## Timed runs

These are CPU float64 medians over five calls on macOS 15.7.4 arm64
with PyTorch 2.15.0 and one Torch thread. After one warmup per method,
the benchmark alternates method order and leaves the fixture weights
unchanged. Each dual invocation starts at zero multipliers. Each
explicit invocation rebuilds the product and then decodes it. Timings
include the native solver's independent witness verification. They
exclude enumeration and certificate checks performed by the benchmark's
external oracle, manual counter instrumentation, and live-update controls.

| Family | Paths per word and factor | Implicit (ms) | Explicit build and decode (ms) | Explicit build (ms) |
| --- | ---: | ---: | ---: | ---: |
| Agreement | 2 | 0.438 | 0.658 | 0.220 |
| Agreement | 8 | 0.999 | 3.267 | 1.797 |
| Agreement | 24 | 2.568 | 77.300 | 24.080 |
| Agreement | 48 | 4.912 | 234.442 | 111.588 |
| Adaptive | 2 | 0.527 | 0.332 | 0.167 |
| Adaptive | 8 | 1.454 | 2.941 | 1.620 |
| Adaptive | 24 | 4.088 | 39.853 | 24.734 |
| Adaptive | 48 | 13.187 | 275.552 | 100.488 |

The smallest adaptive case incurs more local calls than the materialized
decoder. Increasing parallel alignments grows the materialized product
quadratically in these two families. The timings have no performance
threshold and describe only this implementation and workload.
