# A contextual edit channel

A channel can assign different edit probabilities to the same symbol
depending on what follows it. In this example, an embedding of the
remaining input predicts the logits for the next edit. The same
parameters score all alignments of all observed pairs.

The channel allows deletion, replacement, and insertion. It allows
one consecutive insertion, including at the end of the input. At
each step, the normalization includes all legal edits, including
edits incompatible with the observed output. The likelihood then
sums just the alignments of the observed pair.

<!-- python: run -->
```python
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.relations import ContextualEditChannel


class InputContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.symbols = nn.Embedding(3, 8, padding_idx=2)
        self.edits = nn.Linear(8, 6)

    def forward(self, words):
        width = max(map(len, words)) + 1
        padded = torch.tensor(
            [word + (2,) * (width - len(word)) for word in words],
            device=self.symbols.weight.device,
        )
        encoded = self.symbols(padded)
        remaining = encoded.flip((1,)).cumsum(dim=1).flip((1,))
        return self.edits(remaining).unsqueeze(2).expand(-1, -1, 2, -1)


tape = StringSet(alphabet=FinSet(name="TwoSymbols", cardinality=2))
channel = ContextualEditChannel(tape, tape, InputContext(), max_insertions=1)
plan = channel.plan(
    ((0, 1), (0, 0), (), (1,)),
    ((0,), (1, 0), (1,), ()),
    max_states=256,
    max_work=512,
)

optimizer = torch.optim.Adam(channel.parameters(), lr=0.03)
negative_log_likelihood = -plan.log_prob().sum()
negative_log_likelihood.backward()
assert all(parameter.grad is not None for parameter in channel.parameters())
assert torch.isfinite(negative_log_likelihood)
optimizer.step()
```

The plan stores which edit alignments are compatible with the
observations. Reusing it after an optimizer step recalculates
probabilities from the updated parameters.
