# 4. Fuzzy logic factorization

Matrix factorization approximates an observed matrix $R$ by a product $F G$ of two smaller matrices. Boolean matrix factorization replaces sum and product by OR and AND [@miettinen2008discrete]. This chapter factorizes a relation with entries in $[0, 1]$ under the `PRODUCT_FUZZY` operations of [chapter 1](01-first-quiver.md#composition), which interpolate between the two: the product [t-norm](https://en.wikipedia.org/wiki/T-norm) $a \otimes b = ab$ as fuzzy AND, and the probabilistic sum (noisy-OR) as fuzzy OR [@klement2000triangular].

## Background

Given an observed relation $R : X \to Z$ with entries in $[0, 1]$, find learnable relations $f : X \to Y$ and $g : Y \to Z$ whose composite approximates $R$:

$$
(f \mathbin{>\!\!>} g)(x, z) = \bigoplus_{y \in Y} f(x, y) \otimes g(y, z) = 1 - \prod_{y \in Y} \bigl(1 - f(x, y)\, g(y, z)\bigr).
$$

Each entry of the composite measures whether *some* intermediate $y$ links $x$ to $z$: the product is the "through $y$" and the noisy-OR the "some". With independent causes, $1 - \prod_y (1 - p_y)$ is the probability that at least one cause succeeds [@pearl1988probabilistic]. Where an ordinary factorization adds the contributions of the latent dimensions, this one treats them as independent chances. The small set $Y$ is a bottleneck of features: if $X$ are users and $Z$ items, $f(x, y)$ says how strongly user $x$ has preference $y$ and $g(y, z)$ how well item $z$ matches it. Permuting the features in both factors leaves the composite unchanged, so the factors are identified only up to that permutation.

## Setup

```python
import torch
from quivers import FinSet, morphism, observed, Program
```

`Y` is smaller than `X` and `Z`:

```python
X = FinSet(name="X", cardinality=6)   # domain: 6 elements
Y = FinSet(name="Y", cardinality=3)   # latent: 3 fuzzy features
Z = FinSet(name="Z", cardinality=8)   # codomain: 8 elements
```

## Observed relation

Build a synthetic relation from two random factors, so that an exact factorization through three features exists, and wrap it with [`observed`](../../api/core/morphisms.md):

```python
torch.manual_seed(42)

# ground-truth factors (unknown to the model)
f_true = torch.sigmoid(torch.randn(6, 3))
g_true = torch.sigmoid(torch.randn(3, 8))

# compose via noisy-OR of products
product = f_true.unsqueeze(-1) * g_true.unsqueeze(0)  # (6, 3, 8)
R_data = 1.0 - (1.0 - product).prod(dim=1)            # (6, 8)

R = observed(X, Z, R_data)
```

## Learnable factorization

Two learnable morphisms, created with [`morphism`](../../api/core/morphisms.md):

```python
f = morphism(X, Y)   # learnable: X -> Y
g = morphism(Y, Z)   # learnable: Y -> Z
```

Each stores unconstrained logits and exposes their sigmoid, in $(0, 1)$. Their composite uses [`ProductFuzzyAlgebra`](../../api/core/algebras.md), the default:

```python
h = f >> g            # composite under PRODUCT_FUZZY: X -> Z
```

```python
model = Program(h)
```

## Training

[`Program.bce_loss`](../../api/program.md) is the binary cross-entropy between the composite's tensor and a target in $[0, 1]$:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(500):
    optimizer.zero_grad()
    loss = model.bce_loss(R_data)
    loss.backward()
    optimizer.step()

    if (step + 1) % 100 == 0:
        print(f"step {step+1:>4d}  loss={loss.item():.4f}")
```

Gradients flow through the noisy-OR and the sigmoid to the logits of `f` and `g`.

## Inspecting the result

```python
with torch.no_grad():
    R_hat = model()

# element-wise reconstruction error
error = (R_hat - R_data).abs()
print(f"max error:  {error.max().item():.4f}")
print(f"mean error: {error.mean().item():.4f}")
```

```python
print("f (X -> Y):")
print(f.tensor.detach())

print("\ng (Y -> Z):")
print(g.tensor.detach())
```

Row $x$ of `f.tensor` gives the membership of $x$ in each of the three features, and column $z$ of `g.tensor` how each feature relates to $z$.

## Alternative algebras

Other [algebras](../../api/core/algebras.md) change what the composite means:

```python
from quivers import BOOLEAN, LUKASIEWICZ, GODEL

# Boolean: AND and OR (crisp relations)
f_bool = morphism(X, Y, algebra=BOOLEAN)

# Lukasiewicz t-norm with bounded sum
f_luk = morphism(X, Y, algebra=LUKASIEWICZ)

# Goedel: min and max
f_godel = morphism(X, Y, algebra=GODEL)
```

`BOOLEAN` composes crisp relations. `LUKASIEWICZ` uses the t-norm of [Łukasiewicz logic](https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic) with the bounded sum $\min(1, \sum_i x_i)$. `GODEL` uses minimum and maximum, the operations of [Gödel–Dummett logic](https://en.wikipedia.org/wiki/G%C3%B6del%E2%80%93Dummett_logic) and of possibility theory. The code is the same for each, and only the meaning of composition changes.

## Next

[Chapter 5](05-variational-inference.md) fits probabilistic programs by variational inference.
