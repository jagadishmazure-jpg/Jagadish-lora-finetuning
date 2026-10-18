# The LoRA math

## Low-rank update

A linear layer maps `x ∈ R^in` to `y ∈ R^out` via `y = Wx`,
`W ∈ R^(out×in)`. Full fine-tuning learns a dense update `ΔW` with
`in·out` parameters. LoRA constrains the update to rank `r`:

```
ΔW = (α/r) · B·A,   A ∈ R^(r×in), B ∈ R^(out×r)
```

Trainable parameters drop from `in·out` to `r·(in+out)`. For
`in = out = 4096`, `r = 8`: 16.8M → 65K (**0.4%**).

The adapted forward pass never materializes `ΔW` during training:

```
y = Wx + (α/r)·B(Ax)
```

Two small matmuls instead of one big one; `W` stays frozen.

## Why B starts at zero

If both `A` and `B` were randomly initialized, the adapter would inject
random noise into the model's outputs at step 0, destroying pretrained
behavior before training even starts. With `B = 0`:

```
ΔW₀ = B₀·A = 0  →  y₀ = Wx
```

The model behaves exactly as the pretrained base on the first forward
pass, and the adapter grows smoothly from there. `A` uses Kaiming
uniform init (`U(-√(6/in), √(6/in))`) so activations keep unit variance
flowing through the adapter path.

A subtle consequence, verified in `test_only_adapter_moves`: with
`B = 0`, the gradient `dA = (s·dy·B)ᵀx` is zero on step 1 — only `B`
moves first, then `A` follows. This is correct LoRA behavior, not a bug.

## Why scale by α/r

Without scaling, changing `r` changes the magnitude of the update
(`‖BA‖` grows with `r`), forcing the learning rate to be retuned per
rank. Dividing by `r` keeps the update magnitude roughly rank-invariant;
`α` then acts as a single effective learning-rate multiplier. This is
the `lora_alpha` / `r` convention from the original LoRA paper
(Hu et al., 2021).

## Merge equivalence (proof sketch)

The adapter path computes `s·B(Ax)` with `s = α/r`. By associativity:

```
s·B(Ax) = (s·B·A)x = ΔW·x
```

So folding is exact in exact arithmetic:

```
W_merged = W + ΔW    →    W_merged·x = Wx + ΔWx = adapted forward
```

In floating point the round-trip `(W + ΔW) - ΔW` recovers `W` up to
last-ulp rounding — hence `unmerge` is verified with `allclose`
(atol/rtol 1e-12), not bitwise equality. `test_merge_equivalence`
checks merged-forward == adapter-forward to 1e-6 on random inputs.

## Backprop through the adapter only

With MSE loss `L = mean((y - t)²)`, `N = batch·out`:

```
dy  = 2(y - t)/N
dB  = s · dyᵀh₂        h₂ = dropout-masked adapter hidden state
dh₂ = s · dy·B
dh  = dh₂ ⊙ mask
dA  = dhᵀx
```

`W` never appears in the backward pass — it is frozen by construction,
and the test suite asserts it is bitwise unchanged after training.
Inverted dropout (`mask / (1-p)`) keeps `E[mask] = 1`, so no rescaling
is needed at inference; dropout is disabled in eval mode.

## DPO: preference optimization without RL

RLHF trains a reward model on human preferences, then optimizes the
policy against it with PPO — two training runs and an RL loop. DPO
(Direct Preference Optimization) observes that the optimal policy for
the RLHF objective has a closed form, and turns the whole thing into a
single classification loss on preference pairs `(x, y_w, y_l)`:

```
L = −log σ(β · (r(x, y_w) − r(x, y_l)))
r(x, y) = log π_θ(y|x) − log π_ref(y|x)
```

`r` is the *implicit reward*: how much more log-likely the adapted
policy rates a completion than the frozen reference. With `B = 0` at
init, `r = 0` everywhere, so the initial loss is exactly `−log σ(0) =
log 2 ≈ 0.693` — verified in `test_dpo_loss_is_log2_at_init`.

Backprop mirrors `train.py`, with one simplification. Let
`m = β·(r_w − r_l)`; then `dL/dm = σ(m) − 1 ≤ 0`, and

```
dy_w = dL/dm · β · (onehot(y_w) − softmax(l_w))
dy_l = −dL/dm · β · (onehot(y_l) − softmax(l_l))
```

Both completions share the same prompt forward pass, so the two
softmax terms cancel:

```
dy = coef · (onehot(y_w) − onehot(y_l)),   coef = (σ(m) − 1)·β/N ≤ 0
```

The gradient pushes the chosen token's logit up and the rejected
token's down, weighted by `|coef|` — pairs the policy already ranks
correctly (`m >> 0`, `σ(m) ≈ 1`) contribute almost nothing, which is
why DPO keeps improving ranking instead of overfitting log-likelihood.
From `dy` the adapter backward pass is identical to SFT:

```
dB = s·dyᵀh₂/N,  dh₂ = s·dy·B/N,  dh = dh₂⊙mask,  dA = dhᵀx/N
```

`W` and the reference layer are never touched — `test_dpo_shifts_policy`
asserts `W` is bitwise unchanged after 200 DPO steps while the mean
margin moves from 0 to > 8 and preference accuracy reaches 1.0.
