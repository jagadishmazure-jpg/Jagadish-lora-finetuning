# LoRA fine-tuning: concepts

## The idea in one paragraph

Full fine-tuning updates every weight — expensive and it forgets. LoRA
freezes the base weights **W** and learns a low-rank update **ΔW = s·B·A**,
where **A** (r×k) and **B** (d×r) are tiny and **s** is a scaling factor.
Forward pass: `y = Wx + s·B(Ax)`. Only A and B train; W never moves.
For rank r ≪ min(d,k), the adapter is a few percent of the parameters —
cheap to train, cheap to store, swappable at inference time.

## The implementation details that matter

- **Init:** A ~ Kaiming (so the product starts with sane variance), B = 0
  (so ΔW = 0 at step 0 — the adapter starts as a no-op and *earns* its
  influence). This is the standard recipe; see `docs/lora-math.md` for
  the rationale.
- **Frozen base:** backprop flows through the adapter path only. The
  base model's knowledge is untouched — no catastrophic forgetting by
  construction.
- **Scaling s:** controls how aggressively the adapter perturbs the base;
  the classic α/r normalization.
- **Merge/unmerge:** for deployment, `W' = W + s·B·A` folds the adapter
  into the weights (zero inference overhead). Merge equivalence is
  verified to 1e-6 here; round-trip merge→unmerge restores exactly.

## What the demo proves

- **Real training:** base loss 57.76 → adapted loss 8.8e-07 on a
  synthetic rank-r task. The adapter actually learns.
- **Adapter swap:** two adapters on one frozen base — task-switching
  without reloading the model. This is the production pattern (one base,
  many task adapters).
- A genuine design bug caught in verification: targets were generated
  from *per-task* base matrices, making them unfittable — fixed to
  rank-r perturbations of the *shared* base, which is what LoRA can
  actually express.

## Numpy here vs. production

| This repo (numpy) | Production |
|---|---|
| Hand-derived backprop through the adapter | HF PEFT / `peft` library |
| Tiny synthetic rank-r tasks, CPU | Real checkpoints, GPUs |
| Merge verified to 1e-6 on small matrices | Same merge math, fused kernels |

The README is upfront about this: production training needs GPUs and HF
PEFT. This repo proves the *engineering* — the math, the merge proof,
the frozen-base training loop.

## Module map

| Module | Role |
|---|---|
| `lorax/config.py` | LoRAConfig (rank, alpha, init) |
| `lorax/lora.py` | LoRALinear: frozen W, `forward = Wx + s·B(Ax)` |
| `lorax/train.py` | adapter-only backprop + training demo |
| `lorax/merge.py` | merge/unmerge with equivalence checks |
| `lorax/eval.py` | adapter-swap demo |

Interview line: *"LoRA says the update you need lives in a low-rank
subspace. Freeze the base, learn B and A, and B starts at zero so the
adapter is a no-op until it earns its influence. One base, many
swappable adapters."*
