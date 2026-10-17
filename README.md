# Jagadish-lora-finetuning

[![ci](https://github.com/jagadishmazure-jpg/Jagadish-lora-finetuning/actions/workflows/ci.yml/badge.svg)](https://github.com/jagadishmazure-jpg/Jagadish-lora-finetuning/actions/workflows/ci.yml)
[![codeql](https://github.com/jagadishmazure-jpg/Jagadish-lora-finetuning/actions/workflows/codeql.yml/badge.svg)](https://github.com/jagadishmazure-jpg/Jagadish-lora-finetuning/actions/workflows/codeql.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**LoRA fine-tuning from scratch in numpy: frozen base weights, trainable low-rank adapters, manual backprop, merge and unmerge, and adapter swapping. CPU only, no GPU, no model weights.**

> Personal learning and portfolio project. Everything runs offline on synthetic,
> fictional data with no API keys. Nothing is deployed. Numbers in this README
> come from real runs and are re-checked by CI (`scripts/render_docs.py --check`).

## Recruiter summary

- Shows I understand parameter-efficient fine-tuning at the level of the maths, not just a library call.
- Real run: a rank-4 adapter fits a synthetic rank-4 change (loss 57.76 to near zero in 300 steps); merging matches the adapter output to 1e-14; the frozen base is provably unchanged.
- Honest about scale: the demo layer trains 50% of its parameters; the same formula gives 0.39% for a 4096x4096 layer at rank 8.

## Purpose

Full fine-tuning updates every weight. LoRA freezes the pretrained weight `W`
and learns a low-rank update `B·A`, scaled by `alpha/r`, which can be merged back
for no inference overhead. I built it by hand in numpy so every step (init,
scaling, gradients, merge) is visible and tested. Derivation:
[docs/lora-math.md](docs/lora-math.md).

## Architecture

```mermaid
flowchart LR
    X["x"] --> W["frozen W"]
    X --> A["A (r x in)<br/>Kaiming init"] --> B["B (out x r)<br/>zero init"]
    W --> SUM["y = Wx + (alpha/r)·B(Ax)"]
    B --> SUM
    SUM --> L["MSE loss"] -->|"grads for A, B only"| T["train_step"]
    T --> A
    T --> B
    SUM -.-> M["merge: W + (alpha/r)·BA"]
```

## How it works

- `LoRAConfig` holds rank, alpha and dropout; scaling is `alpha / rank`.
- `LoRALinear` keeps `W` frozen; `A` starts Kaiming-uniform and `B` starts at
  zero, so the adapter begins as no change.
- `train_step` back-propagates by hand through the adapter path only.
- `merge_adapter` folds `(alpha/r)·B·A` into `W`; `unmerge` reverses it.
- `adapter_swap_demo` trains two adapters on two tasks over one shared base.

Trainable parameters per layer are `r·(in + out)` against `in·out` for full
fine-tuning, so the saving depends on layer size.

## Real code and output

```python
"""Train a LoRA adapter on a synthetic rank-4 task, merge it, and swap adapters on one frozen base."""
import numpy as np

from lorax import LoRAConfig, LoRALinear, benchmark, merge_adapter
from lorax.eval import adapter_swap_demo, make_synthetic_task
from lorax.train import train_step

res = benchmark(steps=300)
print(f"Loss before training: {res.base_loss:.4f}, after 300 steps: {res.adapted_loss:.2e}")
print(f"Trainable params in this 16x16 demo layer: {res.trainable_params} of {res.full_params} "
      f"({100 * res.trainable_params / res.full_params:.0f}%)")

d, r = 4096, 8  # arithmetic only: r*(in+out) vs in*out for one 4096x4096 projection
print(f"Same formula for a 4096x4096 layer at rank 8: {r * 2 * d:,} of {d * d:,} "
      f"({100 * r * 2 * d / (d * d):.2f}%)")

w_base, _, (x, y) = make_synthetic_task(seed=0)
layer = LoRALinear(16, 16, LoRAConfig(rank=4, alpha=8.0, dropout=0.0), rng=np.random.default_rng(1))
layer.W = w_base.copy()
frozen = layer.W.copy()
layer.train()
for _ in range(300):
    train_step(layer, x, y, 0.1)
merged = merge_adapter(layer.eval())
gap = float(np.max(np.abs(merged.forward(x) - layer.forward(x))))
print(f"Merged vs adapter output, max abs difference: {gap:.2e}")
print("Frozen base weight unchanged by training:", bool(np.array_equal(layer.W, frozen)))

swap = adapter_swap_demo()
print("Adapter swap losses on the shared base:",
      {k: f"{v:.2e}" for k, v in swap["swapped_losses"].items()})
```

Output of `python examples/demo.py` (real run, re-checked in CI):

<!-- run: python examples/demo.py -->
```text
Loss before training: 57.7639, after 300 steps: 2.18e-09
Trainable params in this 16x16 demo layer: 128 of 256 (50%)
Same formula for a 4096x4096 layer at rank 8: 65,536 of 16,777,216 (0.39%)
Merged vs adapter output, max abs difference: 1.07e-14
Frozen base weight unchanged by training: True
Adapter swap losses on the shared base: {'task_a': '1.66e-10', 'task_b': '1.44e-22'}
```
<!-- /run -->

## How to run

```bash
git clone https://github.com/jagadishmazure-jpg/Jagadish-lora-finetuning.git
cd Jagadish-lora-finetuning
python -m pip install numpy pytest
python -m pytest -q          # offline, no keys
python examples/demo.py
python scripts/render_docs.py --check   # documented output still matches
```

Test suite size (real collection, re-checked in CI):

<!-- run: python scripts/count_tests.py -->
```text
29 tests collected
```
<!-- /run -->

## Guardrails

- The base weight is never written by training; the demo checks it byte for byte.
- Config validation rejects a rank larger than the layer allows.
- Merge equivalence is tested, so a merged model behaves like the adapter.

## Limitations

- One linear layer on synthetic data; no transformer, tokenizer or real model.
- The synthetic targets are exact rank-r changes, so near-zero loss is expected by design.
- At this demo's size LoRA trains 50% of parameters; the under-1% figure only holds for large layers.
- No quantisation (QLoRA), optimiser state or multi-adapter serving.
- `adapter_swap_demo` returns `base_unchanged: True` as a constant; the demo script checks it directly instead.

## Built vs planned

| Capability | Status | Where |
|---|---|---|
| Config and LoRALinear | Built | `lorax/config.py`, `lorax/lora.py` |
| Manual backprop through the adapter | Built | `lorax/train.py` |
| Merge and unmerge | Built | `lorax/merge.py` |
| Benchmark and adapter swap | Built | `lorax/eval.py` |
| Real model fine-tune with PEFT on a GPU | Planned | docs/implementation-guide.md |
| QLoRA | Planned | not started |

## Implementation guide

The step-by-step path from this offline build to an Azure deployment is in
[docs/implementation-guide.md](docs/implementation-guide.md). None of it has been
run; it is a plan.

## Related repositories

| Repo | How it relates |
|---|---|
| [Jagadish-dspy-prompt-optimizer](https://github.com/jagadishmazure-jpg/Jagadish-dspy-prompt-optimizer) | Adapting behaviour through prompts instead of weights |
| [Jagadish-llm-evals-observability](https://github.com/jagadishmazure-jpg/Jagadish-llm-evals-observability) | Evaluating a fine-tuned model before release |
| [Jagadish-ai-learning-lab](https://github.com/jagadishmazure-jpg/Jagadish-ai-learning-lab) | Other from-scratch learning notebooks |

## Documentation

- [LoRA maths](docs/lora-math.md)
- [Concepts](knowledge/01-concepts.md)
- [Diagrams](knowledge/02-architecture.md)
- [Implementation guide (Azure)](docs/implementation-guide.md)
- [Interview points](docs/interview-points.md)
- [Security policy](SECURITY.md)

## Project layout

```
lorax/
  config.py  LoRAConfig
  lora.py    LoRALinear: frozen W, adapters A and B
  train.py   manual backprop, MSE
  merge.py   merge / unmerge
  eval.py    benchmark, adapter swap
examples/demo.py  the run shown above
tests/       CPU-only pytest suite
```

## License

MIT, see [LICENSE](LICENSE). All data is synthetic; every organisation and person is fictional.
