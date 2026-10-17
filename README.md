# lorax — LoRA fine-tuning from scratch (numpy, no GPU)

Low-Rank Adaptation (LoRA) implemented at code level: frozen base weights,
trainable low-rank adapters, manual backprop, merge/unmerge, and CPU
benchmarks on synthetic tasks. The only dependency is numpy — no torch,
no GPU, no network.

## The problem

Full fine-tuning updates every weight of a model. For a 7B-parameter
model that means training 7B parameters: optimizer states, gradients,
and checkpoints at full scale. LoRA freezes the pretrained weights and
trains a low-rank update instead — typically **<1% of the parameters** —
then merges the adapter back for zero-overhead inference.

## The math

For a frozen weight `W` (out × in) and rank `r`:

```
y = Wx + (α/r) · B(Ax)
```

- `A` is (r × in), Kaiming-initialized — trainable
- `B` is (out × r), **zero-initialized** — trainable, so the adapter
  starts as the identity change
- `α/r` is the LoRA scaling factor

Trainable params per layer: `r·in + r·out` vs `in·out` for full
fine-tuning. Merging folds the adapter in with zero inference overhead:

```
W_merged = W + (α/r)·(B·A)
```

See [docs/lora-math.md](docs/lora-math.md) for the full derivation,
init rationale, and merge-equivalence argument.

## What's here vs production

| This repo | Production (e.g. HF PEFT) |
|---|---|
| numpy, CPU-only, tiny synthetic tasks | PyTorch, GPUs, real models |
| Manual backprop through one linear layer | Autograd through transformers |
| Proves the engineering: init, scaling, merge, frozen base | Adds quantization (QLoRA), multi-adapter serving |

Training at scale needs GPUs; this repo proves the math and the
mechanics are correct — every claim is covered by a CPU test.

## Quickstart

```bash
pip install numpy
python -c "
from lorax import benchmark
res = benchmark(steps=300)
print(f'base loss: {res.base_loss:.4f} -> adapted: {res.adapted_loss:.6f}')
print(f'trainable params: {res.trainable_params} / {res.full_params}')
"
```

## Architecture

```
lorax/
  config.py   LoRAConfig: rank, alpha, dropout, scaling = alpha/rank
  lora.py     LoRALinear: frozen W + trainable A/B, forward, param counts
  train.py    train_step: manual backprop through the adapter path only
  merge.py    merge_adapter / unmerge: fold adapter into W and back
  eval.py     benchmark + adapter_swap_demo on synthetic rank-r tasks
```

## Tests

```bash
pip install numpy pytest
python -m pytest tests/ -q   # 29 tests, CPU only, <1s
```

## License

MIT — synthetic data only, no model weights included.

## Knowledge

Distilled notes and architecture diagrams: [knowledge/](knowledge/README.md)
