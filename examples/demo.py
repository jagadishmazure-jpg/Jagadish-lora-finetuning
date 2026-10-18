"""Train a LoRA adapter on a synthetic rank-4 task, merge it, swap adapters on one frozen base, and run DPO on an adapter."""
import numpy as np

from lorax import LoRAConfig, LoRALinear, benchmark, merge_adapter
from lorax.dpo import dpo_loss, dpo_step, make_preference_dataset, preference_margin
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

# DPO on the adapter: 16 preference pairs, 200 steps, frozen base and reference.

cfg = LoRAConfig(rank=2, alpha=8.0, dropout=0.0)
policy = LoRALinear(8, 6, cfg, rng=np.random.default_rng(21))
ref = LoRALinear(8, 6, cfg, rng=np.random.default_rng(1021))
ref.W = policy.W.copy()
policy.train()
x, y_w, y_l = make_preference_dataset(n=16, seed=22)
w0 = policy.W.copy()
loss0 = dpo_loss(policy, ref, x, y_w, y_l)
for _ in range(200):
    dpo_step(policy, ref, x, y_w, y_l, beta=1.0, lr=0.5)
m = preference_margin(policy, ref, x, y_w, y_l)
print(f"DPO: loss {loss0:.4f} -> {dpo_loss(policy, ref, x, y_w, y_l):.4f}, mean margin {m.mean():.2f}, "
      f"pairs preferring the chosen answer {(m > 0).sum()}/16, base W unchanged: {np.array_equal(policy.W, w0)}")
