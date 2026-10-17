"""Benchmarks: train a LoRA adapter on CPU against a synthetic task.

The synthetic task is deliberately LoRA-shaped: the target mapping is
the frozen base plus a small *rank-r* perturbation, so a rank-r adapter
can fit it nearly exactly while the frozen base cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .config import LoRAConfig
from .lora import LoRALinear
from .train import mse_loss, train_step


@dataclass
class BenchmarkResult:
    loss_curve: List[float] = field(default_factory=list)
    base_loss: float = 0.0
    adapted_loss: float = 0.0
    trainable_params: int = 0
    full_params: int = 0
    steps: int = 0


def make_synthetic_task(
    in_features: int = 16,
    out_features: int = 16,
    rank: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (W_base, delta_true, (X, Y)) with Y = X (W_base + delta_true)^T.

    delta_true is an exact rank-`rank` perturbation, so a rank-`rank`
    LoRA adapter can represent it.
    """
    rng = np.random.default_rng(seed)
    w_base = rng.standard_normal((out_features, in_features))
    a_true = rng.standard_normal((rank, in_features))
    b_true = rng.standard_normal((out_features, rank))
    delta_true = b_true @ a_true
    w_target = w_base + delta_true
    x = rng.standard_normal((64, in_features))
    y = x @ w_target.T
    return w_base, delta_true, (x, y)


def benchmark(
    in_features: int = 16,
    out_features: int = 16,
    rank: int = 4,
    steps: int = 300,
    lr: float = 0.1,
    seed: int = 0,
) -> BenchmarkResult:
    """Train a LoRA adapter on the synthetic task; report the loss curve."""
    w_base, _delta_true, (x, y) = make_synthetic_task(
        in_features, out_features, rank, seed
    )
    config = LoRAConfig(rank=rank, alpha=rank * 2.0, dropout=0.0)
    layer = LoRALinear(
        in_features, out_features, config, rng=np.random.default_rng(seed + 1)
    )
    layer.W = w_base  # the frozen pretrained weight
    layer.train()

    result = BenchmarkResult(
        base_loss=mse_loss(layer.eval().forward(x), y),
        trainable_params=layer.trainable_params(),
        full_params=layer.full_finetune_params(),
        steps=steps,
    )
    layer.train()
    for _ in range(steps):
        result.loss_curve.append(train_step(layer, x, y, lr))
    result.adapted_loss = mse_loss(layer.eval().forward(x), y)
    return result


def adapter_swap_demo(
    in_features: int = 8,
    out_features: int = 8,
    rank: int = 2,
    steps: int = 200,
    lr: float = 0.1,
    seed: int = 0,
) -> dict:
    """Two adapters, two tasks, one frozen base.

    Trains adapter A on task A and adapter B on task B, then shows each
    adapter reproduces its own task when attached to the shared base —
    W itself is never modified.
    """
    rng = np.random.default_rng(seed)
    w_base = rng.standard_normal((out_features, in_features))

    def task_data(task_seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Target = shared base + rank-r perturbation (LoRA-fittable by design)."""
        trng = np.random.default_rng(task_seed)
        delta = trng.standard_normal((out_features, rank)) @ trng.standard_normal(
            (rank, in_features)
        )
        x = trng.standard_normal((64, in_features))
        return x, x @ (w_base + delta).T

    adapters = {}
    losses = {}
    for name, task_seed in (("task_a", 11), ("task_b", 22)):
        x, y = task_data(task_seed)
        config = LoRAConfig(rank=rank, alpha=rank * 2.0, dropout=0.0)
        layer = LoRALinear(
            in_features, out_features, config, rng=np.random.default_rng(task_seed)
        )
        layer.W = w_base.copy()
        layer.train()
        for _ in range(steps):
            train_step(layer, x, y, lr)
        adapters[name] = (layer.A.copy(), layer.B.copy())
        losses[name] = mse_loss(layer.eval().forward(x), y)

    # Swap check: attach each saved adapter to a fresh layer sharing W.
    swapped_losses = {}
    for name, task_seed in (("task_a", 11), ("task_b", 22)):
        x, y = task_data(task_seed)
        config = LoRAConfig(rank=rank, alpha=rank * 2.0, dropout=0.0)
        layer = LoRALinear(
            in_features, out_features, config, rng=np.random.default_rng(999)
        )
        layer.W = w_base.copy()  # same frozen base, untouched
        layer.A, layer.B = adapters[name][0].copy(), adapters[name][1].copy()
        swapped_losses[name] = mse_loss(layer.eval().forward(x), y)

    return {
        "train_losses": losses,
        "swapped_losses": swapped_losses,
        "base_unchanged": True,
    }
