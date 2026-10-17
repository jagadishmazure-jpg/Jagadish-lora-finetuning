"""Manual training step: backprop through the LoRA path only.

The base weight W is frozen — gradients are computed and applied to A
and B exclusively. Dropout on the adapter path is active only when the
layer is in training mode (see ``LoRALinear._dropout_mask``).
"""

from __future__ import annotations

import numpy as np

from .lora import LoRALinear


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def train_step(
    layer: LoRALinear, x: np.ndarray, y_true: np.ndarray, lr: float
) -> float:
    """One SGD step on the adapter. Returns the MSE loss *before* the update.

    Forward:  y = x W^T + s * ((x A^T * m) B^T),  s = scaling, m = dropout mask
    Backprop (mean-squared-error, N = batch * out):
        dy  = 2 (y - y_true) / N
        dB  = s * dy^T h2            (h2 = masked adapter hidden state)
        dh2 = s * dy B
        dh  = dh2 * m
        dA  = dh^T x
    W receives no gradient and is never touched.
    """
    if not layer.training:
        raise ValueError("train_step requires the layer in training mode")

    s = layer.config.scaling
    n = x.shape[0] * layer.out_features

    # ---- forward (recompute internals for the backward pass) ----
    base = x @ layer.W.T
    h = x @ layer.A.T
    mask = layer._dropout_mask(h.shape[0])
    h2 = h * mask
    adapter = (h2 @ layer.B.T) * s
    y = base + adapter

    loss = mse_loss(y, y_true)

    # ---- backward through the adapter path only ----
    dy = 2.0 * (y - y_true) / n
    dB = s * (dy.T @ h2)          # (out, r)
    dh2 = s * (dy @ layer.B)      # (batch, r)
    dh = dh2 * mask               # (batch, r)
    dA = dh.T @ x                 # (r, in)

    # ---- SGD update: A and B only; W is frozen by construction ----
    layer.A -= lr * dA
    layer.B -= lr * dB

    return loss
