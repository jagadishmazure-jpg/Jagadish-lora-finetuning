"""Adapter merge / unmerge.

Merging folds the adapter into the base weight so inference runs as a
single matmul with no adapter overhead::

    W_merged = W + scaling * (B @ A)

Unmerge restores the original frozen weight exactly (the delta is kept
aside at merge time).
"""

from __future__ import annotations

import numpy as np

from .lora import LoRALinear


class MergedLayer:
    """A plain linear layer produced by folding a LoRA adapter into W."""

    def __init__(self, weight: np.ndarray) -> None:
        self.W = weight

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W.T


def merge_adapter(layer: LoRALinear) -> MergedLayer:
    """Fold the adapter into a new base weight. The source layer is untouched."""
    merged_w = layer.W + layer.adapter_delta()
    merged = MergedLayer(merged_w)
    # Stash the delta so unmerge can restore the exact original weight.
    merged._delta = layer.adapter_delta().copy()  # type: ignore[attr-defined]
    return merged


def unmerge(merged: MergedLayer) -> MergedLayer:
    """Restore the original base weight from a merged layer.

    Returns a new layer whose weight equals the pre-merge W exactly.
    """
    if not hasattr(merged, "_delta"):
        raise ValueError("unmerge requires a layer produced by merge_adapter")
    restored = MergedLayer(merged.W - merged._delta)  # type: ignore[attr-defined]
    return restored
