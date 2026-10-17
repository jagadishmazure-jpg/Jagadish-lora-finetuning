"""LoRA linear layer: frozen base weight plus a trainable low-rank adapter.

The adapted forward pass is::

    y = W x + scaling * B (A x)

where W is frozen, A is (rank x in_features) with Kaiming-style init,
B is (out_features x rank) initialized to zeros (so the adapter starts
as the identity change), and scaling = alpha / rank.
"""

from __future__ import annotations

import numpy as np

from .config import LoRAConfig


class LoRALinear:
    """A linear layer with a frozen base weight and a LoRA adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        config.validate_for(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

        # Frozen base weight (never updated by train_step).
        self.W = self.rng.standard_normal((out_features, in_features))

        # Trainable adapter: A ~ Kaiming uniform, B = zeros.
        # Kaiming uniform bound for a matrix with fan_in inputs.
        bound = float(np.sqrt(6.0 / in_features))
        self.A = self.rng.uniform(-bound, bound, size=(config.rank, in_features))
        self.B = np.zeros((out_features, config.rank))

        self.training = True

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------
    def train(self) -> "LoRALinear":
        self.training = True
        return self

    def eval(self) -> "LoRALinear":
        self.training = False
        return self

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def _dropout_mask(self, batch: int) -> np.ndarray:
        p = self.config.dropout
        if not self.training or p <= 0.0:
            return np.ones((batch, self.config.rank))
        keep = 1.0 - p
        mask = (self.rng.random((batch, self.config.rank)) < keep).astype(float)
        return mask / keep  # inverted dropout keeps expectation stable

    def forward(self, x: np.ndarray) -> np.ndarray:
        """y = Wx + scaling * B(dropout(Ax))."""
        base = x @ self.W.T
        h = x @ self.A.T
        h = h * self._dropout_mask(h.shape[0])
        adapter = (h @ self.B.T) * self.config.scaling
        return base + adapter

    def adapter_delta(self) -> np.ndarray:
        """The effective weight update B @ A * scaling (out x in)."""
        return (self.B @ self.A) * self.config.scaling

    # ------------------------------------------------------------------
    # Parameter accounting
    # ------------------------------------------------------------------
    def trainable_params(self) -> int:
        """Number of trainable parameters: |A| + |B|."""
        return int(self.A.size + self.B.size)

    def full_finetune_params(self) -> int:
        """Parameters a full fine-tune of W would update."""
        return int(self.W.size)

    def trainable_fraction(self) -> float:
        return self.trainable_params() / self.full_finetune_params()
