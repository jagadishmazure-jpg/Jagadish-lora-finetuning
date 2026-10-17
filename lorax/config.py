"""LoRA configuration: rank, scaling, dropout, and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter.

    rank: inner dimension r of the low-rank update (r << min(in, out)).
    alpha: LoRA scaling hyperparameter; effective scale is alpha / rank.
    dropout: dropout probability applied to the adapter path during training.
    target_modules: names of linear layers the adapter attaches to
        (informational in this numpy implementation).
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError(f"rank must be a positive int, got {self.rank!r}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout!r}")

    @property
    def scaling(self) -> float:
        """Effective scale applied to the low-rank update: alpha / rank."""
        return self.alpha / self.rank

    def validate_for(self, in_features: int, out_features: int) -> None:
        """Check the rank is a genuine low-rank choice for a layer shape."""
        if self.rank >= min(in_features, out_features):
            raise ValueError(
                f"rank {self.rank} must be < min(in_features, out_features) "
                f"= {min(in_features, out_features)} to be low-rank"
            )
