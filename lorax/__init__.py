"""lorax: LoRA fine-tuning from scratch in numpy. No GPU required."""

from .config import LoRAConfig
from .dpo import (
    dpo_loss,
    dpo_step,
    implicit_reward,
    make_preference_dataset,
    preference_margin,
)
from .eval import BenchmarkResult, adapter_swap_demo, benchmark, make_synthetic_task
from .lora import LoRALinear
from .merge import MergedLayer, merge_adapter, unmerge
from .train import mse_loss, train_step

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "MergedLayer",
    "merge_adapter",
    "unmerge",
    "mse_loss",
    "train_step",
    "dpo_loss",
    "dpo_step",
    "implicit_reward",
    "preference_margin",
    "make_preference_dataset",
    "BenchmarkResult",
    "adapter_swap_demo",
    "benchmark",
    "make_synthetic_task",
]
