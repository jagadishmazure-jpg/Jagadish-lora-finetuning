# Knowledge

Distilled notes on LoRA (Low-Rank Adaptation) and how this repo derives
it from scratch in numpy, plus architecture diagrams. Written for you
(interview prep) and first-time repo readers.

- [`01-concepts.md`](01-concepts.md) — the LoRA math in plain words,
  merge/unmerge, adapter-only training, and the numpy→production table.
- [`02-architecture.md`](02-architecture.md) — forward pass and training
  diagrams (Mermaid, renders on GitHub).

The full derivation lives in [`docs/lora-math.md`](../docs/lora-math.md).
CPU-only — no GPU needed; synthetic rank-r tasks.
