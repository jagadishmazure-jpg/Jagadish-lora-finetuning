# When to finetune vs RAG vs prompt engineering

A practical decision tree. The three techniques solve different problems;
picking the wrong one is the most common expensive mistake in AI
engineering: teams finetune to memorize facts (use RAG), or bolt RAG
onto a style problem (use a prompt, or finetune).

## The 30-second decision tree

```
Do you need the model to KNOW something new (facts, docs, private data)?
├── Yes → can the knowledge change after deploy, or must answers cite sources?
│   ├── Yes → RAG. Facts live in the index, not the weights.
│   └── No, stable knowledge, and retrieval keeps failing → finetune (SFT).
└── No → is the task about HOW it behaves (tone, format, tool use, judgment)?
    ├── Small behavior nudge, few examples → prompt engineering.
    ├── Consistent behavior across many inputs → finetune (SFT/LoRA).
    └── Aligning to human PREFERENCES (helpful, safe, on-brand) → DPO/RLHF.
```

## The four dimensions

### 1. Behavior-change depth
- **Surface** (wording, format, a few rules): prompt engineering. Cheapest,
  fastest to iterate, zero training.
- **Structural** (reliable JSON, consistent tool-call discipline, domain
  jargon, a house style across thousands of inputs): finetune. Prompts
  degrade as instructions pile up; weights absorb the pattern.
- **Preference** (which of two good answers a human would pick): DPO.
  This is what SFT cannot do — SFT teaches *a* right answer, DPO teaches
  *better vs worse*.

### 2. Data volume
- **0–50 examples**: prompt engineering (few-shot). Finetuning on this
  little data mostly memorizes noise.
- **50–10k examples**: LoRA SFT sweet spot. This repo's `lorax` trains a
  rank-8 adapter on a synthetic task in seconds on CPU; real adapters
  train in minutes-to-hours on one GPU.
- **10k+ preference pairs**: DPO starts to beat SFT-only pipelines.
- **Millions of tokens of knowledge**: don't finetune it in — that's
  what RAG is for.

### 3. Latency
- Finetuning (especially LoRA, merged at deploy) adds **zero** inference
  latency: the knowledge is in the weights.
- RAG adds retrieval latency (vector search + longer context) on every
  call. Prompt engineering with long few-shot blocks does the same via
  context length.
- If your p99 budget is tight and the knowledge is stable, finetuning
  wins. If knowledge changes daily, eat the RAG latency.

### 4. Cost
- Prompt engineering: $0 to try, but long prompts tax every inference
  forever.
- RAG: index build + embedding + retrieval infra; per-query cost grows
  with context.
- Finetuning: upfront training cost (LoRA keeps it small — this repo's
  demo goes 57.76 → 8.8e-07 loss on CPU), then inference at base-model
  price. DPO costs one more training run on preference data.

## Comparison

| | Prompt eng. | RAG | SFT / LoRA | DPO |
|---|---|---|---|---|
| Changes knowledge | No | Yes (at query time) | Yes (in weights) | No |
| Changes behavior/style | Shallow | No | Deep, consistent | Preference-level |
| Needs training data | No (few-shot) | Corpus + index | 50–10k examples | Preference pairs |
| Latency impact | Longer prompts | Retrieval + context | None (merged) | None |
| Staleness risk | None | Low (re-index) | High (re-train) | Medium |
| Cites sources | No | Yes | No | No |

## Worked examples

1. **Support bot must sound like your brand.** 20 example replies →
   prompt engineering. 2,000 graded replies and the tone still drifts →
   LoRA SFT.
2. **Answer questions over 10k internal docs.** RAG, full stop.
   Finetuning the docs into weights gives no citations and goes stale.
3. **Model must emit valid JSON for a tool API.** A prompt begs; a
   finetune guarantees. (This repo: see the DPO module — preference
   tuning is the same idea applied to *which* valid output is best.)
4. **Two good answers, humans prefer the shorter one.** SFT can't
   express "prefer A over B" — that's DPO's loss function exactly:
   `−log σ(β·(r(x,y_w) − r(x,y_l)))`.

## Anti-patterns

- **Finetuning to memorize facts.** Weights are a lossy, uncitable,
  hard-to-update database. RAG exists for this.
- **RAG for style.** Retrieving tone examples at query time is
  expensive and inconsistent; bake style into weights or prompts.
- **Prompt-stuffing a behavior problem.** Past ~10 instructions, prompts
  get brittle and slow. That's the finetune threshold.
- **DPO without SFT first.** DPO sharpens preferences; it doesn't teach
  the task. SFT first, then DPO — the standard post-training stack.

## Where this repo fits

`lorax` implements the finetuning column from scratch in numpy:
`lora.py` (the adapter), `train.py` (SFT backprop), `dpo.py`
(preference optimization), plus the math in `docs/lora-math.md`.
Use it to build intuition for *when* each technique earns its cost —
then reach for the production frameworks with the decision already made.
