# Interview points

Short, honest talking points for Jagadish-lora-finetuning. Every number quoted is from the
real run shown in the README.

- B starts at zero so the adapter begins as no change; A gets Kaiming init so gradients flow.
- alpha/r keeps the update size stable when you change rank.
- Merging folds the adapter in, so inference cost is unchanged; I verified it to 1e-14.
- The parameter saving depends on layer size: 50% in my toy layer, 0.39% at 4096x4096, rank 8.

## Likely follow-up questions

- Why low rank? (Fine-tuning updates tend to be low rank in practice.)
- Which layers would you adapt? (Usually attention projections first; measure.)
- LoRA or prompt optimisation? (Prompts first; fine-tune when they plateau and data exists.)
