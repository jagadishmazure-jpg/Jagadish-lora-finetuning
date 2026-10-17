# Implementation guide: from offline build to Azure

> Status: **planned**. Nothing here has been deployed. The offline build in this
> repository is what exists today; this guide is how it would be taken to a
> real Azure environment.

## Azure service mapping

| Offline component | Azure service (planned) | Notes |
|---|---|---|
| Training | Azure ML job on a GPU compute cluster with Hugging Face PEFT | Low-priority VMs to save cost |
| Managed alternative | Azure OpenAI or Azure AI Foundry fine-tuning |  |
| Adapters | Azure ML model registry, versioned per task |  |
| Serving | Managed online endpoint with merged weights or multi-adapter serving |  |

## Steps

1. Pick a small open model and a labelled task dataset.
2. Fine-tune with PEFT LoRA on an Azure ML GPU job; log loss and eval.
3. Compare against the base model and a prompt-only baseline on held-out data.
4. Register the adapter; serve merged or swap adapters per tenant.

## Guardrails to keep when going live

- The base weight is never written by training; the demo checks it byte for byte.
- Config validation rejects a rank larger than the layer allows.
- Merge equivalence is tested, so a merged model behaves like the adapter.

## What to measure before calling it production-ready

- Held-out task accuracy against the base model and a prompt baseline.
- Trainable parameter share and GPU hours per run.
- Regression on general benchmarks after merging.
