# LoRA fine-tuning: architecture

## Forward pass (adapter path)

```mermaid
flowchart LR
    X["x"] --> W["frozen W"]
    X --> A["A (r×k)\nKaiming init"]
    A --> B["B (d×r)\nzeros init"]
    B --> S["× s (scale)"]
    W --> ADD["+"]
    S --> ADD
    ADD --> Y["y = Wx + s·B(Ax)"]
```

At init, B = 0 → the adapter is a no-op. Backprop trains A and B only.

## Training loop (adapter-only)

```mermaid
flowchart TD
    F["forward:\ny = Wx + s·B(Ax)"] --> L["loss vs target"]
    L --> BP["backprop through\nadapter path only"]
    BP --> U["update A, B\n(W frozen)"]
    U -->|"converge"| DONE["adapted model"]
    U -->|"not yet"| F
```

## Deployment: merge / swap

```mermaid
flowchart LR
    M["merge:\nW' = W + s·B·A\n(zero inference overhead)"] 
    SW["swap:\none frozen base,\nmany task adapters"]
```
