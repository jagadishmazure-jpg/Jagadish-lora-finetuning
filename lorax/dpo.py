"""Direct Preference Optimization from scratch in numpy.

DPO replaces the RLHF reward-model + PPO pipeline with a single
classification loss on preference pairs::

    L = -log σ(β · (r(x, y_w) − r(x, y_l)))

where the *implicit reward* is

    r(x, y) = log π_θ(y|x) − log π_ref(y|x)

i.e. how much more (log-)likely the adapted policy rates a completion
than the frozen reference does. The reference is the same LoRALinear
with B = 0 (never trained) sharing W, so r = 0 at step 0 and the loss
only rewards *changes* induced by the adapter — the policy cannot
improve the loss by exploiting the base model.

The policy here is a toy sequence head: one LoRALinear maps a prompt
embedding x to logits over a small vocabulary, and
log π(y|x) = log-softmax(logits)[y]. Gradients flow through the adapter
path only; W is frozen by construction, exactly as in train.py.

Backprop (batch mean, N = batch size, m_i = β·(r_w − r_l) per example):
    dL/dm_i = σ(m_i) − 1
    dy_w    = dL/dm_i · β · (onehot(y_w) − softmax(l_w))
    dy_l    = −dL/dm_i · β · (onehot(y_l) − softmax(l_l))
    dy      = dy_w + dy_l          (shared forward pass per prompt;
                                   the softmax terms cancel:
                                   dy = coef · (onehot(y_w) − onehot(y_l)))
    dB  = s · dyᵀh₂ / N,   dh₂ = s · dy·B / N,
    dh  = dh₂ ⊙ mask,      dA  = dhᵀx / N
"""

from __future__ import annotations

import numpy as np

from .lora import LoRALinear


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _check_beta(beta: float) -> float:
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta!r}")
    return float(beta)


def implicit_reward(
    layer: LoRALinear, ref: LoRALinear, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Per-example implicit reward r(x, y) = log π_θ(y|x) − log π_ref(y|x).

    Zero for every example when the adapter is at its B = 0 init.
    """
    logp = _log_softmax(layer.forward(x))
    logp_ref = _log_softmax(ref.forward(x))
    rows = np.arange(x.shape[0])
    return logp[rows, y] - logp_ref[rows, y]


def preference_margin(
    layer: LoRALinear,
    ref: LoRALinear,
    x: np.ndarray,
    y_w: np.ndarray,
    y_l: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """Per-example margin β·(r(x, y_w) − r(x, y_l)).

    Positive means the policy currently prefers the chosen completion.
    """
    _check_beta(beta)
    r_w = implicit_reward(layer, ref, x, y_w)
    r_l = implicit_reward(layer, ref, x, y_l)
    return beta * (r_w - r_l)


def dpo_loss(
    layer: LoRALinear,
    ref: LoRALinear,
    x: np.ndarray,
    y_w: np.ndarray,
    y_l: np.ndarray,
    beta: float = 1.0,
) -> float:
    """Mean DPO loss −log σ(margin) = softplus(−margin) over the batch."""
    margin = preference_margin(layer, ref, x, y_w, y_l, beta)
    return float(np.mean(np.logaddexp(0.0, -margin)))


def dpo_step(
    layer: LoRALinear,
    ref: LoRALinear,
    x: np.ndarray,
    y_w: np.ndarray,
    y_l: np.ndarray,
    beta: float = 1.0,
    lr: float = 0.1,
) -> float:
    """One SGD step of DPO on the adapter. Returns the loss *before* the update.

    Only A and B move; W and the reference layer are never touched.
    """
    if not layer.training:
        raise ValueError("dpo_step requires the layer in training mode")
    _check_beta(beta)

    s = layer.config.scaling
    n = x.shape[0]
    rows = np.arange(n)

    # ---- forward through the adapter path (recompute internals) ----
    base = x @ layer.W.T
    h = x @ layer.A.T
    mask = layer._dropout_mask(h.shape[0])
    h2 = h * mask
    logits = base + s * (h2 @ layer.B.T)
    ref_logits = ref.forward(x)  # frozen reference: no gradient

    # ---- loss ----
    logp = _log_softmax(logits)
    logp_ref = _log_softmax(ref_logits)
    margin = beta * (
        (logp[rows, y_w] - logp_ref[rows, y_w])
        - (logp[rows, y_l] - logp_ref[rows, y_l])
    )
    loss = float(np.mean(np.logaddexp(0.0, -margin)))

    # ---- backward: dL/dlogits, then through the adapter path only ----
    coef = (_sigmoid(margin) - 1.0) * beta / n  # (n,), ≤ 0
    one_w = np.zeros_like(logits)
    one_l = np.zeros_like(logits)
    one_w[rows, y_w] = 1.0
    one_l[rows, y_l] = 1.0
    # (onehot(y_w) − softmax) − (onehot(y_l) − softmax): softmax cancels.
    dy = coef[:, None] * (one_w - one_l)

    dB = s * (dy.T @ h2)      # (out, r)
    dh2 = s * (dy @ layer.B)  # (batch, r)
    dh = dh2 * mask           # (batch, r)
    dA = dh.T @ x             # (r, in)

    # ---- SGD update: A and B only ----
    layer.A -= lr * dA
    layer.B -= lr * dB

    return loss


def make_preference_dataset(
    n: int = 16,
    in_features: int = 8,
    vocab_size: int = 6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiny seeded preference dataset: (X, y_w, y_l).

    Prompts are random embeddings; preferred/rejected tokens are distinct
    random vocabulary items. The pairs are arbitrary by construction — the
    point is to verify the optimizer moves probability mass from y_l to y_w,
    not to learn anything meaningful.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, in_features))
    y_w = rng.integers(0, vocab_size, size=n)
    y_l = rng.integers(0, vocab_size, size=n)
    clash = y_l == y_w
    y_l[clash] = (y_w[clash] + 1) % vocab_size
    return x, y_w, y_l
