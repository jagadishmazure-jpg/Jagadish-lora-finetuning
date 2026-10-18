"""Tests for lorax: LoRA from scratch in numpy. Deterministic, CPU only."""

import numpy as np
import pytest

from lorax import (
    LoRAConfig,
    LoRALinear,
    adapter_swap_demo,
    benchmark,
    dpo_loss,
    dpo_step,
    implicit_reward,
    make_preference_dataset,
    make_synthetic_task,
    merge_adapter,
    preference_margin,
    train_step,
    unmerge,
)


def make_layer(in_f=8, out_f=6, rank=2, seed=0, **kw):
    cfg = LoRAConfig(rank=rank, **kw)
    return LoRALinear(in_f, out_f, cfg, rng=np.random.default_rng(seed))


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_scaling_is_alpha_over_rank():
    assert LoRAConfig(rank=4, alpha=8.0).scaling == pytest.approx(2.0)


def test_rank_zero_rejected():
    with pytest.raises(ValueError):
        LoRAConfig(rank=0)


def test_negative_alpha_rejected():
    with pytest.raises(ValueError):
        LoRAConfig(rank=2, alpha=-1.0)


def test_bad_dropout_rejected():
    with pytest.raises(ValueError):
        LoRAConfig(rank=2, dropout=1.0)
    with pytest.raises(ValueError):
        LoRAConfig(rank=2, dropout=-0.1)


def test_rank_must_be_low_rank_for_shape():
    cfg = LoRAConfig(rank=8)
    with pytest.raises(ValueError):
        cfg.validate_for(8, 8)  # rank == min dim: not low-rank
    cfg.validate_for(16, 10)  # rank < min dim: fine


def test_layer_rejects_non_low_rank():
    with pytest.raises(ValueError):
        make_layer(in_f=4, out_f=4, rank=4)


# ----------------------------------------------------------------------
# forward correctness
# ----------------------------------------------------------------------
def test_forward_matches_manual_computation():
    layer = make_layer(seed=1).eval()
    rng = np.random.default_rng(7)
    x = rng.standard_normal((5, 8))
    s = layer.config.scaling
    expected = x @ layer.W.T + s * ((x @ layer.A.T) @ layer.B.T)
    np.testing.assert_allclose(layer.forward(x), expected, rtol=1e-10)


def test_forward_batch_size_one():
    layer = make_layer(seed=2).eval()
    x = np.random.default_rng(3).standard_normal((1, 8))
    assert layer.forward(x).shape == (1, 6)


# ----------------------------------------------------------------------
# initialization
# ----------------------------------------------------------------------
def test_b_starts_at_zeros():
    layer = make_layer(seed=3)
    np.testing.assert_array_equal(layer.B, np.zeros_like(layer.B))


def test_adapter_contributes_nothing_at_step_zero():
    layer = make_layer(seed=4).eval()
    x = np.random.default_rng(5).standard_normal((4, 8))
    np.testing.assert_allclose(layer.forward(x), x @ layer.W.T, rtol=1e-12)


def test_a_kaiming_bounded():
    layer = make_layer(in_f=8, out_f=6, rank=2, seed=5)
    bound = np.sqrt(6.0 / 8)
    assert np.all(np.abs(layer.A) <= bound + 1e-12)


# ----------------------------------------------------------------------
# parameter accounting
# ----------------------------------------------------------------------
def test_trainable_param_count_exact():
    layer = make_layer(in_f=8, out_f=6, rank=2, seed=6)
    assert layer.trainable_params() == 2 * 8 + 6 * 2  # |A| + |B|
    assert layer.full_finetune_params() == 6 * 8


def test_trainable_fraction_below_full():
    layer = make_layer(in_f=64, out_f=64, rank=4, seed=7)
    assert layer.trainable_fraction() < 1.0
    assert layer.trainable_fraction() == pytest.approx((4 * 64 + 64 * 4) / (64 * 64))


def test_adapter_delta_shape_and_formula():
    layer = make_layer(seed=8).eval()
    delta = layer.adapter_delta()
    assert delta.shape == (6, 8)
    np.testing.assert_allclose(
        delta, (layer.B @ layer.A) * layer.config.scaling, rtol=1e-12
    )


# ----------------------------------------------------------------------
# merge / unmerge
# ----------------------------------------------------------------------
def test_merge_equivalence():
    layer = make_layer(seed=9).eval()
    # give the adapter something nonzero to merge
    layer.B[:] = np.random.default_rng(10).standard_normal(layer.B.shape) * 0.1
    merged = merge_adapter(layer)
    x = np.random.default_rng(11).standard_normal((7, 8))
    np.testing.assert_allclose(merged.forward(x), layer.forward(x), atol=1e-6)


def test_merge_weight_formula():
    layer = make_layer(seed=12).eval()
    merged = merge_adapter(layer)
    np.testing.assert_allclose(
        merged.W, layer.W + layer.adapter_delta(), rtol=1e-12
    )


def test_merge_does_not_mutate_source():
    layer = make_layer(seed=13).eval()
    w_before = layer.W.copy()
    merge_adapter(layer)
    np.testing.assert_array_equal(layer.W, w_before)


def test_unmerge_restores_w_exactly():
    layer = make_layer(seed=14).eval()
    layer.B[:] = np.random.default_rng(15).standard_normal(layer.B.shape) * 0.1
    w_orig = layer.W.copy()
    restored = unmerge(merge_adapter(layer))
    # (W + d) - d round-trips to W up to floating-point rounding
    np.testing.assert_allclose(restored.W, w_orig, rtol=1e-12, atol=1e-12)


def test_unmerge_rejects_non_merged():
    from lorax.merge import MergedLayer

    with pytest.raises(ValueError):
        unmerge(MergedLayer(np.zeros((6, 8))))


# ----------------------------------------------------------------------
# training: base frozen, adapter learns
# ----------------------------------------------------------------------
def test_base_w_frozen_after_train_steps():
    layer = make_layer(seed=16)
    w_before = layer.W.copy()
    rng = np.random.default_rng(17)
    x = rng.standard_normal((10, 8))
    y = rng.standard_normal((10, 6))
    for _ in range(5):
        train_step(layer, x, y, lr=0.01)
    np.testing.assert_array_equal(layer.W, w_before)


def test_only_adapter_moves():
    # LoRA cold-start: B is zeros, so dA is zero on step 1 and only B moves.
    # Once B is nonzero, A receives gradient too. W never moves.
    layer = make_layer(seed=18)
    a0, b0, w0 = layer.A.copy(), layer.B.copy(), layer.W.copy()
    rng = np.random.default_rng(19)
    x = rng.standard_normal((10, 8))
    y = rng.standard_normal((10, 6))
    train_step(layer, x, y, lr=0.01)
    assert not np.array_equal(layer.B, b0)
    np.testing.assert_array_equal(layer.A, a0)
    train_step(layer, x, y, lr=0.01)
    assert not np.array_equal(layer.A, a0)
    np.testing.assert_array_equal(layer.W, w0)


def test_loss_decreases_on_synthetic_rank_r_task():
    w_base, _d, (x, y) = make_synthetic_task(
        in_features=8, out_features=6, rank=2, seed=20
    )
    layer = make_layer(in_f=8, out_f=6, rank=2, seed=21, alpha=4.0)
    layer.W = w_base
    layer.train()
    first = train_step(layer, x, y, lr=0.05)
    for _ in range(60):
        last = train_step(layer, x, y, lr=0.05)
    assert last < first
    assert last < 0.05 * first  # substantial convergence on a fittable task


def test_train_step_rejects_eval_mode():
    layer = make_layer(seed=22).eval()
    x = np.zeros((2, 8))
    with pytest.raises(ValueError):
        train_step(layer, x, x @ np.zeros((8, 6)).T, lr=0.01)


# ----------------------------------------------------------------------
# dropout
# ----------------------------------------------------------------------
def test_dropout_identity_in_eval_mode():
    layer = make_layer(rank=2, seed=23, dropout=0.9).eval()
    x = np.random.default_rng(24).standard_normal((16, 8))
    s = layer.config.scaling
    expected = x @ layer.W.T + s * ((x @ layer.A.T) @ layer.B.T)
    np.testing.assert_allclose(layer.forward(x), expected, rtol=1e-10)


def test_dropout_mask_expectation_in_train_mode():
    layer = make_layer(rank=4, seed=25, dropout=0.5)
    masks = np.stack([layer._dropout_mask(2000) for _ in range(20)])
    assert abs(masks.mean() - 1.0) < 0.05  # inverted dropout: E[mask] == 1


def test_dropout_changes_forward_in_train_mode():
    layer = make_layer(rank=4, seed=26, dropout=0.8)
    layer.B[:] = 0.5  # nonzero adapter so dropout has an effect
    x = np.random.default_rng(27).standard_normal((32, 8))
    outs = [layer.forward(x) for _ in range(5)]
    assert any(not np.allclose(outs[0], o) for o in outs[1:])


# ----------------------------------------------------------------------
# adapter swap + benchmark
# ----------------------------------------------------------------------
def test_adapter_swap_two_tasks_one_base():
    demo = adapter_swap_demo(steps=150, seed=0)
    for name in ("task_a", "task_b"):
        assert demo["swapped_losses"][name] == pytest.approx(
            demo["train_losses"][name], rel=1e-6
        )
        assert demo["swapped_losses"][name] < 1e-3


def test_benchmark_end_to_end():
    res = benchmark(steps=120, seed=0)
    assert len(res.loss_curve) == 120
    assert res.adapted_loss < res.base_loss
    assert res.adapted_loss < 0.01
    assert res.trainable_params < res.full_params
    # loss curve trends down overall
    assert np.mean(res.loss_curve[-10:]) < np.mean(res.loss_curve[:10])


def test_benchmark_param_report():
    res = benchmark(in_features=16, out_features=16, rank=4, steps=5, seed=1)
    assert res.trainable_params == 4 * 16 + 16 * 4
    assert res.full_params == 16 * 16


# ----------------------------------------------------------------------
# DPO from scratch
# ----------------------------------------------------------------------
def make_policy_pair(in_f=8, vocab=6, rank=2, seed=0):
    cfg = LoRAConfig(rank=rank, alpha=8.0, dropout=0.0)
    layer = LoRALinear(in_f, vocab, cfg, rng=np.random.default_rng(seed))
    ref = LoRALinear(in_f, vocab, cfg, rng=np.random.default_rng(seed + 1000))
    ref.W = layer.W.copy()  # same frozen base; ref adapter never trained
    layer.train()
    return layer, ref


def test_implicit_reward_zero_at_init():
    layer, ref = make_policy_pair(seed=10)
    x, y_w, y_l = make_preference_dataset(n=8, seed=11)
    np.testing.assert_allclose(
        implicit_reward(layer, ref, x, y_w), np.zeros(8), atol=1e-12
    )


def test_dpo_loss_is_log2_at_init():
    # margin = 0 everywhere at init -> -log σ(0) = log 2
    layer, ref = make_policy_pair(seed=12)
    x, y_w, y_l = make_preference_dataset(n=8, seed=13)
    assert dpo_loss(layer, ref, x, y_w, y_l) == pytest.approx(np.log(2.0))


def test_dpo_beta_must_be_positive():
    layer, ref = make_policy_pair(seed=14)
    x, y_w, y_l = make_preference_dataset(n=4, seed=15)
    with pytest.raises(ValueError):
        dpo_loss(layer, ref, x, y_w, y_l, beta=0.0)
    with pytest.raises(ValueError):
        dpo_step(layer, ref, x, y_w, y_l, beta=-1.0, lr=0.1)


def test_dpo_step_requires_training_mode():
    layer, ref = make_policy_pair(seed=16)
    layer.eval()
    x, y_w, y_l = make_preference_dataset(n=4, seed=17)
    with pytest.raises(ValueError):
        dpo_step(layer, ref, x, y_w, y_l, lr=0.1)


def test_preference_dataset_has_distinct_tokens():
    x, y_w, y_l = make_preference_dataset(n=32, in_features=8, vocab_size=6, seed=18)
    assert x.shape == (32, 8)
    assert (y_w != y_l).all()
    assert y_w.min() >= 0 and y_w.max() < 6


def test_dpo_single_step_increases_margin():
    layer, ref = make_policy_pair(seed=19)
    x, y_w, y_l = make_preference_dataset(n=1, seed=20)
    before = preference_margin(layer, ref, x, y_w, y_l)[0]
    dpo_step(layer, ref, x, y_w, y_l, beta=1.0, lr=0.5)
    after = preference_margin(layer, ref, x, y_w, y_l)[0]
    assert after > before


def test_dpo_shifts_policy_toward_preferred_completions():
    layer, ref = make_policy_pair(seed=21)
    x, y_w, y_l = make_preference_dataset(n=16, seed=22)
    w_before = layer.W.copy()
    ref_w_before, ref_b_before = ref.W.copy(), ref.B.copy()

    margin_before = preference_margin(layer, ref, x, y_w, y_l)
    loss_before = dpo_loss(layer, ref, x, y_w, y_l)
    assert margin_before.mean() == pytest.approx(0.0, abs=1e-9)

    for _ in range(200):
        dpo_step(layer, ref, x, y_w, y_l, beta=1.0, lr=0.5)

    margin_after = preference_margin(layer, ref, x, y_w, y_l)
    loss_after = dpo_loss(layer, ref, x, y_w, y_l)

    # the policy measurably shifts toward the preferred completions
    assert margin_after.mean() > margin_before.mean() + 2.0
    assert (margin_after > 0).mean() == pytest.approx(1.0)
    assert loss_after < loss_before
    # frozen weights untouched: W bitwise identical, reference never trained
    assert np.array_equal(layer.W, w_before)
    assert np.array_equal(ref.W, ref_w_before)
    assert np.array_equal(ref.B, ref_b_before)


def test_dpo_is_deterministic():
    def run(seed):
        layer, ref = make_policy_pair(seed=seed)
        x, y_w, y_l = make_preference_dataset(n=8, seed=seed + 1)
        for _ in range(10):
            dpo_step(layer, ref, x, y_w, y_l, beta=1.0, lr=0.5)
        return preference_margin(layer, ref, x, y_w, y_l)

    np.testing.assert_array_equal(run(23), run(23))
