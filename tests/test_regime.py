import numpy as np

from bayesian_ach.regime import (
    RegimeRecoveryConfig,
    ring_transition_kernels,
    run_regime_recovery,
)


def test_regime_benchmark_recovers_both_model_classes() -> None:
    result = run_regime_recovery(
        RegimeRecoveryConfig(
            n_sequences_per_class=20,
            pre_length=48,
            post_length=80,
            seed=17,
        )
    )

    assert result.summary["balanced_accuracy"] >= 0.90
    margins = result.summary["median_context_minus_changepoint"]
    assert margins["known_context_switch"] > 0.0
    assert margins["structural_reset"] < 0.0


def test_regime_benchmark_is_deterministic() -> None:
    config = RegimeRecoveryConfig(
        n_sequences_per_class=3,
        pre_length=12,
        post_length=16,
        seed=31,
    )
    first = run_regime_recovery(config)
    second = run_regime_recovery(config)
    assert first.summary == second.summary
    assert first.sequences == second.sequences
    assert first.trials == second.trials


def test_novel_similarity_preserves_valid_kernels() -> None:
    forward, backward, novel = ring_transition_kernels(5, novel_similarity=0.6)
    np.testing.assert_allclose(forward.sum(axis=1), 1.0)
    np.testing.assert_allclose(backward.sum(axis=1), 1.0)
    np.testing.assert_allclose(novel.sum(axis=1), 1.0)
    assert np.all(novel > 0.0)
