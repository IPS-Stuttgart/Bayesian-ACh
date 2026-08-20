import numpy as np
import pytest

from bayesian_ach.closed_loop import (
    CausalTriggerPolicy,
    ClosedLoopDataset,
    ClosedLoopFitConfig,
    EligibilityKernelSpec,
    TriggerPolicyConfig,
    causal_window,
    eligibility_kernel,
    fit_closed_loop_models,
)
from bayesian_ach.closed_loop_benchmark import ClosedLoopBenchmarkConfig, run_closed_loop_benchmark


def test_eligibility_kernels_are_causal_and_normalized() -> None:
    lags = np.array([-0.1, 0.0, 0.08, 0.3, 1.0])
    exponential = eligibility_kernel(
        lags,
        EligibilityKernelSpec("exponential", tau_decay=0.35),
    )
    assert exponential[0] == 0.0
    assert exponential[1] == pytest.approx(1.0)
    assert np.all(np.diff(exponential[1:]) < 0.0)

    alpha_spec = EligibilityKernelSpec("alpha", tau_rise=0.08, tau_decay=0.60)
    grid = np.linspace(0.0, 2.0, 10001)
    alpha = eligibility_kernel(grid, alpha_spec)
    assert alpha[0] == pytest.approx(0.0)
    assert float(np.max(alpha)) == pytest.approx(1.0, abs=1e-6)

    boxcar = eligibility_kernel(
        np.array([0.0, 0.45, 0.4501]),
        EligibilityKernelSpec("boxcar", width=0.45),
    )
    np.testing.assert_array_equal(boxcar, [1.0, 1.0, 0.0])
    start, end = causal_window(alpha_spec)
    assert 0.0 < start < end


def test_trigger_policy_applies_uncertainty_refractory_and_false_triggers() -> None:
    policy = CausalTriggerPolicy(
        TriggerPolicyConfig(
            threshold=1.0,
            max_uncertainty=0.3,
            actuation_delay=0.1,
            jitter_std=0.0,
            refractory_period=0.5,
            missed_trigger_probability=0.0,
            false_trigger_probability=1.0,
        ),
        seed=1,
    )
    first = policy.observe(0.0, 1.2, 0.1, command_latency=0.2)
    assert first.accepted and first.reason == "triggered"
    assert first.command_time == pytest.approx(0.2)
    assert first.effective_time == pytest.approx(0.3)

    refractory = policy.observe(0.3, 1.2, 0.1, command_latency=0.0)
    assert not refractory.accepted and refractory.reason == "refractory"

    uncertain = policy.observe(1.0, 1.2, 0.5, command_latency=0.0)
    assert not uncertain.accepted and uncertain.reason == "uncertainty_gate"

    false = policy.observe(1.0, 0.2, 0.1, command_latency=0.0)
    assert false.accepted and false.reason == "false_trigger"


def test_test_outcomes_do_not_change_training_posterior_coefficients() -> None:
    rng = np.random.default_rng(2)
    n = 80
    train = np.arange(n) < 48
    latency = np.linspace(0.0, 1.4, n)
    amplitude = np.ones(n)
    true = np.exp(-(latency + 0.08) / 0.35)
    outcome = true + rng.normal(0.0, 0.1, n)
    common = dict(
        actual_command_latency=latency,
        eligibility_amplitude=amplitude,
        train_mask=train,
        subject_ids=np.repeat(np.arange(4), 20).astype(np.int64),
        session_ids=np.repeat(np.arange(8), 10).astype(np.int64),
        pair_ids=np.arange(n, dtype=np.int64),
        true_event=np.ones(n, dtype=bool),
        false_trigger=np.zeros(n, dtype=bool),
    )
    config = ClosedLoopFitConfig(difference_noise_std=0.1, actuation_delay=0.08)
    first = fit_closed_loop_models(
        ClosedLoopDataset(pair_difference=outcome, **common),
        config,
    )
    modified = outcome.copy()
    modified[~train] += 100.0
    second = fit_closed_loop_models(
        ClosedLoopDataset(pair_difference=modified, **common),
        config,
    )
    first_by_name = {fit.model: fit for fit in first.fits}
    second_by_name = {fit.model: fit for fit in second.fits}
    for name in first_by_name:
        np.testing.assert_equal(
            first_by_name[name].main_effect_mean,
            second_by_name[name].main_effect_mean,
        )
        np.testing.assert_equal(
            first_by_name[name].eligibility_effect_mean,
            second_by_name[name].eligibility_effect_mean,
        )


def test_reduced_benchmark_recovers_all_generators() -> None:
    result = run_closed_loop_benchmark(
        ClosedLoopBenchmarkConfig(
            n_subjects=5,
            sessions_per_subject=4,
            train_sessions_per_subject=2,
            opportunities_per_session=72,
            seed=7,
        )
    )
    assert result.summary["all_generators_recovered"] is True
    assert result.summary["recovery_count"] == 5
    assert result.summary["accepted_false_trigger_count"] > 0
    assert result.summary["trigger_reason_counts"]["refractory"] > 0
    assert result.summary["maximum_active_sham_command_time_difference"] == 0.0

    for session_id in range(5 * 4):
        latencies = [
            float(row["nominal_command_latency"])
            for row in result.opportunities
            if int(row["session_id"]) == session_id
        ]
        counts = [latencies.count(value) for value in sorted(set(latencies))]
        assert len(counts) == 10
        assert max(counts) - min(counts) <= 1


def test_exponential_transport_delay_is_confounded_with_amplitude() -> None:
    tau = 0.35
    delay = 0.08
    command_lags = np.linspace(0.0, 1.2, 13)
    spec = EligibilityKernelSpec("exponential", tau_decay=tau)
    delayed = eligibility_kernel(command_lags + delay, spec)
    rescaled = np.exp(-delay / tau) * eligibility_kernel(command_lags, spec)
    np.testing.assert_allclose(delayed, rescaled, atol=1e-14, rtol=1e-14)
