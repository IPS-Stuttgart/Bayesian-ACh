import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_ach.design_mixture_diagnostic import (
    CandidateThreshold,
    CrossFitScores,
    DiagnosticThresholds,
    MixtureDiagnosticConfig,
    _call,
    _geometry_rows,
    crossfit_scores,
)
from bayesian_ach.design_mixture_diagnostic_cli import (
    _BASELINE_CHECKSUMS_SHA256,
    _BASELINE_MANIFEST_SHA256,
    _write_artifact,
    verify_mixture_diagnostic_artifact,
)


def test_locked_config_uses_fresh_disjoint_streams_and_power_gate() -> None:
    config = MixtureDiagnosticConfig()
    config.validate()
    assert (
        config.threshold_seed,
        config.calibration_audit_seed,
        config.evaluation_seed,
    ) == (196613, 262147, 324949)
    assert config.minimum_pure_power_wilson_lower == 0.70
    assert config.folds == 3
    with pytest.raises(ValueError, match="distinct"):
        MixtureDiagnosticConfig(
            calibration_audit_seed=config.threshold_seed
        ).validate()
    with pytest.raises(ValueError, match="restricted"):
        MixtureDiagnosticConfig(budget=45).validate()


def test_crossfit_pairwise_cone_recovers_noiseless_positive_mixture() -> None:
    rng = np.random.default_rng(42)
    signals = rng.normal(size=(60, 6))
    signals = (signals - signals.mean(axis=0)) / signals.std(axis=0)
    response = 0.5 * (signals[:, 0] + signals[:, 1])
    scores = crossfit_scores(
        signals,
        response,
        folds=3,
        rng=np.random.default_rng(17),
    )
    assert scores.composite_pair == (0, 1)
    assert scores.composite_score > float(np.max(scores.pure_scores))
    assert scores.pure_residual_ratios[scores.winner] > 1.0


def test_call_requires_power_signal_separation_composite_and_gof() -> None:
    thresholds = DiagnosticThresholds(
        pure_over_null=1.0,
        winner_over_runner=1.0,
        candidates=tuple(
            CandidateThreshold(
                composite_over_pure=1.0,
                residual_ratio=2.0,
            )
            for _ in range(6)
        ),
    )
    scores = CrossFitScores(
        pure_scores=np.array([5.0, 3.0, 2.0, 1.0, 0.0, -1.0]),
        null_score=0.0,
        composite_score=5.5,
        composite_pair=(0, 1),
        pure_residual_ratios=np.ones(6),
    )
    assert _call(scores, thresholds, (True,) * 6) == (0, "pure_call")
    assert _call(scores, thresholds, (False,) + (True,) * 5) == (
        None,
        "candidate_underpowered",
    )
    composite = CrossFitScores(
        pure_scores=scores.pure_scores,
        null_score=scores.null_score,
        composite_score=6.1,
        composite_pair=scores.composite_pair,
        pure_residual_ratios=scores.pure_residual_ratios,
    )
    assert _call(composite, thresholds, (True,) * 6) == (
        None,
        "pairwise_composite_better",
    )
    lack_of_fit = CrossFitScores(
        pure_scores=scores.pure_scores,
        null_score=scores.null_score,
        composite_score=scores.composite_score,
        composite_pair=scores.composite_pair,
        pure_residual_ratios=np.array([2.1, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    assert _call(lack_of_fit, thresholds, (True,) * 6) == (
        None,
        "residual_lack_of_fit",
    )


def test_geometry_reports_pair_representability_and_finite_power_index() -> None:
    rng = np.random.default_rng(9)
    full_signals = rng.normal(size=(80, 6))
    full_signals = (
        full_signals - full_signals.mean(axis=0)
    ) / full_signals.std(axis=0)
    indices = np.arange(60, dtype=np.int64)
    rows = _geometry_rows(
        full_signals,
        indices,
        config=MixtureDiagnosticConfig(),
    )
    assert len(rows) == 15
    assert max(float(row["true_pair_affine_residual"]) for row in rows) < 1.0e-20
    assert all(
        np.isfinite(float(row["crossfit_oracle_log_score_gap_index"]))
        for row in rows
    )


def _dummy_result() -> SimpleNamespace:
    row = ({"name": "one", "value": 1},)
    return SimpleNamespace(
        summary={
            "schema_version": 1,
            "technical_gates": {
                "streams_disjoint": True,
                "all_fifteen_pairwise_composites": True,
                "three_fold_cross_fitting": True,
                "candidate_power_gate_applied": True,
                "evaluation_not_used_for_thresholds": True,
            },
        },
        thresholds=row,
        calibration_audit=row,
        pure_evaluation=row,
        null_evaluation=row,
        mixture_evaluation=row,
        out_of_span_evaluation=row,
        geometry=row,
    )


def test_artifact_verifier_binds_config_baseline_and_tampering(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    baseline = {
        "kind": "immutable_original_design_stress_failure",
        "producer_commit": "c71695fda83ae93407599a909097962ee3fa9e0e",
        "checksum_table_sha256": _BASELINE_CHECKSUMS_SHA256,
        "manifest_sha256": _BASELINE_MANIFEST_SHA256,
        "verified_payload_count": 9,
    }
    _write_artifact(
        output,
        result=_dummy_result(),
        config=MixtureDiagnosticConfig(),
        code_sha="a" * 40,
        inputs=(baseline,),
    )
    verified = verify_mixture_diagnostic_artifact(output)
    assert verified["producer_commit"] == "a" * 40
    assert verified["verified_payload_count"] == 9

    summary = output / "summary.json"
    value = json.loads(summary.read_text(encoding="utf-8"))
    value["tampered"] = True
    summary.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="locked byte count mismatch|SHA-256 mismatch",
    ):
        verify_mixture_diagnostic_artifact(output)
