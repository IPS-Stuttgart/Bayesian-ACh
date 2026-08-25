import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bayesian_ach.design_stress as stress
import bayesian_ach.design_stress_cli as stress_cli
from bayesian_ach.design_stress import DesignStressConfig, run_design_stress
from bayesian_ach.design_stress_cli import (
    _load_certified_allocation,
    _load_locked_design_allocation,
    _write_artifact,
)


def test_gate_requires_signal_separation_and_pure_adequacy() -> None:
    thresholds = stress._Thresholds(
        pure_over_null=1.0,
        winner_over_runner=1.0,
        flexible_over_pure=1.0,
    )
    decisive = stress._Scores(
        winner=2,
        runner=1,
        best_pure_score=5.0,
        runner_score=3.0,
        null_score=0.0,
        flexible_score=5.5,
        ridge_lambda=0.1,
    )
    assert stress._call(decisive, thresholds) == (2, "pure_call")

    no_signal = replace(decisive, best_pure_score=1.0)
    ambiguous = replace(decisive, runner_score=4.0)
    misspecified = replace(decisive, flexible_score=6.01)
    assert stress._call(no_signal, thresholds) == (None, "null_not_rejected")
    assert stress._call(ambiguous, thresholds) == (None, "pure_ambiguity")
    assert stress._call(misspecified, thresholds) == (None, "flexible_model_better")


def test_conformal_threshold_and_wilson_interval_are_conservative() -> None:
    assert stress._upper_conformal_quantile(list(range(20)), alpha=0.05) == 19.0
    assert stress._upper_conformal_quantile(list(range(10)), alpha=0.05) == np.inf
    lower, upper = stress._wilson_interval(0, 100, 0.95)
    assert lower == pytest.approx(0.0)
    assert 0.03 < upper < 0.04


def test_stress_config_rejects_nonfinite_and_duplicate_settings() -> None:
    with pytest.raises(ValueError, match="finite, positive, and unique"):
        DesignStressConfig(budget_factors=(1.0, 1.0)).validate()
    with pytest.raises(ValueError, match="target_log_score_gap"):
        DesignStressConfig(target_log_score_gap=np.inf).validate()
    with pytest.raises(ValueError, match="ridge_lambdas"):
        DesignStressConfig(ridge_lambdas=(0.0, np.nan)).validate()


def test_threshold_calibration_is_finite_with_minimum_replicates() -> None:
    rng = np.random.default_rng(17)
    signals = rng.normal(size=(36, 6))
    signals = (signals - signals.mean(axis=0)) / signals.std(axis=0)
    config = DesignStressConfig(
        fixed_budgets=(),
        budget_factors=(1.0,),
        calibration_replicates=20,
        calibration_audit_replicates=20,
        evaluation_replicates=20,
        inner_folds=2,
    )
    thresholds = stress._calibrate_thresholds(
        signals,
        config=config,
        design_index=0,
        budget=36,
    )
    assert np.isfinite(
        [
            thresholds.pure_over_null,
            thresholds.winner_over_runner,
            thresholds.flexible_over_pure,
        ]
    ).all()
    assert thresholds.flexible_over_pure >= 0.0


def test_nonlinear_probe_is_outside_full_grid_linear_span() -> None:
    _, _, signals = stress.generate_transition_design_grid()
    probe, scale, maximum_inner_product = stress._out_of_span_probe(signals)
    assert probe.shape == (signals.shape[0],)
    assert scale > 0.0
    assert np.std(probe) == pytest.approx(1.0)
    assert maximum_inner_product < 1.0e-12


def _mock_small_run(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(23)
    signals = rng.normal(size=(20, 6))
    signals = (signals - signals.mean(axis=0)) / signals.std(axis=0)
    rows = tuple({"point_id": index} for index in range(20))
    monkeypatch.setattr(
        stress,
        "generate_transition_design_grid",
        lambda: (rows, signals.copy(), signals.copy()),
    )
    monkeypatch.setattr(
        stress,
        "_base_targets",
        lambda *_: {name: 8 for name in stress.STRESS_DESIGNS},
    )

    def counts(*args: object) -> np.ndarray:
        result = np.zeros(20, dtype=np.int64)
        result[: int(args[3])] = 1
        return result

    monkeypatch.setattr(stress, "_design_counts", counts)
    monkeypatch.setattr(
        stress,
        "_calibrate_thresholds",
        lambda *args, **kwargs: stress._Thresholds(1.0, 1.0, 1.0),
    )
    monkeypatch.setattr(
        stress,
        "_calibration_audit_rows",
        lambda design, _index, factor, budget, *_args: [
            {
                "design": design,
                "budget_factor": factor,
                "budget": budget,
                "scenario": "null",
                "generator": "null",
                "replicates": 20,
                "correct_pure_calls": 0,
                "wrong_pure_calls": 0,
                "abstentions": 20,
                "rate": 0.0,
                "wilson_lower": 0.0,
                "wilson_upper": 0.16,
            }
        ],
    )

    def evaluations(
        design: str,
        _index: int,
        factor: float | None,
        budget: int,
        *_args: object,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        pure = [
            {
                "design": design,
                "budget_factor": factor,
                "budget": budget,
                "generator": name,
                "replicates": 20,
                "correct_pure_calls": 10,
                "wrong_pure_calls": 0,
                "abstentions": 10,
                "correct_call_rate": 0.5,
                "wilson_lower": 0.30,
                "wilson_upper": 0.70,
                "raw_closed_set_winner_rate": 0.7,
                "abstention_reasons": {"pure_call": 10, "pure_ambiguity": 10},
            }
            for name in stress.DESIGN_CANDIDATE_NAMES
        ]
        null = [
            {
                "design": design,
                "budget_factor": factor,
                "budget": budget,
                "replicates": 20,
                "false_pure_calls": 1,
                "abstentions": 19,
                "false_pure_call_rate": 0.05,
                "wilson_lower": 0.01,
                "wilson_upper": 0.24,
                "abstention_reasons": {"null_not_rejected": 19, "pure_call": 1},
            }
        ]
        mixture = [
            {
                "design": design,
                "budget_factor": factor,
                "budget": budget,
                "first_candidate": f"first-{index}",
                "second_candidate": f"second-{index}",
                "mixture_definition": "test",
                "replicates": 20,
                "false_pure_calls": 2,
                "constituent_pure_calls": 2,
                "abstentions": 18,
                "false_pure_call_rate": 0.1,
                "wilson_lower": 0.03,
                "wilson_upper": 0.30,
                "abstention_reasons": {"flexible_model_better": 18, "pure_call": 2},
            }
            for index in range(15)
        ]
        return pure, null, mixture

    monkeypatch.setattr(stress, "_evaluation_rows", evaluations)
    monkeypatch.setattr(
        stress,
        "_out_of_span_rows",
        lambda design, _index, factor, budget, *_args: [
            {
                "design": design,
                "budget_factor": factor,
                "budget": budget,
                "probe": "test",
                "probe_definition": "test",
                "full_grid_prestandardization_residual_sd": 0.1,
                "full_grid_maximum_absolute_mean_inner_product": 0.0,
                "replicates": 20,
                "false_pure_calls": 1,
                "abstentions": 19,
                "false_pure_call_rate": 0.05,
                "wilson_lower": 0.01,
                "wilson_upper": 0.24,
                "pure_call_counts": {},
                "abstention_reasons": {"null_not_rejected": 19, "pure_call": 1},
            }
        ],
    )


def test_stress_orchestration_covers_every_design_and_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_small_run(monkeypatch)
    result = run_design_stress(
        DesignStressConfig(
            fixed_budgets=(),
            budget_factors=(1.0,),
            calibration_replicates=20,
            calibration_audit_replicates=20,
            evaluation_replicates=20,
        )
    )
    assert len(result.thresholds) == 3
    assert len(result.pure_recovery) == 18
    assert len(result.null_evaluation) == 3
    assert len(result.mixture_evaluation) == 45
    assert result.summary["technical_gates"] == {
        "calibration_and_evaluation_seeds_disjoint": True,
        "all_fifteen_mixtures_per_design_budget": True,
        "one_out_of_span_probe_per_design_budget": True,
        "all_thresholds_finite": True,
    }


def test_optimizer_cap_does_not_apply_to_declared_comparators() -> None:
    counts = np.zeros(20, dtype=np.int64)
    counts[:5] = 12
    accepted = stress._validated_override(
        counts,
        design="coupled_novelty",
        point_count=20,
        budget=60,
        maximum_count=9,
    )
    np.testing.assert_array_equal(accepted, counts)
    with pytest.raises(ValueError, match="maximin allocation override"):
        stress._validated_override(
            counts,
            design="maximin_optimized",
            point_count=20,
            budget=60,
            maximum_count=9,
        )


def test_unused_certified_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_small_run(monkeypatch)
    counts = np.zeros(20, dtype=np.int64)
    counts[:9] = 1
    with pytest.raises(ValueError, match="did not match"):
        run_design_stress(
            DesignStressConfig(
                fixed_budgets=(),
            budget_factors=(1.0,),
                calibration_replicates=20,
                calibration_audit_replicates=20,
                evaluation_replicates=20,
            ),
            allocation_overrides={("maximin_optimized", 9): counts},
        )


def test_locked_primary_allocation_is_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_counts = np.array([1, 0, 1, 0], dtype=np.int64)
    grid_rows = tuple({"point_id": index} for index in range(4))
    standardized = np.arange(24, dtype=float).reshape(4, 6)
    monkeypatch.setattr(
        stress_cli,
        "generate_transition_design_grid",
        lambda: (grid_rows, standardized.copy(), standardized.copy()),
    )
    monkeypatch.setattr(
        stress_cli,
        "coupled_novelty_design",
        lambda *_args, **_kwargs: expected_counts.copy(),
    )
    monkeypatch.setattr(
        stress_cli,
        "uniform_factorial_design",
        lambda *_args, **_kwargs: expected_counts.copy(),
    )
    monkeypatch.setattr(
        stress_cli,
        "optimize_maximin_design",
        lambda *_args, **_kwargs: SimpleNamespace(counts=expected_counts.copy()),
    )
    path = tmp_path / "locked.csv"
    rows = [
        {"design": design, "point_id": point, "count": 1}
        for design in stress.STRESS_DESIGNS
        for point in (0, 2)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("design", "point_id", "count"),
        )
        writer.writeheader()
        writer.writerows(rows)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    overrides, provenance = _load_locked_design_allocation(
        path,
        expected_sha256=expected,
        source_code_sha="b" * 40,
        allocation_seed=7,
    )
    assert set(overrides) == {(design, 2) for design in stress.STRESS_DESIGNS}
    assert provenance["allocation_sha256"] == expected
    assert provenance["allocation_seed"] == 7
    assert provenance["allocation_seed_source"] == "explicit_cli_metadata"
    assert provenance["allocation_file_seed_field_present"] is False
    assert provenance["construction_contract"] == {
        "allocation_seed": 7,
        "maximin_max_point_fraction": 0.15,
        "maximin_maximum_count_by_budget": {"2": 1},
        "comparator_cap_semantics": (
            "the maximin cap does not apply to the deterministic "
            "coupled-novelty or uniform-factorial constructors"
        ),
        "all_three_allocations_reconstructed": True,
    }

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_locked_design_allocation(
            path,
            expected_sha256="0" * 64,
            source_code_sha="b" * 40,
            allocation_seed=7,
        )
    with pytest.raises(ValueError, match="seed does not match"):
        _load_locked_design_allocation(
            path,
            expected_sha256=expected,
            source_code_sha="b" * 40,
            allocation_seed=8,
        )

    altered_rows = [
        {"design": "coupled_novelty", "point_id": 0, "count": 2},
        *[
            {"design": design, "point_id": point, "count": 1}
            for design in ("uniform_factorial", "maximin_optimized")
            for point in (0, 2)
        ],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("design", "point_id", "count"),
        )
        writer.writeheader()
        writer.writerows(altered_rows)
    altered_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="coupled_novelty.*frozen constructor"):
        _load_locked_design_allocation(
            path,
            expected_sha256=altered_sha,
            source_code_sha="b" * 40,
            allocation_seed=7,
        )


def _write_certificate_package(directory: Path) -> Path:
    directory.mkdir()
    allocation = directory / "certified_allocation.csv"
    allocation.write_text("point_id,allocation\n0,2\n2,1\n", encoding="utf-8")
    summary = directory / "certificate_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "certified_finite_grid_maximin_allocation",
                "code_sha": "1" * 40,
                "grid_point_count": 4,
                "certified": True,
                "direct_geometry_matches_lower_bound": True,
                "lower_bound": 0.2,
                "upper_bound": 0.2,
                "absolute_gap": 0.0,
                "config": {"budget": 3, "integer": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (directory / "SHA256SUMS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file", "bytes", "sha256"))
        writer.writeheader()
        for path in (allocation, summary):
            writer.writerow(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return allocation


def test_certified_override_loader_checks_package_and_tampering(
    tmp_path: Path,
) -> None:
    allocation = _write_certificate_package(tmp_path / "certificate")
    key, counts, provenance = _load_certified_allocation(allocation)
    assert key == ("maximin_optimized", 3)
    np.testing.assert_array_equal(counts, np.array([2, 0, 1, 0]))
    assert provenance["certificate_budget"] == 3

    allocation.write_text("point_id,allocation\n0,1\n2,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _load_certified_allocation(allocation)


def test_artifact_writer_hashes_every_payload(tmp_path: Path) -> None:
    one_row = ({"name": "row", "nested": {"b": 2, "a": 1}},)
    result = SimpleNamespace(
        summary={"schema_version": 1, "technical_gates": {"ok": True}},
        thresholds=one_row,
        calibration=one_row,
        pure_recovery=one_row,
        null_evaluation=one_row,
        mixture_evaluation=one_row,
        out_of_span_evaluation=one_row,
        allocations=one_row,
    )
    output = tmp_path / "artifact"
    config = DesignStressConfig()
    _write_artifact(
        output,
        result=result,
        config=config,
        code_sha="a" * 40,
        input_provenance=(),
    )
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert manifest["producer_commit"] == "a" * 40
    assert manifest["producer_git_dirty"] is False
    assert len(manifest["files"]) == 8
    checksums = list(csv.DictReader((output / "SHA256SUMS.csv").open()))
    assert {row["path"] for row in checksums} == {
        "summary.json",
        "thresholds.csv",
        "calibration_audit.csv",
        "pure_recovery.csv",
        "null_evaluation.csv",
        "mixture_evaluation.csv",
        "out_of_span_evaluation.csv",
        "allocations.csv",
        "artifact_manifest.json",
    }
    for row in checksums:
        assert hashlib.sha256((output / row["path"]).read_bytes()).hexdigest() == row["sha256"]
