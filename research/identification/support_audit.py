"""Finite-support identification audit under a common unknown increasing link.

The default audit compares exact weak orders of the supplied floating-point
values. It is a structural diagnostic, not a statistical test or an analytical
proof of equality. Near differences are reported separately, never silently
collapsed into ties. A measured-family overlap check requires NumPy/SciPy.

Run ``python3 support_audit.py --demo`` or pass a JSON file containing a mapping
from candidate names to equal-length arrays (optionally under ``candidates``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Sequence
from pathlib import Path


def _validate(values: Sequence[float], label: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain at least one finite number")
    return result


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def weak_order_audit(
    candidate_a: Sequence[float],
    candidate_b: Sequence[float],
    *,
    near_tie_tolerance: float = 1e-10,
) -> dict:
    """Return equivalence or a minimal two-condition rank/tie witness.

    Equivalence is only on this supplied support, for strictly increasing links
    unconstrained in amplitude. All pairwise signs, including zeros, must agree.
    The tolerance flags fragile nonzero differences; it does not define ties.
    """
    a, b = _validate(candidate_a, "candidate_a"), _validate(candidate_b, "candidate_b")
    if len(a) != len(b):
        raise ValueError("candidate arrays must have equal length")
    if not math.isfinite(near_tie_tolerance) or near_tie_tolerance < 0:
        raise ValueError("near_tie_tolerance must be finite and nonnegative")
    counts = {"order_reversal": 0, "tie_split": 0}
    witnesses: dict[str, dict] = {}
    near_pairs = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        da, db = a[j] - a[i], b[j] - b[i]
        sa, sb = _sign(da), _sign(db)
        fragile = (0 < abs(da) <= near_tie_tolerance) or (0 < abs(db) <= near_tie_tolerance)
        near_pairs += fragile
        if sa == sb:
            continue
        kind = "tie_split" if sa == 0 or sb == 0 else "order_reversal"
        counts[kind] += 1
        witness = {
            "condition_indices": [i, j],
            "values_a": [a[i], a[j]],
            "values_b": [b[i], b[j]],
            "difference_a": da,
            "difference_b": db,
            "nonzero_difference_near_tolerance": bool(fragile),
        }
        # Prefer a witness that is not a near floating-point tie.
        if kind not in witnesses or (
            witnesses[kind]["nonzero_difference_near_tolerance"] and not fragile
        ):
            witnesses[kind] = witness
    return {
        "n_conditions": len(a),
        "same_weak_order_on_supplied_values": not any(counts.values()),
        "candidate_a_levels": len(set(a)),
        "candidate_b_levels": len(set(b)),
        "witness_counts": counts,
        "witnesses": witnesses,
        "near_nonzero_pair_count": near_pairs,
        "near_tie_tolerance": near_tie_tolerance,
    }


def support_audit(candidates: dict[str, Sequence[float]]) -> dict:
    """Partition candidates by identical rank-and-tie structure."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    checked = {name: _validate(values, name) for name, values in candidates.items()}
    lengths = {len(values) for values in checked.values()}
    if len(lengths) != 1:
        raise ValueError("all candidate arrays must have equal length")
    classes: dict[tuple[int, ...], list[str]] = {}
    for name, values in checked.items():
        ranks = {value: rank for rank, value in enumerate(sorted(set(values)))}
        signature = tuple(ranks[value] for value in values)
        classes.setdefault(signature, []).append(name)
    pairs = {
        f"{a}__{b}": weak_order_audit(checked[a], checked[b])
        for a, b in itertools.combinations(checked, 2)
    }
    return {
        "assumptions": [
            "one candidate-specific strictly increasing link shared across conditions",
            "unrestricted link offset and dynamic range",
            "predictions evaluated at fixed parameters on this finite support",
            "premeasurement means; sensor and nuisance need a separate audit",
            "numerical equality is not an analytical or experimental equality certificate",
        ],
        "n_conditions": next(iter(lengths)),
        "equivalence_classes": list(classes.values()),
        "pairs": pairs,
    }


def digamma_positive(x: float) -> float:
    """Digamma for x > 0 using recurrence and a Bernoulli expansion."""
    if not math.isfinite(x) or x <= 0:
        raise ValueError("digamma argument must be finite and positive")
    result = 0.0
    while x < 12.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    z = inv * inv
    return (
        result
        + math.log(x)
        - 0.5 * inv
        - z
        * (
            1.0 / 12.0
            - z
            * (
                1.0 / 120.0
                - z * (1.0 / 252.0 - z * (1.0 / 240.0 - z * (1.0 / 132.0 - z * 691.0 / 32760.0)))
            )
        )
    )


def table1_values(
    q: float,
    concentration: float,
    *,
    hazard: float = 0.1,
    reset_predictive: float = 0.5,
    probabilities: Sequence[float] | None = None,
    observed_index: int = 0,
) -> dict[str, float]:
    """Evaluate a one-row Dirichlet update and a fixed-hazard reset mixture.

    Defaults to the binary slice p=(q, 1-q), observed outcome zero. The reset
    row is only the stated one-step mixture, not a history-dependent reset
    filter. KL is KL[Dir(alpha + e_j) || Dir(alpha)].
    """
    if not (0 < q < 1 and math.isfinite(concentration) and concentration > 0):
        raise ValueError("require 0 < q < 1 and finite concentration > 0")
    if not (0 < hazard < 1 and 0 < reset_predictive <= 1):
        raise ValueError("require 0 < hazard < 1 and 0 < reset_predictive <= 1")
    if probabilities is None:
        if observed_index != 0:
            raise ValueError("observed_index must be zero on the default binary slice")
        innovation = math.sqrt(2.0) * (1.0 - q)
    else:
        p = _validate(probabilities, "probabilities")
        if len(p) < 2 or min(p) <= 0 or not math.isclose(sum(p), 1.0, abs_tol=1e-12):
            raise ValueError("probabilities must be a positive simplex vector")
        if not 0 <= observed_index < len(p) or p[observed_index] != q:
            raise ValueError("q must equal the observed-outcome probability")
        innovation = math.sqrt(
            sum(((1.0 if i == observed_index else 0.0) - v) ** 2 for i, v in enumerate(p))
        )
    gain = 1.0 / (concentration + 1.0)
    surprise = -math.log(q)
    information_gain = (
        surprise + digamma_positive(concentration * q + 1.0) - digamma_positive(concentration + 1.0)
    )
    return {
        "innovation": innovation,
        "surprise": surprise,
        "gain": gain,
        "update": innovation * gain,
        "information_gain": information_gain,
        "reset_fixed_hazard": hazard
        * reset_predictive
        / ((1.0 - hazard) * q + hazard * reset_predictive),
    }


def _candidate_columns(rows: list[dict[str, float]]) -> dict[str, list[float]]:
    return {key: [row[key] for row in rows] for key in rows[0]}


def demonstration() -> dict:
    """Construct support, not sampled or fitted biological evidence."""
    supports = {
        "fixed_confidence_binary": [(q, 20.0) for q in (0.1, 0.3, 0.7, 0.9)],
        "matched_probability_confidence": [(0.3, a) for a in (2.0, 20.0, 200.0)],
        "minimal_three_condition_binary": [(0.8, 5.0), (0.1, 50.0), (0.8, 50.0)],
        "factorial_binary": [(q, a) for q in (0.1, 0.3, 0.7, 0.9) for a in (2.0, 20.0, 200.0)],
    }
    result = {}
    for name, support in supports.items():
        values = _candidate_columns([table1_values(q, a) for q, a in support])
        result[name] = {
            "support": [{"q": q, "concentration": a} for q, a in support],
            "candidates": values,
            "audit": support_audit(values),
        }
    multi = [
        table1_values(0.5, 20.0, probabilities=p) for p in ((0.5, 0.49, 0.01), (0.5, 0.25, 0.25))
    ]
    values = _candidate_columns(multi)
    result["three_outcome_equal_probability"] = {
        "support": [
            {"probabilities": p, "observed_index": 0, "concentration": 20.0}
            for p in ((0.5, 0.49, 0.01), (0.5, 0.25, 0.25))
        ],
        "candidates": values,
        "audit": support_audit(values),
    }
    extended_support = [
        {"probabilities": [0.8, 0.1, 0.1], "concentration": 5.0, "hazard": 0.1},
        {"probabilities": [0.1, 0.45, 0.45], "concentration": 50.0, "hazard": 0.1},
        {"probabilities": [0.8, 0.19, 0.01], "concentration": 50.0, "hazard": 0.8},
    ]
    rows = []
    for condition in extended_support:
        row = table1_values(
            condition["probabilities"][0],
            condition["concentration"],
            probabilities=condition["probabilities"],
            hazard=condition["hazard"],
            reset_predictive=1.0 / 3.0,
        )
        row["reset_variable_hazard"] = row.pop("reset_fixed_hazard")
        rows.append(row)
    values = _candidate_columns(rows)
    result["minimal_three_condition_six_quantities"] = {
        "support": extended_support,
        "observed_index": 0,
        "reset_predictive": 1.0 / 3.0,
        "candidates": values,
        "audit": support_audit(values),
    }
    return result


def measured_overlap(
    candidate_a: Sequence[float],
    candidate_b: Sequence[float],
    sensor_operator: Sequence[Sequence[float]],
    nuisance_design: Sequence[Sequence[float]] | None = None,
    *,
    tolerance: float = 1e-7,
) -> dict:
    """Check overlap of increasing-link mean families by linear feasibility.

    The fixed known sensor H maps n event amplitudes to m measured samples.
    The nuisance matrix B has m rows, unrestricted linear coefficients. Solve
    H(T_a u - T_b v) + B gamma = 0 directly, with free gamma and adjacent level
    increments at least one. Homogeneity makes this equivalent to strictly
    positive increments, provided links have no amplitude bound. We normalize
    H by one global scale and each B column separately: this preserves the
    nuisance span and avoids amplifying projection-cancellation roundoff.
    Solver results are numerical diagnostics, not exact rational certificates.
    """
    import numpy as np
    from scipy.optimize import linprog

    a, b = _validate(candidate_a, "candidate_a"), _validate(candidate_b, "candidate_b")
    if len(a) != len(b):
        raise ValueError("candidate arrays must have equal length")
    h = np.asarray(sensor_operator, dtype=float)
    if h.ndim != 2 or h.shape[0] == 0 or h.shape[1] != len(a) or not np.isfinite(h).all():
        raise ValueError("sensor_operator must be finite, nonempty, and have n_conditions columns")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    nuisance = np.empty((h.shape[0], 0))
    if nuisance_design is not None:
        nuisance = np.asarray(nuisance_design, dtype=float)
        if nuisance.ndim != 2 or nuisance.shape[0] != h.shape[0] or not np.isfinite(nuisance).all():
            raise ValueError("nuisance_design must be finite and have n_measurements rows")
    sensor_scale = float(np.max(np.abs(h))) or 1.0
    normalized_sensor = h / sensor_scale
    nuisance_scales = np.max(np.abs(nuisance), axis=0)
    normalized_nuisance = np.divide(
        nuisance,
        nuisance_scales,
        out=np.zeros_like(nuisance),
        where=nuisance_scales > 0,
    )
    nuisance_rank = int(np.linalg.matrix_rank(normalized_nuisance)) if nuisance.shape[1] else 0

    def levels(values):
        unique = sorted(set(values))
        index = {value: i for i, value in enumerate(unique)}
        return np.eye(len(unique))[[index[value] for value in values]], len(unique)

    ta, ka = levels(a)
    tb, kb = levels(b)
    # Keep the nuisance coefficients in the LP. In particular, do not project
    # H out of col(B) and normalize the residual: an exact-zero projection can
    # contain floating-point noise, which row normalization would turn into a
    # spurious identifying constraint.
    equality = np.concatenate(
        (normalized_sensor @ ta, -normalized_sensor @ tb, normalized_nuisance), axis=1
    )
    n_variables = ka + kb + nuisance.shape[1]
    inequalities = []
    for offset, count in ((0, ka), (ka, kb)):
        for i in range(count - 1):
            row = np.zeros(n_variables)
            row[offset + i], row[offset + i + 1] = 1.0, -1.0
            inequalities.append(row)
    inequality = np.asarray(inequalities) if inequalities else None
    result = linprog(
        np.zeros(n_variables),
        A_ub=inequality,
        b_ub=-np.ones(len(inequalities)) if inequalities else None,
        A_eq=equality,
        b_eq=np.zeros(equality.shape[0]),
        bounds=[(None, None)] * n_variables,
        method="highs",
    )
    output = {
        "solver_status": int(result.status),
        "solver_message": result.message,
        "nuisance_rank": nuisance_rank,
        "formulation": "direct_sensor_plus_free_nuisance",
        "sensor_global_scale": sensor_scale,
        "nuisance_column_scales": nuisance_scales.tolist(),
        "interpretation": "fixed known linear sensor and unrestricted increasing links only",
    }
    if result.success:
        residual = float(np.max(np.abs(equality @ result.x), initial=0.0))
        minimum_gap = float(np.min(-inequality @ result.x)) if inequalities else None
        checked = residual <= tolerance and (minimum_gap is None or minimum_gap >= 1.0 - tolerance)
        output.update(
            {
                "overlap": True if checked else None,
                "max_normalized_equality_residual": residual,
                "minimum_level_gap": minimum_gap,
                "latent_mean_a": (ta @ result.x[:ka]).tolist(),
                "latent_mean_b": (tb @ result.x[ka : ka + kb]).tolist(),
                "nuisance_difference_normalized": result.x[ka + kb :].tolist(),
            }
        )
    else:
        output["overlap"] = False if result.status == 2 else None
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        payload = demonstration()
    elif args.input:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        payload = support_audit(source.get("candidates", source))
    else:
        parser.error("pass a JSON input or --demo")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
