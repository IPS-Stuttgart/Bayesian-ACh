"""Deterministic tests; no empirical outcomes are used."""

import importlib.util
import math
import unittest

from support_audit import (
    demonstration,
    digamma_positive,
    measured_overlap,
    support_audit,
    table1_values,
    weak_order_audit,
)


class WeakOrderTests(unittest.TestCase):
    def test_increasing_reparameterization_and_preserved_ties(self):
        self.assertTrue(
            weak_order_audit([1, 2, 2, 4], [0, 1, 1, 100])["same_weak_order_on_supplied_values"]
        )

    def test_reversal(self):
        audit = weak_order_audit([1, 2], [2, 1])
        self.assertFalse(audit["same_weak_order_on_supplied_values"])
        self.assertEqual(audit["witness_counts"]["order_reversal"], 1)

    def test_tie_split(self):
        audit = weak_order_audit([1, 1], [1, 2])
        self.assertEqual(audit["witness_counts"]["tie_split"], 1)

    def test_near_differences_are_not_silently_tied(self):
        audit = weak_order_audit([1, 1 + 1e-12], [1, 1])
        self.assertFalse(audit["same_weak_order_on_supplied_values"])
        self.assertEqual(audit["near_nonzero_pair_count"], 1)

    def test_partition_is_transitive(self):
        audit = support_audit({"a": [1, 2, 2], "b": [2, 5, 5], "c": [3, 9, 9], "d": [1, 2, 3]})
        self.assertEqual(audit["equivalence_classes"], [["a", "b", "c"], ["d"]])

    def test_rejects_invalid_inputs(self):
        for a, b in (([], []), ([1], [1, 2]), ([math.nan], [1]), ([1], [math.inf])):
            with self.assertRaises(ValueError):
                weak_order_audit(a, b)


class CandidateTests(unittest.TestCase):
    def test_digamma_known_values_and_recurrence(self):
        euler_gamma = 0.5772156649015329
        self.assertAlmostEqual(digamma_positive(1), -euler_gamma, places=13)
        self.assertAlmostEqual(digamma_positive(0.5), -euler_gamma - 2 * math.log(2), places=13)
        for x in (0.01, 0.2, 1.0, 10.0, 30.0):
            self.assertAlmostEqual(digamma_positive(x + 1) - digamma_positive(x), 1 / x, places=11)

    def test_kl_known_uniform_beta_case(self):
        # KL[Beta(2,1) || Beta(1,1)] = log(2) - 1/2.
        self.assertAlmostEqual(
            table1_values(0.5, 2)["information_gain"], math.log(2) - 0.5, places=13
        )

    def test_demo_binary_classes(self):
        result = demonstration()
        self.assertEqual(
            result["fixed_confidence_binary"]["audit"]["equivalence_classes"],
            [
                ["innovation", "surprise", "update", "information_gain", "reset_fixed_hazard"],
                ["gain"],
            ],
        )
        self.assertEqual(
            result["factorial_binary"]["audit"]["equivalence_classes"],
            [
                ["innovation", "surprise", "reset_fixed_hazard"],
                ["gain"],
                ["update"],
                ["information_gain"],
            ],
        )

    def test_multicategory_boundary(self):
        values = demonstration()["three_outcome_equal_probability"]["candidates"]
        self.assertEqual(values["surprise"][0], values["surprise"][1])
        self.assertNotEqual(values["innovation"][0], values["innovation"][1])
        self.assertEqual(
            weak_order_audit(values["innovation"], values["surprise"])["witness_counts"][
                "tie_split"
            ],
            1,
        )

    def test_minimal_three_condition_support(self):
        result = demonstration()["minimal_three_condition_binary"]
        self.assertEqual(
            result["audit"]["equivalence_classes"],
            [
                ["innovation", "surprise", "reset_fixed_hazard"],
                ["gain"],
                ["update"],
                ["information_gain"],
            ],
        )
        values = result["candidates"]
        self.assertEqual(values["innovation"][0], values["innovation"][2])
        self.assertEqual(values["gain"][1], values["gain"][2])
        self.assertTrue(values["update"][2] < values["update"][1] < values["update"][0])
        self.assertTrue(
            values["information_gain"][2]
            < values["information_gain"][0]
            < values["information_gain"][1]
        )

    def test_kl_decreases_with_concentration_on_matched_q_grid(self):
        for q in (0.001, 0.01, 0.1, 0.3, 0.7, 0.99):
            values = [
                table1_values(q, concentration)["information_gain"]
                for concentration in (0.1, 0.5, 1, 2, 10, 20, 100, 1000)
            ]
            self.assertTrue(all(a > b for a, b in zip(values, values[1:], strict=False)))

    def test_three_conditions_separate_six_declared_quantities(self):
        result = demonstration()["minimal_three_condition_six_quantities"]
        self.assertEqual(len(result["audit"]["equivalence_classes"]), 6)
        self.assertTrue(all(len(group) == 1 for group in result["audit"]["equivalence_classes"]))
        values = result["candidates"]
        self.assertTrue(values["innovation"][0] < values["innovation"][2] < values["innovation"][1])
        self.assertEqual(values["surprise"][0], values["surprise"][2])
        self.assertEqual(values["gain"][1], values["gain"][2])
        self.assertTrue(values["update"][2] < values["update"][1] < values["update"][0])
        self.assertTrue(
            values["information_gain"][2]
            < values["information_gain"][0]
            < values["information_gain"][1]
        )
        self.assertTrue(
            values["reset_variable_hazard"][0]
            < values["reset_variable_hazard"][1]
            < values["reset_variable_hazard"][2]
        )

    def test_fixed_confidence_kl_monotonicity(self):
        for concentration in (0.1, 0.5, 1, 2, 10, 20, 100, 1000):
            values = [
                table1_values(q, concentration)["information_gain"]
                for q in (0.001, 0.01, 0.1, 0.3, 0.7, 0.99)
            ]
            self.assertTrue(all(a > b for a, b in zip(values, values[1:], strict=False)))


@unittest.skipUnless(
    importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"),
    "NumPy/SciPy unavailable",
)
class MeasuredOverlapTests(unittest.TestCase):
    def test_identity_preserves_reversal(self):
        result = measured_overlap([0, 1], [1, 0], [[1, 0], [0, 1]])
        self.assertFalse(result["overlap"])

    def test_identity_preserves_tie_split(self):
        self.assertFalse(measured_overlap([0, 0], [0, 1], [[1, 0], [0, 1]])["overlap"])

    def test_same_weak_order_overlaps(self):
        self.assertTrue(
            measured_overlap([0, 1, 1], [5, 10, 10], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])["overlap"]
        )

    def test_aggregation_destroys_witness(self):
        self.assertTrue(measured_overlap([0, 1], [1, 0], [[1, 1]])["overlap"])

    def test_intercept_nuisance_does_not_destroy_reversal(self):
        self.assertFalse(measured_overlap([0, 1], [1, 0], [[1, 0], [0, 1]], [[1], [1]])["overlap"])

    def test_difference_nuisance_destroys_reversal(self):
        self.assertTrue(measured_overlap([0, 1], [1, 0], [[1, 0], [0, 1]], [[1], [-1]])["overlap"])

    def test_causal_full_rank_convolution_preserves_reversal(self):
        self.assertFalse(measured_overlap([0, 1], [1, 0], [[1, 0], [0.5, 1]])["overlap"])

    def test_sensor_units_do_not_change_structural_result(self):
        for scale in (1e-12, 1.0, 1e12):
            self.assertFalse(measured_overlap([0, 1], [1, 0], [[scale, 0], [0, scale]])["overlap"])

    def test_full_rank_nuisance_destroys_all_distinctions(self):
        self.assertTrue(
            measured_overlap([0, 1], [1, 0], [[1, 0], [0, 1]], [[1, 0], [0, 1]])["overlap"]
        )

    def test_sensor_in_intercept_span_has_no_identifying_residual(self):
        result = measured_overlap([0, 1], [1, 0], [[1, -1], [1, -1], [1, -1]], [[1], [1], [1]])
        self.assertTrue(result["overlap"])
        self.assertLessEqual(result["max_normalized_equality_residual"], 1e-12)

    def test_sensor_in_nonconstant_nuisance_span_has_no_identifying_residual(self):
        result = measured_overlap([0, 1], [1, 0], [[2, -2], [3, -3], [4, -4]], [[2], [3], [4]])
        self.assertTrue(result["overlap"])
        self.assertLessEqual(result["max_normalized_equality_residual"], 1e-12)

    def test_nuisance_units_preserve_exact_span_overlap(self):
        for sensor_scale in (1e-12, 1.0, 1e12):
            for nuisance_scale in (1e-12, 1.0, 1e12):
                result = measured_overlap(
                    [0, 1],
                    [1, 0],
                    [
                        [2 * sensor_scale, -2 * sensor_scale],
                        [3 * sensor_scale, -3 * sensor_scale],
                        [4 * sensor_scale, -4 * sensor_scale],
                    ],
                    [[2 * nuisance_scale], [3 * nuisance_scale], [4 * nuisance_scale]],
                )
                self.assertTrue(result["overlap"])

    def test_zero_nuisance_column_does_not_erase_a_witness(self):
        self.assertFalse(measured_overlap([0, 1], [1, 0], [[1, 0], [0, 1]], [[0], [0]])["overlap"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
