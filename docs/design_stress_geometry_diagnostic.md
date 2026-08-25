# Post-freeze stress geometry diagnostic

This explanatory diagnostic is downstream of the immutable locked-N60 stress package. It
does not alter its thresholds, seeds, allocations, simulations, or endpoints.

`scripts/analyze_design_stress_geometry.py` stratifies the frozen result by allocation and
replays the 15 maximin mixture evaluations to count each decision gate independently. For
each 50/50 pair it also computes, on the locked maximin support, the weighted affine
residual against the best constituent and best pure candidate, the oracle two-component
residual, candidate correlation, covariance condition number, and the population profiled
Gaussian gap for the frozen 21-sample held-out size.

`scripts/verify_design_stress_geometry.py` checks the immutable source-package digest,
producer/script provenance, every payload checksum, pair count, gate/false-call identities,
and the zero-residual two-component oracle. It then reruns all stratification, geometry, and
15 x 200 frozen evaluations and requires byte-identical JSON, CSV, and manifest outputs.

The diagnostic explains whether poor mixture rejection reflects population aliasing on the
locked support or finite-sample decision behavior. It is not a new endpoint, a robustness
claim, a threshold-tuning analysis, or permission to select a replacement gate after seeing
the frozen result.
