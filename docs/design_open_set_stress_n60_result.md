# Frozen locked-N60 open-set stress result

The immutable package in `results/design-open-set-stress-n60/` was produced from clean
commit `c71695fda83ae93407599a909097962ee3fa9e0e`. It byte-verifies and reconstructs the
chronologically locked equal-N60 allocation (SHA-256
`a823be49faf6c6cbebf60b11d4b5ca895cf7734d6e9c577ee98f97a5907b69b2`) from source
commit `1b2028929ac6ebc1cce0882f0c22af9918044342` and explicit allocation seed 7.

The frozen settings use 100 calibration, 100 calibration-audit, and 200 evaluation
replicates; threshold, audit, and evaluation seeds are 104729, 130363, and 155921. The
held-out fraction is 0.35. No threshold or evaluation was changed after inspecting results.

| Allocation | Weakest pure rate (Wilson lower; raw closed-set) | Worst 50/50 mixture false-pure rate (upper) | Null false-pure rate (upper) | Nonlinear probe false-pure rate (upper) |
|---|---:|---:|---:|---:|
| Coupled novelty | 0.030 (0.0138; 0.425) | 0.460 (0.5292) | 0.010 (0.0357) | 0.035 (0.0705) |
| Uniform factorial | 0.425 (0.3585; 0.720) | 0.545 (0.6125) | 0.015 (0.0432) | 0.005 (0.0278) |
| Locked heuristic maximin | 0.750 (0.6857; 0.820) | 0.800 (0.8495) | 0.020 (0.0503) | 0.515 (0.5833) |

For the locked heuristic maximin allocation, the pure-over-null, winner-over-runner, and
flexible-over-pure thresholds are 1.4684378147, 0.7589681024, and 2.0743104877. Its
worst mixture is gain plus update, while its weakest pure generator is surprise.

This is failure-boundary evidence, not open-set robustness. Null control is bounded, and
the locked maximin schedule retains reasonable matched-pure performance, but the current
adequacy rule frequently labels mixtures and the single declared nonlinear probe as pure.
The artifact does not justify claims for arbitrary mixtures, nonlinear alternatives,
serial dependence, sensor dynamics, subject hierarchy, or physical protocol feasibility.

The package checksum-table SHA-256 is
`44a5188c43bda52e6fc9dc7007cf2de44a9671e9c5477ac88c3173c06cfdbd80`; the manifest
SHA-256 is `d840a2ec34f5a386109c7f985033b53144fdcc1b1a0e0592b3274c0e38902b64`.
