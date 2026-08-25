# Post-freeze design abstention stress

This module is a **versioned sensitivity analysis**, not part of the immutable
five-seed matched-generator evidence. It asks whether a deliberately conservative
classifier can avoid confident pure-candidate calls when the generator is null or
is an unmodelled 50/50 combination of two candidates. It does not turn synthetic
recovery into biological evidence.

## Frozen schedule

The default run evaluates the coupled-novelty, uniform-factorial, and maximin
allocation targets at 0.5, 1, and 2 times their population observation-equivalent
index, rounded upward. Those indices are recomputed from each design's 60-draw
geometry using

```math
G(R)=\tfrac12\log\!\left(1+a^2R/\sigma^2\right)
```

with standardized generating signals, `a=1`, `sigma=1`, and target held-out
log-score gap 5. Thus the default nominal budgets are 23/45/90 for maximin,
47/93/186 for uniform factorial, and 557/1113/2226 for coupled novelty. These
are effectively independent Gaussian observations. They are not physical trial,
time-bin, fluorescence-sample, session, or animal counts.

A checksum-bound certified integer allocation may replace an exactly matching
maximin budget. The loader verifies the certificate summary, allocation, and
checksum table, requires an integer certificate and direct geometry agreement,
and rejects unused or duplicate overrides. Other allocations retain their
declared deterministic constructors. A count vector over independently
instantiated grid cells remains an allocation target, not an executable ordered
history: no reset, washout, carry-over, or history-realization protocol is
provided here.

## Train-only scoring and abstention

Every replicate is randomly divided into training and held-out observations.
Each pure candidate is fitted with an intercept, slope, and training-residual
variance. The flexible adequacy model contains an intercept and all six
candidates. Its ridge penalty is selected only inside the training set by
three-fold cross-validation; its residual variance is also fitted on training
data. All comparisons are held-out profiled Gaussian log-score differences.

Three inequalities are required for a pure call:

1. the best pure candidate must beat the intercept-only model by more than the
   upper null-calibration quantile. Because the statistic already maximizes over
   all six pure candidates, this threshold is familywise for that candidate set;
2. the best pure candidate must beat the runner-up by more than a separately
   null-calibrated ambiguity threshold;
3. the all-six flexible model may not beat the best pure model by more than the
   worst-candidate pure-generator adequacy threshold.

The third rule is one-sided. The flexible model nests the pure model, so the two
population scores tie under a correctly specified pure generator; requiring the
pure model to beat the flexible model would be invalid. The adequacy threshold is
the largest conformal upper threshold across the six matched pure generators,
which protects the worst calibrated pure candidate at the declared finite
calibration resolution.

Threshold calibration, calibration audit, and final evaluation use three
disjoint deterministic seeds. The artifact reports the independent calibration
audit rather than treating threshold-training performance as validation.

## Scenarios and interpretation

Matched pure generators are evaluated with correct-call, wrong-call, abstention,
and raw closed-set winner rates. The null reports any non-null pure call as a
false call. All 15 unordered 50/50 candidate pairs are evaluated after equal
coefficients and unit-standard-deviation scaling on the full feasible grid. Any
pure call for a mixture—including a call naming one of its constituents—is
counted as false. Pointwise Wilson intervals accompany all reported rates; they
are not simultaneous confidence bounds over the many scenario cells.

The mixture family is open-set relative to the six pure labels but remains inside
their linear span. One additional, deliberately narrow out-of-span probe takes
`tanh(standardized surprise)`, removes its full-grid OLS projection on an
intercept and all six standardized candidates, and scales the residual to unit
standard deviation. Orthogonality is checked numerically on the full grid; a pure
call is false. This is a diagnostic of one fixed saturation-shaped residual, not
coverage of a biologically defined misspecification class.

The bounded artifact therefore does **not** certify robustness to arbitrary
out-of-span misspecification, other nonlinear/saturating combinations, serial
dependence, subject hierarchy, indicator dynamics, nuisance mismatch, or invalid
sequential histories. Matched-field simulation also cannot diagnose
misspecification of the candidate signals themselves.

## Reproducible artifact

From a clean checkout of the exact stress commit:

```bash
bayesian-ach-design-stress \
  --repo-root . \
  --code-sha <40-character-commit> \
  --output /absolute/path/design-stress \
  --certified-allocation /absolute/path/n45/certified_allocation.csv
```

The command refuses a dirty or mismatched checkout. It writes the configuration,
thresholds, independent calibration audit, pure/null/mixture and fixed nonlinear-probe evaluations,
allocations, a provenance manifest, and `SHA256SUMS.csv`. The manifest binds the
producer commit, canonical configuration digest, every payload file, and every
supplied certificate package.
