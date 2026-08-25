# Changelog

All notable changes to Bayesian-ACh will be documented here.

## Unreleased

- Corrected maximin planning diagnostics for the candidate-specific residual
  variances used by held-out recovery: the exact asymptotic per-trial quantity
  is the profiled Gaussian log-score gap `0.5 log1p(a^2 R / sigma^2)`, not a
  fixed-variance expected log Bayes factor.
- Renamed exported rate/target fields, retained deprecated Python/CLI aliases,
  and documented/tested affine reparameterization equivalence.
## 0.7.0 — 2026-08-23

- Added a transparent finite-grid optimizer for prospective discrimination of
  six transition-level ACh hypotheses.
- Added a maximin objective equal to the smallest ordered-pair projection
  residual, which is proportional to expected Gaussian held-out log-evidence.
- Added deterministic greedy allocation, exchange refinement, a per-condition
  trial cap, complete allocation traces, and pairwise geometry exports.
- Added equal-budget coupled-novelty and uniform-factorial baselines.
- Added held-out recovery showing that optimized 60-trial schedules improve
  worst-generator recovery and reduce candidate collinearity.
- Added `bayesian-ach-design`, public API, documentation, plotting,
  tests, and CI smoke coverage.
- Exposed the existing replay/smoothing benchmark through the public API and
  command line, and removed temporary replay/applicator artifacts.

## 0.6.0 — 2026-08-22

- Added exact finite-state filtering-to-smoothing comparisons and posterior
  pairwise transition revisions.
- Added forward-filtering backward-sampling replay without model mutation.
- Added held-out recovery of online surprise, smoothing information, and replay
  content, plus a negative control for replay-as-independent-pseudodata.

## 0.5.0 — 2026-08-21

- Added a stateful causal trigger policy with signal thresholding, uncertainty
  gating, balanced randomized latency assignment, jitter, refractory periods,
  missed commands, and randomized background commands.
- Added explicit command-to-effective ACh transport delay and separate event,
  command, and effective timestamps.
- Added normalized exponential, rise-and-decay, and boxcar eligibility-trace
  families with reported effective and command-time causal windows.
- Added yoked active/sham scheduling and active-minus-sham outcomes with strict
  training-session fitting and held-out-session joint predictive scoring.
- Added conservative nested comparison of null, latency-independent, and
  latency-dependent causal hypotheses.
- Added an explicit delay-identifiability result: a constant delay is confounded
  with amplitude for a monotone exponential trace and must be independently
  calibrated or handled by sensitivity analysis.
- Added five-way closed-loop mechanism recovery, false-trigger negative
  controls, 30-seed stress evidence, documentation, plotting support, tests,
  and CI smoke coverage.
- Added `bayesian-ach closed-loop-benchmark` and complete CSV/JSON evidence
  exports.

## 0.4.0 — 2026-08-20

- Added a forward ACh measurement model with subject-specific phasic release,
  latent AR(1) tonic release, and shared difference-of-exponentials indicator
  dynamics.
- Added exact conditional AR(3) whitening for filtered tonic-release residuals.
- Added calibration-only inference over a discrete rise/decay/tonic-persistence
  grid and held-out likelihood marginalization over that uncertainty.
- Added movement, acceleration, pupil, theta, and engagement nuisance
  regressors; partially pooled subject intercepts and signal slopes; and
  baseline-only session-offset estimation.
- Added seven candidate event trains generated from one exact multisensory
  belief trajectory.
- Added strict calibration/train/test leakage checks and seven-way synthetic
  recovery of the generating Bayesian event signal.
- Added `bayesian-ach measurement-benchmark`, CSV/JSON evidence exports,
  scientific documentation, plotting support, tests, and CI smoke coverage.

## 0.3.0 — 2026-08-20

- Added exact joint filtering over latent state, transition context, and binary
  sensor-health configurations.
- Added health-, context-, and state-dependent categorical observation models,
  including context-cue channels and missing modalities.
- Added state, context, and sensor-health information gain; context-switch,
  sensor-fault, and sensor-fault-onset posteriors; and multisensory conflict
  diagnostics.
- Added exhaustive-enumeration verification of the exact partial-observation
  update.
- Added prequential recovery of visual sensor faults, known context switches,
  and specified structural transition changes from identical observations.
- Added fault/structural similarity controls, total-variation identifiability
  measures, and explicit warnings for observationally null configurations.
- Added `bayesian-ach observation-benchmark`, trial/sequence evidence exports,
  plotting support, documentation, tests, and CI smoke coverage.

## 0.2.0 — 2026-08-20

- Added exact HMM filtering over known transition contexts.
- Added joint context-transition posteriors, context information gain, and
  posterior switch probability.
- Kept context inference parameter-free by default and added explicit exact
  supervised context learning.
- Added full unpruned Dirichlet Bayesian online change-point detection with
  complete run-length sufficient statistics.
- Added exhaustive-enumeration verification of BOCPD on short sequences.
- Added prequential recovery of known context switches versus novel structural
  resets, including a novel-regime similarity stress parameter.
- Added `bayesian-ach regime-benchmark`, trial/sequence evidence exports,
  documentation, tests, and CI smoke coverage.

## 0.1.0 — 2026-08-19

- Added exact Dirichlet transition-signal calculations.
- Added matched-confidence and factorial synthetic designs.
- Added held-out univariate model recovery for six ACh hypotheses.
- Added command-line evidence generation and CSV/JSON outputs.
- Added scientific documentation, tests, CI, citation metadata, and MIT license.
