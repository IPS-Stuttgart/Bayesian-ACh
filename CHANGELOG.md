# Changelog

All notable changes to Bayesian-ACh will be documented here.

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
