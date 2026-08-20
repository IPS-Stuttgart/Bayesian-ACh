# Changelog

All notable changes to Bayesian-ACh will be documented here.

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
