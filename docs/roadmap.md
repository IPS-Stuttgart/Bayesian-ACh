# Roadmap

## 0.1 — Exact finite-state benchmark — complete

- Exact Dirichlet transition updates.
- Six trial-wise candidate signals.
- Matched-confidence dissociation.
- Factorial synthetic model recovery.
- CLI, tests, CI, and scientific documentation.

## 0.2 — Latent contexts and change points — complete

- Exact HMM filtering over a finite bank of fixed transition contexts.
- Joint previous/current-context posterior and posterior switch probability.
- Explicit separation of context inference from supervised Dirichlet learning.
- Full categorical Bayesian online change-point detection with unpruned
  run-length hypotheses and complete transition sufficient statistics.
- Exhaustive-enumeration verification of the BOCPD posterior.
- Prequential model-evidence recovery of known context switches versus novel
  structural resets.
- Configurable novel-regime similarity for identifiability stress tests.

Acceptance evidence for the default seed-7 benchmark:

- 48 known-context-switch sequences and 48 structural-reset sequences;
- balanced model-class recovery accuracy: 1.000;
- median context-minus-change-point log-evidence margin: +12.387 for known
  switches and -206.679 for structural resets.

These are controlled synthetic recovery results, not biological evidence about
ACh coding.

## 0.3 — Partial observations and sensor fusion — next

- State-space observation model for visual, vestibular, proprioceptive, and cue
  channels.
- Modality-reliability inference.
- Robust sensor-corruption alternatives.
- Dissociation of observation innovation from state mislocalization, context
  change, and transition-model change.
- Simulation-based recovery across sensory conflict and reliability levels.

## 0.4 — ACh measurement model

- Latent phasic/tonic release process.
- Indicator impulse-response convolution.
- Movement, task engagement, arousal, and theta nuisance components.
- Hierarchical subject/session effects with held-out prediction.

## 0.5 — Closed-loop and delayed feedback

- Online estimator and trigger API.
- Eligibility-trace and septo-hippocampal delay model.
- Latency-sweep design and preregistered falsification criteria.

## 0.6 — Replay and smoothing

- Filtering-versus-smoothing belief corrections.
- Smoothed transition-count revisions during replay.
- Explicit distinction between internally sampled trajectories and new evidence.
