# Roadmap

## 0.1 — Exact finite-state benchmark

- Exact Dirichlet transition updates.
- Six trial-wise candidate signals.
- Matched-confidence dissociation.
- Factorial synthetic model recovery.
- CLI, tests, CI, and scientific documentation.

## 0.2 — Latent contexts and change points

- Exact small-state HMM/switching model.
- Separate context-posterior update from transition-parameter learning.
- Full Bayesian online change-point baseline with run-length hypotheses.
- Recovery tests for context switch versus structural reset.

## 0.3 — Partial observations and sensor fusion

- State-space observation model for visual, vestibular, proprioceptive, and cue
  channels.
- Modality-reliability inference.
- Robust sensor-corruption alternatives.
- Dissociation of observation innovation from transition-model change.

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
