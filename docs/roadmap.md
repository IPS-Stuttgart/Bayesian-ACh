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

## 0.3 — Partial observations and sensor fusion — complete

- Exact filtering over latent state, transition context, and the complete binary
  sensor-health configuration.
- Health-, context-, and state-dependent categorical observation models.
- Visual, proprioceptive, and context-cue channels in the identifying benchmark.
- Missing-modality support and modality-only nominal-health posteriors.
- State, context, and sensor-health information gain.
- Per-sensor posterior fault and fault-onset probabilities.
- Posterior context-switch probability under partial observation.
- Jensen--Shannon diagnostics for state and context disagreement across sensors.
- Exhaustive-enumeration verification of the one-step joint posterior.
- Three-way prequential recovery of visual sensor fault, known context switch,
  and a specified structural transition change.
- Configurable fault and structural similarity with explicit total-variation
  identifiability warnings.

Acceptance evidence for the default seed-7 benchmark:

- 36 visual-fault, 36 known-context-switch, and 36 structural-change sequences;
- balanced model-class recovery accuracy: 1.000;
- 108/108 sequence decisions correct;
- median winning log-evidence margins: 57.786, 77.689, and 58.761 respectively;
- minimum sequence evidence margin: 35.188.

The structural arm is exact for a preregistered candidate kernel. Arbitrary
unseen transition learning under hidden-state uncertainty remains an explicit
open-set problem rather than an implied conjugate update.

## 0.4 — ACh measurement model — next

- Latent phasic and tonic release processes.
- Indicator impulse-response convolution and deconvolution uncertainty.
- Movement, acceleration, task engagement, arousal, pupil, and theta nuisance
  components.
- Candidate computational event trains generated from the same latent belief
  trajectory.
- Hierarchical subject/session effects.
- Strict train/test separation for sensor-kernel estimation and ACh hypothesis
  comparison.
- Simulation-based recovery of release timescale, nuisance effects, and the
  generating Bayesian signal.

## 0.5 — Closed-loop and delayed feedback

- Online estimator and trigger API.
- Eligibility-trace and septo-hippocampal delay model.
- Latency-sweep design and preregistered falsification criteria.

## 0.6 — Replay and smoothing

- Filtering-versus-smoothing belief corrections.
- Smoothed transition-count revisions during replay.
- Explicit distinction between internally sampled trajectories and new evidence.
