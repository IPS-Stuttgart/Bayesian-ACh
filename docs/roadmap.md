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

## 0.3 — Partial observations and sensor fusion — complete

- Exact filtering over latent state, transition context, and the complete binary
  sensor-health configuration.
- Health-, context-, and state-dependent categorical observation models.
- Visual, proprioceptive, and context-cue channels in the identifying benchmark.
- Missing-modality support and modality-only nominal-health posteriors.
- State, context, and sensor-health information gain.
- Per-sensor posterior fault and fault-onset probabilities.
- Posterior context-switch probability under partial observation.
- Jensen–Shannon diagnostics for state and context disagreement across sensors.
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

## 0.4 — ACh measurement model — complete

- Seven candidate computational event trains generated from the same exact
  partial-observation belief trajectory.
- Subject-specific phasic release coefficients with Gaussian-ridge partial
  pooling.
- Latent stationary AR(1) tonic release.
- Shared causal difference-of-exponentials indicator dynamics for phasic and
  tonic release.
- Exact conditional AR(3) whitening of the tonic sensor residual.
- Calibration-only discrete posterior over rise, decay, and tonic-persistence
  hypotheses.
- Movement, acceleration, pupil, theta, and engagement nuisance regressors.
- Baseline-only session-offset estimation, including held-out sessions.
- Strict separation of calibration, training-task, and held-out-task samples.
- Held-out candidate scoring marginalized over calibration-grid uncertainty.
- Explicit limitations for plug-in regression coefficients, finite grids, and
  omitted independent white sensor noise.

Acceptance evidence for the default seed-7 benchmark:

- 6 subjects, 5 sessions per subject, and 2 held-out sessions per subject;
- 7/7 computational generators recovered;
- median and minimum held-out log-evidence margins: 771.532 and 421.096;
- calibration MAP exactly recovers $`\tau_r=0.4`$, $`\tau_d=1.6`$, and
  $`\rho=0.97`$;
- median nuisance-coefficient MAE: 0.00749;
- median subject-signal correlation: 0.99769;
- maximum absolute post-sensor candidate correlation: 0.88880.

These are controlled synthetic recovery results, not biological evidence about
ACh coding.

## 0.5 — Closed-loop and delayed feedback — complete

- Stateful online candidate-signal and uncertainty trigger API.
- Balanced randomized command latencies independent of event generation.
- Explicit event, command, and effective timestamps with calibrated transport
  delay and realized latency jitter.
- Uncertainty gating, refractory suppression, missed commands, and randomized
  below-threshold background commands.
- Normalized exponential, rise-and-decay, and finite boxcar eligibility traces.
- Yoked active/sham policies receiving identical schedules and command times.
- Active-minus-sham causal outcomes with training-session coefficient fitting
  and held-out-session joint predictive comparison.
- Conservative nested claim threshold separating no effect, an untimed main
  effect, and latency-dependent eligibility gating.
- Explicit proof and sensitivity interface for delay/amplitude confounding.

Acceptance evidence for the default seed-7 benchmark:

- 8 subjects, 5 sessions per subject, and 96 opportunities per session;
- 2,236 accepted yoked active/sham pairs, including 29 false-positive or
  randomized background events;
- 5/5 causal generators recovered: null, latency independent, exponential,
  rise-and-decay, and boxcar;
- minimum and median conservative decision margins: 3.474 and 376.937 log units
  beyond the preregistered claim boundary;
- active and sham command timestamps identical to numerical precision;
- 30-seed stress check: 150/150 generator decisions correct, with minimum
  decision margin 1.963 log units beyond the claim boundary.

These are controlled synthetic recovery results, not biological evidence for a
particular eligibility family or timescale.

## 0.6 — Replay and smoothing — next

- Filtering-versus-smoothing belief corrections.
- Smoothed transition-count revisions during replay.
- Explicit distinction between internally sampled trajectories and new evidence.
