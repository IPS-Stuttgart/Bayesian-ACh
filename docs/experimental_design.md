# Decisive experimental design

## Objective

Distinguish six correlated explanations of trial-wise hippocampal ACh:
transition mismatch, surprise, learning gain, posterior update magnitude,
parameter information gain, and evidence for structural change.

A simple novel-versus-familiar contrast is insufficient because all six
quantities can increase together.

## Recommended first experiment: controlled graph navigation

Use a virtual graph-navigation task with explicit state identities and
well-controlled action-conditioned transitions. The experiment should contain
four orthogonal manipulations.

### A. Confidence at matched prediction

Train two transition relations to the same predictive probability but with very
different evidence mass. Present the same transition in paired low- and
high-confidence conditions.

- Raw innovation: matched.
- Predictive surprise: matched.
- Bayesian gain: different.
- Posterior update magnitude: different.
- Information gain: different.

This is the primary identifying contrast.

### B. Rare stochastic outcome versus structural change

Compare an equally surprising outcome in:

1. a stable, explicitly learned stochastic transition kernel; and
2. the first trials after an abrupt topology change.

Raw surprise can be matched while change evidence and rational parameter update
differ.

### C. Sensory corruption versus transition change

Introduce a visual teleportation or cue corruption while keeping vestibular and
physical transition evidence consistent. A future partially observed model
should infer whether the event is sensor failure, state mislocalization, or
transition-model change.

### D. Context cue revealing a known mode switch

Switch between two already learned transition kernels. This should update the
posterior over context without requiring destructive relearning of either
kernel. It dissociates latent-mode inference from parameter learning.

## Measurements

The strongest implementation combines:

- hippocampal ACh sensor measurements;
- hippocampal population activity or decoded theta-sequence predictions;
- behaviour sufficient to fit subject-specific transition beliefs;
- movement, acceleration, pupil, theta power, and task-engagement nuisance
  variables;
- event timestamps with enough precision to estimate sensor and circuit delay.

## Analysis protocol

1. Fit subject-specific transition beliefs using training trials only.
2. Freeze the estimator before predicting ACh on held-out trials.
3. Generate all candidate signals from the same latent belief trajectory.
4. Convolve candidate event series with an independently estimated sensor kernel
   or jointly infer a constrained observation model.
5. Compare held-out likelihood, not only in-sample correlation.
6. Run simulation-based model recovery under the exact trial schedule.
7. Report candidate correlations and non-identifiable parameter regimes.
8. Validate calibration using posterior predictive checks.

## Closed-loop extension

If ACh acts as a global gain on locally stored eligibility traces, stimulation
should have a delay-dependent effect. Trigger stimulation from online candidate
signals and sweep latency relative to the critical transition. The predicted
plasticity effect should track the eligibility-trace time course rather than a
broad, untimed enhancement of learning.

## Minimum falsification criteria

The Bayesian update hypothesis loses support if, under matched prediction and
observation:

- ACh is invariant to confidence with adequate power;
- raw innovation or surprise consistently wins held-out model comparison;
- apparent gain effects disappear after sensor-dynamics and movement controls;
- model recovery shows that the planned experiment cannot identify the winning
  candidate.
