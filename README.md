# Bayesian-ACh

**Bayesian-ACh** is a falsifiable computational benchmark for testing what
hippocampal acetylcholine (ACh) encodes during predictive learning.

The project starts from the state-transition prediction-error hypothesis of
[de Cothi, Shipley, and Barry (2026)](https://doi.org/10.1038/s41583-026-01058-w)
and asks a sharper estimation-theoretic question:

> Does ACh encode a raw sensory mismatch, a state-belief correction, a sensor
> fault, retrieval of a latent context, or the Bayesian evidence that the world
> model itself should change?

Version 0.3 implements four exact finite-state inference layers:

1. conjugate Dirichlet learning for one transition model;
2. HMM filtering over a bank of already learned transition contexts;
3. Bayesian online change-point detection over genuinely new fully observed
   regimes;
4. joint multisensory filtering over latent state, context, and binary sensor
   health.

The original six scalar hypotheses remain available:

1. raw innovation magnitude;
2. predictive surprise;
3. Bayesian gain;
4. posterior-mean update magnitude;
5. parameter information gain;
6. local reset/change evidence.

Versions 0.2 and 0.3 add context information gain, posterior context-switch
probability, a full run-length posterior, sensor-health information gain,
per-sensor fault-onset probability, and multisensory state/context conflict.
This repository is a computational hypothesis-testing project. It does **not**
claim that any candidate has already been established as the biological ACh
signal.

## Core distinctions

### Error is not update magnitude

For a categorical transition row with

\[
\boldsymbol\theta \sim \operatorname{Dir}(\boldsymbol\alpha),
\qquad
\widehat{\boldsymbol p}=\frac{\boldsymbol\alpha}{\alpha_0},
\]

and observed next state \(j\), define

\[
\boldsymbol\nu = \boldsymbol e_j-\widehat{\boldsymbol p}.
\]

The raw-error proposal is proportional to \(\lVert\boldsymbol\nu\rVert\). The
exact posterior-mean change is instead

\[
\Delta\widehat{\boldsymbol p}
= \frac{1}{\alpha_0+1}\boldsymbol\nu.
\]

Thus the same prediction and observation can imply different rational updates
when model confidence differs.

### Context inference is not parameter learning

For known context \(m_t\), the switching filter computes

\[
q_t^-(m)=\sum_{m'}q_{t-1}(m')\Pi_{m'm},
\qquad
q_t(m)\propto q_t^-(m)
  p(x_{t+1}\mid x_t,u_t,m).
\]

This can retrieve an already learned transition model without changing any
transition parameter. `SwitchingContextFilter.observe(...)` therefore performs
inference only by default. An exact Dirichlet update is applied only when an
external context label is supplied explicitly through `learn_context`.

### Known switch is not structural reset

`DirichletBOCPD` maintains the complete run-length posterior

\[
p(r_t\mid x_{0:t},u_{0:t-1})
\]

and complete Dirichlet sufficient statistics for every run-length hypothesis.
It provides a full structural-change baseline rather than the one-step local
reset score used in version 0.1.

### Observation mismatch is not world change

Version 0.3 retains the exact joint posterior

\[
q_t(m,x,\boldsymbol h)
=
p(m_t=m,x_t=x,\boldsymbol h_t=\boldsymbol h
\mid y_{0:t},u_{0:t-1}),
\]

where \(\boldsymbol h_t\) contains one binary health state per sensor. The joint
prediction is

\[
q_t^-(m,x,\boldsymbol h)
=
\sum_{m',x',\boldsymbol h'}
q_{t-1}(m',x',\boldsymbol h')
\Pi_{m'm}P_m^u(x',x)Q(\boldsymbol h',\boldsymbol h).
\]

Health-, context-, and state-dependent sensor models then determine whether an
unexpected observation should revise the state, context, or sensor-health
belief. A conflicting visual observation can therefore be assigned to visual
corruption without forcing a transition-model change.

## Installation

```bash
python -m pip install -e '.[dev]'
```

Python 3.10 or newer is supported.

## Run the benchmarks

Run the matched-confidence and six-signal benchmark:

```bash
bayesian-ach dissociate --output results/dissociation --seed 7
bayesian-ach benchmark --output results/benchmark --seed 7
```

Run known-context versus structural-reset recovery with fully observed states:

```bash
bayesian-ach regime-benchmark \
  --output results/regime-recovery \
  --seed 7
```

Run the partial-observation causal-attribution benchmark:

```bash
bayesian-ach observation-benchmark \
  --output results/observation-attribution \
  --seed 7
```

The observation benchmark compares three exact models receiving identical
visual, proprioceptive, and context-cue observations:

- a fixed world with latent visual-sensor health;
- healthy sensors with a switch to a known transition context;
- healthy sensors with a preregistered structural-transition alternative.

It writes:

- `observation_trials.csv`: trial-wise state, context, health, conflict, and
  model-class posteriors;
- `observation_sequences.csv`: sequence-wise evidence, decisions, margins, and
  state-decoding diagnostics;
- `summary.json`: confusion matrix, balanced accuracy, and identifiability
  measures.

The default seed-7 benchmark recovers all 108 sequences. Median winning evidence
margins are 57.786 for visual faults, 77.689 for known context switches, and
58.761 for structural transition changes. These are controlled synthetic
model-recovery results, not biological evidence.

Use `--fault-similarity` and `--structural-similarity` to move the candidate
mechanisms towards observational non-identifiability. The summary reports total
variation separations and warnings instead of silently interpreting an
unidentifiable configuration.

## Python example

```python
import numpy as np

from bayesian_ach import MultisensoryContextFilter

transition = np.array(
    [
        [[0.90, 0.10], [0.10, 0.90]],
        [[0.20, 0.80], [0.80, 0.20]],
    ]
)
state_sensor = np.array([[0.90, 0.10], [0.10, 0.90]])
visual_models = np.stack((state_sensor, state_sensor[:, ::-1]))
proprioceptive_models = np.stack((state_sensor, state_sensor))
health_transition = np.array(
    [
        [[0.98, 0.02], [0.10, 0.90]],
        [[1.00, 0.00], [1.00, 0.00]],
    ]
)

filter_ = MultisensoryContextFilter(
    transition_probabilities=transition,
    context_transition=[[0.98, 0.02], [0.02, 0.98]],
    emission_probabilities=(visual_models, proprioceptive_models),
    sensor_health_transition=health_transition,
    initial_context=[1.0, 0.0],
    initial_sensor_health=[[1.0, 0.0], [1.0, 0.0]],
)
filter_.initialize((0, 0))
step = filter_.step((1, 0))

assert np.isclose(step.posterior_joint.sum(), 1.0)
assert step.sensor_fault_probabilities.shape == (2,)
```

## Scientific programme

The repository follows a staged programme:

- **Stage 1 — complete:** exact finite-state transition learning, calibrated
  scalar hypotheses, matched-confidence dissociation, and model recovery;
- **Stage 2 — complete:** exact filtering over known contexts, explicit
  separation of mode inference from parameter learning, full categorical
  BOCPD, and context-switch-versus-reset recovery;
- **Stage 3 — complete:** exact joint filtering of latent state, transition
  context, and sensor health; context-dependent observation channels;
  multisensory conflict diagnostics; and prequential recovery of sensor fault,
  known context switch, and specified structural change;
- **Stage 4 — next:** ACh observation dynamics, including phasic/tonic release,
  indicator convolution, movement, arousal, and hierarchical session effects;
- **Stage 5:** delayed closed-loop perturbation and eligibility-trace tests;
- **Stage 6:** replay as smoothing-based revision rather than unconstrained
  internally generated prediction error.

See [`docs/partial_observation.md`](docs/partial_observation.md) for the new
multisensory derivation, [`docs/switching_model.md`](docs/switching_model.md) for
context and change-point inference, [`docs/experimental_design.md`](docs/experimental_design.md)
for the proposed VR experiment, and [`docs/model.md`](docs/model.md) for the
original transition signals.

## Repository layout

```text
src/bayesian_ach/       exact models, simulations, model recovery, CLI
tests/                  unit, exhaustive-enumeration, and end-to-end tests
docs/                   scientific model, experiment, data contract, roadmap
examples/               reproducible plotting examples
data/                    instructions for external datasets; no data vendored
results/                 generated evidence; only documentation is tracked
```

## Development

```bash
ruff check .
mypy src/bayesian_ach
pytest --cov=bayesian_ach --cov-report=term-missing
python -m build
```

GitHub Actions runs linting, typing, build checks, and tests on Python 3.10–3.13.

## Citation

Please cite both the motivating Perspective and this software. Repository
citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
