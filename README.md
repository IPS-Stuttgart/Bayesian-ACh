# Bayesian-ACh

**Bayesian-ACh** is a falsifiable computational benchmark for testing what
hippocampal acetylcholine (ACh) encodes during predictive learning.

The project starts from the state-transition prediction-error hypothesis of
[de Cothi, Shipley, and Barry (2026)](https://doi.org/10.1038/s41583-026-01058-w)
and asks a sharper estimation-theoretic question:

> Does ACh encode a raw sensory mismatch, a rational parameter update, a
> state-belief correction, a sensor fault, retrieval of a latent context, or
> evidence that the world model itself should change—and can that event signal
> still be identified after realistic release and measurement dynamics?

Version 0.7 implements seven computational layers:

1. conjugate Dirichlet learning for one transition model;
2. exact HMM filtering over already learned transition contexts;
3. exact categorical Bayesian online change-point detection for fully observed
   regimes;
4. exact joint multisensory filtering over latent state, context, and binary
   sensor health;
5. a calibration-separated ACh measurement model with phasic and tonic release,
   indicator convolution, nuisance regressors, subject effects, session
   baselines, and held-out candidate comparison;
6. causal closed-loop triggering with explicit transport delay, randomized
   latency, local eligibility traces, yoked sham controls, and held-out recovery
   of the causal stimulation window;
7. a versioned, leakage-audited replay artifact contract and a prespecified
   animal-level comparison of behavioral filtering-to-smoothing revision against
   online-surprise, content, location, recency, prospective, and TD-error fields,
   with recovery gates and explicit abstention.

This repository is a computational hypothesis-testing project. It does **not**
claim that any candidate has already been established as the biological ACh
signal.

## Core distinctions

### Error is not rational update magnitude

For a categorical transition row,

```math
\boldsymbol\theta\sim\mathrm{Dir}(\boldsymbol\alpha),
\qquad
\widehat{\boldsymbol p}=\frac{\boldsymbol\alpha}{\alpha_0},
```

and observed next state $`j`$, the raw innovation is

```math
\boldsymbol\nu=\boldsymbol e_j-\widehat{\boldsymbol p}.
```

The exact posterior-mean change is

```math
\Delta\widehat{\boldsymbol p}
=
\frac{1}{\alpha_0+1}\boldsymbol\nu.
```

The same prediction and observation can therefore produce the same mismatch and
surprise but a very different rational update when confidence differs.

### Context inference is not parameter learning

For known context $`m_t`$, the exact switching filter computes

```math
q_t^-(m)=\sum_{m'}q_{t-1}(m')\Pi_{m'm},
\qquad
q_t(m)\propto q_t^-(m)
 p(x_{t+1}\mid x_t,u_t,m).
```

This can retrieve an already learned transition model without changing any
transition parameter. `SwitchingContextFilter.observe(...)` performs inference
only unless an external context label is explicitly supplied through
`learn_context`.

### A known switch is not a structural reset

`DirichletBOCPD` maintains the full run-length posterior

```math
p(r_t\mid x_{0:t},u_{0:t-1})
```

and complete Dirichlet sufficient statistics for each run-length hypothesis. It
therefore separates retrieval of a stored context from evidence that a new
piecewise-stationary regime began.

### Observation mismatch is not world change

The multisensory filter retains

```math
q_t(m,x,\boldsymbol h)
=
p(m_t=m,x_t=x,\boldsymbol h_t=\boldsymbol h
\mid y_{0:t},u_{0:t-1}),
```

where $`\boldsymbol h_t`$ contains one binary health state per sensor. A
conflicting visual observation can update state, context, or visual-health
belief instead of automatically forcing transition learning.

### A computational event is not the measured trace

Version 0.4 models

```math
z_t
=
b_j
+
(h*a_s c_k)_t
+
\boldsymbol\beta^\top\boldsymbol q_t
+
(h*u)_t,
```

where

- $`c_{k,t}`$ is one candidate Bayesian event train;
- $`a_s`$ is a partially pooled subject-specific signal coefficient;
- $`u_t=\rho u_{t-1}+\epsilon_t`$ is latent tonic release;
- $`h`$ is a causal difference-of-exponentials indicator response;
- $`\boldsymbol q_t`$ contains movement and arousal nuisance regressors;
- $`b_j`$ is a baseline-only session offset.

The sensor and tonic timescales are inferred from a known calibration input in
training sessions only. Candidate coefficients are fitted on training task
samples, and the final comparison uses held-out task samples only.

### A stimulation main effect is not eligibility gating

For effective event-to-perturbation lag $`\ell_i`$, the closed-loop model tests

```math
D_i
=
Y_i^{\mathrm{active}}-Y_i^{\mathrm{sham}}
=
\beta_0+\beta_E A_i E(\ell_i;\theta)+\epsilon_i.
```

The null, latency-independent, and latency-dependent models are compared on
held-out sessions. Eligibility gating is claimed only when a causal timing
interaction clears a preregistered log-evidence threshold over the selected
simpler model. Event, command, and effective timestamps are separate, and the
command-to-effective delay is treated as independently calibrated rather than
silently absorbed into the inferred trace.

## Installation

```bash
python -m pip install -e '.[dev]'
```

Python 3.10 or newer is supported.

## Run the benchmarks

### Matched confidence and scalar-signal recovery

```bash
bayesian-ach dissociate --output results/dissociation --seed 7
bayesian-ach benchmark --output results/benchmark --seed 7
```

### Known context versus structural reset

```bash
bayesian-ach regime-benchmark \
  --output results/regime-recovery \
  --seed 7
```

### Partial-observation causal attribution

```bash
bayesian-ach observation-benchmark \
  --output results/observation-attribution \
  --seed 7
```

The default observation benchmark recovers all 108 sequences: 36 visual sensor
faults, 36 known-context switches, and 36 specified structural changes.

### ACh measurement-model recovery

```bash
bayesian-ach measurement-benchmark \
  --output results/measurement-recovery \
  --seed 7
```

The measurement benchmark asks whether the generating computational event train
can be recovered after phasic and tonic release, indicator smoothing, nuisance
confounding, subject variation, session offsets, and held-out sessions.

It writes:

- `measurement_generators.csv`: one recovery summary per generating signal;
- `measurement_fits.csv`: held-out scores for every generator/candidate pair;
- `measurement_kernel_posterior.csv`: calibration-only posterior over rise,
  decay, and tonic-persistence hypotheses;
- `measurement_samples.csv`: candidate, nuisance, split, and synthetic trace
  values;
- `summary.json`: acceptance metrics and split provenance.

For the default seed-7 benchmark:

- all 7/7 generating signals are recovered;
- the median held-out evidence margin is 771.532 log units;
- the minimum evidence margin is 421.096 log units;
- the calibration MAP is exactly
  $`(\tau_r,\tau_d,\rho)=(0.4,1.6,0.97)`$;
- median nuisance-coefficient MAE is 0.00749;
- median subject-signal correlation is 0.99769;
- maximum absolute correlation between sensor-convolved candidates is 0.88880.

These are controlled synthetic model-recovery results, not biological evidence.

### Closed-loop eligibility-window recovery

```bash
bayesian-ach closed-loop-benchmark \
  --output results/closed-loop-recovery \
  --seed 7
```

The default benchmark compares five causal generators: no effect, an untimed
main effect, exponential eligibility, rise-and-decay eligibility, and a finite
boxcar window. It uses balanced randomized latencies, uncertainty gating,
refractory suppression, missed and background commands, explicit actuation
delay, and yoked active/sham command times.

For the default seed-7 benchmark:

- all 5/5 causal generators are recovered;
- 2,236 active/sham pairs are accepted, including 29 false-positive or
  randomized background events;
- the minimum conservative decision margin is 3.474 log units beyond the
  preregistered claim boundary;
- the median decision margin is 376.937 log units;
- active and sham command timestamps agree exactly;
- a 30-seed stress check recovers all 150 generator instances.

The benchmark writes generator summaries, all candidate fits, pair-level
outcomes, opportunity-level trigger decisions, and a provenance-rich JSON
summary. These remain controlled synthetic recovery results.

## Python example

```python
import numpy as np

from bayesian_ach import (
    MeasurementDataset,
    MeasurementFitConfig,
    fit_measurement_models,
)

n = 400
sessions = np.repeat(np.arange(4), n // 4)
subjects = np.repeat([0, 0, 1, 1], n // 4)
train = np.isin(sessions, [0, 2])
calibration = np.tile(np.r_[np.ones(40, dtype=bool), np.zeros(60, dtype=bool)], 4)
task = ~calibration
baseline = np.tile(np.r_[np.ones(8, dtype=bool), np.zeros(92, dtype=bool)], 4)

rng = np.random.default_rng(7)
dataset = MeasurementDataset(
    observed=rng.normal(size=n),
    calibration_event=np.zeros(n),
    candidate_events=rng.normal(size=(n, 2)),
    nuisance=rng.normal(size=(n, 1)),
    subject_ids=subjects,
    session_ids=sessions,
    train_mask=train,
    calibration_mask=calibration,
    task_mask=task,
    baseline_mask=baseline,
    candidate_names=("surprise", "context_information_gain"),
    nuisance_names=("movement",),
)

result = fit_measurement_models(dataset, MeasurementFitConfig(dt=0.2))
print(result.winner.candidate)
```

Real analyses should use a meaningful exogenous calibration event and inspect
whether the discrete timescale posterior is concentrated and stable.

## Scientific programme

- **Stage 1 — complete:** exact transition learning, matched-confidence
  dissociation, and six-way scalar model recovery.
- **Stage 2 — complete:** exact context filtering, supervised parameter learning,
  full categorical BOCPD, and known-switch-versus-reset recovery.
- **Stage 3 — complete:** exact partial-observation filtering over state, context,
  and sensor health, with three-way sensor/world attribution.
- **Stage 4 — complete:** calibration-only sensor and tonic-timescale inference,
  phasic/tonic forward measurement modeling, nuisance and subject separation,
  and seven-way held-out event-signal recovery.
- **Stage 5 — complete:** causal online triggering, independently calibrated
  delay, randomized timing, yoked active/sham perturbation, eligibility-family
  recovery, and falsification against null and latency-independent effects.
- **Stage 6 — implemented, claim-gated:** a leakage-audited replay artifact
  contract, filtering-to-smoothing spatial-field comparison, animal-level
  simultaneous contrasts, and post-decoder recovery gates. A strict real-data
  freeze is still required before making any biological replay claim.

See [`docs/pf_replay_spatial_revision.md`](docs/pf_replay_spatial_revision.md)
for the replay contract, claim boundary, and fixed analysis; [`docs/closed_loop.md`](docs/closed_loop.md) for causal triggering and
eligibility-window recovery, [`docs/measurement_model.md`](docs/measurement_model.md)
for the ACh measurement derivation,
[`docs/partial_observation.md`](docs/partial_observation.md) for multisensory
inference, [`docs/switching_model.md`](docs/switching_model.md) for context and
change-point inference, and [`docs/model.md`](docs/model.md) for the original
transition signals.

## Repository layout

```text
src/bayesian_ach/       exact models, measurement and closed-loop benchmarks, CLI
tests/                  unit, exhaustive-enumeration, leakage, and recovery tests
docs/                   derivations, experiment design, data contract, roadmap
examples/               reproducible plotting examples
data/                    external-data instructions; no data vendored
results/                 generated evidence; only documentation is tracked
```

## Development

```bash
ruff check .
mypy src/bayesian_ach
pytest --cov=bayesian_ach --cov-report=term-missing
python -m build
```

GitHub Actions runs linting, typing, build checks, the complete test suite on
Python 3.10–3.13, and smoke tests for all benchmark commands.

## Citation

Please cite both the motivating Perspective and this software. Repository
citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
