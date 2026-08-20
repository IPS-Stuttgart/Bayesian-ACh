# Bayesian-ACh

**Bayesian-ACh** is a falsifiable computational benchmark for testing what
hippocampal acetylcholine (ACh) encodes during predictive learning.

The project starts from the state-transition prediction-error hypothesis of
[de Cothi, Shipley, and Barry (2026)](https://doi.org/10.1038/s41583-026-01058-w)
and asks a sharper estimation-theoretic question:

> Does ACh encode the raw transition mismatch, the update of a latent context,
> or the Bayesian quantity that determines whether the transition model itself
> should change?

Version 0.2 implements three exact finite-state inference layers:

1. conjugate Dirichlet learning for one transition model;
2. HMM filtering over a bank of already learned transition contexts;
3. Bayesian online change-point detection (BOCPD) over genuinely new regimes.

The original six scalar hypotheses remain available:

1. raw innovation magnitude;
2. predictive surprise;
3. Bayesian gain;
4. posterior-mean update magnitude;
5. parameter information gain;
6. local reset/change evidence.

Version 0.2 adds context-belief information gain, posterior context-switch
probability, a full run-length posterior, and structural-change probability.
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
external context label is passed explicitly through `learn_context`.

### Known switch is not structural reset

`DirichletBOCPD` maintains the complete run-length posterior

\[
p(r_t\mid x_{0:t},u_{0:t-1})
\]

and complete Dirichlet sufficient statistics for every run-length hypothesis.
It provides a full structural-change baseline rather than the one-step local
reset score used in version 0.1.

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

Run the new model-class recovery benchmark:

```bash
bayesian-ach regime-benchmark \
  --output results/regime-recovery \
  --seed 7
```

The regime benchmark compares prequential post-change evidence under two
models:

- retrieval of a known alternative transition context;
- a newly learned piecewise-stationary transition regime.

It writes:

- `regime_trials.csv`: trial-wise context and run-length signals;
- `regime_sequences.csv`: sequence-wise model-evidence margins and decisions;
- `summary.json`: per-class and balanced recovery accuracy.

The `--novel-similarity` option continuously moves the new regime towards a
mixture of the known context kernels and can be used as an identifiability
stress test.

## Python example

```python
import numpy as np

from bayesian_ach import DirichletBOCPD, SwitchingContextFilter

context_kernels = np.array(
    [
        [[0.90, 0.10], [0.80, 0.20]],
        [[0.20, 0.80], [0.10, 0.90]],
    ]
)
context_filter = SwitchingContextFilter(
    alpha=100.0 * context_kernels,
    context_transition=[[0.98, 0.02], [0.02, 0.98]],
    initial_context=[1.0, 0.0],
)

before = context_filter.alpha.copy()
context_step = context_filter.observe(state=0, next_state=1)
assert np.array_equal(context_filter.alpha, before)
assert context_step.context_kl > 0.0

change_detector = DirichletBOCPD(n_states=2, hazard=0.02)
change_step = change_detector.observe(state=0, next_state=1)
assert np.isclose(change_step.run_length_probabilities.sum(), 1.0)
```

## Scientific programme

The repository follows a staged programme:

- **Stage 1 — complete:** exact finite-state transition learning, calibrated
  scalar hypotheses, matched-confidence dissociation, and model recovery;
- **Stage 2 — complete:** exact filtering over known contexts, explicit
  separation of mode inference from parameter learning, full categorical
  BOCPD, and context-switch-versus-reset recovery;
- **Stage 3 — next:** partial observations and multisensory reliability, to
  separate sensor corruption and state uncertainty from transition change;
- **Stage 4:** ACh observation dynamics, including phasic/tonic release,
  indicator convolution, movement, arousal, and hierarchical session effects;
- **Stage 5:** delayed closed-loop perturbation and eligibility-trace tests;
- **Stage 6:** replay as smoothing-based revision rather than unconstrained
  internally generated prediction error.

See [`docs/switching_model.md`](docs/switching_model.md) for the new derivations,
[`docs/experimental_design.md`](docs/experimental_design.md) for the proposed VR
experiment, and [`docs/model.md`](docs/model.md) for the original transition
signals.

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
