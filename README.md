# Bayesian-ACh

**Bayesian-ACh** is a falsifiable computational benchmark for testing what
hippocampal acetylcholine (ACh) encodes during predictive learning.

The project starts from the state-transition prediction-error hypothesis of
[de Cothi, Shipley, and Barry (2026)](https://doi.org/10.1038/s41583-026-01058-w)
and asks a sharper estimation-theoretic question:

> Does ACh encode the raw transition mismatch, or the Bayesian quantity that
> determines whether and how strongly that mismatch should update the internal
> world model?

The initial release implements an exact finite-state Dirichlet transition model,
a decisive matched-confidence experiment, and held-out model-recovery tests for
six candidate signals:

1. raw innovation magnitude;
2. predictive surprise;
3. Bayesian gain;
4. posterior-mean update magnitude;
5. parameter information gain;
6. local reset/change evidence.

This repository is a computational hypothesis-testing project. It does **not**
claim that any candidate has already been established as the biological ACh
signal.

## Core distinction

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

Thus the same predicted transition and the same observation can imply very
different rational updates when model confidence differs. Bayesian-ACh makes
that dissociation executable and testable.

## Installation

```bash
python -m pip install -e '.[dev]'
```

Python 3.10 or newer is supported.

## Run the decisive synthetic benchmark

```bash
bayesian-ach benchmark --output results/benchmark --seed 7
```

This writes:

- `trials.csv`: trial-wise candidate regressors;
- `model_recovery.csv`: held-out fit statistics for every generating and fitted
  hypothesis;
- `summary.json`: recovery winners and experimental metadata.

Run the minimal matched-confidence demonstration:

```bash
bayesian-ach dissociate --output results/dissociation --seed 7
```

The low- and high-confidence conditions have identical predictive probabilities
and paired observations. Therefore raw innovation and surprise are matched,
while gain and update magnitude differ.

## Python example

```python
import numpy as np

from bayesian_ach import compute_transition_signals

probabilities = np.array([0.70, 0.20, 0.10])
observed_next_state = 1

low = compute_transition_signals(
    alpha=5.0 * probabilities,
    observed_index=observed_next_state,
)
high = compute_transition_signals(
    alpha=100.0 * probabilities,
    observed_index=observed_next_state,
)

assert np.isclose(low.surprise, high.surprise)
assert np.isclose(low.innovation_l2, high.innovation_l2)
assert low.update_l2 > high.update_l2
```

## Scientific programme

The repository follows a staged programme:

- **Stage 1 — exact model recovery:** finite-state, fully observed transition
  learning with calibrated synthetic experiments;
- **Stage 2 — partial observability:** separate sensory corruption, state
  uncertainty, latent-context inference, and structural change;
- **Stage 3 — ACh observation model:** infer latent phasic and tonic release from
  sensor-convolved photometry while controlling movement and arousal;
- **Stage 4 — closed-loop experiments:** compute trial-wise candidate signals
  online and test delay-dependent cholinergic perturbations.

See [`docs/experimental_design.md`](docs/experimental_design.md) for the proposed
VR experiment and [`docs/model.md`](docs/model.md) for the derivations.

## Repository layout

```text
src/bayesian_ach/       exact models, simulations, model recovery, CLI
tests/                  unit and end-to-end regression tests
docs/                   scientific model, experiment, data contract, roadmap
examples/               reproducible plotting example
data/                    instructions for external datasets; no data vendored
results/                 generated evidence; only documentation is tracked
```

## Development

```bash
ruff check .
pytest --cov=bayesian_ach --cov-report=term-missing
python -m build
```

GitHub Actions runs linting and tests on Python 3.10–3.13.

## Citation

Please cite both the motivating Perspective and this software. Repository
citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
