# Contributing

Bayesian-ACh is intended to support falsifiable computational neuroscience, not
post-hoc relabelling of correlated regressors. Contributions should preserve
that standard.

## Workflow

1. Create a focused branch from `main`.
2. Add or update tests with every behavioural change.
3. Run `ruff check .` and `pytest` locally.
4. Open a pull request describing the scientific claim, assumptions, and checks.
5. Prefer squash merging after CI passes.

## Scientific requirements

A new candidate ACh signal should include:

- a mathematical definition;
- an explicit generative assumption;
- at least one experimental manipulation that dissociates it from existing
  candidates;
- simulation-based model recovery;
- held-out evaluation rather than in-sample correlation alone;
- documentation of non-identifiability or near-collinearity when present.

## Code requirements

- Keep the exact finite-state core dependency-light.
- Validate shapes, positivity, and normalization at public boundaries.
- Use deterministic random seeds in tests and documented examples.
- Do not commit restricted or identifying experimental data.
- Keep generated outputs out of version control unless they are designated,
  provenance-locked evidence artifacts.
