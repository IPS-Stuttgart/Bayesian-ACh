# Paper repository and frozen evidence

Paper-facing material for Bayesian-ACh lives in a separate repository:

- **Paper, evidence, figures, notes, and submission material:**
  https://github.com/FlorianPfaff/2026-08-Bayesian-ACh-Paper
- **Reusable implementation, tests, package code, and technical documentation:**
  this repository (`IPS-Stuttgart/Bayesian-ACh`).

## Frozen manuscript provenance

The paper-ready computational evidence is frozen against this exact source revision:

```text
IPS-Stuttgart/Bayesian-ACh
commit 53530aa4f59940cfdb41dbf8acd2ee8bd9dccbaa
version 0.5.0
```

That revision includes the transition-signal, latent-context/change-point,
partial-observation attribution, ACh measurement, and closed-loop eligibility
benchmarks used by the manuscript.

The paper repository verifies its claim registry against this exact commit in CI.
Consequently, later development in this repository cannot silently change a
published numerical claim.

## Scope boundary

Replay/smoothing work, including PR #6, is later experimental development and is
**not** part of the frozen paper evidence. It should enter a future manuscript
revision only after it is independently cleaned, validated, and explicitly added
to a new evidence freeze.

## Reproducing the paper

Clone the paper repository and follow its README. Its CI checks out this repository
at the frozen SHA above, verifies the committed evidence and executable claims,
generates figures, and builds the manuscript/submission PDFs.

The separation is deliberate: scientific software remains reusable and reviewable
here, while manuscript text, evidence capsules, submission notes, and generated
paper artifacts remain versioned with the paper.
