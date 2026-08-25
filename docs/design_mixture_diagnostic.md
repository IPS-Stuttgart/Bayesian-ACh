# Post-failure mixture-aware design diagnostic

This diagnostic is intentionally separate from the immutable N=60 stress failure
frozen at commit `d1251dddeaa706d79feaaec99185dcc4236aa4a3`. It does not
replace, tune, or reinterpret that result. Its method and random streams are
committed before the new evaluation is run.

## Locked method

The diagnostic is restricted to the chronologically earlier heuristic maximin
allocation at N=60. It uses three-fold cross-fitting, so every observation is
scored only by models fitted without that observation. The pure family contains
the six prespecified single-candidate regressions.

The targeted composite family contains all 15 unordered candidate pairs. For
each training fold, an intercept and two nonnegative slopes are fitted. A free
total amplitude makes this a nonnegative cone over each pairwise convex simplex;
a pure candidate is a boundary case. The maximum held-out score across all 15
pairs is the familywise composite statistic.

For each possible pure winner, calibration under that matched pure generator
sets separate upper thresholds for:

- the maximum pairwise-composite score improvement over the pure model; and
- a cross-fitted residual lack-of-fit ratio, defined as validation squared error
  divided by the variance fitted on the corresponding training fold.

Every upper threshold uses the one-based order statistic
`ceil((n+1)(1-alpha))`. With `n=200` and familywise `alpha=0.05`, this is
rank 191. The pure-over-null statistic is the maximum over all six pure
candidates under the null. The composite statistic is the maximum over all 15
pairs, calibrated separately under each matched pure candidate. The
winner/runner statistic is the best-minus-second-best pure score under the null;
the residual threshold is candidate-specific.

The independent audit measures four distinct forms of power, each with a 95%
Wilson interval:

- correct pure-call retention, separately for each pure candidate;
- correct abstention under the null;
- correct abstention, separately for each of the 15 mixture pairs; and
- correct abstention for the fixed out-of-span probe.

A candidate or contrast is enabled only when its own audit Wilson lower bound is
at least 0.70. An underpowered pure winner is forced to abstain. Pair/null/probe
evaluation rates remain descriptive but receive the status
`mandatory_abstain_underpowered` when their corresponding audit gate fails.
Thus an easy contrast cannot license a claim for a difficult pair. These power
rules were fixed before evaluation and are not relaxed after seeing its output.

The fixed streams and replicate counts are:

| Purpose | Seed | Replicates |
|---|---:|---:|
| threshold calibration | 196613 | 200 |
| independent calibration audit | 262147 | 200 |
| one-time evaluation | 324949 | 200 |

All three streams are disjoint. Evaluation data do not set thresholds, choose
models, change the power rule, or alter the allocation.

## Scope and identifiability

The extension is targeted to positive two-candidate mixtures. It is not a
general open-set classifier. The nonnegative sign restriction follows the
declared positive-effect simulation and would need separate justification for
an empirical model with unoriented effects.

Cross-fitting uses all 60 observations once as held-out predictions, but it
cannot create information absent from the design. The immutable diagnostic
already shows that several mixtures are only weakly separated from their best
pure affine approximation on this support. The artifact therefore reports a
population oracle log-score-gap index for every pair. Low finite-sample audit
power forces abstention; it is not treated as evidence that the pure hypothesis
is correct.

The fixed orthogonalized nonlinear residual remains one bounded out-of-span
probe. The residual lack-of-fit gate is calibrated for the matched Gaussian
simulation, not arbitrary biological misspecification, serial dependence,
animal hierarchy, indicator dynamics, or an executable sequential protocol.
Passing this diagnostic would support only the declared simulation family and
would not constitute a main-paper robustness claim without an independent
freeze.

## Reproduction

From a clean checkout of the exact producer commit:

```bash
bayesian-ach-mixture-diagnostic \
  --repo-root . \
  --code-sha <producer-commit> \
  --baseline-artifact results/design-open-set-stress-n60 \
  --locked-allocation /absolute/path/optimal_design_allocation_seed7.csv \
  --locked-allocation-sha256 a823be49faf6c6cbebf60b11d4b5ca895cf7734d6e9c577ee98f97a5907b69b2 \
  --locked-design-code-sha 1b2028929ac6ebc1cce0882f0c22af9918044342 \
  --locked-allocation-seed 7 \
  --output /absolute/path/mixture-aware-diagnostic
```

The command verifies the immutable baseline, exact allocation hash, explicit
seed metadata, source commit, deterministic allocation reconstruction, clean
worktree, and producer commit. It writes checksum-bound tables and immediately
runs the independent artifact verifier.
