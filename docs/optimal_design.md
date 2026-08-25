# Maximin prospective experimental design

## Motivation

Bayesian-ACh originally supplied hand-built dissociations such as matched
confidence and sensor-versus-world changes. Those contrasts are scientifically
useful, but a prospective study still has to decide how many trials to allocate
to each feasible condition. A one-dimensional novel-versus-familiar schedule can
leave the candidate signals almost collinear even when the total number of
trials is large.

Version 0.7 adds a finite-design optimizer that converts the candidate equations
into an auditable trial allocation.

## Feasible design grid

For a three-state categorical transition, one design point controls:

- probability of the observed next state;
- distribution of the remaining probability mass;
- Dirichlet concentration, or evidence mass;
- probability assigned to the observation by a reset model;
- prior reset hazard.

Every point is evaluated by the same exact transition model and yields the six
candidate signals:

```text
innovation_l2
surprise
gain
update_l2
information_gain
change_probability
```

The default grid contains 240 conditions. The optimizer never invents a
condition outside this declared feasible set.

## Maximin objective

Let $`x_k(d)`$ be the globally standardized value of candidate $`k`$ at design
point $`d`$, and let $`w_d`$ be the fraction of trials allocated to that point.
For an ordered generator/alternative pair $`(k,l)`$, regress $`x_k`$ on an
intercept and $`x_l`$ under the design weights. The residual variance is

```math
R_{k\mid l}(w)
=
\mathrm{Var}_w(x_k)
-
\frac{\mathrm{Cov}_w(x_k,x_l)^2}
     {\mathrm{Var}_w(x_l)}.
```

Under the Gaussian response model

```math
y=a x_k+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,\sigma^2),
```

the generating candidate has residual variance $`\sigma^2`$, whereas the
alternative's population-optimal profiled residual variance is
$`\sigma^2+a^2R_{k\mid l}(w)`$. Because the recovery code estimates a separate
training residual variance for every candidate, the corresponding expected
held-out Gaussian log-score gap per trial is

```math
G_{k\mid l}(w)
=
\frac{1}{2}\log\!\left(
1+\frac{a^2R_{k\mid l}(w)}{\sigma^2}
\right).
```

The earlier linear expression $`a^2R/(2\sigma^2)`$ is only the first-order
small-residual expansion of this profiled-variance gap; it is not a Bayes
factor.
The primary design criterion is therefore

```math
\max_w\min_{k\ne l} R_{k\mid l}(w).
```

This directly optimizes the worst candidate confusion rather than average
variance or a global determinant that can hide one nearly indistinguishable
pair. For fixed $`a/\sigma`$, $`G`$ is strictly increasing in $`R`$, so the
maximin allocation and all residual-ratio comparisons are unchanged by the
profiled-variance correction.

### Affine-equivalence and identification proposition

For $`\mathrm{Var}_w(x_l)>0`$, the ordered residual has the projection
interpretation

```math
R_{k\mid l}(w)
=
\min_{b,c}\;\mathbb E_w[(x_k-b-cx_l)^2].
```

Hence $`R_{k\mid l}=0`$ if and only if $`x_k=b+cx_l`$ almost surely on the
positive-weight design support. If $`x_l`$ is constant, it adds nothing beyond
the fitted intercept and the implementation sets
$`R_{k\mid l}=\mathrm{Var}_w(x_k)`$; a constant generator consequently has
zero residual against every alternative.

Every candidate recovery fit contains an intercept and a free slope. Replacing
a candidate column by $`b+c x`$ with $`c\ne0`$ therefore leaves its affine
column space, fitted predictions, candidate-specific residual variance, and
held-out Gaussian log score unchanged. Independently z-standardizing the
declared candidate columns also leaves every residual geometry and the optimized
allocation unchanged under such affine reparameterizations (including sign
reversal). This invariance does not justify unequal biological amplitudes:
$`a`$ remains a prespecified effect per standardized candidate unit.
## Integer allocation algorithm

`optimize_maximin_design` uses deterministic greedy allocation followed by
exchange refinement:

1. add the feasible point that maximizes the current lexicographic objective;
2. use minimum pairwise residual variance as the primary score;
3. use minimum candidate variance, covariance log determinant, and maximum
   correlation as deterministic tie breakers while the design is rank deficient;
4. perform remove-one/add-one exchanges until no allowed exchange improves the
   score;
5. enforce a configurable maximum fraction of trials at any one condition.

The algorithm records every addition and exchange. It is not a black-box neural
optimizer, and the complete selected support is exported as CSV.

## Baselines

The benchmark compares equal trial budgets:

- `coupled_novelty`: improbability, weak confidence, reset compatibility, hazard,
  and representational skew increase together;
- `uniform_factorial`: a seeded uniform sample of the feasible grid;
- `maximin_optimized`: the proposed worst-case allocation.

All schedules are evaluated with the same held-out model-recovery code.

## Default controlled evidence

With 60 trials, unit candidate amplitude, unit Gaussian noise, and 200 recovery
replicates per generator, the seed-7 controlled benchmark gives approximately:

| Design | Worst generator recovery | Mean recovery | Minimum pairwise residual |
|---|---:|---:|---:|
| Coupled novelty | 0.380 | 0.536 | 0.0090 |
| Uniform factorial | 0.705 | 0.793 | 0.1141 |
| Maximin optimized | 0.830 | 0.895 | 0.2500 |

Across prespecified seeds 7, 11, 19, 23, and 31, the minimum recovery rate over
all generators and seeds was 0.310 for coupled novelty, 0.635 for uniform
factorial, and 0.750 for the optimized design. The optimized minimum residual
was 27.69 times the novelty value and at least 1.91 times the seeded uniform
factorial value.

These are controlled design-recovery results. They do not establish that the
Gaussian response model or any candidate is biologically correct.

## Quantitative trial guidance

The geometry also converts a prespecified signal-to-noise ratio into a transparent
asymptotic trial target. For desired cumulative expected profiled Gaussian
log-score gap $`B`$, the worst-pair diagnostic is

```math
N_{B}
=
\left\lceil
\frac{B}
     {\frac12\log\left(
       1+a^2\min_{k\ne l}R_{k\mid l}(w)/\sigma^2
     \right)}
\right\rceil.
```

At unit standardized amplitude, unit noise, and $`B=5`$, the default residuals
correspond to 45 trials for the maximin design, 93 for the seeded uniform
factorial design, and 1,113 for the coupled-novelty design. These values are
planning diagnostics, not retrospective power guarantees: serial dependence,
subject variation, sensor convolution, missing trials, and model misspecification
must be included in a study-specific simulation before animal numbers are fixed.
## Scaling assumption and sensitivity requirement

Global standardization gives each computational candidate one unit of variation
over the declared feasible grid. This is an explicit equal-standardized-effect
comparison; it does not assert that one physical unit of every candidate produces
the same cholinergic amplitude. Before freezing a biological protocol, investigators
should rerun the optimizer over a plausible set of candidate-specific amplitudes
and noise levels, or optimize the minimum criterion jointly over that uncertainty
set. A design whose advantage disappears under modest rescaling should not be
presented as robust.

Likewise, the current Gaussian criterion is local to a scalar linear readout.
Photometry kinetics, nuisance regressors, subject-level random effects, and
closed-loop timing can be incorporated by replacing the candidate matrix with
the corresponding forward-model predictions while retaining the same auditable
finite-allocation structure.

## Certified continuous and integer follow-up

The original `maximin_optimized` allocation is a deterministic greedy
construction with at most three one-for-one exchanges. Its trace is auditable,
but it is not a certificate of global optimality.

`certify_maximin_design` supplies a separate global certificate. For each
ordered pair,

```math
R_{k\mid l}(w)
=
\inf_{b,c}\sum_d w_d[x_k(d)-b-cx_l(d)]^2.
```

For any fixed `(b,c)`, the loss is linear in the allocation. The certificate
iteratively solves a HiGHS LP or MILP master problem, evaluates exact weighted
least squares for all ordered pairs, and adds every violated loss cut. The
finite master gives an upper bound; the directly evaluated allocation gives a
feasible lower bound. Results report both bounds and stop as certified only
when their declared tolerance is met.

The integer mode certifies the stated budget and per-cell count cap. The
continuous mode certifies the capped-simplex relaxation and therefore also
provides an upper bound for every integer allocation. Neither mode constructs a
sequential behavioral history. The 240 cells are independently instantiated
belief conditions; reset, washout, carry-over, and physical ordering constraints
must be encoded before calling any count vector an executable protocol.

```bash
bayesian-ach-design-certify \
  --output results/certified-design-n60 \
  --code-sha <exact-commit> \
  --budget 60 \
  --mode integer \
  --require-certificate
```

Certificate artifacts are versioned separately from the previously frozen
greedy/exchange evidence. A changed certified allocation must trigger new
recovery evidence; it must never silently replace the earlier artifact.

Objective certification and planning-index certification are reported
separately. If rigorous residual bounds are `[R_L,R_U]`, monotonicity of the
profiled Gaussian gap gives

```math
\left\lceil\frac{B}{G(R_U)}\right\rceil
\le N_{\mathrm{eff}} \le
\left\lceil\frac{B}{G(R_L)}\right\rceil.
```

The rounded `N_eff` index is certified when these endpoints agree, even if a
continuous objective run has not met a much tighter numerical residual-gap
tolerance. This does not relabel that continuous objective as certified. The
claim-bearing finite schedules use exact integer certificates.

Frozen integer packages can be independently checked against their SHA-256
table, allocation budget and cap, canonical 240-cell geometry, objective gap,
cut trace, and `N_eff` rounding:

```bash
bayesian-ach-design-certificate-verify \
  results/certified-maximin-design/n60 \
  results/certified-maximin-design/n45
```

## Use

```bash
bayesian-ach-design \
  --output results/optimal-design \
  --budget 60 \
  --replicates 200 \
  --seed 7
```

The command writes the feasible grid, selected allocation, design diagnostics,
ordered pairwise geometry, held-out recovery results, complete optimization
trace, and a JSON summary.

## Scope boundaries

The default optimizer does not automatically represent animal-welfare costs,
trial carry-over, asymmetric intervention risk, sensor kinetics, unknown
multiplexing, or hardware restrictions. Such constraints must be encoded by
removing infeasible grid points, changing the per-condition trial cap, or
extending the objective before a biological protocol is frozen.
