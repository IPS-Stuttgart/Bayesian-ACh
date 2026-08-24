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

Let $x_k(d)$ be the globally standardized value of candidate $k$ at design
point $d$, and let $w_d$ be the fraction of trials allocated to that point.
For an ordered generator/alternative pair $(k,l)$, regress $x_k$ on an
intercept and $x_l$ under the design weights. The residual variance is

$$
R_{k\mid l}(w)
=
\operatorname{Var}_w(x_k)
-
\frac{\operatorname{Cov}_w(x_k,x_l)^2}
     {\operatorname{Var}_w(x_l)}.
$$

Under the Gaussian response model

$$
y=a x_k+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,\sigma^2),
$$

the expected held-out log-evidence separation from candidate $l$, per trial,
is

$$
\frac{a^2}{2\sigma^2}R_{k\mid l}(w).
$$

The primary design criterion is therefore

$$
\max_w\min_{k\ne l} R_{k\mid l}(w).
$$

This directly optimizes the worst candidate confusion rather than average
variance or a global determinant that can hide one nearly indistinguishable
pair.

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
first-order trial target. For desired expected log Bayes factor $B$, the
worst-pair approximation is

$$
N_{B}
=
\left\lceil
\frac{2\sigma^2 B}
     {a^2\min_{k\ne l}R_{k\mid l}(w)}
\right\rceil.
$$

At unit standardized amplitude, unit noise, and $B=5$, the default residuals
correspond to approximately 40 trials for the maximin design, 88 for the seeded
uniform factorial design, and 1,112 for the coupled-novelty design. These values
are planning diagnostics, not retrospective power guarantees: serial dependence,
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
