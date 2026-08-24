# Replay as fixed-interval Bayesian revision

## Objective

Replay must not be treated as if an internally generated trajectory were a new
external observation. Bayesian-ACh 0.6 instead separates:

1. information available during online filtering;
2. retrospective corrections produced by later external observations;
3. trajectories sampled from the already conditioned fixed-interval posterior.

Only the second operation changes a past belief because it introduces
information that was not available online. The third is a readout of that
posterior and carries no additional evidence by itself.

## Finite-state model

For latent state $x_t\in\{1,\ldots,K\}$, define

$$
\pi_i=p(x_0=i),
\qquad
P_t(i,j)=p(x_{t+1}=j\mid x_t=i),
\qquad
\ell_t(i)=p(y_t\mid x_t=i).
$$

The implementation permits a different transition matrix at every step. A
missing observation is represented by a likelihood row of ones. It therefore
contributes predictive probability one and surprise zero; it is not incorrectly
converted into a uniform categorical observation with surprise $\log K$.

## Online filter

The one-step state prediction and normalized filtered posterior are

$$
\bar\alpha_t(j)
=
\sum_i \alpha_{t-1}(i)P_{t-1}(i,j),
$$

$$
c_t
=
\sum_j \bar\alpha_t(j)\ell_t(j),
\qquad
\alpha_t(j)
=
\frac{\bar\alpha_t(j)\ell_t(j)}{c_t}.
$$

Here $c_t=p(y_t\mid y_{0:t-1})$, so the online predictive surprise is

$$
s_t^{\mathrm{online}}=-\log c_t.
$$

The complete interval evidence is

$$
\log p(y_{0:T})=\sum_{t=0}^{T}\log c_t.
$$

## Scaled backward information

The backward message is initialized with $\beta_T(i)=1$ and propagated by

$$
\beta_t(i)
=
\frac{1}{c_{t+1}}
\sum_j
P_t(i,j)\ell_{t+1}(j)\beta_{t+1}(j).
$$

The scaling by the corresponding forward normalizer keeps the recursion stable
and yields the fixed-interval state posterior

$$
\gamma_t(i)
=
p(x_t=i\mid y_{0:T})
\propto
\alpha_t(i)\beta_t(i).
$$

## State hindsight correction

Bayesian-ACh reports two complementary state revisions:

$$
G_t^{\mathrm{state}}
=
D_{\mathrm{KL}}\!\left(
\gamma_t\,\middle\|\,\alpha_t
\right),
$$

$$
R_t^{\mathrm{state}}
=
\lVert\gamma_t-\alpha_t\rVert_1.
$$

These quantities answer how strongly later external observations revise the
state belief that was rational online. They are zero at the terminal time because
there is no later evidence beyond $T$.

## Transition-pair revision

After observing $y_{t+1}$, the online pair posterior is

$$
\xi_t^{\mathrm{filter}}(i,j)
=
p(x_t=i,x_{t+1}=j\mid y_{0:t+1})
\propto
\alpha_t(i)P_t(i,j)\ell_{t+1}(j).
$$

The fixed-interval pair posterior is

$$
\xi_t^{\mathrm{smooth}}(i,j)
=
p(x_t=i,x_{t+1}=j\mid y_{0:T})
\propto
\xi_t^{\mathrm{filter}}(i,j)\beta_{t+1}(j).
$$

The transition hindsight signal is

$$
G_t^{\mathrm{transition}}
=
D_{\mathrm{KL}}\!\left(
\xi_t^{\mathrm{smooth}}
\,\middle\|
\xi_t^{\mathrm{filter}}
\right).
$$

An L1 revision is exported as a diagnostic. It is deliberately not promoted to
an additional benchmark generator because it is strongly correlated with the
transition KL in this design.

Expected transition counts are

$$
\bar N_{ij}^{\mathrm{smooth}}
=
\sum_{t=0}^{T-1}
\xi_t^{\mathrm{smooth}}(i,j).
$$

The difference from the sum of online pair posteriors quantifies retrospective
count revision. These expectations are posterior diagnostics and the sufficient
statistics used by an EM M-step. Under hidden states, they are not an exact
factorized Dirichlet posterior update.

## Posterior replay sampling

`FiniteStateSmoother.sample_smoothed_trajectories(...)` implements
forward-filtering backward-sampling. It first draws

$$
x_T^{(r)}\sim\gamma_T,
$$

then recursively draws

$$
p(x_t=i\mid x_{t+1}=j,y_{0:T})
=
p(x_t=i\mid x_{t+1}=j,y_{0:t})
\propto
\alpha_t(i)P_t(i,j).
$$

The implementation vectorizes samples sharing the same next state and never
modifies the smoother or any posterior result. Tests verify both posterior
marginals and exact read-only behavior.

### Why samples are not new evidence

A replay path is drawn from

$$
p(x_{0:T}\mid y_{0:T}).
$$

It is therefore a stochastic representation of information already contained
in $y_{0:T}$. Updating transition parameters again as though the sample were
an independent observation would count the same measurement interval twice.
The repository exposes posterior samples for decoding and mechanistic comparison,
not as an automatic learning update.

## Replay-content surprise

For each transition, the expected model surprisal of posterior replay content is

$$
S_t^{\mathrm{content}}
=
\sum_{i,j}
\xi_t^{\mathrm{smooth}}(i,j)
\left[-\log P_t(i,j)\right].
$$

This measures what kind of transition content the posterior is expected to
replay. It is mathematically distinct from both online predictive surprise and
filtering-to-smoothing information gain. It is not an increment of the data
likelihood.

## Identifying benchmark

Each synthetic sequence contains:

- a randomly parameterized transition kernel;
- high- and low-reliability observations;
- an interval of genuinely missing observations represented by unit likelihoods;
- a later high-reliability landmark that can revise earlier latent-state and
  transition beliefs;
- naturally occurring rare transitions that can be surprising online without
  requiring a large hindsight correction.

The benchmark compares four candidate replay-epoch event signals generated from
one exact posterior trajectory:

1. online predictive surprise;
2. state smoothing information gain;
3. transition smoothing information gain;
4. replay-content surprise.

Whole sequences, rather than individual time points, are assigned to the
training or held-out split. For each synthetic generator, univariate Gaussian
models are fitted on training sequences only and ranked by held-out predictive
likelihood.

## Default controlled evidence

For seed 7, the default benchmark uses 96 sequences, 64 transitions per
sequence, six latent states, 67 training sequences, and 29 held-out sequences.
It obtains:

- 4/4 generating candidates recovered;
- minimum held-out evidence margin: 762.905 log units;
- median held-out evidence margin: 917.011 log units;
- maximum absolute candidate correlation: 0.63705;
- maximum posterior mutation after replay sampling: exactly 0.0;
- median FFBS state-marginal mean absolute error with 128 samples: 0.01094.

A 30-seed stress check recovers all 120/120 generator decisions. The smallest
held-out evidence margin is 702.949 log units, the largest absolute candidate
correlation is 0.67638, and posterior mutation remains exactly zero.

These are controlled synthetic recovery results. They do not establish that a
biological replay-associated ACh signal encodes any one candidate.

## Experimental interpretation

The model supports a sharper replay experiment:

- fit the animal's online filter using behavior and neural decoding;
- freeze the generative model before analyzing the replay epoch;
- compute state and transition smoothing corrections from later external
  evidence;
- decode replay content without feeding decoded or sampled trajectories back as
  observations;
- compare replay-associated ACh against online surprise, hindsight correction,
  and replay-content surprise on held-out sessions.

Evidence for smoothing-based revision would require replay ACh to track the
filtering-to-smoothing correction after controlling for online surprise and
replay content. A signal explained only by replay-content surprise would support
content-dependent modulation, but not the claim that internally generated
samples provide new model-error evidence.

## Current scope

Version 0.6 assumes a known finite-state generative model. It does not yet
include:

- smoothing over the full state-context-sensor-health model from version 0.3;
- uncertain transition parameters inside the smoother;
- continuous-state or manifold smoothing;
- neural sequence-decoder uncertainty;
- replay event detection and censoring;
- a full measurement model jointly fitted to real sleep data.

Those extensions can reuse the same conceptual contract: future external data
may revise past beliefs, whereas posterior replay samples are read-only unless a
separate biological mechanism supplies genuinely new evidence.
