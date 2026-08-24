# Closed-loop perturbation and eligibility-trace recovery

## Scientific question

A stimulation main effect is not evidence that acetylcholine gates a local
synaptic eligibility trace. The identifying prediction is temporal:
stimulation should alter learning only when its **effective** arrival overlaps
the trace left by the transition event.

Version 0.5 formalizes that claim as a causal scheduling and held-out
model-recovery problem. It distinguishes:

1. no causal stimulation effect;
2. a latency-independent stimulation effect;
3. an exponentially decaying eligibility trace;
4. a rise-and-decay eligibility trace;
5. a finite boxcar eligibility window.

The benchmark is synthetic. It establishes that the proposed protocol can
recover these mechanisms under its assumptions; it does not establish the
biological mechanism.

## Online trigger

At event time $`t_i`$, an online estimator emits a candidate signal $`c_i`$ and
an uncertainty value $`u_i`$. A command is eligible only when

```math
c_i \ge \gamma,
\qquad
u_i \le u_{\max},
```

where $`\gamma`$ is a preregistered trigger threshold. The stateful trigger also
models:

- a refractory period;
- missed commands;
- randomized background commands below threshold;
- command-latency jitter;
- an independently calibrated command-to-effective transport delay.

Let $`d_i`$ be the randomized nominal command latency, $`\epsilon_i`$ the
realized non-negative jitter, and $`\delta`$ the calibrated biological and
hardware transport delay. The effective event-to-perturbation lag is

```math
\ell_i = d_i + \epsilon_i + \delta.
```

The implementation records event, command, and effective timestamps separately.
Latency assignments are balanced and randomized within each session using a
random stream independent of the synthetic event stream.

## Eligibility families

All eligibility kernels are causal and normalized to unit peak.

### Exponential trace

```math
E_{\exp}(\ell;\tau)
=
\mathbf 1_{\ell\ge 0}\exp(-\ell/\tau).
```

### Rise-and-decay trace

```math
E_{\alpha}(\ell;\tau_r,\tau_d)
=
\mathbf 1_{\ell\ge0}
\frac{
  \exp(-\ell/\tau_d)-\exp(-\ell/\tau_r)
}{
  \max_{s\ge0}
  [\exp(-s/\tau_d)-\exp(-s/\tau_r)]
},
\qquad
0<\tau_r<\tau_d.
```

This family allows a delayed peak rather than assuming that eligibility is
maximal at the transition time.

### Boxcar window

```math
E_{\mathrm{box}}(\ell;w)
=
\mathbf 1_{0\le\ell\le w}.
```

For reporting, the causal window is the interval on which a normalized kernel
is at least 10% of its peak. The corresponding command-time window subtracts
the calibrated transport delay.

## Yoked active and sham design

Each accepted opportunity creates a matched active/sham pair with the same:

- observation and candidate-signal history;
- event time;
- randomized nominal latency;
- realized command timestamp;
- subject and session assignment.

Only the intervention arm differs. In the synthetic benchmark the pair shares
subject, session, and pair-level baseline components, while arm-specific noise
is independent. The analyzed outcome is

```math
D_i = Y_i^{\mathrm{active}}-Y_i^{\mathrm{sham}},
```

which removes those shared components exactly in the simulation.

A real experiment cannot observe both potential outcomes for one physical
transition. The direct analogue is a masked, randomized, matched-repeat design
with identical scheduling software and hardware timestamps in active and sham
trials. The synthetic yoking is a protocol and analysis benchmark, not a claim
that counterfactual outcomes can be observed simultaneously.

## Causal outcome hypotheses

The null model is

```math
M_0:\qquad D_i=\varepsilon_i.
```

The nonspecific stimulation model is

```math
M_{\mathrm{main}}:\qquad D_i=\beta_0+\varepsilon_i.
```

An eligibility-gated model is

```math
M_E(\theta):\qquad
D_i
=
\beta_0
+
\beta_E A_i E(\ell_i;\theta)
+
\varepsilon_i,
```

where $`A_i`$ is a causally available candidate-amplitude estimate. The
simulation separately retains the latent true eligibility amplitude so that
accepted false-positive triggers can contaminate the observed predictor without
receiving an oracle effect. Ground-truth labels are never used to schedule
commands or construct the fitted candidate amplitude.

The preregistered grid contains:

- exponential $`\tau\in\{0.20,0.35,0.70,1.20\}`$ s;
- rise/decay pairs $`(0.08,0.60)`$, $`(0.16,0.90)`$, and
  $`(0.25,1.40)`$ s;
- boxcar widths $`0.25`$, $`0.45`$, and $`0.75`$ s.

## Bayesian fitting and held-out evidence

For each model, training-session outcomes obey

```math
\boldsymbol D_{\mathrm{tr}}
=
X_{\mathrm{tr}}\boldsymbol\beta+\boldsymbol\varepsilon,
\qquad
\boldsymbol\beta\sim\mathcal N(0,\sigma_\beta^2 I),
\qquad
\boldsymbol\varepsilon\sim\mathcal N(0,\sigma_D^2 I).
```

The exact Gaussian posterior is

```math
S
=
\left(
\sigma_\beta^{-2}I+
\sigma_D^{-2}X_{\mathrm{tr}}^\top X_{\mathrm{tr}}
\right)^{-1},
```

```math
m
=
S\sigma_D^{-2}X_{\mathrm{tr}}^\top
\boldsymbol D_{\mathrm{tr}}.
```

Held-out sessions are scored jointly under

```math
\boldsymbol D_{\mathrm{te}}
\sim
\mathcal N(
X_{\mathrm{te}}m,
\sigma_D^2I+X_{\mathrm{te}}SX_{\mathrm{te}}^\top
).
```

Thus test outcomes do not alter coefficients, timescales, or scheduling. A
regression test changes held-out outcomes by a large amount and verifies that
all training posterior coefficients remain unchanged.

## Conservative nested decision

The raw maximum held-out score is retained, but the scientific claim follows a
nested decision with a preregistered threshold $`B`$, defaulting to five log
units:

1. claim a stimulation effect only when $`\log p(D_{\mathrm{te}}\mid M_{\mathrm{main}})-\log p(D_{\mathrm{te}}\mid M_0)>B`$;
2. claim eligibility gating only when the best latency-dependent model exceeds
   the selected simpler model by more than $`B`$.

This prevents a slightly better flexible kernel from being described as causal
temporal gating.

## Falsification logic

The result is interpreted as follows:

- **Null wins:** no supported causal stimulation effect.
- **Latency-independent wins:** stimulation has an effect, but the data do not
  support eligibility gating.
- **Eligibility model wins:** the active-minus-sham effect depends on effective
  latency in a held-out session and clears the simpler-model threshold.

A broad active-versus-sham difference without the latency interaction does not
support the eligibility-trace hypothesis.

## Transport-delay identifiability

The benchmark distinguishes the randomized command latency from the calibrated
transport delay. It does not jointly infer a free delay and every trace family.
For an exponential trace,

```math
\beta_E\exp[-(d_i+\delta)/\tau]
=
\left(\beta_E\exp[-\delta/\tau]\right)
\exp(-d_i/\tau),
```

so a constant unknown $`\delta`$ is exactly confounded with effect amplitude.
Rise-and-decay and finite-window boundaries can contain more timing information,
but hardware and biological delay should still be independently calibrated and
sensitivity analyses should be reported.

The command-line benchmark exposes both `true_actuation_delay` and
`assumed_actuation_delay` so this sensitivity is explicit rather than hidden.

## Default controlled recovery evidence

With seed 7, the default benchmark uses:

- 8 subjects;
- 5 sessions per subject, of which 3 are training sessions;
- 96 opportunities per session;
- ten randomized nominal latency values from 0 to 1.6 s;
- uncertainty gating, refractory suppression, missed commands, jitter, and
  randomized background commands.

The resulting schedule contains 2,236 accepted active/sham pairs, including 29
accepted false-positive/background events. All five generating mechanisms are
recovered. The minimum conservative decision margin is 3.474 log units beyond
the preregistered claim boundary, and the median is 376.937. Active and sham
command timestamps agree exactly by construction.

A 30-seed stress check recovers all five generators for every seed. The minimum
conservative decision margin over those runs is 1.963 log units beyond the
claim boundary.

These values are controlled synthetic evidence tied to the version 0.5
implementation. They are not estimates of a biological eligibility timescale.

## Recommended experimental implementation

A collaborating laboratory should preregister:

1. the online candidate signal and uncertainty computation;
2. threshold, uncertainty gate, refractory period, and permitted exclusion
   reasons;
3. a balanced randomized latency schedule spanning the expected eligibility
   window and clearly late negative-control latencies;
4. hardware timestamp definitions and independent transport-delay calibration;
5. active/sham masking and matched-repeat construction;
6. the eligibility family grid and claim threshold;
7. training sessions used for fitting and untouched held-out sessions used for
   model comparison;
8. primary plasticity outcomes, such as transition-specific place-field,
   sequence-probability, or subsequent-choice changes.

## Scope boundaries

Version 0.5 uses known Gaussian arm noise and Gaussian-ridge coefficient
posteriors. It does not yet include:

- adaptive latency allocation;
- non-Gaussian or heteroscedastic behavioural outcomes;
- drift in transport delay;
- interference between overlapping eligibility traces;
- animal-level hierarchical uncertainty over kernel family;
- simultaneous ACh measurement-model and intervention-model fitting;
- replay-dependent backward credit assignment.

Those additions should follow only when the randomized latency protocol retains
adequate support across subjects and held-out sessions.
