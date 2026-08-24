# Partial observations and multisensory causal attribution

## 1. Why the observation model is necessary

An animal does not observe its allocentric state directly. It receives visual,
vestibular, proprioceptive, tactile, olfactory, and contextual-cue observations.
A surprising visual event can therefore mean at least four different things:

1. the animal was uncertain about its current state;
2. one sensory channel became unreliable;
3. the task switched to an already learned transition context;
4. the physical transition dynamics changed.

A model that equates an experimenter-defined visual mismatch with a world-model
error cannot distinguish these explanations. Version 0.3 introduces an exact
small-state observation model before fitting any biological ACh measurement.

## 2. Joint latent state

Let

- $`x_t\in\{1,\ldots,S\}`$ be the latent task or spatial state;
- $`m_t\in\{1,\ldots,M\}`$ be the transition context;
- $`h_{r,t}\in\{0,1\}`$ be the health of sensor $`r`$, with zero nominal and
  one faulty;
- $`y_{r,t}`$ be the observation from sensor $`r`$;
- $`u_{t-1}`$ be the action preceding the transition.

The exact filter retains

```math
q_t(m,x,\boldsymbol h)
=
p(m_t=m,x_t=x,\boldsymbol h_t=\boldsymbol h
\mid y_{0:t},u_{0:t-1}).
```

Context and sensor-health dynamics are specified by

```math
\Pi_{m'm}=p(m_t=m\mid m_{t-1}=m'),
```

```math
Q_r(h'_r,h_r)=p(h_{r,t}=h_r\mid h_{r,t-1}=h'_r).
```

Sensor health is conditionally independent across modalities, so

```math
Q(\boldsymbol h' ,\boldsymbol h)
=
\prod_r Q_r(h'_r,h_r).
```

The context-indexed controlled transition model is

```math
P_m^{u}(x',x)=p(x_t=x\mid x_{t-1}=x',u_{t-1}=u,m_t=m).
```

## 3. Exact prediction and update

The joint prediction is

```math
q_t^-(m,x,\boldsymbol h)
=
\sum_{m',x',\boldsymbol h'}
q_{t-1}(m',x',\boldsymbol h')
\Pi_{m'm}
P_m^{u_{t-1}}(x',x)
Q(\boldsymbol h',\boldsymbol h).
```

Each modality has a health-, context-, and state-dependent categorical
observation model

```math
B_{r,h_r}(y\mid m,x)
=
p(y_{r,t}=y\mid h_{r,t}=h_r,m_t=m,x_t=x).
```

This general form includes ordinary state sensors and explicit context cues.
Conditionally independent observations yield

```math
L_t(m,x,\boldsymbol h)
=
\prod_{r\in\mathcal O_t}
B_{r,h_r}(y_{r,t}\mid m,x),
```

where $`\mathcal O_t`$ contains the modalities present at time $`t`$. Missing
modalities contribute a factor of one. The exact posterior is

```math
q_t(m,x,\boldsymbol h)
=
\frac{q_t^-(m,x,\boldsymbol h)L_t(m,x,\boldsymbol h)}
{\sum_{\bar m,\bar x,\bar{\boldsymbol h}}
 q_t^-(\bar m,\bar x,\bar{\boldsymbol h})
 L_t(\bar m,\bar x,\bar{\boldsymbol h})}.
```

The denominator is the prequential multisensory evidence used for model-class
comparison.

## 4. Trial-wise computational quantities

`MultisensoryContextFilter` reports quantities that should not be conflated:

- state information gain: $`D_{\mathrm{KL}}(q_t(x)\|q_t^-(x))`$;
- context information gain: $`D_{\mathrm{KL}}(q_t(m)\|q_t^-(m))`$;
- sensor-health information gain: $`D_{\mathrm{KL}}(q_t(\boldsymbol h)\|q_t^-(\boldsymbol h))`$;
- posterior context-switch probability: $`P(m_t\neq m_{t-1}\mid y_{0:t})`$;
- posterior fault probability for every sensor;
- posterior probability that each fault began at the current observation;
- state and context entropy;
- combined predictive surprise.

The filter also computes modality-only state and context posteriors under the
nominal observation models. Their Jensen--Shannon divergence is a diagnostic of
raw sensory conflict. It is not used as a substitute for the full posterior.

## 5. Exactness and computational scope

Every binary sensor-health configuration is represented explicitly. With
$`R`$ sensors, the joint health state has $`2^R`$ configurations. One update
therefore scales with the state, context, and complete health configuration
spaces. The implementation is exact for small hypothesis-driven modality sets,
not a replacement for high-dimensional learned perception.

The one-step posterior, joint previous/current-context posterior, and joint
previous/current-health posterior are checked against exhaustive enumeration in
the test suite.

## 6. Three-way causal-attribution benchmark

The controlled benchmark contains three post-change mechanisms. Every sequence
begins with the same forward transition context and healthy sensors.

### Visual sensor fault

The world remains in the forward context, but the visual channel changes to a
systematically shifted observation model. Proprioception and the contextual cue
remain nominal.

### Known context switch

The transition kernel switches from the learned forward relation to the learned
backward relation. All sensors remain healthy, and the context cue changes.

### Structural transition change

The transition kernel switches to a preregistered jump relation absent from the
known-context model. All sensors remain healthy, and the manipulation is uncued.

Three exact model classes receive identical observations:

```math
\mathcal M_F=\text{fixed world with latent visual health},
```

```math
\mathcal M_C=\text{healthy sensors with forward/backward contexts},
```

```math
\mathcal M_S=\text{healthy sensors with forward/structural contexts}.
```

The post-change evidence for model $`k`$ is

```math
\mathcal L_k
=
\sum_{t\in\mathrm{post}}
\log p(y_t\mid y_{0:t-1},u_{0:t-1},\mathcal M_k).
```

Equal model priors are updated prequentially. No discriminative classifier is
fit, and simulated true states are used only for post-hoc decoding diagnostics.

Run the benchmark with

```bash
bayesian-ach observation-benchmark \
  --output results/observation-attribution \
  --seed 7
```

The output contains trial-wise latent-cause posteriors, sequence-wise evidence,
a confusion matrix, state-decoding diagnostics, and explicit identifiability
measures.

## 7. Identifiability controls

`--fault-similarity` interpolates the faulty visual emission towards the healthy
visual emission. At one, the configured fault is observationally null.

`--structural-similarity` interpolates the structural transition kernel towards
a mixture of the known transition kernels. The summary reports:

- mean total variation between healthy and faulty visual emissions;
- minimum total variation between the structural and known transition kernels;
- total variation supplied by the binary context cue;
- warnings for exactly non-identifiable configurations.

Strong recovery under the default benchmark is meaningful only together with
these stress controls.

## 8. Important limitation

The structural model is exact for a specified candidate transition kernel. This
matches experiments in which the topology or transition intervention is known
in advance. Exact online learning of an arbitrary unseen transition kernel while
also marginalizing uncertain latent-state trajectories is combinatorial; a
scalable open-set extension will require an explicitly labelled approximation
or a restricted exact enumeration. Version 0.3 does not disguise that problem
as conjugate Dirichlet learning.

## 9. ACh predictions enabled by the model

The model permits experiments that match visual surprise while changing its
causal explanation. Candidate ACh regressors can now be compared for sensitivity
to:

- sensory conflict;
- state-belief revision;
- inferred sensor-fault onset;
- retrieval of a known context;
- evidence for a specified structural world change.

This is the required computational layer before interpreting sensor-convolved
ACh measurements as errors in the world model itself.
