# Latent contexts and structural change

Version 0.2 separates three operations that can all follow an unexpected
transition but have different computational and biological interpretations:

1. updating belief about which known context is active;
2. updating parameters inside an identified context;
3. inferring that no stored context is adequate and a new regime began.

## 1. Exact filtering over known contexts

Let $`m_t\in\{1,\ldots,M\}`$ be a latent context with Markov transition matrix
$`\Pi`$. For current state $`i`$, action $`u`$, and observed next state $`j`$,
context $`m`$ has Dirichlet transition parameters
$`\boldsymbol\alpha_{m,u,i}`$. Its posterior-predictive likelihood is

```math
L_m(j\mid i,u)
=\frac{\alpha_{m,u,i,j}}
       {\sum_k\alpha_{m,u,i,k}}.
```

The context prediction is

```math
q_t^-(m)
=\sum_{m'}q_{t-1}(m')\Pi_{m'm},
```

and the exact HMM filtering update for fixed context-specific predictive kernels
is

```math
q_t(m)
=\frac{q_t^-(m)L_m(j\mid i,u)}
       {\sum_n q_t^-(n)L_n(j\mid i,u)}.
```

`SwitchingContextFilter` also retains the joint posterior

```math
q_t(m_{t-1},m_t)
\propto q_{t-1}(m_{t-1})
\Pi_{m_{t-1}m_t}L_{m_t}(j\mid i,u),
```

which yields a posterior switch probability

```math
P(m_t\neq m_{t-1}\mid\mathcal D_{1:t}).
```

The context-belief update signal is

```math
G_t^{\mathrm{context}}
=D_{\mathrm{KL}}\!\left[q_t(m)\,\|\,q_t^-(m)\right].
```

This is a hidden-state update. It is not evidence that the stored transition
parameters should be overwritten.

## 2. Parameter learning is an explicit, separate operation

By default, `SwitchingContextFilter.observe(...)` leaves every Dirichlet
parameter unchanged. This permits a context cue or surprising transition to
retrieve a previously learned model without destructive relearning.

If an external label establishes context $`m^*`$, an exact supervised conjugate
update can be requested:

```math
\boldsymbol\alpha_{m^*,u,i}'
=\boldsymbol\alpha_{m^*,u,i}+\boldsymbol e_j.
```

The API intentionally does not perform an unlabelled fractional Dirichlet
update. Such a soft-responsibility update would be an assumed-density or
variational approximation, not exact Bayesian learning in the latent-context
model.

For hypothesis generation, the filter reports the posterior expectation of the
context-conditional posterior-mean update magnitude without applying it:

```math
\mathbb E_{q_t(m)}
\left[
  \left\|
  \frac{\boldsymbol e_j-\widehat{\boldsymbol p}_{m,u,i}}
       {\alpha_{m,u,i,0}+1}
  \right\|_2
\right].
```

This supplies a parameter-learning candidate that is distinct from context
information gain.

## 3. Full Bayesian online change-point detection

A known context bank cannot represent a genuinely new transition regime.
`DirichletBOCPD` implements exact Bayesian online change-point detection for a
piecewise-stationary categorical transition model.

Let $`r_t`$ be the run length and $`h`$ the constant change hazard. Every
run-length hypothesis stores a complete Dirichlet tensor over
$`(u,i,j)`$. After the first observation starts the initial segment, the
recursion is

```math
\widetilde R_t(0)
=h\,p_0(j_t\mid i_t,u_t),
```

```math
\widetilde R_t(r+1)
=(1-h)R_{t-1}(r)
 p_r(j_t\mid i_t,u_t),
```

followed by normalization

```math
R_t(r)=\frac{\widetilde R_t(r)}
             {\sum_s\widetilde R_t(s)}.
```

Here $`p_0`$ is the prior predictive distribution for a new regime and $`p_r`$
is the predictive distribution from the sufficient statistics associated with
run length $`r`$.

The principal structural signals are

```math
P(c_t=1\mid\mathcal D_{1:t})=R_t(0)
```

and

```math
G_t^{\mathrm{run}}
=D_{\mathrm{KL}}\!\left[R_t(r)\,\|\,R_t^-(r)\right].
```

No run-length pruning or moment merging is used. Inference is therefore exact
under the stated model, with $`O(T)`$ memory and update cost at time $`T`$, and
$`O(T^2)`$ total work for a sequence.

## 4. Model-class recovery benchmark

The benchmark uses three distinct ring transition kernels:

- context A: predominantly forward transitions;
- context B: predominantly backward transitions and already stored;
- regime C: a novel jump kernel absent from the context bank.

Every sequence begins in context A. The post-change segment is generated either
by stored context B or novel regime C.

Two models receive the same fully observed transitions:

- $`M_{\mathrm{context}}`$: exact filtering over known contexts A and B;
- $`M_{\mathrm{change}}`$: exact BOCPD with a symmetric Dirichlet reset prior.

The sequence decision uses only prequential post-change evidence:

```math
\Delta\mathcal L
=\sum_{t\in\mathrm{post}}
 \log p(y_t\mid y_{<t},M_{\mathrm{context}})
-\sum_{t\in\mathrm{post}}
 \log p(y_t\mid y_{<t},M_{\mathrm{change}}).
```

Positive margins favour retrieval of a known context; negative margins favour a
new regime. No fitted discriminative classifier is required.

`novel_similarity` blends regime C towards the average known kernel and exposes
an explicit identifiability stress parameter. As a genuinely new regime becomes
observationally equivalent to a distribution representable by the context
model, model-class recovery must eventually become prior-dependent or
non-identifiable rather than being forced to return a confident answer.

## 5. Interpretation for ACh experiments

The implementation creates trial-wise quantities that can be compared without
semantic conflation:

| Operation | Candidate quantity |
|---|---|
| Unexpected transition | predictive surprise or raw innovation |
| Retrieve a stored context | context posterior and context KL |
| Infer a context switch | posterior switch probability |
| Revise a known transition row | expected or applied parameter-update magnitude |
| Infer a new regime | BOCPD change probability |
| Revise structural belief | run-length KL |

A controlled experiment should independently manipulate whether an unexpected
transition is:

- rare but valid within the active stochastic context;
- diagnostic of another already learned context;
- evidence for a genuinely novel transition regime.

## 6. Scope and limitations

Version 0.2 assumes fully observed discrete states and actions. The context bank
and BOCPD model are compared as separate generative explanations; a unified
nonparametric model that can switch among known contexts while spawning new
contexts is not yet implemented. Other deliberate exclusions are:

- uncertain or corrupted sensory observations;
- context-dependent observation models;
- non-constant hazards;
- hierarchical sharing between contexts;
- photometry sensor dynamics and movement covariates;
- delayed septo-hippocampal feedback;
- replay and fixed-interval smoothing.

The next milestone introduces an observation model so that sensory corruption,
state mislocalization, context switching, and structural transition change can
be compared within one partially observed inference problem.
