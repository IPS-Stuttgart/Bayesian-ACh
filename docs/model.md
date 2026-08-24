# Bayesian model

## 1. Transition row

For current state $`i`$, action $`u`$, and categorical next state $`j`$, let

```math
\boldsymbol\theta_{iu} \sim \mathrm{Dir}(\boldsymbol\alpha_{iu}).
```

The posterior predictive distribution is

```math
\widehat{\boldsymbol p}_{iu}
= \mathbb E[\boldsymbol\theta_{iu}\mid\mathcal D]
= \frac{\boldsymbol\alpha_{iu}}{\alpha_{iu,0}}.
```

After observing next state $`j`$, the exact conjugate update is

```math
\boldsymbol\alpha'_{iu}=\boldsymbol\alpha_{iu}+\boldsymbol e_j.
```

## 2. Candidate ACh signals

### Raw innovation magnitude

```math
\boldsymbol\nu=\boldsymbol e_j-\widehat{\boldsymbol p},
\qquad
s_{\mathrm{innovation}}=\lVert\boldsymbol\nu\rVert_2.
```

This is close to a literal state-transition mismatch. It depends on the full
predictive vector and on the chosen representation metric.

### Predictive surprise

```math
s_{\mathrm{surprise}}=-\log \widehat p_j.
```

This measures how improbable the observation was under the current predictive
model, but not how much the model should change.

### Bayesian gain

```math
g=\frac{1}{\alpha_0+1}.
```

For a Dirichlet transition row, concentration $`\alpha_0`$ is effective evidence
mass. Gain is therefore large for weakly established relations and small for
well-established relations.

### Rational posterior update magnitude

The posterior predictive mean changes by

```math
\Delta\widehat{\boldsymbol p}
=\frac{1}{\alpha_0+1}
  (\boldsymbol e_j-\widehat{\boldsymbol p})
=g\boldsymbol\nu.
```

The corresponding scalar candidate is

```math
s_{\mathrm{update}}=\lVert g\boldsymbol\nu\rVert_2.
```

This is the leading Bayesian-ACh hypothesis: ACh may encode the magnitude of the
rational structural update, while local eligibility traces retain its direction.

### Parameter information gain

```math
s_{\mathrm{IG}}
=D_{\mathrm{KL}}\!\left[
\mathrm{Dir}(\boldsymbol\alpha+\boldsymbol e_j)
\,\|\,
\mathrm{Dir}(\boldsymbol\alpha)
\right].
```

Information gain measures how much the complete posterior over transition
parameters changes, not merely its mean.

### Local reset/change evidence

Let $`\boldsymbol q`$ be a reset predictive model and $`h`$ a prior reset hazard.
A one-step two-model posterior is

```math
P(c=1\mid j)
=\frac{h q_j}{(1-h)\widehat p_j+h q_j}.
```

This quantity asks whether the observation is better explained by a structural
reset than by the current model. It is intentionally labelled **local**: a full
Bayesian online change-point model must maintain run-length or mode hypotheses.

## 3. Why matched confidence is decisive

Choose two Dirichlet rows

```math
\boldsymbol\alpha^{(L)}=c_L\boldsymbol p,
\qquad
\boldsymbol\alpha^{(H)}=c_H\boldsymbol p,
\qquad c_L<c_H.
```

They produce exactly the same predictive distribution $`\boldsymbol p`$. Pair
the same observation $`j`$ across conditions. Innovation and surprise are then
identical, whereas gain, posterior update, and information gain differ. A neural
signal sensitive to the latter quantities cannot be explained by raw mismatch
alone.

## 4. Current scope and exclusions

Version 0.1 assumes fully observed discrete states and independent Dirichlet
rows. It does not yet model:

- uncertain sensory observations;
- latent contexts or switching behavioural policies;
- hierarchical sharing between transition rows;
- sensor convolution and movement-correlated photometry;
- delayed septo-hippocampal feedback;
- replay or fixed-interval smoothing.

Those extensions are staged only after the finite-state hypotheses pass
simulation-based identifiability and model-recovery checks.
