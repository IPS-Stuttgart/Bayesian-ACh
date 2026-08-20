# ACh release and measurement model

## 1. Purpose

The exact belief-state models in Bayesian-ACh produce trial-wise computational
quantities such as predictive surprise, state information gain, context
information gain, sensor-health information gain, posterior context-switch
probability, visual-fault-onset probability, and multisensory conflict. A
photometry trace is not one of those quantities directly.

A measured ACh signal can contain at least four distinct layers:

1. a phasic event-dependent release drive;
2. a slowly varying tonic release process;
3. indicator and acquisition dynamics;
4. movement, arousal, and session-dependent nuisance effects.

Version 0.4 introduces a forward measurement model so candidate computational
signals are compared only after they have passed through the same measurement
physics and nuisance-separation procedure.

## 2. Event candidates from one belief trajectory

All candidate event trains are generated from the same exact
`MultisensoryContextFilter` trajectory. For sample index \(t\), the benchmark
retains

\[
\boldsymbol c_t =
\begin{bmatrix}
\text{predictive surprise} \\
D_{\mathrm{KL}}(q_t(x)\|q_t^-(x)) \\
D_{\mathrm{KL}}(q_t(m)\|q_t^-(m)) \\
D_{\mathrm{KL}}(q_t(h)\|q_t^-(h)) \\
P(m_t\ne m_{t-1}\mid y_{0:t}) \\
P(h^{\mathrm{visual}}_{t-1}=0,h^{\mathrm{visual}}_t=1\mid y_{0:t}) \\
\operatorname{JS}(q_t^{(1)}(x),\ldots,q_t^{(R)}(x))
\end{bmatrix}.
\]

The seven columns differ only in the proposed computational interpretation.
They do not come from separately fitted latent-state models. This prevents a
candidate from gaining an advantage through a different upstream estimator.

## 3. Phasic release and indicator dynamics

For candidate \(k\), subject \(s\), and sample interval \(\Delta t\), the
phasic release drive is

\[
r^{\mathrm{phasic}}_{s,t}
= a_s c_{k,t},
\]

where \(a_s\) is a partially pooled subject-specific signal coefficient.

The indicator impulse response is a causal difference of exponentials,

\[
h_j(\tau_r,\tau_d)
=
\frac{
\exp(-j\Delta t/\tau_d)-\exp(-j\Delta t/\tau_r)
}{
\max_{\ell\ge 0}
\left[
\exp(-\ell\Delta t/\tau_d)-\exp(-\ell\Delta t/\tau_r)
\right]
},
\qquad
0<\tau_r<\tau_d.
\]

It is normalized to unit peak so the signal coefficients remain interpretable
on a common scale. Convolution is performed independently within every session;
there is no state or signal leakage across session boundaries.

The phasic sensor contribution is

\[
s^{\mathrm{phasic}}_{s,t}
=(h*r^{\mathrm{phasic}}_s)_t.
\]

## 4. Latent tonic release

Tonic release is represented by a stationary AR(1) process within each session,

\[
u_t=\rho u_{t-1}+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,\sigma_\epsilon^2),
\qquad 0\le\rho<1.
\]

The same indicator dynamics act on tonic and phasic release. This is important:
a slow measured trace need not imply a slow biological release process because
both release persistence and indicator decay contribute to the observed
timescale.

Let

\[
a_r=\exp(-\Delta t/\tau_r),
\qquad
a_d=\exp(-\Delta t/\tau_d).
\]

The sampled difference-of-exponentials filter and AR(1) tonic process have the
combined denominator

\[
(1-a_rL)(1-a_dL)(1-\rho L),
\]

where \(L\) is the lag operator. Therefore the tonic sensor residual obeys a
conditional AR(3) representation. Define

\[
\phi_1=a_r+a_d+\rho,
\]

\[
\phi_2=a_ra_d+a_r\rho+a_d\rho,
\]

\[
\phi_3=a_ra_d\rho.
\]

After conditioning on the first three samples of each fitted session segment,

\[
e_t
-
\phi_1 e_{t-1}
+
\phi_2 e_{t-2}
-
\phi_3 e_{t-3}
\]

is proportional to the independent tonic-release innovations. Version 0.4 uses
this exact conditional whitening relation rather than treating the strongly
autocorrelated residual as independent noise.

## 5. Complete conditional measurement equation

For subject \(s\), session \(j\), and candidate \(k\), the measured signal is
modeled as

\[
z_t
=
b_j
+
\left(h * a_s c_k\right)_t
+
\boldsymbol\beta^\top\boldsymbol q_t
+
\left(h*u\right)_t,
\]

where

- \(b_j\) is a session offset estimated only from the pre-task baseline;
- \(a_s\) is a subject-specific signal coefficient;
- \(\boldsymbol q_t\) contains nuisance regressors;
- \(\boldsymbol\beta\) contains nuisance coefficients;
- \(u_t\) is the latent tonic release.

The identifying synthetic benchmark includes movement, acceleration, pupil,
theta, and engagement regressors. Several are deliberately correlated with the
computational candidates. Successful recovery therefore requires nuisance
separation, not merely matching an isolated synthetic waveform.

## 6. Partial pooling

For each candidate and sensor-grid point, the regression contains

- one global signal coefficient;
- fixed nuisance coefficients;
- subject-specific intercept deviations;
- subject-specific signal-slope deviations.

Subject deviations receive a Gaussian ridge penalty. In mixed-model language,
this is a penalized best linear unbiased prediction calculation with a fixed
penalty rather than a fully integrated hierarchical posterior. Test subjects
must have training sessions, but test sessions themselves are never used to fit
candidate coefficients.

Session offsets are handled differently. Each session contains a pre-task
baseline segment with no candidate event. Its mean is subtracted from the
response and nuisance regressors. For held-out sessions, no task sample is used
to estimate this offset.

## 7. Calibration-only timescale posterior

The sensor and tonic timescales are represented by a discrete grid

\[
\lambda_g=(\tau_{r,g},\tau_{d,g},\rho_g).
\]

The default grid contains 27 points:

- \(\tau_r\in\{0.25,0.40,0.60\}\) seconds;
- \(\tau_d\in\{1.00,1.60,2.40\}\) seconds;
- \(\rho\in\{0.92,0.97,0.99\}\).

A known exogenous calibration event train is presented in training sessions.
For each grid point, the calibration block is fitted under the conditional AR(3)
likelihood. With a uniform grid prior,

\[
\log w_g
=
\log p(z_{\mathrm{cal,train}}\mid\lambda_g)
-
\operatorname{logsumexp}_{g'}
\log p(z_{\mathrm{cal,train}}\mid\lambda_{g'}).
\]

No task sample, held-out session, or candidate identity enters this posterior.
The tests explicitly modify the held-out task trace and verify that calibration
weights remain bitwise unchanged.

## 8. Strict train/test protocol

The complete protocol is:

1. **Training calibration only:** infer \(w_g\) over sensor and tonic timescales.
2. **Training task only:** fit candidate, nuisance, and subject coefficients for
   every \((k,g)\).
3. **Held-out task only:** evaluate conditional predictive log likelihood.
4. **Held-out pre-task baseline only:** estimate the held-out session offset.

For candidate \(k\), the final held-out score marginalizes the calibration grid,

\[
\log p(z_{\mathrm{test}}\mid k)
=
\operatorname{logsumexp}_g
\left[
\log w_g
+
\log p(z_{\mathrm{test}}\mid k,\lambda_g,
\widehat{\boldsymbol\gamma}_{k,g})
\right],
\]

where \(\widehat{\boldsymbol\gamma}_{k,g}\) denotes the training-only ridge
coefficient estimate. Candidate ranking is based on this held-out marginal
score, not on deconvolved peak correlation or in-sample fit.

## 9. Default recovery benchmark

The default benchmark uses

- 6 subjects;
- 5 sessions per subject;
- 3 training and 2 held-out sessions per subject;
- 112 calibration samples and 144 task samples per session;
- \(\Delta t=0.2\) seconds;
- true \(\tau_r=0.4\) seconds, \(\tau_d=1.6\) seconds, and \(\rho=0.97\).

Each of the seven candidate signals is used once as the true phasic generator,
while the belief trajectory, tonic process, nuisance traces, subject effects,
session offsets, and calibration input are otherwise shared.

For seed 7, version 0.4 obtains:

- 7/7 generating signals recovered;
- median held-out evidence margin: 771.532 log units;
- minimum evidence margin: 421.096 log units;
- calibration MAP exactly equal to the true grid point;
- median nuisance-coefficient mean absolute error: 0.00749;
- median subject-signal correlation: 0.99769;
- maximum absolute correlation between sensor-convolved candidates: 0.88880.

High candidate correlation is intentional. The benchmark asks whether strict
forward modeling can recover the generator despite shared event structure,
measurement smoothing, tonic fluctuations, and nuisance confounding.

## 10. Identifiability and limitations

### Calibration is essential

Without an exogenous input, the roots associated with indicator decay and tonic
persistence can trade off. A slowly varying trace alone does not identify which
part arose from \(\tau_d\) and which from \(\rho\). The calibration block is
therefore part of the generative design, not an optional visualization aid.

### Conditional likelihood

The likelihood conditions on the first three samples of each fitted segment.
Those samples are not scored. This avoids imposing an incorrect stationary
initial-state density after baseline subtraction.

### Discrete rather than continuous uncertainty

Timescale uncertainty is represented by a finite grid. A real-data analysis
should expand the grid, verify posterior stability, and compare against a
continuous parameter treatment.

### Plug-in regression coefficients

The calibration grid is marginalized, but candidate, nuisance, and subject
coefficients are plug-in Gaussian-ridge estimates. A fully Bayesian hierarchical
analysis would integrate those coefficients and their variance components.

### No independent white sensor-noise term

Version 0.4 assigns residual stochasticity to latent tonic-release innovations.
It does not add an independent white observation-noise term after the indicator.
Adding that term requires a linear-Gaussian state-space likelihood or equivalent
structured covariance calculation and is a priority for real photometry.

### Synthetic evidence only

The recovery results establish that the proposed experimental and statistical
pipeline can distinguish its own candidate generators under the documented
simulation. They are not biological evidence that ACh encodes any candidate.
