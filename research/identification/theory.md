# What can a finite task identify under an unknown neural encoding?

## Scope and model

Let a finite task support comprise conditions `i=1,...,n`, and let candidate `f` assign scalar predictions `f_i`. A condition can include outcome probability, confidence, cue identity, context, and trial history. The candidate is evaluated at fixed parameters; alternatively its latent parameters must be constrained by an independent behavioral analysis. Allow a *single*, candidate-specific, strictly increasing encoding `h` shared by all conditions. There is no constraint on its offset, dynamic range, or curvature. Before measurement, the possible mean vectors are

`C_f = { z : z_i=h(f_i), h strictly increasing on the candidate range }`.

For fixed, known linear event-to-sample/sensor operator `H` and nuisance matrix `B`, the measured family is

`M_f = { H z + B beta : z in C_f, beta unrestricted }`.

The noise distribution must also be shared between compared candidates. The statements concern the conditional mean family, not arbitrary hypothesis-specific residual models. The shared-across-conditions link is crucial: an unrestricted condition-specific encoding removes these constraints. An increasing orientation is also substantive. Unknown sign requires allowing order reversal; nondecreasing links with flat responses give all candidates at least a common constant-response submodel.

## Proposition 1: finite-support equivalence is exactly equality of weak order

The following are equivalent:

1. `sign(f_i-f_j)=sign(g_i-g_j)` for every pair of conditions, including equality.
2. There is a strictly increasing reparameterization `phi` on the observed levels of `f` with `g_i=phi(f_i)`.
3. `C_f=C_g`.
4. `C_f` and `C_g` have a common vector.

Thus two candidates either yield exactly the same premeasurement family or have disjoint premeasurement families under these assumptions. The latter statement does *not* imply a positive statistical separation margin.

Proof: a strictly increasing map preserves all inequalities and equalities, proving necessity and `(4) -> (1)`. For sufficiency, assign each distinct observed level of `f` the corresponding level of `g`. Equality of tie classes makes the assignment well-defined and equality of ordering makes it strictly increasing. Piecewise-linear interpolation (with positive-slope extensions outside the observed range, if needed) supplies `phi`. Composing either candidate's arbitrary increasing encoding with `phi` or its inverse proves equality of the two families. A one-level support is included: any value can be assigned to that single level.

### Minimal witnesses and what they mean

If the weak orders differ, some *two-condition* witness suffices: either opposite nonzero differences (order reversal), or equality for one candidate and a nonzero difference for the other (tie split). These are necessary and sufficient witnesses of a structural distinction on the given support. A correlation smaller than one is not the relevant condition under arbitrary monotone links: two nonlinearly related candidates can have imperfect Pearson correlation but the same entire family of encoded means.

The witness is against a *pair* of candidates. More conditions may be required to separate a candidate collection. With two conditions there are only three weak-order signatures (`<`, `=`, `>`), so at most three increasing-link classes can be separated. With one condition there is only one class.

Floating-point or empirically estimated near-equality is not an exact tie. `support_audit.py` audits the supplied numeric values and flags near differences separately. An experimental tie claim requires matching/estimating the relevant latent quantity, uncertainty analysis, and a prespecified equivalence tolerance; nonsignificance is not an equality certificate.

## Consequences for the six Table 1 quantities

Write `q=p_j` for the probability of the observed outcome and `a=sum(alpha)` for Dirichlet concentration. In the *binary* case, after observing category `j`,

`I = sqrt(2)(1-q)`,

`S = -log(q)`,

`G = 1/(a+1)`,

`U = sqrt(2)(1-q)/(a+1)`,

`K = KL[Dir(alpha+e_j) || Dir(alpha)] = -log(q) + psi(aq+1) - psi(a+1)`.

For a *one-step, fixed-hazard* stable/reset mixture with fixed reset predictive probability `r_j>0` and `0<hazard<1`,

`R = hazard*r_j / [(1-hazard)*q + hazard*r_j]`.

The last expression is not a general identity for a history-dependent reset filter with changing hazard or reset predictive distribution. That broader model can leave the equivalence class.

### Fixed-confidence binary supports collapse five candidates into one class

At fixed `a`, each of `I,S,U,K,R` is strictly decreasing in `q`; `G` is constant. For `K`, differentiate:

`dK/dq = -1/q + a*psi_1(aq+1) < 0`.

The inequality follows from `psi_1(x+1)=sum_{k=1}^infinity 1/(x+k)^2 < integral_x^infinity t^-2 dt = 1/x`, for `x>0`. All other derivatives are immediate. Consequently, arbitrarily many repetitions of such conditions cannot distinguish the five nonconstant candidates if their neural encoding can be any increasing function. A preference between their linear fits is a preference within the *linear-encoding* model class, not encoding-invariant identification of the latent computational quantity.

### Confidence variation can break this equivalence, but not every ambiguity

Across binary conditions with `q` and `a` varied independently, `I,S,R` remain in one class under the fixed-reset assumptions. `G,U,K` can each be separated from it and from each other. At matched `q`, `G,U,K` all decrease with `a`; a confidence contrast alone consequently need not distinguish these three.

For completeness, `K` decreases with `a` because

`dK/da = q*psi_1(aq+1)-psi_1(a+1) = [F(aq)-F(a)]/a < 0`,

where `F(x)=x*psi_1(x+1)` is strictly increasing on `x>0`. One direct proof uses

`F(x)=integral_0^infinity u*exp(-u) / [x*(exp(u/x)-1)] du`.

For each fixed `u>0`, the denominator strictly decreases with `x`: `(exp(v)-1)/v` strictly increases in `v>0`. The integrand, and hence the integral, strictly increases.

### A minimal three-condition support separates the four attainable classes

Consider the three conditions below. The numeric values are deterministic theoretical predictions, not empirical estimates.

| Condition | q | a | I | G | U | K |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.8 | 5 | 0.28284271 | 0.16666667 | 0.04714045 | 0.02314355 |
| B | 0.1 | 50 | 1.27279221 | 0.01960784 | 0.02495671 | 0.08671309 |
| C | 0.8 | 50 | 0.28284271 | 0.01960784 | 0.00554594 | 0.00248125 |

The increasing-link signatures are:

- `I,S,R`: `A=C<B`;
- `G`: `B=C<A`;
- `U`: `C<B<A`;
- `K`: `C<A<B`.

These are four distinct weak orders. Three conditions suffice; the three-signature bound above proves that two cannot suffice for four classes. This is minimality of *support cardinality under the stated binary/fixed-reset model*, not an optimal animal allocation, a power calculation, or a guarantee that animals possess the intended `q,a` values.

The distinguishing `U` versus `K` comparison is not a curvature/linear-fit effect: conditions A and B reverse their ordering. Here `K_A=log(5/4)-1/5`, `K_B=log(10)+H_5-H_50`, and `K_C=log(5/4)+H_40-H_50`, with `H_m` the harmonic number.

### The binary and fixed-reset boundaries are experimentally meaningful

For three or more outcomes, `I = sqrt[(1-q)^2 + sum_{k != j}p_k^2]` need not be a function of `q` alone. At `p=(0.5,0.49,0.01)` versus `p=(0.5,0.25,0.25)`, observing category 1 yields the same surprise and (at matched `a`) the same Dirichlet KL, but different innovations (`sqrt(0.4902)` versus `sqrt(0.375)`). This is an equal-probability tie witness separating innovation from surprise without committing to a neural response curve. Similarly, changing independently established reset hazard or reset predictive probability at matched `q` can separate the restricted reset posterior from surprise. Both contrasts need checks that their other manipulations do not independently change the neural measurement.

An extended three-condition construction separates all six declared scalar quantities when these dimensions can vary independently. With observed category 1 and fixed reset predictive `r_j=1/3`, let A have `p=(.8,.1,.1),a=5,hazard=.1`, B have `p=(.1,.45,.45),a=50,hazard=.1`, and C have `p=(.8,.19,.01),a=50,hazard=.8`. Then the orders are `A<C<B` for innovation, `A=C<B` for surprise, `B=C<A` for gain, `C<B<A` for update, `C<A<B` for information gain, and `A<B<C` for the one-step reset posterior. Since two conditions permit only three signatures, three is also cardinality-minimal for these six quantities on this enlarged support. This constructive scalar-model result does not establish that the latent quantities can be independently manipulated in an animal, or that a history-dependent reset model has been identified. It also assumes an increasing orientation: in this particular construction the update and reset orders reverse, so allowing each an unrestricted response sign restores their equivalence on this support.

## Proposition 2: the sensor/nuisance operator defines a second quotient

Let `P` project onto the orthogonal complement of `col(B)` and let `A=P H`. For a known common `H,B`,

`M_f` and `M_g` overlap iff there exist `z in C_f`, `w in C_g` with `A(z-w)=0`.

They are identical iff `A C_f = A C_g`. Equal premeasurement weak orders therefore imply observational equivalence for *every* common sensor/nuisance operator. The converse need not hold: aggregation or nuisance removal can erase an order/tie witness. For example, `f=(0,1)` and `g=(1,0)` are separated before measurement, but `H=(1,1)` observes only their summed response and makes their unrestricted increasing-link mean families identical. The same happens with `H=I` and nuisance direction `B=(1,-1)^T`. An intercept-only nuisance does not erase this reversal.

A sufficient condition for all premeasurement distinctions to survive is `ker(A) subset span(1)`: equality after measurement then implies latent vectors differ only by a common offset, which cannot change their weak order. Full column rank of `H` is sufficient with no nuisance. A slow but full-rank sensor need not destroy exact structural identification; it can instead make the inference arbitrarily ill-conditioned. Fixed invertible filtering and non-injective temporal aggregation must not be conflated.

### A finite linear feasibility diagnostic

Let `T_f` assign each condition to its distinct ordered candidate level, so `z=T_f u`, and similarly `w=T_g v`. Overlap is equivalent to

`A(T_f u - T_g v)=0`, `u_(k+1)-u_k>0`, `v_(l+1)-v_l>0`.

Because these equalities and strict inequalities are homogeneous, any feasible solution can be rescaled so that every adjacent gap is at least 1. Thus linear feasibility with those non-strict lower bounds characterizes the overlap exactly. Constant candidates have no adjacent-gap constraints. For numerical stability, the implementation solves the equivalent direct equation `H(T_f u - T_g v) + B gamma = 0` with free nuisance coefficients, globally normalizing `H` and individually normalizing columns of `B`. It does not form and then normalize a projected residual: cancellation roundoff from a sensor lying entirely in the nuisance span could otherwise become a spurious identifying constraint. The HiGHS/SciPy check is numerical, with explicit equality/gap verification; it is not an exact arithmetic certificate. With bounded dynamic range, known link slopes, or an unknown fitted sensor, use the corresponding constrained/union model instead of this homogeneous program. Separate fitting of kernel parameters for each candidate takes a union over sensor operators and can create additional equivalences.

## Proposition 3: structural distinction does not give uniform finite-noise power

Even when the two premeasurement families are disjoint, they have zero separation margin without a lower response-scale constraint. For any candidate vector `f`, `h_epsilon(x)=b+epsilon*x` is strictly increasing for every `epsilon>0`, and its encoded vector tends to the common constant `b*1` as `epsilon -> 0`.

For shared Gaussian noise covariance `sigma^2 I`, two such encoded means give

`KL[N(H(b*1+epsilon*f),sigma^2 I) || N(H(b*1+epsilon*g),sigma^2 I)] = epsilon^2 ||H(f-g)||^2/(2 sigma^2) -> 0`.

Consequently their total variation distance tends to zero. Any level-alpha discrimination test has worst-case power no greater than alpha over unrestricted increasing links at any fixed finite sample size. Replication can identify fixed nonzero effects, but cannot supply a positive *uniform* power guarantee over effects allowed to approach zero. Power and robust-design claims therefore require an independently justified minimum neural dynamic range/slope or an explicit effect-size class, in addition to witness-support and measurement checks.

## Implication for the present paper

The positive result is a support-level diagnostic and constructive witness design, not a claim that existing ACh recordings have identified one computation. More observations on the same equivalent support cannot repair Proposition 1. Better measurement helps only ambiguities introduced by Proposition 2. More repetitions help finite-noise uncertainty only after an effect-size class has been stated. The existing linear-response recovery studies remain valid for their stated encoding model; they must not be interpreted as general six-way biological identification.

These finite-support results are elementary order/linear-algebra consequences. The intended substantive contribution is their joint application to the exact Table 1 candidates, the minimal three-condition separation result and its boundaries, and a reproducible audit of the supports and measurement assumptions used in this ACh framework—not a claim to have invented monotone invariance or structural identifiability.
