# Supported-set diagnostic: analysis contract

This is a new methods diagnostic following the already observed frozen stress
failure. It does not alter or replace the old analyses, and it is not a
preregistration of those analyses. This specification is written before this
diagnostic's simulations are evaluated.

## Fixed scope and endpoint

Use the six Table 1 signals, globally standardized over the existing 240-point
transition grid. Reuse the immutable N=60 heuristic allocation, and retain
independent Gaussian noise with known standard deviation 1 and signal amplitude
1. The matrix is fixed before observing neural responses. An intercept is a
nuisance regressor. No sensor convolution is applied in this specific
simulation; the algorithm accepts already convolved/whitened designs, and
separate algebraic tests check that measurement operators can induce aliasing.

Allowed classes are the null, the six nonnegative pure rays, and all 15 positive
two-candidate mixtures. Each mixture has coefficient fraction w in [0.25, 0.75]
in the declared full-grid standardized units, and any nonnegative total
amplitude. The lower fraction is a substantive identifiability restriction,
not a fitted quantity. The simulated mixtures use w=0.25, 0.50 and 0.75; all
three weights are evaluated, without choosing the most favorable afterward.

After nuisance projection, construct one simultaneous 95% confidence ball for
the mean in the span of all six projected signals using the exact chi-square
quantile and known Gaussian covariance. Retain every class whose mean set
intersects that ball. Declare a pure code only when exactly one pure ray is
retained, no mixture is retained, and the null is excluded. Additionally reject
adequacy of the candidate span at a separate 5% chi-square residual threshold;
such rejection can only remove pure calls. No generator-specific power tag is
used to make a decision on an observation of unknown origin.

The analytic target is at most 5% probability of certifying purity under any
member of the declared continuous mixture class, not only under the tested
weights. The Monte Carlo audit is descriptive, not the proof. Report correct
pure certification, false pure certification, true-class support, support-set
size, null retention and none-adequate frequencies. Do not interpret retained
classes as posterior probabilities or verified mechanisms.

## Locked numerical settings

- PRNG: the explicit xorshift32 implementation in supported_set.js, Box-Muller
  normal transform; stream keyed by seed, generator and replicate.
- Evaluation seed: 20260930; 2,000 replicates per generator (six pure, 45 mixture,
  one null); no data-dependent stopping or threshold adjustment.
- alpha=0.05 for the common confidence ball; beta=0.05 for the separate
  full-span lack-of-fit check; mixture fraction floor eta=0.25.
- Additional descriptive budget sensitivity: exact repetitions of the same
  allocation by factors 1, 4 and 16 (N=60, 240 and 960). These are independent
  ideal Gaussian samples, not animals, time bins or an executable protocol.
- Evaluate the existing fixed nonlinear probe as a descriptive boundary only.
  No universal guarantee against arbitrary unknown alternatives is claimed.

## Source lock

Existing allocation: source commit 1b2028929ac6ebc1cce0882f0c22af9918044342,
original CSV SHA-256 a823be49faf6c6cbebf60b11d4b5ca895cf7734d6e9c577ee98f97a5907b69b2.
The 15 nonzero heuristic counts are copied verbatim from
results/design-open-set-stress-n60/allocations.csv at repository commit
5dc8c9280c5d3e042ce25d926c0d856ed9776ad9. This follow-up uses fresh random
streams and an analytically set threshold; the old calibration is untouched.
