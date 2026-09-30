# Post-run numerical robustness correction

An independent implementation review found unit-scale sensitivity outside the
locked unit-noise experiment: an absolute numerical-rank cutoff and absolute
decision tolerances could erase informative directions when all measurement
and noise units were jointly rescaled. This was conservative in the explicit
counterexample but inappropriate for a general known-covariance interface.

The corrected implementation divides design and observations by the supplied
known noise SD before computing inference. Numerical-rank and decision checks
are relative. Clearly unresolved near-rank-deficient and nearly collinear cone
calculations stop explicitly instead of silently advertising a certificate.
Exact degeneracies are handled, subject to ordinary floating-point precision.
This numerical implementation is not an exact arithmetic rank certificate.

New tests jointly rescale measurements and noise by 1e-12 through 1e12, retain
small independently informative columns, and reject unresolved near-degeneracy.
The frozen experiment uses sigma=1 and well-separated finite candidate columns;
its full result is required to remain byte-identical. The original source
manifest is retained as results/original_manifest.json. No simulation setting,
endpoint, threshold level, seed, replicate count or frozen result has changed.

Repository integration subsequently formatted the independent Python verifier,
wrapped its embedded JavaScript declaration and marked unused grid-loop names
with underscores to satisfy repository lint rules. This is formatting-only:
the check order, generated numerical inputs and verification results are
unchanged. The current manifest records the verifier's updated source hash;
the original manifest and frozen result bytes remain unmodified.
