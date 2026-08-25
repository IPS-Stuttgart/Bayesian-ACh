# Paired heuristic-versus-certified recovery diagnostic

The exact integer certificate maximizes the declared finite-grid worst ordered-pair
residual. It does not follow that the certified allocation must outperform the earlier
heuristic allocation in a finite held-out recovery simulation.

`bayesian-ach-design-paired-recovery` therefore compares the chronologically locked
heuristic N=60 allocation with the exact certified N=60 allocation. The comparison uses
the same five seeds (`7, 11, 19, 23, 31`), 200 replicates per generating candidate and
seed, a 0.35 held-out fraction, unit effect and noise scales, and an identically reset RNG
stream for both allocations. Both source allocations are checksum-bound; the heuristic
constructor and integer certificate are independently reverified before simulation.

`bayesian-ach-design-paired-recovery-verify` rechecks every output checksum, both source
allocations, certificate geometry, all 60 recovery rows, five seed summaries, and every
headline allocation/recovery diagnostic by rerunning the deterministic simulation.

This is a diagnostic sensitivity analysis. It does not replace the chronologically locked
paper allocation, select a post-result schedule, establish empirical superiority, guarantee
recovery power, or specify a physical trial, time-bin, or animal protocol.
