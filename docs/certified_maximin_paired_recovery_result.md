# Frozen paired N=60 recovery result

The checksum-bound paired diagnostic was produced from clean commit
`c6dea54d1e8a1b702c212c83535af8a5de3ab3ea`. It compares the chronologically locked
heuristic allocation (SHA-256 `a823be49faf6c6cbebf60b11d4b5ca895cf7734d6e9c577ee98f97a5907b69b2`)
with the exact certified allocation (SHA-256
`694a84107c1ba94f39146a05675def1b5813621fab486d7567f6711510a6fc08`).

The schedules are materially different: allocation L1 distance 100, support 15 versus 12,
support overlap 4, and maximum absolute cell change 9. With identical random streams for
five seeds and 200 replicates per generating candidate, the minimum recovery rate changed
from 0.750 for the heuristic schedule to 0.665 for the certified schedule. The certified
schedule's minimum was lower for every seed.

This result is a negative sensitivity boundary. The certificate remains an exact statement
about the declared asymptotic worst-residual objective; it is not evidence that the certified
schedule is superior under finite recovery. The diagnostic does not replace the locked
heuristic schedule, select a post-result design, guarantee power, or define a physical
trial, time-bin, or animal protocol.

The frozen evidence is in `results/certified-maximin-design/paired-recovery/`. Its verifier
recomputes both inputs, every recovery row, all headline diagnostics, and every checksum.
