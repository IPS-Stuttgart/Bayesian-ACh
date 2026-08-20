# Results

Generated benchmark outputs are ignored by default.

Create local evidence with:

```bash
bayesian-ach dissociate --output results/dissociation
bayesian-ach benchmark --output results/benchmark
bayesian-ach regime-benchmark --output results/regime-recovery
bayesian-ach observation-benchmark --output results/observation-attribution
bayesian-ach measurement-benchmark --output results/measurement-recovery
bayesian-ach closed-loop-benchmark --output results/closed-loop-recovery
```

A result should be committed only when it is intentionally designated as a
versioned evidence artifact and accompanied by configuration, random seed,
source revision, train/test split provenance, and checksums.
