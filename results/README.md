# Results

Generated benchmark outputs are ignored by default.

Create local evidence with:

```bash
bayesian-ach dissociate --output results/dissociation
bayesian-ach benchmark --output results/benchmark
```

A result should be committed only when it is intentionally designated as a
versioned evidence artifact and accompanied by configuration, random seed,
source revision, and checksums.
