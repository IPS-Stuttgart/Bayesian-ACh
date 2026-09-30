# Composite-code identification work package

Completed local methods diagnostic: analytic bounded-family false-pure control,
verified reference implementation, independent numerical checks and a frozen
318,000-replicate simulation audit. Existing frozen stress and paired-recovery
results are not changed or retuned.

Read `theory_and_result.md` for scientific claims and limitations, `protocol.md`
for the specification written before simulation, and `manuscript_insert.tex`
for integration text (check the target manuscript's candidate-table label).

From the repository root, run:

```sh
node research/composite_codes/test_supported_set.js
python3 research/composite_codes/independent_verify.py
node research/composite_codes/verify_supported_set.js
```

The new procedure issues a certificate only relative to its explicitly bounded
family. The low false-pure rate has a substantial pure-certification cost:
.0060 worst pure at N60, .1375 at N240, .7770 at N960. These are ideal independent
observations under known Gaussian covariance, not real-data mechanism findings.
