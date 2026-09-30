# Identification result — working package

This package develops the finite-support identification result for the Bayesian-ACh manuscript. It does not claim a biological winner. See `theory.md` for assumptions and proofs and `support_audit.py` for the numerical diagnostic.

## Contents

- `theory.md`: assumptions, three propositions and proofs, minimal support constructions, and scope limits.
- `identification_section.tex`: integration fragment for the manuscript (not a standalone document). Check the destination's Table 1 label; the draft uses `tab:candidates`.
- `support_audit.py`: standard-library weak-order auditor and canonical candidate examples; optional NumPy/SciPy measured-family overlap program.
- `test_support_audit.py`: deterministic correctness tests, not empirical significance tests.

## Reproduce

From this directory:

```sh
python3 -m unittest -v test_support_audit.py
python3 support_audit.py --demo
python3 support_audit.py path/to/candidate_columns.json
```

Input JSON maps candidate names to equal-length numeric arrays, optionally under a `candidates` key. Conditions/trials must be aligned and computed at the same declared latent parameters. The demo includes fixed-confidence collapse, matched-confidence support, four-class and six-quantity minimal three-condition constructions, and the multi-outcome boundary.

The rank audit assumes a shared strictly increasing encoding per candidate. It never interprets near differences as exact ties. The optional measured-overlap function assumes a fixed known sensor operator, unrestricted linear nuisance, and unbounded link dynamic range; an unknown sensor or bounded links require an expanded model. The LP retains free nuisance coefficients directly, with one global sensor normalization and columnwise nuisance normalization, avoiding projection-cancellation artifacts. HiGHS feasibility is checked numerically, not certified symbolically.

## Verification

The package was tested with Python 3.14.4 in the current workspace. All 27 deterministic tests passed, including 13 measured-family tests using NumPy/SciPy. The tests verify the candidate equivalence partitions, exact reference KL/digamma values, both three-condition witnesses, temporal aggregation and nuisance counterexamples, and invariance to measurement-unit scaling. Regression cases cover sensors lying exactly in constant and nonconstant nuisance spans, joint sensor/nuisance rescaling, and a zero nuisance column. No new biological data were accessed by this package.
