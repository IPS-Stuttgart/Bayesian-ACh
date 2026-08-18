# Real-data contract

Bayesian-ACh does not vendor experimental data. A real-data adapter should map
source-specific files to one trial table and, optionally, one continuous sensor
table.

## Trial table

Required columns:

| Column | Type | Meaning |
|---|---|---|
| `subject_id` | string | Stable pseudonymous subject identifier |
| `session_id` | string | Session identifier |
| `trial_id` | integer/string | Trial identifier within session |
| `time` | float | Transition time in seconds |
| `state` | integer | Inferred or experiment-defined current state |
| `action` | integer | Action or policy input |
| `next_state` | integer | Observed next state |
| `split` | string | `train`, `validation`, or `test` |

Recommended columns include context cue, transition-kernel condition, sensory
reliability manipulation, movement speed, acceleration, pupil, theta power, and
behavioural choice.

## Continuous ACh table

| Column | Type | Meaning |
|---|---|---|
| `subject_id` | string | Subject identifier matching the trial table |
| `session_id` | string | Session identifier |
| `time` | float | Sample time in seconds |
| `ach_signal` | float | Preprocessed but not hypothesis-regressed ACh measurement |
| `reference_signal` | float/optional | Isosbestic or reference channel |

## Provenance manifest

Every adapter should record:

- public accession or controlled-access identifier;
- license and usage restrictions;
- source checksums;
- preprocessing code revision;
- excluded sessions and reasons;
- time-base transformations;
- train/test split before model fitting.

Direct identifiers and restricted raw data must never be committed.
