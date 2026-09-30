# Data

The course dataset used in the paper is not distributed in this repository.
It contains learner assessment records and submitted source code. Obtain the
appropriate institutional approval and anonymize learner identifiers before
using or sharing comparable data.

The files in `data/example/` are small, fully synthetic examples created only
to document the required schemas. They are **not** observations from the paper,
must not be used to reproduce the reported results, and are too small for a
meaningful model comparison.

## Synthetic example files

| File | Purpose |
|---|---|
| `train.csv`, `valid.csv`, `test.csv` | Example response splits for the model workflows |
| `item_text.csv` | Example assessment prompts |
| `attempts.csv` | Example learner submissions for the full model only |
| `q_matrix.csv` | Example 18-skill Q-matrix for model input |
| `mid_scores.csv`, `final_scores.csv` | Example downstream assessment scores |
| `q_matrix_mid.csv`, `q_matrix_final.csv` | Example Q-matrices for RQ2 target assessments |

The notebook configuration cells deliberately use names such as
`REPLACE_WITH_CP2_THETA_CSV`. Replace each placeholder with an explicit local
path, for example:

```python
THETA_CP2_PATH = Path("GLLMCDM/full_model/example_outputs/CP2/theta_new.csv")
```

Paths may be absolute or relative. Relative paths are resolved from the
directory in which Jupyter was started; starting Jupyter at the repository
root is recommended.

## Required schemas

Response logs use long format:

```text
user_id,item_id,score
```

Scores must be numeric values in `[0, 10]`.

Item text files use:

```text
item_id,item_text
```

The full-model attempt file uses:

```text
user_id,item_id,attempt_clean
```

Q-matrices can be NumPy arrays of shape `(n_items, n_skills)` or CSV files
with `item_id` followed by skill columns. The experiments in the paper use 18
skills.

The CP2 and CP4 mastery files used by RQ2 and RQ3 are generated outputs, not
input data. Run the appropriate model's diagnosis script after training. The
generated CSVs contain one row per learner with `user_id` and `K0` through
`K17`. Synthetic output examples live inside each model directory under
`example_outputs/`. Mid and Final target files use the response-log schema
above.
