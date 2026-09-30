# GLLMCDM

Research code for **Learning to Assess Learner Skills with LLM-Based Scoring
in Cognitive Diagnostic Models**.

This repository contains the original model and experiment code used in the
project. The public layout changes file locations and documentation only; it
does not replace the research workflows with newly written implementations.
See `SOURCE_PROVENANCE.md` for the source of every model directory.

## Repository layout

```text
GLLMCDM/
├── GLLMCDM/
│   ├── full_model/          # GLLMCDM prompt with learner attempt
│   └── without_attempt/     # Ablation prompt without learner attempt
├── baselines/
│   ├── gncdm_poly/          # G-NCDM continuous-score baseline
│   └── ncdm_poly/           # NCDM continuous-score baseline
├── experiments/
│   ├── RQ1_Paired_ZTest/
│   ├── RQ2_Diagnostic_Alignment/
│   ├── RQ3_Cross_Assessment/
│   └── RQ3_TTest/
└── data/README.md
```

The RQ1 reconstruction MAE is produced by each model's prediction workflow.
The additional RQ1 notebooks perform the paired one-tailed Z-tests on the
saved `pred_test.csv` files. RQ2 and RQ3 remain in their original Jupyter
Notebook form.

RQ2 and RQ3 require diagnosed CP2 and CP4 mastery profiles. These theta files
are outputs of each trained model, not raw data. Use `diagnose.py` in each
GLLMCDM directory, and `diagnose.py`/`diagnose_s3.py` in the G-NCDM directory,
before running the downstream notebooks. Synthetic output examples are stored
under each model's `example_outputs/` directory.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For an LLM-backed GLLMCDM run, copy `.env.example` into the selected model
directory and add your own key:

```bash
cp .env.example GLLMCDM/full_model/.env
```

Never commit `.env`. See the README inside each model directory for the exact
input schemas and commands.

## Experiments

Start Jupyter from the repository root:

```bash
jupyter lab
```

Every experiment notebook begins with a path-configuration cell. Replace the
`REPLACE_WITH_...` values with paths to your own authorized inputs. This makes
the required artifact explicit without assuming a particular private folder
layout. The paths may be absolute or relative to the directory from which
Jupyter was started.

The private course dataset and learner submissions are not included. Follow
`data/README.md` when supplying authorized, anonymized data. The repository
includes schema-only synthetic examples under `data/example/`; they are not
the data used for the paper.

## Third-party notice

The G-NCDM baseline is adapted from Generative Cognitive Diagnosis. Its source
files retain restrictive upstream notices, and the upstream repository does
not provide a redistribution license. Read `THIRD_PARTY_NOTICES.md` before
publishing this repository publicly.

## Citation

Citation metadata is provided in `CITATION.cff` and should be completed with
all authors and final publication information before release.
# GLLMCDM
