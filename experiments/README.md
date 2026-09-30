# Experiment notebooks

These are the original Jupyter Notebooks used for the research analyses.

- `RQ1_Paired_ZTest`: paired one-tailed Z-tests applied to model-generated
  `pred_test.csv` files. RQ1 reconstruction MAE itself is generated during
  model evaluation.
- `RQ2_Diagnostic_Alignment`: Spearman analyses for base, implicit, and
  explicit mastery representations where the model provides those branches.
- `RQ3_Cross_Assessment`: item-wise Ridge regression with 30 repeated
  user-level holdouts.
- `RQ3_TTest`: the original statistical comparison notebooks for RQ3 outputs.

Start Jupyter from the repository root. Edit only each notebook's path/config
cell: every required input uses an explicit `REPLACE_WITH_...` placeholder, so
readers can see whether the notebook expects a theta CSV, score CSV, prediction
CSV, Q-matrix, or result directory. Paths may be absolute or relative to the
Jupyter working directory.

Raw course data and learner attempts are intentionally not distributed. The
files in `../data/example/` are synthetic input-schema examples, not paper
results. Theta files are model outputs: first train the selected model, then
run its diagnosis script separately on CP2 and CP4 evidence before starting
the RQ2 or RQ3 notebooks. Each model README documents the exact commands.
