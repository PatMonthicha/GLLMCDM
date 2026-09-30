# G-NCDM-poly baseline

This directory contains the continuous-score G-NCDM baseline used for S1,
strict S2, and strict S3 comparisons. The implementation is copied from the
research working directory `GNCDM_reg_me_datanew`.

## Files

- `model.py`: G-NCDM architecture and open-world diagnostic branches.
- `train.py`: datasets, training, validation, cold-start evaluation, and
  mastery export.
- `run.py`: S1/S2/S3 data preparation and the full train/test workflow.
- `diagnose.py`: S1/S2 post-training mastery export for evidence expressed in
  the model's seen-item coordinate system.
- `diagnose_s3.py`: strict-S3 mastery export for unseen-item evidence using the
  protocol described in the paper.
- `model_parser.py`: command-line options.
- `tools.py`: logging and training utilities.
- `example_outputs/`: synthetic examples of generated base-mastery CSVs.

## Run

```bash
python baselines/gncdm_poly/run.py \
  --eval_setting S1 \
  --train_file path/to/train.csv \
  --valid_file path/to/valid.csv \
  --test_file path/to/test.csv \
  --Q_matrix path/to/full_q_matrix.npy \
  --save_path outputs/gncdm_poly/S1 \
  --n_user 460 --n_item 15 --n_know 18 \
  --user_dim 32 --item_dim 32 --alpha 0.5 \
  --training_config baselines/gncdm_poly/config/training_config_Exp_200epoch.json \
  --n_epoch 200 --use_early_stopping \
  --early_stop_patience 5 --early_stop_min_delta 0.0 \
  --monitor_metric rmse
```

Change `--eval_setting` to `S2` or `S3` for the corresponding strict protocol.
For those settings, training entities are derived from the training split by
the original workflow.

## Export diagnosed mastery for RQ2 and RQ3

For S1 or S2, diagnose CP2 and CP4 separately using the matching trained
checkpoint:

```bash
python baselines/gncdm_poly/diagnose.py \
  --evidence_file path/to/test_CP2_responses.csv \
  --model_path outputs/gncdm_poly/S2/params_32_32.pt \
  --output_path outputs/diagnosis/gncdm_poly/S2/CP2 \
  --profile_label CP2

python baselines/gncdm_poly/diagnose.py \
  --evidence_file path/to/test_CP4_responses.csv \
  --model_path outputs/gncdm_poly/S2/params_32_32.pt \
  --output_path outputs/diagnosis/gncdm_poly/S2/CP4 \
  --profile_label CP4
```

S3 requires its dedicated script because unseen items cannot be inserted into
G-NCDM's training-item-indexed implicit input. The script uses the mean
training implicit mastery and learner-specific explicit evidence, matching the
paper's S3 protocol:

```bash
python baselines/gncdm_poly/diagnose_s3.py \
  --train_file path/to/S3_train.csv \
  --evidence_file path/to/test_CP2_responses.csv \
  --model_path outputs/gncdm_poly/S3/params_32_32.pt \
  --Q_matrix path/to/full_q_matrix.npy \
  --output_path outputs/diagnosis/gncdm_poly/S3/CP2 \
  --profile_label CP2
```

Repeat the S3 command with the CP4 evidence and `--profile_label CP4`. These
workflows produce `theta.csv`/`theta_base.csv`, `theta_implicit.csv`, and
`theta_explicit.csv`, together with NumPy and audit files.

The upstream source carries restrictive notices. Consult
`../../THIRD_PARTY_NOTICES.md` before redistribution.
