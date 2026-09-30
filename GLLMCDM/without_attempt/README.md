# GLLMCDM without learner attempts

This is the no-attempt ablation used in the paper. It uses a distinct prompt
that excludes learner-submission content. The prompt remains inside
`train.py`; this workflow has no `--attempt_csv` argument.

## Files

- `model.py`: fixed-size skill-evidence diagnosis network and policy head.
- `train.py`: data banks, evidence construction, no-attempt prompt,
  REINFORCE training, evaluation, theta export, and prediction export.
- `run.py`: command-line entry point, configuration, checkpointing, and output
  management.
- `diagnose.py`: post-training deterministic diagnosis used to export the CP2
  and CP4 mastery profiles for RQ2 and RQ3.
- `example_outputs/`: synthetic examples of generated base-mastery CSVs.

## Run

```bash
python GLLMCDM/without_attempt/run.py \
  --train_csv path/to/train.csv \
  --valid_csv path/to/valid.csv \
  --test_csv path/to/test.csv \
  --item_text_csv path/to/item_text.csv \
  --q_matrix_path path/to/q_matrix.csv \
  --save_dir outputs/without_attempt/S1 \
  --use_openai \
  --openai_model gpt-5.4-nano \
  --openai_reasoning_effort high \
  --openai_verbosity low \
  --n_know 18 --hidden 128 --alpha 0.5 \
  --batch_size 8 --lr 0.0001 --n_epoch 20 \
  --reward_scale_c 100 --baseline_beta 0.05 \
  --use_early_stopping --early_stop_patience 5 \
  --early_stop_min_delta 0.0 --monitor_metric rmse \
  --seed 2026 --device cpu
```

Add `--export_theta --theta_evidence train` when a deterministic mastery CSV
is required for the downstream RQ2 or RQ3 notebooks.

## Export CP2 and CP4 diagnosis outputs

The `--export_theta` option above exports mastery for one selected training or
validation evidence split. For the time-matched CP2 and CP4 profiles used in
RQ2 and RQ3, run the original post-training diagnosis script separately.

CP2 example:

```bash
python GLLMCDM/without_attempt/diagnose.py \
  --evidence_file path/to/test_CP2_responses.csv \
  --q_matrix_path path/to/q_matrix.csv \
  --model_path outputs/without_attempt/S1/gllmcdm_stage1_policy.pt \
  --run_args_json outputs/without_attempt/S1/run_args.json \
  --output_path outputs/diagnosis/without_attempt/S1/CP2 \
  --device cpu
```

CP4 example:

```bash
python GLLMCDM/without_attempt/diagnose.py \
  --evidence_file path/to/test_CP4_responses.csv \
  --q_matrix_path path/to/q_matrix.csv \
  --model_path outputs/without_attempt/S1/gllmcdm_stage1_policy.pt \
  --run_args_json outputs/without_attempt/S1/run_args.json \
  --output_path outputs/diagnosis/without_attempt/S1/CP4 \
  --device cpu
```

Each run creates `theta_new.csv` (base mastery), `theta_implicit.csv`, and
`theta_explicit.csv`, with matching NumPy files. Diagnosis itself uses score
evidence and Q-vectors; the difference between the two GLLMCDM variants comes
from their separately trained checkpoints.
