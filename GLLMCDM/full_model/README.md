# GLLMCDM full model

This is the full model used in the paper. Its LLM scorer prompt includes the
item text, required skills, diagnosed mastery profile, and the learner's
submitted attempt. The prompt remains inside `train.py`.

## Files

- `model.py`: fixed-size skill-evidence diagnosis network and policy head.
- `train.py`: data banks, evidence construction, full prompt, REINFORCE
  training, evaluation, and `pred_test.csv` generation.
- `run.py`: command-line entry point, configuration, checkpointing, and output
  management.
- `diagnose.py`: post-training deterministic diagnosis used to export the CP2
  and CP4 mastery profiles for RQ2 and RQ3.
- `example_outputs/`: synthetic examples of the generated base-mastery CSVs;
  these are outputs, not required input data.

## Input schemas

```text
train/valid/test: user_id,item_id,score
item text:        item_id,item_text
attempts:         user_id,item_id,attempt_clean
Q-matrix CSV:     item_id,K0,...,K17
```

The Q-matrix may alternatively be a NumPy array whose rows correspond to
zero-based item IDs.

## Run

From the repository root:

```bash
python GLLMCDM/full_model/run.py \
  --train_csv path/to/train.csv \
  --valid_csv path/to/valid.csv \
  --test_csv path/to/test.csv \
  --item_text_csv path/to/item_text.csv \
  --q_matrix_path path/to/q_matrix.csv \
  --attempt_csv path/to/attempt_master.csv \
  --save_dir outputs/full_model/S1 \
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

Omit `--use_openai` for the original dummy-scorer pipeline check. The output
directory contains the model state, training history, test metrics, and
`pred_test.csv` used for RQ1.

## Export diagnosed mastery for RQ2 and RQ3

Training produces the checkpoint. Diagnosis is a separate post-training step:

```text
CP2 response evidence -> diagnose.py -> CP2 theta -> Mid analyses
CP4 response evidence -> diagnose.py -> CP4 theta -> Final analyses
```

The diagnosis step uses only `user_id,item_id,score` and the corresponding
Q-matrix. Learner attempts are used by the full model's LLM scorer during
training/evaluation, but they are not inputs to deterministic diagnosis.

CP2 example:

```bash
python GLLMCDM/full_model/diagnose.py \
  --evidence_file path/to/test_CP2_responses.csv \
  --q_matrix_path path/to/q_matrix.csv \
  --model_path outputs/full_model/S1/gllmcdm_stage1_policy.pt \
  --run_args_json outputs/full_model/S1/run_args.json \
  --output_path outputs/diagnosis/full_model/S1/CP2 \
  --device cpu
```

CP4 example:

```bash
python GLLMCDM/full_model/diagnose.py \
  --evidence_file path/to/test_CP4_responses.csv \
  --q_matrix_path path/to/q_matrix.csv \
  --model_path outputs/full_model/S1/gllmcdm_stage1_policy.pt \
  --run_args_json outputs/full_model/S1/run_args.json \
  --output_path outputs/diagnosis/full_model/S1/CP4 \
  --device cpu
```

Each run creates `theta_new.csv` (base mastery), `theta_implicit.csv`, and
`theta_explicit.csv`, with matching NumPy files. Repeat with the appropriate
S2 or S3 checkpoint and evidence files for those settings.
