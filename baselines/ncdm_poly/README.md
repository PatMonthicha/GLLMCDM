# NCDM-poly baseline

This directory contains the continuous-score NCDM baseline used in the paper.
S1 uses standard learned student and item parameters. S2 evaluates unseen
students with the original mean-student fallback. S3 adds the original
Q-vector cosine item fallback for unseen items.

## Files

- `model.py`: continuous-score NeuralCDM network.
- `data_loader.py`: response and Q-matrix loading.
- `train.py` / `predict.py`: S1 workflow.
- `train_s2.py` / `predict_s2.py`: strict S2 workflow.
- `train_s3.py` / `predict_s3.py`: strict S3 workflow.
- `fallback.py` and `fallback_old.py`: fallback functions used by the original
  S2/S3 scripts.

## S1

```bash
python baselines/ncdm_poly/train.py \
  --train-file path/to/S1_train.csv \
  --valid-file path/to/S1_valid.csv \
  --q-matrix path/to/q_matrix.npy \
  --student-n 460 \
  --model-dir outputs/ncdm_poly/S1/model \
  --result-dir outputs/ncdm_poly/S1/results \
  --epochs 200 --batch-size 8 --lr 0.002 \
  --patience 5 --min-delta 0.0 --seed 42 --device cpu

python baselines/ncdm_poly/predict.py \
  --test-file path/to/S1_test.csv \
  --q-matrix path/to/q_matrix.npy \
  --checkpoint outputs/ncdm_poly/S1/model/best_model.pt \
  --result-dir outputs/ncdm_poly/S1/results \
  --batch-size 8 --device cpu
```

## S2 and S3

Use `train_s2.py` with `predict_s2.py`, or `train_s3.py` with
`predict_s3.py`, keeping the same training options. Run each command with
`--help` for its setting-specific checkpoint and Q-matrix arguments.
