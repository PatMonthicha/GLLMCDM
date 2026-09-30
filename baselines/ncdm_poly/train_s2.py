"""Train NCDM for strict S2 evaluation with unseen learners and seen items."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import ResponseDataset, load_q_matrix
from fallback_old import (
    evaluate_s2_mean_student,
    export_fallback_theta_by_user,
    export_mean_train_theta,
    prediction_range,
    regression_metrics,
)
from model import Net
from train import export_best_theta, set_seed


def train_s2(args):
    """Train the backbone and validate S2 with a fixed mean-theta fallback."""
    set_seed(args.seed)
    device = torch.device(args.device)
    q_matrix = load_q_matrix(args.q_matrix)
    train_dataset = ResponseDataset(args.train_file, q_matrix)
    valid_dataset = ResponseDataset(args.valid_file, q_matrix)

    train_users = sorted(set(int(x) for x in train_dataset.user_ids))
    valid_users = sorted(set(int(x) for x in valid_dataset.user_ids))
    overlap = sorted(set(train_users).intersection(valid_users))
    if overlap:
        raise ValueError(
            'S2 requires disjoint train/validation students; overlapping IDs: '
            f'{overlap[:10]}'
        )

    inferred_student_n = int(train_dataset.user_ids.max()) + 1
    student_n = args.student_n or inferred_student_n
    if student_n < inferred_student_n:
        raise ValueError('--student-n is smaller than a train user_id.')

    exer_n, knowledge_n = q_matrix.shape
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = Net(student_n, exer_n, knowledge_n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.MSELoss()

    model_dir = Path(args.model_dir)
    result_dir = Path(args.result_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / 'best_model.pt'

    best_rmse = float('inf')
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    history = []

    print('training NCDM S2 backbone (strict no-retrain validation)...')
    for epoch in range(1, args.epochs + 1):
        model.train()
        labels_all, predictions_all = [], []
        loss_sum = 0.0
        response_count = 0

        for user_ids, item_ids, knowledge, labels in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            knowledge = knowledge.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, item_ids, knowledge).view(-1)
            loss = loss_function(predictions, labels.view(-1))
            loss.backward()
            optimizer.step()
            model.apply_clipper()

            batch_count = labels.numel()
            loss_sum += loss.item() * batch_count
            response_count += batch_count
            labels_all.extend(labels.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())

        train_result = regression_metrics(labels_all, predictions_all)
        train_result['loss'] = loss_sum / response_count
        train_result['raw_prediction'] = prediction_range(predictions_all)
        valid_result = evaluate_s2_mean_student(
            model,
            valid_dataset,
            train_users,
            device,
            batch_size=args.batch_size,
        )
        history.append(
            {
                'epoch': epoch,
                'train': train_result,
                'validation': valid_result,
            }
        )
        print(
            f'epoch={epoch}, train_raw_rmse={train_result["rmse"]:.6f}, '
            f'valid_s2_clipped_rmse={valid_result["rmse"]:.6f}, '
            f'valid_s2_clipped_mae={valid_result["mae"]:.6f}'
        )

        current_rmse = valid_result['rmse']
        if current_rmse < best_rmse - args.min_delta:
            best_rmse = current_rmse
            best_epoch = epoch
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    'model_state_dict': best_state,
                    'student_n': student_n,
                    'exer_n': exer_n,
                    'knowledge_n': knowledge_n,
                    'best_epoch': best_epoch,
                    'best_validation': valid_result,
                    'train_user_ids': train_users,
                    'train_item_ids': sorted(
                        set(int(x) for x in train_dataset.item_ids)
                    ),
                    'scenario': 'S2_unseen_student_seen_item',
                    'student_fallback': 'mean_train_theta',
                    'score_range': [0.0, 10.0],
                    'seed': args.seed,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if args.patience is not None and bad_epochs >= args.patience:
                print(
                    f'early stopping at epoch {epoch}; best epoch={best_epoch}, '
                    f'clipped RMSE={best_rmse:.6f}'
                )
                break

    if best_state is None:
        raise RuntimeError('Training finished without a best checkpoint.')

    model.load_state_dict(best_state)
    best_validation = evaluate_s2_mean_student(
        model,
        valid_dataset,
        train_users,
        device,
        batch_size=args.batch_size,
        prediction_file=result_dir / 'pred_valid.csv',
    )
    theta_export = export_best_theta(model, train_users, result_dir)
    mean_theta_export = export_mean_train_theta(model, train_users, result_dir)
    valid_theta_export = export_fallback_theta_by_user(
        model,
        train_users,
        valid_dataset.user_ids,
        result_dir / 'theta_valid_fallback.csv',
        split_name='validation',
    )
    summary = {
        'scenario': 'S2: unseen student + seen item',
        'best_epoch': best_epoch,
        'best_validation': best_validation,
        'checkpoint': str(best_path),
        'best_theta': theta_export,
        'mean_train_theta': mean_theta_export,
        'validation_fallback_theta': valid_theta_export,
        'protocol': {
            'training': 'raw MSE on train only',
            'validation_student': 'fixed mean theta from train students',
            'validation_item': 'learned seen-item parameters',
            'parameter_updates_on_validation': False,
            'early_stopping_metric': 'clipped validation RMSE',
        },
        'history': history,
    }
    with (result_dir / 'training_result.json').open(
        'w', encoding='utf-8'
    ) as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(f'saved best checkpoint: {best_path}')
    print(f'best epoch={best_epoch}, clipped validation RMSE={best_rmse:.6f}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train NCDM for strict S2 unseen-student evaluation.'
    )
    parser.add_argument(
        '--train-file', default='data/Exp_data/S2_split/train.csv'
    )
    parser.add_argument(
        '--valid-file', default='data/Exp_data/S2_split/valid.csv'
    )
    parser.add_argument(
        '--q-matrix', default='data/Exp_data/q_matrix_CP1toCP4.npy'
    )
    parser.add_argument('--model-dir', default='model/NCDM_S2_fallback')
    parser.add_argument('--result-dir', default='result/NCDM_S2_fallback')
    parser.add_argument('--student-n', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    train_s2(parse_args())



'''
python train_s2.py \
  --train-file data/Exp_data/S2_split/train.csv \
  --valid-file data/Exp_data/S2_split/valid.csv \
  --q-matrix data/Exp_data/q_matrix_CP1toCP4.npy \
  --student-n 460 \
  --model-dir model/NCDM_S2_fallback \
  --result-dir result/NCDM_S2_fallback \
  --device cpu \
  --epochs 200 \
  --batch-size 8 \
  --lr 0.002 \
  --patience 5 \
  --min-delta 0.0 \
  --seed 42
'''
