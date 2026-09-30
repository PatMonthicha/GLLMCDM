"""Train the S1 continuous-score NCDM baseline and export its best mastery."""

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import ResponseDataset, load_q_matrix
from model import Net


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    errors = predictions - labels
    mse = float(np.mean(errors ** 2))
    return {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'mae': float(np.mean(np.abs(errors))),
    }


def prediction_range(predictions):
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    return {
        'min': float(predictions.min()),
        'max': float(predictions.max()),
        'n_below_0': int(np.sum(predictions < 0.0)),
        'n_above_10': int(np.sum(predictions > 10.0)),
    }


def evaluate(model, data_loader, device, prediction_file=None):
    """Evaluate with clipped primary metrics and raw diagnostic metrics."""
    model.eval()
    user_all, item_all = [], []
    label_all, raw_pred_all = [], []

    with torch.no_grad():
        for user_ids, item_ids, knowledge, labels in data_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            knowledge = knowledge.to(device)
            raw_predictions = model(user_ids, item_ids, knowledge).view(-1)

            user_all.extend(user_ids.cpu().tolist())
            item_all.extend(item_ids.cpu().tolist())
            label_all.extend(labels.tolist())
            raw_pred_all.extend(raw_predictions.cpu().tolist())

    raw_predictions = np.asarray(raw_pred_all, dtype=np.float64)
    clipped_predictions = np.clip(raw_predictions, 0.0, 10.0)
    result = regression_metrics(label_all, clipped_predictions)
    result['raw'] = regression_metrics(label_all, raw_predictions)
    result['raw_prediction'] = prediction_range(raw_predictions)

    if prediction_file is not None:
        prediction_file = Path(prediction_file)
        prediction_file.parent.mkdir(parents=True, exist_ok=True)
        with prediction_file.open('w', newline='', encoding='utf-8') as output:
            writer = csv.writer(output)
            writer.writerow(
                ['user_id', 'item_id', 'score', 'pred', 'pred_raw', 'pred_clipped']
            )
            writer.writerows(
                zip(
                    user_all,
                    item_all,
                    label_all,
                    clipped_predictions.tolist(),
                    raw_predictions.tolist(),
                    clipped_predictions.tolist(),
                )
            )

    return result


def export_best_theta(model, train_user_ids, result_dir):
    """Export mastery theta = sigmoid(student embedding) from the best epoch."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    train_user_ids = set(int(user_id) for user_id in train_user_ids)

    raw_embeddings = model.student_emb.weight.detach().cpu().numpy()
    theta = torch.sigmoid(
        model.student_emb.weight.detach()
    ).cpu().numpy()

    raw_path = result_dir / 'student_embedding_best.npy'
    theta_npy_path = result_dir / 'theta_best.npy'
    theta_csv_path = result_dir / 'theta_best.csv'
    np.save(raw_path, raw_embeddings)
    np.save(theta_npy_path, theta)

    with theta_csv_path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(
            ['user_id', 'seen_in_train']
            + [f'theta_{knowledge_id}' for knowledge_id in range(theta.shape[1])]
        )
        for user_id, theta_row in enumerate(theta):
            writer.writerow(
                [user_id, int(user_id in train_user_ids)]
                + theta_row.astype(float).tolist()
            )

    return {
        'theta_definition': 'sigmoid(student_emb.weight)',
        'theta_shape': list(theta.shape),
        'theta_npy': str(theta_npy_path),
        'theta_csv': str(theta_csv_path),
        'raw_student_embedding_npy': str(raw_path),
        'n_students_seen_in_train': len(train_user_ids),
    }


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    q_matrix = load_q_matrix(args.q_matrix)
    train_dataset = ResponseDataset(args.train_file, q_matrix)
    valid_dataset = ResponseDataset(args.valid_file, q_matrix)

    inferred_student_n = max(
        int(train_dataset.user_ids.max()),
        int(valid_dataset.user_ids.max()),
    ) + 1
    student_n = args.student_n or inferred_student_n
    if student_n < inferred_student_n:
        raise ValueError(
            f'--student-n={student_n} is smaller than the largest ID in the data '
            f'({inferred_student_n - 1}).'
        )

    exer_n, knowledge_n = q_matrix.shape
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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

    print('training NCDM continuous-score regression model...')
    for epoch in range(1, args.epochs + 1):
        model.train()
        label_all, raw_pred_all = [], []
        loss_sum = 0.0
        response_count = 0

        for user_ids, item_ids, knowledge, labels in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            knowledge = knowledge.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            raw_predictions = model(user_ids, item_ids, knowledge).view(-1)
            loss = loss_function(raw_predictions, labels.view(-1))
            loss.backward()
            optimizer.step()
            model.apply_clipper()

            batch_size = labels.numel()
            loss_sum += loss.item() * batch_size
            response_count += batch_size
            label_all.extend(labels.detach().cpu().tolist())
            raw_pred_all.extend(raw_predictions.detach().cpu().tolist())

        train_result = regression_metrics(label_all, raw_pred_all)
        train_result['loss'] = loss_sum / response_count
        train_result['raw_prediction'] = prediction_range(raw_pred_all)
        valid_result = evaluate(model, valid_loader, device)

        row = {
            'epoch': epoch,
            'train': train_result,
            'validation': valid_result,
        }
        history.append(row)
        print(
            f'epoch={epoch}, train_raw_rmse={train_result["rmse"]:.6f}, '
            f'valid_clipped_rmse={valid_result["rmse"]:.6f}, '
            f'valid_clipped_mae={valid_result["mae"]:.6f}'
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
                    'score_range': [0.0, 10.0],
                    'seed': args.seed,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if args.patience is not None and bad_epochs >= args.patience:
                print(
                    f'early stopping at epoch {epoch}; '
                    f'best epoch={best_epoch}, clipped RMSE={best_rmse:.6f}'
                )
                break

    if best_state is None:
        raise RuntimeError('Training finished without a best checkpoint.')

    model.load_state_dict(best_state)
    best_validation = evaluate(
        model,
        valid_loader,
        device,
        prediction_file=result_dir / 'pred_valid.csv',
    )
    theta_export = export_best_theta(
        model,
        train_dataset.user_ids,
        result_dir,
    )
    summary = {
        'best_epoch': best_epoch,
        'best_validation': best_validation,
        'checkpoint': str(best_path),
        'best_theta': theta_export,
        'protocol': {
            'training_loss': 'raw MSE',
            'evaluation': 'clip predictions to [0, 10]',
            'early_stopping_metric': 'clipped validation RMSE',
        },
        'history': history,
    }
    with (result_dir / 'training_result.json').open('w', encoding='utf-8') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(f'saved best checkpoint: {best_path}')
    print(f'best epoch={best_epoch}, clipped validation RMSE={best_rmse:.6f}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the NCDM continuous-score regression baseline.'
    )
    parser.add_argument(
        '--train-file',
        default='data/Exp_data/S2_split/train.csv',
    )
    parser.add_argument(
        '--valid-file',
        default='data/Exp_data/S2_split/valid.csv',
    )
    parser.add_argument(
        '--q-matrix',
        default='data/Exp_data/q_matrix_CP1toCP4.npy',
    )
    parser.add_argument('--model-dir', default='model/ncdm_regression')
    parser.add_argument('--result-dir', default='result/ncdm_regression')
    parser.add_argument('--student-n', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())




'''
NCDM S1: train

python train.py \
  --train-file data/Exp_data/S1_split/S1_train_random_response.csv \
  --valid-file data/Exp_data/S1_split/S1_valid_random_response.csv \
  --q-matrix data/Exp_data/q_matrix_CP1toCP4.npy \
  --student-n 460 \
  --model-dir model/NCDM_S1 \
  --result-dir result/NCDM_S1 \
  --device cpu \
  --epochs 200 \
  --batch-size 8 \
  --lr 0.002 \
  --patience 5 \
  --min-delta 0.0 \
  --seed 42
'''


'''
NCDM S2: train
python train.py \
  --train-file data/Exp_data/S2_split/train.csv \
  --valid-file data/Exp_data/S2_split/valid.csv \
  --q-matrix data/Exp_data/q_matrix_CP1toCP4.npy \
  --student-n 460 \
  --model-dir model/NCDM_S2_backbone \
  --result-dir result/NCDM_S2_backbone \
  --device cpu \
  --epochs 200 \
  --patience 5 \
  --min-delta 0.0
'''
