"""Generate strict S2 predictions using the trained mean-student fallback."""

import argparse
import json
from pathlib import Path

import torch

from data_loader import ResponseDataset, load_q_matrix
from fallback import evaluate_s2_mean_student, export_fallback_theta_by_user
from model import Net


def predict_s2(args):
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get('scenario') != 'S2_unseen_student_seen_item':
        raise ValueError('Checkpoint is not an S2 strict-fallback checkpoint.')

    q_matrix = load_q_matrix(args.q_matrix)
    expected_shape = (
        int(checkpoint['exer_n']),
        int(checkpoint['knowledge_n']),
    )
    if q_matrix.shape != expected_shape:
        raise ValueError(
            f'Q-matrix shape {q_matrix.shape} does not match {expected_shape}.'
        )
    dataset = ResponseDataset(args.test_file, q_matrix)
    train_users = set(int(x) for x in checkpoint['train_user_ids'])
    overlap = sorted(train_users.intersection(int(x) for x in dataset.user_ids))
    if overlap:
        raise ValueError(
            'S2 test students must be unseen in train; overlapping IDs: '
            f'{overlap[:10]}'
        )
    if int(dataset.item_ids.max()) >= int(checkpoint['exer_n']):
        raise ValueError('S2 test contains an unseen item; use S3 instead.')

    model = Net(
        int(checkpoint['student_n']),
        int(checkpoint['exer_n']),
        int(checkpoint['knowledge_n']),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_s2_mean_student(
        model,
        dataset,
        checkpoint['train_user_ids'],
        device,
        batch_size=args.batch_size,
        prediction_file=result_dir / 'pred_test.csv',
    )
    test_theta_export = export_fallback_theta_by_user(
        model,
        checkpoint['train_user_ids'],
        dataset.user_ids,
        result_dir / 'theta_test_fallback.csv',
        split_name='test',
    )
    result.update(
        {
            'scenario': 'S2: unseen student + seen item',
            'checkpoint': str(args.checkpoint),
            'best_epoch': int(checkpoint['best_epoch']),
            'parameter_updates_on_test': False,
            'test_fallback_theta': test_theta_export,
        }
    )
    with (result_dir / 'test_result.json').open(
        'w', encoding='utf-8'
    ) as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(
        f'S2 test clipped RMSE={result["rmse"]:.6f}, '
        f'clipped MAE={result["mae"]:.6f}, '
        f'raw RMSE={result["raw"]["rmse"]:.6f}'
    )
    print(f'saved predictions: {result_dir / "pred_test.csv"}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Strict no-retrain S2 NCDM prediction.'
    )
    parser.add_argument(
        '--test-file', default='data/Exp_data/S2_split/test.csv'
    )
    parser.add_argument(
        '--q-matrix', default='data/Exp_data/q_matrix_CP1toCP4.npy'
    )
    parser.add_argument(
        '--checkpoint',
        default='model/NCDM_S2_fallback/best_model.pt',
    )
    parser.add_argument('--result-dir', default='result/NCDM_S2_fallback')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    predict_s2(parse_args())


'''
python predict_s2.py \
  --test-file data/Exp_data/S2_split/test.csv \
  --q-matrix data/Exp_data/q_matrix_CP1toCP4.npy \
  --checkpoint model/NCDM_S2_fallback/best_model.pt \
  --result-dir result/NCDM_S2_fallback \
  --device cpu \
  --batch-size 8
'''
