"""Generate strict S3 predictions using Q-vector cosine item fallback."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_loader import ResponseDataset
from fallback import (
    build_combined_q_matrix,
    evaluate_s3_cosine_ties,
    export_fallback_theta_by_user,
    save_item_fallback_metadata,
)
from model import Net


def predict_s3(args):
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get('scenario') != 'S3_unseen_student_unseen_item':
        raise ValueError('Checkpoint is not an S3 strict-fallback checkpoint.')
    if checkpoint.get('item_fallback') != 'maximum_positive_cosine_ties':
        raise ValueError(
            'Checkpoint was not selected with the maximum-cosine-ties '
            'validation protocol. Retrain with the new train_s3.py first.'
        )

    q_matrix = build_combined_q_matrix(*args.q_csv)
    if q_matrix.shape[1] != int(checkpoint['knowledge_n']):
        raise ValueError(
            f'Combined Q-matrix has {q_matrix.shape[1]} skills but checkpoint '
            f'has {checkpoint["knowledge_n"]}.'
        )
    dataset = ResponseDataset(args.test_file, q_matrix)
    train_users = set(int(x) for x in checkpoint['train_user_ids'])
    train_items = set(int(x) for x in checkpoint['train_item_ids'])
    user_overlap = sorted(
        train_users.intersection(int(x) for x in dataset.user_ids)
    )
    item_overlap = sorted(
        train_items.intersection(int(x) for x in dataset.item_ids)
    )
    if user_overlap:
        raise ValueError(
            'S3 test students must be unseen in train; overlapping IDs: '
            f'{user_overlap[:10]}'
        )
    if item_overlap:
        raise ValueError(
            'S3 test items must be unseen in train; overlapping IDs: '
            f'{item_overlap[:10]}'
        )

    model = Net(
        int(checkpoint['student_n']),
        int(checkpoint['exer_n']),
        int(checkpoint['knowledge_n']),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / 'combined_q_matrix.npy', q_matrix)

    # CHANGED: no best_k is read; test uses the exact same cosine/ties rule
    # that selected the checkpoint during validation.
    result = evaluate_s3_cosine_ties(
        model,
        dataset,
        q_matrix,
        checkpoint['train_user_ids'],
        checkpoint['train_item_ids'],
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
            'scenario': 'S3: unseen student + unseen item',
            'checkpoint': str(args.checkpoint),
            'best_epoch': int(checkpoint['best_epoch']),
            'item_fallback': 'maximum_positive_cosine_ties',
            'q_csv': [str(path) for path in args.q_csv],
            'parameter_updates_on_test': False,
            'test_fallback_theta': test_theta_export,
        }
    )
    with (result_dir / 'test_result.json').open(
        'w', encoding='utf-8'
    ) as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    save_item_fallback_metadata(
        result,
        result_dir / 'test_item_fallbacks.json',
    )
    print(
        f'S3 test clipped RMSE={result["rmse"]:.6f}, '
        f'clipped MAE={result["mae"]:.6f}, '
        f'raw RMSE={result["raw"]["rmse"]:.6f}'
    )
    print(f'saved predictions: {result_dir / "pred_test.csv"}')
    print(
        'saved neighbor audit: '
        f'{result_dir / "test_item_fallbacks.json"}'
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Strict no-retrain S3 NCDM prediction using maximum cosine ties.'
        )
    )
    parser.add_argument(
        '--test-file',
        default='data/Exp_data/S3_split/test.csv',
    )
    parser.add_argument(
        '--q-csv',
        nargs='+',
        default=['data/Exp_data/q_matrix_CP1toCP4.csv'],
        help='Q CSVs containing train and unseen test items with global IDs.',
    )
    parser.add_argument(
        '--checkpoint',
        default='model/NCDM_S3_fallback/best_model.pt',
    )
    parser.add_argument('--result-dir', default='result/NCDM_S3_fallback')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    predict_s3(parse_args())


'''
python predict_s3.py \
  --test-file data/Exp_data/S3_split/test.csv \
  --q-csv data/Exp_data/q_matrix_CP1toCP4.csv \
  --checkpoint model/NCDM_S3_fallback/best_model.pt \
  --result-dir result/NCDM_S3_fallback \
  --device cpu \
  --batch-size 8
'''
