"""Generate S1 NCDM predictions and reconstruction metrics."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_loader import ResponseDataset, load_q_matrix
from model import Net
from train import evaluate


def test(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    q_matrix = load_q_matrix(args.q_matrix)

    expected_shape = (
        int(checkpoint['exer_n']),
        int(checkpoint['knowledge_n']),
    )
    if q_matrix.shape != expected_shape:
        raise ValueError(
            f'Q-matrix shape {q_matrix.shape} does not match checkpoint '
            f'{expected_shape}.'
        )

    dataset = ResponseDataset(args.test_file, q_matrix)
    if int(dataset.user_ids.max()) >= int(checkpoint['student_n']):
        raise ValueError(
            'Test data contains a user_id outside the checkpoint embedding table.'
        )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = Net(
        int(checkpoint['student_n']),
        int(checkpoint['exer_n']),
        int(checkpoint['knowledge_n']),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(
        model,
        data_loader,
        device,
        prediction_file=result_dir / 'pred_test.csv',
    )
    result['checkpoint'] = str(args.checkpoint)
    result['best_epoch'] = int(checkpoint['best_epoch'])
    with (result_dir / 'test_result.json').open('w', encoding='utf-8') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)

    print(
        f'test clipped RMSE={result["rmse"]:.6f}, '
        f'clipped MAE={result["mae"]:.6f}, '
        f'raw RMSE={result["raw"]["rmse"]:.6f}'
    )
    print(f'saved predictions: {result_dir / "pred_test.csv"}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the NCDM continuous-score regression baseline.'
    )
    parser.add_argument(
        '--test-file',
        default='data/Exp_data/S1_split/S1_test_random_response.csv',
    )
    parser.add_argument(
        '--q-matrix',
        default='data/Exp_data/q_matrix_CP1toCP4.npy',
    )
    parser.add_argument(
        '--checkpoint',
        default='model/NCDM_S1/best_model.pt',
    )
    parser.add_argument('--result-dir', default='result/NCDM_S1')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    test(parse_args())



'''
NCDM S1: test reconstruction
python predict.py \
  --test-file data/Exp_data/S1_split/S1_test_random_response.csv \
  --q-matrix data/Exp_data/q_matrix_CP1toCP4.npy \
  --checkpoint model/NCDM_S1/best_model.pt \
  --result-dir result/NCDM_S1 \
  --device cpu \
  --batch-size 8
'''
