"""Provide the original S2 fallback functions used by train_s2.py."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


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


def mean_train_theta(model, train_user_ids):
    """Skill-wise mean mastery from students that occur in train only."""
    train_user_ids = sorted(set(int(user_id) for user_id in train_user_ids))
    if not train_user_ids:
        raise ValueError('train_user_ids must not be empty.')
    if train_user_ids[-1] >= model.student_emb.num_embeddings:
        raise ValueError('A train user ID exceeds the student embedding table.')

    indices = torch.tensor(
        train_user_ids,
        dtype=torch.long,
        device=model.student_emb.weight.device,
    )
    return torch.sigmoid(model.student_emb.weight[indices]).mean(
        dim=0,
        keepdim=True,
    )


def export_mean_train_theta(model, train_user_ids, result_dir):
    """Save the deterministic unseen-student fallback from the best epoch."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    theta = mean_train_theta(model, train_user_ids).detach().cpu().numpy()
    npy_path = result_dir / 'mean_train_theta.npy'
    csv_path = result_dir / 'mean_train_theta.csv'
    np.save(npy_path, theta)
    with csv_path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(
            [f'theta_{knowledge_id}' for knowledge_id in range(theta.shape[1])]
        )
        writer.writerow(theta[0].astype(float).tolist())
    return {
        'definition': 'mean(sigmoid(student_emb.weight[train_user_ids]), axis=0)',
        'shape': list(theta.shape),
        'n_train_students': len(set(int(x) for x in train_user_ids)),
        'npy': str(npy_path),
        'csv': str(csv_path),
    }


def export_fallback_theta_by_user(
    model,
    train_user_ids,
    target_user_ids,
    output_file,
    split_name,
):
    """Repeat the best-epoch train mean theta for each unseen target user."""
    target_user_ids = sorted(set(int(x) for x in target_user_ids))
    if not target_user_ids:
        raise ValueError('target_user_ids must not be empty.')

    theta = mean_train_theta(model, train_user_ids).detach().cpu().numpy()[0]
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(
            ['user_id']
            + [f'theta_{knowledge_id}' for knowledge_id in range(len(theta))]
            + ['theta_source', 'split']
        )
        for user_id in target_user_ids:
            writer.writerow(
                [user_id]
                + theta.astype(float).tolist()
                + ['train_population_mean', split_name]
            )
    return {
        'definition': 'best-epoch mean train theta repeated per target user',
        'n_users': len(target_user_ids),
        'theta_is_constant_across_users': True,
        'csv': str(output_file),
    }


def forward_with_effective_parameters(
    model,
    theta,
    difficulty,
    discrimination,
    knowledge,
):
    """Run the unchanged NCDM prediction network with supplied effective traits."""
    input_x = discrimination * (theta - difficulty) * knowledge
    input_x = model.drop_1(torch.sigmoid(model.prednet_full1(input_x)))
    input_x = model.drop_2(torch.sigmoid(model.prednet_full2(input_x)))
    return model.prednet_full3(input_x)


def _evaluation_result(labels, raw_predictions):
    raw_predictions = np.asarray(raw_predictions, dtype=np.float64)
    clipped_predictions = np.clip(raw_predictions, 0.0, 10.0)
    result = regression_metrics(labels, clipped_predictions)
    result['raw'] = regression_metrics(labels, raw_predictions)
    result['raw_prediction'] = prediction_range(raw_predictions)
    return result, clipped_predictions


def _save_predictions(
    prediction_file,
    user_ids,
    item_ids,
    labels,
    raw_predictions,
    clipped_predictions,
):
    prediction_file = Path(prediction_file)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    with prediction_file.open('w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(
            ['user_id', 'item_id', 'score', 'pred', 'pred_raw', 'pred_clipped']
        )
        writer.writerows(
            zip(
                user_ids,
                item_ids,
                labels,
                clipped_predictions,
                raw_predictions,
                clipped_predictions,
            )
        )


@torch.no_grad()
def evaluate_s2_mean_student(
    model,
    dataset,
    train_user_ids,
    device,
    batch_size=8,
    prediction_file=None,
):
    """Evaluate unseen students with a train-derived mean theta and seen items."""
    model.eval()
    theta_mean = mean_train_theta(model, train_user_ids).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    user_all, item_all, label_all, raw_pred_all = [], [], [], []

    for user_ids, item_ids, knowledge, labels in loader:
        item_ids_device = item_ids.to(device)
        knowledge = knowledge.to(device)
        batch_theta = theta_mean.expand(len(user_ids), -1)
        difficulty = torch.sigmoid(model.k_difficulty(item_ids_device))
        discrimination = (
            torch.sigmoid(model.e_discrimination(item_ids_device)) * 10.0
        )
        raw_predictions = forward_with_effective_parameters(
            model,
            batch_theta,
            difficulty,
            discrimination,
            knowledge,
        ).view(-1)

        user_all.extend(user_ids.tolist())
        item_all.extend(item_ids.tolist())
        label_all.extend(labels.tolist())
        raw_pred_all.extend(raw_predictions.cpu().tolist())

    result, clipped_predictions = _evaluation_result(label_all, raw_pred_all)
    result['fallback'] = {
        'student': 'skill-wise mean theta of train students',
        'n_train_students': len(set(int(x) for x in train_user_ids)),
        'parameter_updates_on_evaluation_data': False,
    }
    if prediction_file is not None:
        _save_predictions(
            prediction_file,
            user_all,
            item_all,
            label_all,
            raw_pred_all,
            clipped_predictions.tolist(),
        )
    return result


def jaccard_similarity(q_target, q_candidates):
    q_target = np.asarray(q_target, dtype=np.float32) > 0
    q_candidates = np.asarray(q_candidates, dtype=np.float32) > 0
    intersection = np.logical_and(q_candidates, q_target).sum(axis=1)
    union = np.logical_or(q_candidates, q_target).sum(axis=1)
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


@torch.no_grad()
def q_neighbor_item_parameters(
    model,
    q_matrix,
    target_item_ids,
    candidate_item_ids,
    k,
    exclude_target_from_candidates=False,
):
    """Create deterministic effective item parameters using Q similarity only."""
    q_matrix = np.asarray(q_matrix, dtype=np.float32)
    candidate_item_ids = sorted(set(int(x) for x in candidate_item_ids))
    target_item_ids = sorted(set(int(x) for x in target_item_ids))
    if not candidate_item_ids:
        raise ValueError('candidate_item_ids must not be empty.')
    if k <= 0:
        raise ValueError('k must be positive.')
    if max(candidate_item_ids) >= model.exer_n:
        raise ValueError('Candidate items must have learned model parameters.')
    if max(target_item_ids + candidate_item_ids) >= q_matrix.shape[0]:
        raise ValueError('Q-matrix does not cover all target/candidate items.')

    device = model.k_difficulty.weight.device
    all_candidate_index = torch.tensor(
        candidate_item_ids,
        dtype=torch.long,
        device=device,
    )
    all_difficulty = torch.sigmoid(
        model.k_difficulty(all_candidate_index)
    ).detach().cpu().numpy()
    all_discrimination = (
        torch.sigmoid(model.e_discrimination(all_candidate_index)) * 10.0
    ).detach().cpu().numpy()
    parameter_by_item = {
        item_id: (all_difficulty[pos], all_discrimination[pos])
        for pos, item_id in enumerate(candidate_item_ids)
    }

    fallback_by_item = {}
    for target_item_id in target_item_ids:
        available = [
            item_id
            for item_id in candidate_item_ids
            if not (
                exclude_target_from_candidates
                and item_id == target_item_id
            )
        ]
        if not available:
            raise ValueError(
                f'No fallback candidates remain for item {target_item_id}.'
            )

        q_target = q_matrix[target_item_id]
        q_available = q_matrix[available]
        exact_mask = np.all(q_available == q_target, axis=1)

        if exact_mask.any():
            selected = np.asarray(available, dtype=np.int64)[exact_mask]
            similarities = np.ones(len(selected), dtype=np.float64)
            method = 'exact_q_mean'
        else:
            all_similarity = jaccard_similarity(q_target, q_available)
            ranked_positions = sorted(
                range(len(available)),
                key=lambda pos: (-all_similarity[pos], available[pos]),
            )
            chosen_positions = ranked_positions[:min(k, len(ranked_positions))]
            selected = np.asarray(
                [available[pos] for pos in chosen_positions],
                dtype=np.int64,
            )
            similarities = np.asarray(
                [all_similarity[pos] for pos in chosen_positions],
                dtype=np.float64,
            )
            method = 'jaccard_top_k'

        if similarities.sum() > 0:
            weights = similarities / similarities.sum()
        else:
            selected = np.asarray(available, dtype=np.int64)
            similarities = np.zeros(len(selected), dtype=np.float64)
            weights = np.full(len(selected), 1.0 / len(selected))
            method = 'global_train_item_mean'

        selected_difficulty = np.stack(
            [parameter_by_item[int(item_id)][0] for item_id in selected]
        )
        selected_discrimination = np.stack(
            [parameter_by_item[int(item_id)][1] for item_id in selected]
        )
        difficulty = np.sum(selected_difficulty * weights[:, None], axis=0)
        discrimination = np.sum(
            selected_discrimination * weights[:, None],
            axis=0,
        )
        fallback_by_item[target_item_id] = {
            'difficulty': torch.tensor(
                difficulty,
                dtype=torch.float32,
                device=device,
            ),
            'discrimination': torch.tensor(
                discrimination,
                dtype=torch.float32,
                device=device,
            ),
            'method': method,
            'neighbor_item_ids': selected.astype(int).tolist(),
            'similarities': similarities.astype(float).tolist(),
            'weights': weights.astype(float).tolist(),
        }
    return fallback_by_item


@torch.no_grad()
def evaluate_s3_q_neighbor(
    model,
    dataset,
    q_matrix,
    train_user_ids,
    train_item_ids,
    k,
    device,
    batch_size=8,
    prediction_file=None,
    exclude_target_from_candidates=False,
):
    """Evaluate mean-student + Q-neighbor item fallback without updates."""
    model.eval()
    theta_mean = mean_train_theta(model, train_user_ids).to(device)
    target_item_ids = sorted(set(int(x) for x in dataset.item_ids.tolist()))
    item_fallbacks = q_neighbor_item_parameters(
        model,
        q_matrix,
        target_item_ids,
        train_item_ids,
        k,
        exclude_target_from_candidates=exclude_target_from_candidates,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    user_all, item_all, label_all, raw_pred_all = [], [], [], []

    for user_ids, item_ids, knowledge, labels in loader:
        knowledge = knowledge.to(device)
        batch_theta = theta_mean.expand(len(user_ids), -1)
        difficulty = torch.stack(
            [item_fallbacks[int(item_id)]['difficulty'] for item_id in item_ids]
        ).to(device)
        discrimination = torch.stack(
            [
                item_fallbacks[int(item_id)]['discrimination']
                for item_id in item_ids
            ]
        ).to(device)
        raw_predictions = forward_with_effective_parameters(
            model,
            batch_theta,
            difficulty,
            discrimination,
            knowledge,
        ).view(-1)

        user_all.extend(user_ids.tolist())
        item_all.extend(item_ids.tolist())
        label_all.extend(labels.tolist())
        raw_pred_all.extend(raw_predictions.cpu().tolist())

    result, clipped_predictions = _evaluation_result(label_all, raw_pred_all)
    result['fallback'] = {
        'student': 'skill-wise mean theta of train students',
        'item': 'exact-Q mean, otherwise Jaccard top-k weighted mean',
        'k': int(k),
        'exclude_target_from_candidates': bool(
            exclude_target_from_candidates
        ),
        'parameter_updates_on_evaluation_data': False,
        'items': {
            str(item_id): {
                key: value
                for key, value in fallback.items()
                if key not in {'difficulty', 'discrimination'}
            }
            for item_id, fallback in item_fallbacks.items()
        },
    }
    if prediction_file is not None:
        _save_predictions(
            prediction_file,
            user_all,
            item_all,
            label_all,
            raw_pred_all,
            clipped_predictions.tolist(),
        )
    return result


def build_combined_q_matrix(*q_csv_files):
    frames = [pd.read_csv(path) for path in q_csv_files]
    knowledge_columns = sorted(
        {
            column
            for frame in frames
            for column in frame.columns
            if column.startswith('K') and column[1:].isdigit()
        },
        key=lambda column: int(column[1:]),
    )
    if not knowledge_columns:
        raise ValueError('No K0..Kn knowledge columns were found.')

    max_item_id = max(int(frame['item_id'].max()) for frame in frames)
    q_matrix = np.zeros(
        (max_item_id + 1, len(knowledge_columns)),
        dtype=np.float32,
    )
    seen_item_ids = set()
    for frame in frames:
        required = {'item_id', *knowledge_columns}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f'Q-matrix CSV is missing: {sorted(missing)}')
        for _, row in frame.iterrows():
            item_id = int(row['item_id'])
            values = row[knowledge_columns].to_numpy(dtype=np.float32)
            if item_id in seen_item_ids and not np.array_equal(
                q_matrix[item_id],
                values,
            ):
                raise ValueError(f'Conflicting Q-vectors for item {item_id}.')
            q_matrix[item_id] = values
            seen_item_ids.add(item_id)
    return q_matrix


def save_item_fallback_metadata(result, output_file):
    """Save the auditable item-neighbor choices without model tensors."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as output:
        json.dump(
            result['fallback']['items'],
            output,
            ensure_ascii=False,
            indent=2,
        )
