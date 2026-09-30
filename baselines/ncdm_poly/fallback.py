"""Implement the strict S2/S3 cold-start evaluation fallbacks."""

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



def cosine_similarity(q_target, q_candidates):
    """Compute cosine similarity from one Q-vector to candidate Q-vectors.

    Q-vectors may be binary or non-negative numeric vectors. A zero-norm
    target or candidate receives similarity 0. Neighbor selection uses only
    the Q-matrix; no validation/test scores or target-specific parameters are
    used.
    """
    q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
    q_candidates = np.asarray(q_candidates, dtype=np.float64)
    if q_candidates.ndim != 2:
        raise ValueError('q_candidates must be a 2-D array.')
    if q_candidates.shape[1] != q_target.shape[0]:
        raise ValueError('Q-vector dimensions do not match.')

    target_norm = np.linalg.norm(q_target)
    candidate_norms = np.linalg.norm(q_candidates, axis=1)
    denominators = candidate_norms * target_norm
    numerators = q_candidates @ q_target
    return np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=np.float64),
        where=denominators > 0.0,
    )


@torch.no_grad()
def maximum_cosine_tie_item_parameters(
    model,
    q_matrix,
    target_item_ids,
    candidate_item_ids,
):
    """Impute unseen-item parameters from maximum positive cosine ties.

    For each target item, cosine similarity is computed against every
    training-item Q-vector. All training items tied at the maximum positive
    similarity are selected, and their effective difficulty and
    discrimination parameters are averaged uniformly. If no positive overlap
    exists, the global means over training-item parameters are used.
    """
    q_matrix = np.asarray(q_matrix, dtype=np.float32)
    candidate_item_ids = sorted(set(int(x) for x in candidate_item_ids))
    target_item_ids = sorted(set(int(x) for x in target_item_ids))

    if not candidate_item_ids:
        raise ValueError('candidate_item_ids must not be empty.')
    if not target_item_ids:
        raise ValueError('target_item_ids must not be empty.')
    if max(candidate_item_ids) >= model.exer_n:
        raise ValueError('A candidate item has no learned NCDM parameters.')
    if max(target_item_ids + candidate_item_ids) >= q_matrix.shape[0]:
        raise ValueError('Q-matrix does not cover all target/candidate items.')

    device = model.k_difficulty.weight.device
    candidate_index = torch.tensor(
        candidate_item_ids,
        dtype=torch.long,
        device=device,
    )
    candidate_difficulty = torch.sigmoid(
        model.k_difficulty(candidate_index)
    ).detach().cpu().numpy()
    candidate_discrimination = (
        torch.sigmoid(model.e_discrimination(candidate_index)) * 10.0
    ).detach().cpu().numpy()

    q_candidates = q_matrix[candidate_item_ids]
    fallback_by_item = {}

    for target_item_id in target_item_ids:
        similarities_all = cosine_similarity(
            q_matrix[target_item_id],
            q_candidates,
        )
        max_similarity = float(similarities_all.max())

        if max_similarity > 0.0:
            selected_positions = np.flatnonzero(
                np.isclose(
                    similarities_all,
                    max_similarity,
                    rtol=1e-7,
                    atol=1e-10,
                )
            )
            method = 'maximum_positive_cosine_ties'
        else:
            selected_positions = np.arange(len(candidate_item_ids))
            method = 'global_train_item_mean'

        selected_item_ids = np.asarray(candidate_item_ids, dtype=np.int64)[
            selected_positions
        ]
        selected_similarities = similarities_all[selected_positions]
        weights = np.full(
            len(selected_positions),
            1.0 / len(selected_positions),
            dtype=np.float64,
        )

        difficulty = np.average(
            candidate_difficulty[selected_positions],
            axis=0,
            weights=weights,
        )
        discrimination = np.average(
            candidate_discrimination[selected_positions],
            axis=0,
            weights=weights,
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
            'max_cosine_similarity': max_similarity,
            'neighbor_item_ids': selected_item_ids.astype(int).tolist(),
            'similarities': selected_similarities.astype(float).tolist(),
            'weights': weights.astype(float).tolist(),
            'all_candidate_similarities': {
                str(item_id): float(similarity)
                for item_id, similarity in zip(
                    candidate_item_ids,
                    similarities_all,
                )
            },
        }

    return fallback_by_item


@torch.no_grad()
def evaluate_s3_cosine_ties(
    model,
    dataset,
    q_matrix,
    train_user_ids,
    train_item_ids,
    device,
    batch_size=8,
    prediction_file=None,
):
    """Evaluate S3 with mean-train theta and cosine-tie item fallback."""
    model.eval()
    theta_mean = mean_train_theta(model, train_user_ids).to(device)
    target_item_ids = sorted(set(int(x) for x in dataset.item_ids.tolist()))
    item_fallbacks = maximum_cosine_tie_item_parameters(
        model,
        q_matrix,
        target_item_ids,
        train_item_ids,
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
        'item': 'maximum positive cosine-similarity ties with uniform mean',
        'similarity': 'cosine over Q-vectors',
        'tie_rule': 'all training items tied at the maximum positive value',
        'no_positive_overlap': 'global mean of training-item parameters',
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
