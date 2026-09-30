"""Export G-NCDM mastery components for the strict S3 protocol.

This is the original research workflow for unseen learners and unseen items.
It combines the mean training implicit profile with learner-specific explicit
mastery computed from the supplied evidence and external Q-vectors.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import model
import train


def _infer_profile_label(path: str):
    name = Path(path).name.upper()
    if 'CP2' in name:
        return 'CP2'
    if 'CP4' in name:
        return 'CP4'
    return None


def _make_external_matrix(df, full_q):
    required = {'user_id', 'item_id', 'score'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'evidence_file missing columns: {sorted(missing)}')
    if df.duplicated(['user_id', 'item_id']).any():
        raise ValueError('Duplicate user_id-item_id rows found in S3 evidence.')

    user_ids = sorted(df['user_id'].astype(int).unique().tolist())
    item_ids = sorted(df['item_id'].astype(int).unique().tolist())
    user_map = {old: new for new, old in enumerate(user_ids)}
    item_map = {old: new for new, old in enumerate(item_ids)}

    log_mat = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
    mask_mat = np.zeros_like(log_mat)
    for _, row in df.iterrows():
        score = float(row['score'])
        if np.isnan(score):
            continue
        u = user_map[int(row['user_id'])]
        i = item_map[int(row['item_id'])]
        log_mat[u, i] = (score / 10.0) * 2.0 - 1.0
        mask_mat[u, i] = 1.0

    full_q = np.asarray(full_q, dtype=np.float32)
    if min(item_ids) < 0 or max(item_ids) >= full_q.shape[0]:
        raise IndexError('S3 item_id lies outside the full Q-matrix.')
    q_new = full_q[item_ids]
    return user_ids, item_ids, log_mat, mask_mat, q_new


def _save(df, arr, out_dir, stem):
    df.to_csv(os.path.join(out_dir, f'{stem}.csv'), index=False)
    np.save(os.path.join(out_dir, f'{stem}.npy'), arr.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', required=True,
                        help='original-ID S3 training CSV')
    parser.add_argument('--evidence_file', required=True,
                        help='S3 unseen-item evidence CSV for CP2 or CP4 profile')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--Q_matrix', required=True,
                        help='full original-item Q-matrix')
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--user_map_file', default=None)
    parser.add_argument('--profile_label', choices=['CP2', 'CP4'], default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    train_df_original = pd.read_csv(args.train_file)
    evidence_df = pd.read_csv(args.evidence_file)
    full_q = np.load(args.Q_matrix).astype(np.float32)

    net = torch.load(args.model_path, map_location='cpu', weights_only=False)
    net.eval()
    net.device = torch.device('cpu')

    # Recreate exactly the training-only coordinate system used by run.py.
    train_user_ids = sorted(train_df_original['user_id'].astype(int).unique().tolist())
    train_item_ids = sorted(train_df_original['item_id'].astype(int).unique().tolist())
    if len(train_user_ids) != int(net.n_user):
        raise ValueError(
            f'Train users ({len(train_user_ids)}) do not match model.n_user ({net.n_user}).'
        )
    if len(train_item_ids) != int(net.n_item):
        raise ValueError(
            f'Train items ({len(train_item_ids)}) do not match model.n_item ({net.n_item}).'
        )

    train_user_map = {old: new for new, old in enumerate(train_user_ids)}
    train_item_map = {old: new for new, old in enumerate(train_item_ids)}
    train_df = train_df_original.copy()
    train_df['user_id'] = train_df['user_id'].astype(int).map(train_user_map)
    train_df['item_id'] = train_df['item_id'].astype(int).map(train_item_map)
    if train_df[['user_id', 'item_id']].isna().any().any():
        raise ValueError('Failed to remap S3 training coordinates.')
    train_df[['user_id', 'item_id']] = train_df[['user_id', 'item_id']].astype(int)

    # Mean implicit prior from training learners only.
    theta_imp_train, theta_exp_train, theta_base_train = \
        train.compute_train_theta_components(net, train_df, args.batch_size)
    mean_train_imp = theta_imp_train.mean(dim=0)

    # Explicit diagnosis from unseen-item responses + their external Q-vectors.
    user_ids, new_item_ids, log_mat, mask_mat, q_new_np = \
        _make_external_matrix(evidence_df, full_q)
    new_log = torch.tensor(log_mat, dtype=torch.float32)
    new_mask = torch.tensor(mask_mat, dtype=torch.float32)
    q_new = torch.tensor(q_new_np, dtype=torch.float32)

    with torch.no_grad():
        theta_imp, theta_exp, theta_base = net.diagnose_theta_s3(
            new_log, new_mask, q_new, mean_train_imp
        )

    imp_np = theta_imp.cpu().numpy()
    exp_np = theta_exp.cpu().numpy()
    base_np = theta_base.cpu().numpy()
    columns = [f'K{i}' for i in range(net.n_know)]

    def make_df(arr):
        out = pd.DataFrame(arr, columns=columns)
        out.insert(0, 'user_id', np.asarray(user_ids, dtype=int))
        return out

    df_imp = make_df(imp_np)
    df_exp = make_df(exp_np)
    df_base = make_df(base_np)

    if args.user_map_file is not None and os.path.exists(args.user_map_file):
        map_df = pd.read_csv(args.user_map_file)
        if {'student_id', 'user_id'} <= set(map_df.columns):
            map_df = map_df[['student_id', 'user_id']].drop_duplicates('user_id')
            df_imp = df_imp.merge(map_df, on='user_id', how='left', validate='one_to_one')
            df_exp = df_exp.merge(map_df, on='user_id', how='left', validate='one_to_one')
            df_base = df_base.merge(map_df, on='user_id', how='left', validate='one_to_one')
            order = ['student_id', 'user_id'] + columns
            df_imp = df_imp[order]
            df_exp = df_exp[order]
            df_base = df_base[order]

    os.makedirs(args.output_path, exist_ok=True)
    _save(df_imp, imp_np, args.output_path, 'theta_implicit')
    _save(df_exp, exp_np, args.output_path, 'theta_explicit')
    _save(df_base, base_np, args.output_path, 'theta')
    _save(df_base, base_np, args.output_path, 'theta_base')

    label = args.profile_label or _infer_profile_label(args.evidence_file)
    if label:
        df_base.to_csv(os.path.join(args.output_path, f'theta_test_{label}.csv'), index=False)
        df_base.to_csv(os.path.join(args.output_path, f'theta_experiment2_{label}.csv'), index=False)
        df_base.to_csv(os.path.join(args.output_path, f'theta_experiment3_{label}.csv'), index=False)
        df_imp.to_csv(os.path.join(args.output_path, f'theta_implicit_{label}.csv'), index=False)
        df_exp.to_csv(os.path.join(args.output_path, f'theta_explicit_{label}.csv'), index=False)
        df_base.to_csv(os.path.join(args.output_path, f'theta_base_{label}.csv'), index=False)

    pd.DataFrame([mean_train_imp.numpy()], columns=columns).to_csv(
        os.path.join(args.output_path, 'mean_train_theta_implicit.csv'), index=False
    )

    # Psi is not needed to export theta, but this audit records exactly which
    # already-trained item representations Experiment 1 will use.
    psi_audit = []
    for item_id, q_vec in zip(new_item_ids, q_new):
        _, info = train._nearest_q_psi(
            net, q_vec, train_original_item_ids=train_item_ids
        )
        info['new_original_item_id'] = int(item_id)
        info['new_q_vector'] = [float(x) for x in q_vec.tolist()]
        psi_audit.append(info)
    pd.DataFrame(psi_audit).to_csv(
        os.path.join(args.output_path, 'psi_Q_neighbor_fallback_audit.csv'),
        index=False,
    )

    summary = pd.DataFrame([{
        'profile_label': label,
        'n_profiles': len(user_ids),
        'n_unseen_items': len(new_item_ids),
        'unseen_item_ids': ','.join(map(str, new_item_ids)),
        'n_skills': int(net.n_know),
        'alpha': float(net.alpha),
        'n_unique_theta_implicit': np.unique(np.round(imp_np, 8), axis=0).shape[0],
        'n_unique_theta_explicit': np.unique(np.round(exp_np, 8), axis=0).shape[0],
        'n_unique_theta_base': np.unique(np.round(base_np, 8), axis=0).shape[0],
        'protocol': 'mean_train_implicit + external_Q_explicit',
    }])
    summary.to_csv(
        os.path.join(args.output_path, 'theta_generation_summary.csv'), index=False
    )

    print('Saved S3 theta outputs to:', args.output_path)
    print('theta_implicit is the same train-only prior for every profile.')
    print('theta_explicit and theta_base are learner-specific when responses differ.')
    print('Psi neighbor audit:', os.path.join(
        args.output_path, 'psi_Q_neighbor_fallback_audit.csv'
    ))


if __name__ == '__main__':
    main()


