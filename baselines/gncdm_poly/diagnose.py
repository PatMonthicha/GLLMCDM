"""Export G-NCDM mastery components for S1 or S2 evidence.

This is the original research workflow for seen-item coordinates. It writes
base, implicit, and explicit mastery profiles for downstream RQ2 and RQ3.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import model
from train import IDCDataset


def _infer_profile_label(path: str):
    name = Path(path).name.upper()
    if 'CP2' in name:
        return 'CP2'
    if 'CP4' in name:
        return 'CP4'
    return None


def _save_csv_and_npy(df, array, output_path, stem):
    np.save(os.path.join(output_path, f'{stem}.npy'), array.astype(np.float32))
    df.to_csv(os.path.join(output_path, f'{stem}.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence_file', required=True,
                        help='long-format log: user_id,item_id,score')
    parser.add_argument('--model_path', required=True,
                        help='trusted full-model checkpoint saved with torch.save(net)')
    parser.add_argument('--output_path', required=True)

    # CHANGED: optional. Values are inferred safely when omitted, so the short
    # S2 commands in the project can be used directly.
    parser.add_argument('--n_user', type=int, default=None,
                        help='matrix row capacity; default=max evidence user_id + 1')
    parser.add_argument('--n_item', type=int, default=None,
                        help='default=model.n_item')
    parser.add_argument('--n_know', type=int, default=None,
                        help='default=model.n_know')
    parser.add_argument('--user_map_file', default=None)
    parser.add_argument('--profile_label', choices=['CP2', 'CP4'], default=None,
                        help='optional alias suffix; inferred from evidence filename')
    args = parser.parse_args()

    df_log = pd.read_csv(args.evidence_file)
    required = {'user_id', 'item_id', 'score'}
    missing = required - set(df_log.columns)
    if missing:
        raise ValueError(f'evidence_file missing columns: {sorted(missing)}')
    if df_log.empty:
        raise ValueError('evidence_file is empty.')

    device = torch.device('cpu')
    net = torch.load(
        args.model_path, map_location='cpu', weights_only=False
    )
    net.eval()
    if hasattr(net, 'device'):
        net.device = device

    n_item = int(args.n_item if args.n_item is not None else net.n_item)
    n_know = int(args.n_know if args.n_know is not None else net.n_know)
    inferred_n_user = int(df_log['user_id'].astype(int).max()) + 1
    n_user = int(args.n_user if args.n_user is not None else inferred_n_user)

    if n_item != int(net.n_item):
        raise ValueError(
            f'n_item={n_item} does not match the trained model n_item={net.n_item}. '
            'S2 evidence must use the seen-item coordinate system.'
        )
    if n_know != int(net.n_know):
        raise ValueError(
            f'n_know={n_know} does not match model n_know={net.n_know}.'
        )
    if int(df_log['item_id'].max()) >= n_item or int(df_log['item_id'].min()) < 0:
        raise ValueError(
            'Evidence contains an item_id outside the trained S2 item space. '
            'Use the dedicated S3 notebook for unseen items.'
        )
    if n_user <= int(df_log['user_id'].max()):
        raise ValueError('n_user must be greater than the maximum evidence user_id.')

    dataset = IDCDataset(df_log, n_user=n_user, n_item=n_item)
    log_mat = dataset.log_mat
    mask_mat = dataset.obs_mat

    uids = np.sort(df_log['user_id'].astype(int).unique())
    print(f'Users in evidence: {len(uids)} (min={uids.min()}, max={uids.max()})')
    print(f'Model item space: {n_item}; knowledge dimensions: {n_know}')
    print(f'Using device: {device}')

    theta_list, theta_imp_list, theta_exp_list = [], [], []
    for u in tqdm(uids, desc='Diagnosing theta'):
        user_log = torch.from_numpy(log_mat[u][None, :]).float().to(device)
        user_mask = torch.from_numpy(mask_mat[u][None, :]).float().to(device)

        with torch.no_grad():
            # CHANGED: use the model's component methods. This preserves the
            # original S2 formula and exports all three representations.
            if hasattr(net, 'diagnose_theta_components'):
                theta_imp_u, theta_exp_u, theta_u = \
                    net.diagnose_theta_components(user_log, user_mask)
            else:
                # Backward-compatible fallback for an older model.py.
                theta_imp_u = net.f_nn(user_log)
                evid_count_raw = user_mask @ net.Q_mat
                evid_sum = (user_log * user_mask) @ net.Q_mat
                skill_score = torch.where(
                    evid_count_raw > 0,
                    evid_sum / evid_count_raw.clamp(min=1e-6),
                    torch.zeros_like(evid_sum),
                )
                theta_exp_u = torch.sigmoid(skill_score)
                theta_u = theta_imp_u * (1-net.alpha) + theta_exp_u * net.alpha

        theta_imp_list.append(theta_imp_u.cpu().numpy())
        theta_exp_list.append(theta_exp_u.cpu().numpy())
        theta_list.append(theta_u.cpu().numpy())

    theta_mat = np.concatenate(theta_list, axis=0)
    theta_imp_mat = np.concatenate(theta_imp_list, axis=0)
    theta_exp_mat = np.concatenate(theta_exp_list, axis=0)
    cols = [f'K{i}' for i in range(n_know)]

    def make_df(arr):
        out = pd.DataFrame(arr, columns=cols)
        out.insert(0, 'user_id', uids.astype(int))
        return out

    df_theta = make_df(theta_mat)
    df_imp = make_df(theta_imp_mat)
    df_exp = make_df(theta_exp_mat)

    if args.user_map_file is not None and os.path.exists(args.user_map_file):
        map_df = pd.read_csv(args.user_map_file)
        if {'student_id', 'user_id'} <= set(map_df.columns):
            # Keep one mapping row per profile user_id to prevent accidental
            # row duplication during merge.
            map_df = map_df[['student_id', 'user_id']].drop_duplicates('user_id')
            for name, frame in [('base', df_theta), ('implicit', df_imp), ('explicit', df_exp)]:
                merged = frame.merge(map_df, on='user_id', how='left', validate='one_to_one')
                merged = merged[['student_id', 'user_id'] + cols]
                if name == 'base':
                    df_theta = merged
                elif name == 'implicit':
                    df_imp = merged
                else:
                    df_exp = merged
            print(f'Using user mapping: {args.user_map_file}')
        else:
            print('Mapping file lacks student_id/user_id; keeping user_id only.')

    os.makedirs(args.output_path, exist_ok=True)
    _save_csv_and_npy(df_theta, theta_mat, args.output_path, 'theta')
    _save_csv_and_npy(df_theta, theta_mat, args.output_path, 'theta_base')
    _save_csv_and_npy(df_imp, theta_imp_mat, args.output_path, 'theta_implicit')
    _save_csv_and_npy(df_exp, theta_exp_mat, args.output_path, 'theta_explicit')
    np.save(os.path.join(args.output_path, 'user_ids.npy'), uids.astype(int))

    label = args.profile_label or _infer_profile_label(args.evidence_file)
    if label:
        # ADDED: aliases matching the downstream Experiment 2/3 filenames.
        df_theta.to_csv(
            os.path.join(args.output_path, f'theta_test_{label}.csv'), index=False
        )
        df_theta.to_csv(
            os.path.join(args.output_path, f'theta_experiment2_{label}.csv'), index=False
        )
        df_theta.to_csv(
            os.path.join(args.output_path, f'theta_experiment3_{label}.csv'), index=False
        )
        df_imp.to_csv(
            os.path.join(args.output_path, f'theta_implicit_{label}.csv'), index=False
        )
        df_exp.to_csv(
            os.path.join(args.output_path, f'theta_explicit_{label}.csv'), index=False
        )
        df_theta.to_csv(
            os.path.join(args.output_path, f'theta_base_{label}.csv'), index=False
        )

    summary = pd.DataFrame([{
        'n_profiles': len(uids),
        'n_skills': n_know,
        'alpha': float(net.alpha),
        'n_unique_theta_implicit': np.unique(np.round(theta_imp_mat, 8), axis=0).shape[0],
        'n_unique_theta_explicit': np.unique(np.round(theta_exp_mat, 8), axis=0).shape[0],
        'n_unique_theta_base': np.unique(np.round(theta_mat, 8), axis=0).shape[0],
        'profile_label': label,
    }])
    summary.to_csv(
        os.path.join(args.output_path, 'theta_generation_summary.csv'), index=False
    )

    print('Saved S1/S2 theta outputs to:', args.output_path)
    print(' - theta.csv / theta_base.csv')
    print(' - theta_implicit.csv')
    print(' - theta_explicit.csv')
    if label:
        print(f' - downstream aliases for {label}')


if __name__ == '__main__':
    main()


