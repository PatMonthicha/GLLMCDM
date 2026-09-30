# Purpose: Run the complete G-NCDM-poly S1, S2, or S3 experiment workflow.
# Provenance: Original research baseline; workflow logic is unchanged.
# run.py
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Jiatong Li
# All rights reserved.
#
# This software is the confidential and proprietary information
# of Jiatong Li. You shall not disclose such confidential
# information and shall use it only in accordance with the terms of
# the license agreement.

from model_parser import parse_args
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import model
import train
import sys
import os
from tools import degree_of_consistency


def add_knowledge_code(data: pd.DataFrame, Q_mat):
    knowledge = []
    for i in range(data.shape[0]):
        knowledge.append(Q_mat[data.loc[i, 'item_id']])
    data['knowledge'] = knowledge
    return data



def _remap_column(df: pd.DataFrame, column: str, mapping: dict) -> pd.DataFrame:
    """Return a copy with one ID column remapped by the supplied dictionary."""
    out = df.copy()
    mapped = out[column].map(mapping)
    if mapped.isna().any():
        missing = sorted(out.loc[mapped.isna(), column].astype(int).unique().tolist())
        raise ValueError(f'Cannot remap {column}; unseen IDs: {missing}')
    out[column] = mapped.astype(int)
    return out


def prepare_open_world_splits(df_train, df_valid, df_test, full_Q_mat, eval_setting):
    """
    ADDED: build the model universe from training entities only.

    S2: valid/test learners are new, but their items must map to train items.
    S3: valid/test learners and items must both be disjoint from training.

    Original IDs are retained for audit and external Q-vector lookup.
    """
    setting = str(eval_setting).upper()
    frames = []
    for df in (df_train, df_valid, df_test):
        out = df.copy().reset_index(drop=True)
        out['original_user_id'] = out['user_id'].astype(int)
        out['original_item_id'] = out['item_id'].astype(int)
        # Kept from the original structure. This column is not used by GNCDM.
        out = add_knowledge_code(out, full_Q_mat)
        frames.append(out)
    train_df, valid_df, test_df = frames

    train_user_ids = sorted(train_df['original_user_id'].unique().tolist())
    train_item_ids = sorted(train_df['original_item_id'].unique().tolist())
    train_user_map = {old: new for new, old in enumerate(train_user_ids)}
    train_item_map = {old: new for new, old in enumerate(train_item_ids)}

    # CHANGED: only train users/items receive model coordinates and buffers.
    train_df['user_id'] = train_df['original_user_id'].map(train_user_map).astype(int)
    train_df['item_id'] = train_df['original_item_id'].map(train_item_map).astype(int)

    for name, split_df in (('valid', valid_df), ('test', test_df)):
        split_user_ids = sorted(split_df['original_user_id'].unique().tolist())
        split_user_map = {old: new for new, old in enumerate(split_user_ids)}
        split_df['user_id'] = split_df['original_user_id'].map(split_user_map).astype(int)

        user_overlap = set(split_user_ids) & set(train_user_ids)
        if setting in {'S2', 'S3'} and user_overlap:
            raise ValueError(f'{setting} requires new {name} users; overlap={sorted(user_overlap)}')

        split_items = set(split_df['original_item_id'].unique().tolist())
        train_items = set(train_item_ids)
        if setting == 'S2':
            unseen_items = split_items - train_items
            if unseen_items:
                raise ValueError(f'S2 requires seen items; unseen {name} items={sorted(unseen_items)}')
            # CHANGED (S2): map seen item IDs to the train-item coordinate system.
            split_df['item_id'] = split_df['original_item_id'].map(train_item_map).astype(int)
        elif setting == 'S3':
            overlap = split_items & train_items
            if overlap:
                raise ValueError(f'S3 requires new items; {name} overlap={sorted(overlap)}')
            # S3 inference never indexes model buffers by these IDs. A local ID is
            # retained only for readable output; original_item_id drives Q lookup.
            local_item_map = {
                old: new for new, old in enumerate(sorted(split_items))
            }
            split_df['item_id'] = split_df['original_item_id'].map(local_item_map).astype(int)

    train_Q_mat = np.asarray(full_Q_mat)[train_item_ids]
    return (
        train_df, valid_df, test_df,
        train_user_ids, train_item_ids, train_Q_mat,
    )

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


def flatten_result_all(result_all):
    """
    """
    rows = []

    for epoch_idx, r in enumerate(result_all):
        train_eval = r.get('train_eval', {})
        valid_eval = r.get('valid_eval', {})

        row = {
            'epoch': epoch_idx,
            'theta_norm': r.get('Theta_norm', np.nan),

            'train_rmse': train_eval.get('rmse', np.nan),
            'train_mae': train_eval.get('mae', np.nan),
            'train_mse': train_eval.get('mse', np.nan),

            'valid_rmse': valid_eval.get('rmse', np.nan),
            'valid_mae': valid_eval.get('mae', np.nan),
            'valid_mse': valid_eval.get('mse', np.nan),

            'early_stopped': r.get('early_stopped', False),
            'best_epoch': r.get('best_epoch', np.nan),
            'best_metric': r.get('best_metric', np.nan),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def save_result_all_flat(result_all, save_path):
    """
    """
    df_flat = flatten_result_all(result_all)
    flat_csv_path = os.path.join(save_path, 'result_all_flat.csv')
    df_flat.to_csv(flat_csv_path, index=False)

    json_path = os.path.join(save_path, 'result_all.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(result_all), f, ensure_ascii=False, indent=2)

    print("Saved readable training summaries:")
    print(" -", flat_csv_path)
    print(" -", json_path)


if __name__ == '__main__':
    args = parse_args()

    df_train = pd.read_csv(args.train_file)
    df_valid = pd.read_csv(args.valid_file)
    df_test = pd.read_csv(args.test_file)
    eval_setting = str(args.eval_setting).upper()

    n_know = int(args.n_know)
    full_Q_mat = np.load(args.Q_matrix) if args.Q_matrix != '' else None
    if full_Q_mat is None:
        if eval_setting == 'S3':
            raise ValueError('S3 requires a full external Q-matrix.')
        if args.n_item is None:
            raise ValueError('n_item is required when Q_matrix is omitted.')
        full_Q_mat = np.ones((int(args.n_item), n_know), dtype=np.float32)

    if eval_setting in {'S2', 'S3'}:
        # CHANGED: no valid/test user or item slot is preallocated.
        (df_train, df_valid, df_test,
         train_original_user_ids, train_original_item_ids,
         Q_mat) = prepare_open_world_splits(
            df_train, df_valid, df_test, full_Q_mat, eval_setting
        )
        n_user = len(train_original_user_ids)
        n_item = len(train_original_item_ids)
        print(f'[OpenWorld {eval_setting}] model n_user={n_user}, n_item={n_item}')
        print('[OpenWorld] train original items:', train_original_item_ids)
    else:
        # Original S1 behavior is retained.
        if args.n_user is None or args.n_item is None:
            raise ValueError('S1 requires --n_user and --n_item.')
        n_user = int(args.n_user)
        n_item = int(args.n_item)
        Q_mat = np.asarray(full_Q_mat)
        train_original_item_ids = sorted(df_train['item_id'].astype(int).unique().tolist())
        df_train = add_knowledge_code(df_train, full_Q_mat)
        df_valid = add_knowledge_code(df_valid, full_Q_mat)
        df_test = add_knowledge_code(df_test, full_Q_mat)

    user_dim = int(args.user_dim)
    item_dim = int(args.item_dim)
    alpha = float(args.alpha)

    with open(args.training_config, 'r') as fp:
        config = json.load(fp)

    batch_size = int(config['batch_size'])
    lr = float(config['lr'])
    epoch = int(args.n_epoch) if args.n_epoch is not None else int(config['n_epoch'])
    device = torch.device(config['device'])
    print(device)

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rmse_all = []
    mae_all = []

    net = model.GNCDM(
        n_user, n_item, n_know, user_dim,
        item_dim, alpha, Q_mat=Q_mat,
        monotonicity_assumption=True, device=device
    )

    result_all = train.train(
        net, df_train, df_valid,
        batch_size=batch_size,
        lr=lr,
        n_epoch=epoch,
        early_stopping_patience=args.early_stop_patience if args.use_early_stopping else None,
        early_stopping_metric=args.monitor_metric,
        early_stopping_min_delta=args.early_stop_min_delta,
        save_best_path=os.path.join(save_path, 'best_model_state.pt'),
        # CHANGED: validation uses the same S2/S3 protocol as final testing.
        eval_setting=eval_setting,
        full_Q_mat=full_Q_mat,
        train_original_item_ids=train_original_item_ids,
    )

    np.save(os.path.join(save_path, 'result_all.npy'), result_all)
    save_result_all_flat(result_all, save_path)

    Theta = net.get_Theta_buf().numpy()
    Psi = net.get_Psi_buf().numpy()
    np.save(os.path.join(save_path, 'Theta_buf.npy'), Theta)
    np.save(os.path.join(save_path, 'Psi_buf.npy'), Psi)
    pd.DataFrame(Theta).to_csv(os.path.join(save_path, 'Theta_buf.csv'), index=False)
    pd.DataFrame(Psi).to_csv(os.path.join(save_path, 'Psi_buf.csv'), index=False)
    print('Saved train-only Theta/Psi buffers.')

    # ADDED FOR S3 AND THETA EXPORT:
    # Save the branch-specific training representations from the final/best
    # checkpoint. S3 uses only the mean implicit vector as a frozen prior.
    train_theta_imp, train_theta_exp, train_theta_base =         train.compute_train_theta_components(
            net, df_train, batch_size=batch_size
        )
    mean_train_theta_imp = train_theta_imp.mean(dim=0)

    np.save(os.path.join(save_path, 'Theta_implicit_train.npy'),
            train_theta_imp.numpy().astype(np.float32))
    np.save(os.path.join(save_path, 'Theta_explicit_train.npy'),
            train_theta_exp.numpy().astype(np.float32))
    np.save(os.path.join(save_path, 'Theta_base_train.npy'),
            train_theta_base.numpy().astype(np.float32))
    np.save(os.path.join(save_path, 'mean_train_theta_implicit.npy'),
            mean_train_theta_imp.numpy().astype(np.float32))
    pd.DataFrame([mean_train_theta_imp.numpy()]).to_csv(
        os.path.join(save_path, 'mean_train_theta_implicit.csv'), index=False
    )
    print('Saved train implicit/explicit/base components and mean implicit prior.')

    pred_test_path = os.path.join(save_path, 'pred_test.csv')
    test_result = train.eval(
        net, df_test, batch_size=batch_size,
        save_pred_path=pred_test_path, split_name='Test',
        eval_setting=eval_setting, full_Q_mat=full_Q_mat,
        train_original_item_ids=train_original_item_ids,
        s3_mean_theta_imp=(mean_train_theta_imp if eval_setting == 'S3' else None),
    )

    rmse_all.append(test_result['rmse'])
    mae_all.append(test_result['mae'])

    with open(os.path.join(save_path, 'cmd.txt'), 'w') as fp:
        fp.write(' '.join(['python'] + sys.argv))

    with open(os.path.join(save_path, 'test_result.json'), 'w') as fp:
        json.dump(to_serializable(test_result), fp, ensure_ascii=False, indent=2)

    torch.save(net, os.path.join(save_path, 'params_%s_%s.pt' % (user_dim, item_dim)))
    gc.collect()


'''
python run.py \
  --eval_setting S1 \
  --train_file data/Exp_data/S1_split/S1_train_random_response.csv \
  --valid_file data/Exp_data/S1_split/S1_valid_random_response.csv \
  --test_file data/Exp_data/S1_split/S1_test_random_response.csv \
  --Q_matrix data/Exp_data/q_matrix_CP1toCP4.npy \
  --save_path result/GNCDM_S1 \
  --n_user 460 \
  --n_item 15 \
  --n_know 18 \
  --user_dim 32 \
  --item_dim 32 \
  --training_config config/training_config_Exp_200epoch.json \
  --alpha 0.5 \
  --use_early_stopping \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.0 \
  --monitor_metric rmse \
  --n_epoch 200



python run.py \
  --eval_setting S3 \
  --train_file data/Exp_data/S3_split/train.csv \
  --valid_file data/Exp_data/S3_split/valid.csv \
  --test_file data/Exp_data/S3_split/test.csv \
  --Q_matrix data/Exp_data/q_matrix_CP1toCP4.npy \
  --save_path result/GNCDM_S3 \
  --n_know 18 \
  --user_dim 32 \
  --item_dim 32 \
  --training_config config/training_config_Exp_200epoch.json \
  --alpha 0.5 \
  --use_early_stopping \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.0 \
  --monitor_metric rmse \
  --n_epoch 200
'''


# S2 example: --n_user/--n_item are intentionally omitted because the model
# universe is computed from the training split only.
# python run.py \
#   --eval_setting S2 \
#   --train_file data/Exp_data/S2_split/train.csv \
#   --valid_file data/Exp_data/S2_split/valid.csv \
#   --test_file data/Exp_data/S2_split/test.csv \
#   --Q_matrix data/Exp_data/q_matrix_CP1toCP4.npy \
#   --save_path result/GNCDM_S2_open_world \
#   --n_know 18 --user_dim 32 --item_dim 32 --alpha 0.5 \
#   --training_config config/training_config_Exp_200epoch.json \
#   --use_early_stopping --early_stop_patience 5 \
#   --early_stop_min_delta 0.0 --monitor_metric rmse --n_epoch 200
#
# S3 example: theta_imp = mean train implicit; theta_exp comes from
# unseen-item responses + external Q; psi = nearest-Q train Psi_buf.
# No valid/test gradient, buffer update, or new entity slot is created.
# python run.py \
#   --eval_setting S3 \
#   --train_file data/Exp_data/S3_split/train.csv \
#   --valid_file data/Exp_data/S3_split/valid.csv \
#   --test_file data/Exp_data/S3_split/test.csv \
#   --Q_matrix data/Exp_data/q_matrix_CP1toCP4.npy \
#   --save_path result/GNCDM_S3_cold_start \
#   --n_know 18 --user_dim 32 --item_dim 32 --alpha 0.5 \
#   --training_config config/training_config_Exp_200epoch.json \
#   --use_early_stopping --early_stop_patience 5 \
#   --early_stop_min_delta 0.0 --monitor_metric rmse --n_epoch 200
