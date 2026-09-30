# Purpose: Train, validate, evaluate, and export mastery from G-NCDM-poly.
# Provenance: Original research baseline; training logic is unchanged.
# train.py
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Jiatong Li
# All rights reserved.
#
# This software is the confidential and proprietary information
# of Jiatong Li. You shall not disclose such confidential
# information and shall use it only in accordance with the terms of
# the license agreement.

import gc
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import GNCDM, UAutoRec, CDAE

torch.set_default_tensor_type(torch.FloatTensor)

import os
import numpy as np

DEBUG_EVAL = os.getenv("DEBUG_EVAL", "0") == "1"

def _stat(name, arr):
    arr = np.array(arr).reshape(-1,)
    return (f"{name}: min={arr.min():.4f} max={arr.max():.4f} "
            f"mean={arr.mean():.4f} std={arr.std():.4f} "
            f"n_unique~={len(np.unique(np.round(arr, 6)))}")


class IDCDataset(Dataset):
    """

      * 0/10 → -1
      * 10/10 → +1
    """
    def __init__(self, df_log: pd.DataFrame, n_user: int, n_item: int,
                 max_score: float = 10.0):
        self.df_log = df_log.reset_index(drop=True)
        self.n_user = n_user
        self.n_item = n_item
        self.max_score = float(max_score)

        self.log_mat = np.zeros((n_user, n_item), dtype=np.float32)
        self.obs_mat = np.zeros((n_user, n_item), dtype=np.float32)

        self.user_id = self.df_log['user_id'].values.astype(int)
        self.item_id = self.df_log['item_id'].values.astype(int)

        self.score = self.df_log['score'].values.astype(float)

        pbar = tqdm(total=self.df_log.shape[0], desc='Loading data')
        for _, row in self.df_log.iterrows():
            u = int(row['user_id'])
            q = int(row['item_id'])
            raw_score = float(row['score'])

            if np.isnan(raw_score):
                pbar.update(1)
                continue

            # encode: 0–10 → 0–1 → [-1,1]
            norm = raw_score / self.max_score          # 0..1
            signed = norm * 2.0 - 1.0                  # -1..1
            self.log_mat[u, q] = signed
            self.obs_mat[u, q] = 1.0

            pbar.update(1)
        pbar.close()

    def __getitem__(self, index):
        u = self.user_id[index]
        q = self.item_id[index]

        user_log = torch.tensor(self.log_mat[u, :], dtype=torch.float32)

        item_log = torch.tensor(self.log_mat[:, q], dtype=torch.float32)

        user_mask = torch.tensor(self.obs_mat[u, :], dtype=torch.float32)

        user_id_t = torch.LongTensor([u])
        item_id_t = torch.LongTensor([q])

        y = torch.tensor([self.score[index]], dtype=torch.float32)

        return user_log, item_log, user_mask, user_id_t, item_id_t, y

    
    def __len__(self):
        return self.user_id.shape[0]



class AEDataset(Dataset):
    def __init__(self, df_log: pd.DataFrame, n_user: int, n_item: int):
        self.log_mat = np.zeros((n_user, n_item))
        self.obs_mat = np.zeros((n_user, n_item))
        self.user_id = np.arange(n_user)
        self.score = df_log['score'].values
        pbar = tqdm(total=df_log.shape[0], desc='Loading data')
        for i, row in df_log.iterrows():
            self.log_mat[int(row['user_id']), int(row['item_id'])] = row['score']
            self.obs_mat[int(row['user_id']), int(row['item_id'])] = 1
            pbar.update(1)
        pbar.close()

    def __getitem__(self, index):
        user_id = self.user_id[index]
        return torch.Tensor(self.log_mat[user_id, :]), \
            torch.Tensor(self.obs_mat[user_id, :]), \
            torch.LongTensor([user_id])

    def __len__(self):
        return self.user_id.shape[0]



def compute_theta_components_from_matrices(model: GNCDM,
                                           log_mat: np.ndarray,
                                           obs_mat: np.ndarray,
                                           batch_size: int = 256):
    """
    Compute train-item-coordinate theta components without changing the model.

    ADDED FOR S3:
    The mean of theta_implicit from *training learners only* is the frozen
    population prior used when S3 learners answer only post-training items.
    """
    model.eval()
    device = model.device
    imp_all, exp_all, base_all = [], [], []
    with torch.no_grad():
        for start in range(0, log_mat.shape[0], batch_size):
            stop = min(start + batch_size, log_mat.shape[0])
            log_batch = torch.tensor(
                log_mat[start:stop], dtype=torch.float32, device=device
            )
            mask_batch = torch.tensor(
                obs_mat[start:stop], dtype=torch.float32, device=device
            )
            theta_imp, theta_exp, theta_base = model.diagnose_theta_components(
                log_batch, mask_batch
            )
            imp_all.append(theta_imp.detach().cpu())
            exp_all.append(theta_exp.detach().cpu())
            base_all.append(theta_base.detach().cpu())

    return (
        torch.cat(imp_all, dim=0),
        torch.cat(exp_all, dim=0),
        torch.cat(base_all, dim=0),
    )


def compute_train_theta_components(model: GNCDM,
                                   train_data: pd.DataFrame,
                                   batch_size: int = 256):
    """Build the train response matrix and return implicit/explicit/base theta."""
    dataset = IDCDataset(train_data, model.n_user, model.n_item)
    return compute_theta_components_from_matrices(
        model, dataset.log_mat, dataset.obs_mat, batch_size=batch_size
    )


def train(model: GNCDM, train_data: pd.DataFrame, valid_data: pd.DataFrame,
          batch_size, lr, n_epoch, early_stopping_patience=None,
          early_stopping_metric='rmse', early_stopping_min_delta=0.0,
          save_best_path=None, eval_setting='S1', full_Q_mat=None,
          train_original_item_ids=None):
    """
    """
    model.train()
    device = model.device
    dataset = IDCDataset(train_data, model.n_user, model.n_item)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=True)
    print(model.Theta_buf.is_leaf, model.Psi_buf.is_leaf)
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}], lr=lr
    )
    result_per_epoch = []

    best_metric = float('inf')
    bad_epoch = 0
    best_epoch = -1
    best_state_dict = None

    for epoch in range(n_epoch):
        # CHANGED: eval() leaves the module in evaluation mode. Restore training
        # mode at every epoch so dropout remains active after validation.
        model.train()
        result_epoch = {}
        pbar = tqdm(total=len(dataloader), desc='Epoch %d' % epoch)
        score_all = []
        pred_all = []
        Theta_old = model.get_Theta_buf().numpy().copy()

        for i, (user_log, item_log, user_mask, user_id, item_id, score) in enumerate(dataloader):

            user_log = user_log.to(device)
            item_log = item_log.to(device)
            user_mask = user_mask.to(device)
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            score = score.to(device)           # 0–10

            pred = model(user_log, item_log, user_mask, user_id, item_id)  # (batch,1) 0–10
            loss = F.mse_loss(pred, score)                                  # regression loss

            score_all += score.detach().cpu().numpy().reshape(-1,).tolist()
            pred_all += pred.detach().cpu().numpy().reshape(-1,).tolist()

            optimizer.zero_grad()
            loss.backward()            #loss.backward(retain_graph=True)
            optimizer.step()
            pbar.update(1)
        pbar.close()

        model.eval()

        for i in range(math.ceil(dataset.log_mat.shape[0] / batch_size)):  # ceil(345/16)=22
            idx = np.arange(i * batch_size,
                            min(dataset.log_mat.shape[0], (i+1) * batch_size))

            log_batch = torch.Tensor(dataset.log_mat[idx, :]).to(device)    # (b, n_item)
            mask_batch = torch.Tensor(dataset.obs_mat[idx, :]).to(device)   # (b, n_item)

            model.update_Theta_buf(
                model.diagnose_theta(log_batch, mask_batch).detach(),
                torch.LongTensor(idx)
            )

        # Update question features (Psi_buf)
        for i in range(math.ceil(dataset.log_mat.shape[1] / batch_size)):
            idx = np.arange(i * batch_size,
                            min(dataset.log_mat.shape[1], (i+1) * batch_size))
            model.update_Psi_buf(
                model.diagnose_psi(
                    torch.Tensor(dataset.log_mat[:, idx].T).to(device)
                ).detach(),
                torch.LongTensor(idx)
            )

        Theta = model.get_Theta_buf().numpy()
        Psi   = model.get_Psi_buf().numpy()
        print("Theta_buf std:", Theta.std(), "Psi_buf std:", Psi.std())
        print("Theta var per dim head:", Theta.var(axis=0)[:10])



        model.train()
        Theta_new = model.get_Theta_buf().numpy().copy()

        score_all = np.array(score_all)
        pred_all = np.array(pred_all)

        Theta_norm = np.sqrt(np.sum(np.abs(Theta_new - Theta_old)))
        mse  = mean_squared_error(score_all, pred_all)
        rmse = np.sqrt(mean_squared_error(score_all, pred_all))
        mae = mean_absolute_error(score_all, pred_all)

        print('Theta_old.head =', Theta_old[:5, 0])
        print('Theta_new.head =', Theta_new[:5, 0])
        print('epoch = %d, theta_norm = %.6f, rmse = %.6f mae = %.6f mse = %.6f'
              % (epoch, Theta_norm, rmse, mae, mse))

        result_epoch['Theta_old_head'] = Theta_old[:5, :5]
        result_epoch['Theta_new_head'] = Theta_new[:5, :5]
        result_epoch['Theta_norm'] = Theta_norm
        result_epoch['train_eval'] = {'rmse': rmse, 'mae': mae, 'mse': mse}

        if valid_data is not None:
            # CHANGED: validation must use the same inference protocol as test.
            # S2 uses theta diagnosed from new-user responses and psi from train Psi_buf.
            # S3 uses train-only mean theta and Q-neighbor psi; no valid parameters are fitted.
            # CHANGED FOR S3: compute the implicit population prior from
            # training learners only at the current epoch. Validation responses
            # are used only by the explicit branch and as reconstruction labels.
            s3_mean_theta_imp = None
            if str(eval_setting).upper() == 'S3':
                train_imp, _, _ = compute_theta_components_from_matrices(
                    model, dataset.log_mat, dataset.obs_mat,
                    batch_size=batch_size
                )
                s3_mean_theta_imp = train_imp.mean(dim=0)

            result_epoch['valid_eval'] = eval(
                model, valid_data, batch_size=batch_size, split_name='Valid',
                eval_setting=eval_setting, full_Q_mat=full_Q_mat,
                train_original_item_ids=train_original_item_ids,
                s3_mean_theta_imp=s3_mean_theta_imp
            )

            if early_stopping_patience is not None:
                current_metric = result_epoch['valid_eval'][early_stopping_metric]

                if current_metric < best_metric - early_stopping_min_delta:
                    best_metric = current_metric
                    bad_epoch = 0
                    best_epoch = epoch
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }

                    if save_best_path is not None:
                        torch.save(best_state_dict, save_best_path)

                    print(f"[EarlyStopping] New best at epoch {epoch}: "
                          f"{early_stopping_metric} = {current_metric:.6f}")
                else:
                    bad_epoch += 1
                    print(f"[EarlyStopping] No improvement: "
                          f"{bad_epoch}/{early_stopping_patience}")

                    if bad_epoch >= early_stopping_patience:
                        print(f"[EarlyStopping] Stop at epoch {epoch}. "
                              f"Best epoch = {best_epoch}, "
                              f"best {early_stopping_metric} = {best_metric:.6f}")

                        if best_state_dict is not None:
                            model.load_state_dict(best_state_dict)

                        result_epoch['early_stopped'] = True
                        result_epoch['best_epoch'] = best_epoch
                        result_epoch['best_metric'] = best_metric
                        result_per_epoch.append(result_epoch)
                        break

        result_per_epoch.append(result_epoch)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return result_per_epoch


def get_eval_result(s_true, s_pred):
    """
    Regression evaluation:
    """
    s_true = np.array(s_true).reshape(-1,)
    s_pred = np.array(s_pred).reshape(-1,)
    mse  = mean_squared_error(s_true, s_pred)
    rmse = np.sqrt(mean_squared_error(s_true, s_pred))
    mae = mean_absolute_error(s_true, s_pred)
    print('rmse = %.4f mae = %.4f mse = %.4f' % (rmse, mae, mse))
    return {'rmse': rmse, 'mae': mae, 'mse': mse}


def get_raw_prediction_diagnostics(s_pred):
    """
    """
    raw_pred = np.asarray(s_pred, dtype=float).reshape(-1,)

    if raw_pred.size == 0:
        return {
            'min': None,
            'max': None,
            'n_below_0': 0,
            'n_above_10': 0,
        }

    return {
        'min': float(raw_pred.min()),
        'max': float(raw_pred.max()),
        'n_below_0': int((raw_pred < 0.0).sum()),
        'n_above_10': int((raw_pred > 10.0).sum()),
    }


def _nearest_q_psi(model: GNCDM, q_new: torch.Tensor,
                   train_original_item_ids=None):
    """
    Deterministic train-only item fallback for a post-training unseen item.

    Neighbor selection:
    1. Compare the unseen item's Q-vector with every training-item Q-vector
       using cosine similarity.
    2. Select every training item tied at the highest *positive* similarity.
    3. Average their already-trained Psi_buf vectors.
    4. If no training item shares any required skill (maximum similarity <= 0),
       use the global mean of train Psi_buf.

    For binary Q-vectors, cosine similarity is:
        skill overlap / sqrt(#skills_new * #skills_train)

    No valid/test score, gradient, parameter fitting, buffer update, or new item
    slot is involved.
    """
    train_q = model.Q_mat.detach()
    train_psi = model.Psi_buf.detach()

    q_new = q_new.to(model.device).float().reshape(1, -1)
    train_norm = torch.linalg.vector_norm(train_q, dim=1).clamp(min=1e-12)
    new_norm = torch.linalg.vector_norm(q_new, dim=1).clamp(min=1e-12)
    sim = (train_q @ q_new.T).reshape(-1) / (train_norm * new_norm)

    max_sim = torch.max(sim)
    use_global_mean = bool(max_sim <= 0)
    if use_global_mean:
        neighbor_indices = torch.arange(
            train_psi.shape[0], device=train_psi.device
        )
        psi_new = train_psi.mean(dim=0)
    else:
        nearest = torch.isclose(sim, max_sim, rtol=1e-6, atol=1e-8)
        neighbor_indices = torch.where(nearest)[0]
        psi_new = train_psi[neighbor_indices].mean(dim=0)

    original_ids = None
    if train_original_item_ids is not None:
        original_ids = [
            int(train_original_item_ids[int(i)])
            for i in neighbor_indices.detach().cpu().tolist()
        ]

    audit = {
        'max_cosine_similarity': float(max_sim.detach().cpu()),
        'neighbor_internal_item_ids': [
            int(i) for i in neighbor_indices.detach().cpu().tolist()
        ],
        'neighbor_original_item_ids': original_ids,
        'used_global_mean_psi': use_global_mean,
        'all_train_similarities': [
            float(x) for x in sim.detach().cpu().tolist()
        ],
    }
    return psi_new, audit


def _build_s3_external_matrices(data: pd.DataFrame, full_Q_mat):
    """
    Build new-user x new-item response/mask matrices from S3 data.

    These matrices are used only by the explicit branch. Their item columns do
    not need to match the training-item coordinates because the matching
    external Q rows are passed alongside them.
    """
    user_col = 'original_user_id' if 'original_user_id' in data.columns else 'user_id'
    item_col = 'original_item_id' if 'original_item_id' in data.columns else 'item_id'

    required = {user_col, item_col, 'score'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'S3 data is missing columns: {sorted(missing)}')

    dup = data.duplicated([user_col, item_col], keep=False)
    if dup.any():
        examples = data.loc[dup, [user_col, item_col]].head().to_dict('records')
        raise ValueError(f'Duplicate S3 user-item responses found: {examples}')

    user_ids = sorted(data[user_col].astype(int).unique().tolist())
    item_ids = sorted(data[item_col].astype(int).unique().tolist())
    user_map = {old: new for new, old in enumerate(user_ids)}
    item_map = {old: new for new, old in enumerate(item_ids)}

    log_mat = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
    mask_mat = np.zeros_like(log_mat)
    for _, row in data.iterrows():
        u = user_map[int(row[user_col])]
        i = item_map[int(row[item_col])]
        score = float(row['score'])
        if np.isnan(score):
            continue
        log_mat[u, i] = (score / 10.0) * 2.0 - 1.0
        mask_mat[u, i] = 1.0

    full_q = np.asarray(full_Q_mat, dtype=np.float32)
    for item_id in item_ids:
        if item_id < 0 or item_id >= len(full_q):
            raise IndexError(
                f'original_item_id={item_id} is outside the full Q-matrix.'
            )
    q_new = full_q[item_ids]
    return user_col, item_col, user_ids, item_ids, log_mat, mask_mat, q_new


def _eval_s3_hybrid_theta(model: GNCDM, data: pd.DataFrame, full_Q_mat,
                          s3_mean_theta_imp: torch.Tensor,
                          save_pred_path=None, split_name='eval',
                          train_original_item_ids=None):
    """
    Open-world S3 reconstruction without new slots or retraining.

    Learner representation:
        theta_imp  = mean implicit theta of training learners only
        theta_exp  = explicit diagnosis from each new learner's observed
                     responses on unseen items and their external Q-vectors
        theta_base = (1-alpha)*theta_imp + alpha*theta_exp

    Item representation:
        psi_new = nearest-Q average of already-trained train Psi_buf vectors

    Valid/test responses enter the learner-specific explicit evidence because
    this experiment is response reconstruction. They never update a parameter
    or create a new item/user buffer slot.
    """
    if full_Q_mat is None:
        raise ValueError('S3 evaluation requires the full external Q-matrix.')
    if s3_mean_theta_imp is None:
        raise ValueError(
            'S3 evaluation requires mean implicit theta from training learners.'
        )

    (user_col, item_col, user_ids, item_ids,
     log_mat, mask_mat, q_new_np) = _build_s3_external_matrices(
        data, full_Q_mat
    )

    if train_original_item_ids is not None:
        overlap = set(item_ids) & set(map(int, train_original_item_ids))
        if overlap:
            raise ValueError(
                f'S3 requires unseen items, but overlap was found: {sorted(overlap)}'
            )

    model.eval()
    device = model.device
    with torch.no_grad():
        new_log = torch.tensor(log_mat, dtype=torch.float32, device=device)
        new_mask = torch.tensor(mask_mat, dtype=torch.float32, device=device)
        q_new = torch.tensor(q_new_np, dtype=torch.float32, device=device)

        theta_imp_users, theta_exp_users, theta_base_users = model.diagnose_theta_s3(
            new_log, new_mask, q_new, s3_mean_theta_imp
        )

        user_to_row = {uid: idx for idx, uid in enumerate(user_ids)}
        item_to_qrow = {iid: idx for idx, iid in enumerate(item_ids)}

        psi_cache = {}
        psi_audit = {}
        for item_id in item_ids:
            q_t = q_new[item_to_qrow[item_id]]
            psi_cache[item_id], info = _nearest_q_psi(
                model, q_t,
                train_original_item_ids=train_original_item_ids
            )
            info['new_original_item_id'] = int(item_id)
            info['new_q_vector'] = [float(x) for x in q_t.detach().cpu().tolist()]
            psi_audit[item_id] = info

        theta_rows, psi_rows, q_rows = [], [], []
        for _, row in data.iterrows():
            uid = int(row[user_col])
            iid = int(row[item_col])
            theta_rows.append(theta_base_users[user_to_row[uid]])
            psi_rows.append(psi_cache[iid])
            q_rows.append(q_new[item_to_qrow[iid]])

        theta_batch = torch.stack(theta_rows, dim=0)
        psi_batch = torch.stack(psi_rows, dim=0)
        q_batch = torch.stack(q_rows, dim=0)
        pred_raw = model.predict_response(
            theta_batch, psi_batch, q_batch
        ).detach().cpu().numpy().reshape(-1,)

    y_true = data['score'].astype(float).to_numpy()
    pred_clipped = np.clip(pred_raw, 0.0, 10.0)
    print(f'{split_name} S3 hybrid-theta reconstruction (clipped):')
    result = get_eval_result(y_true, pred_clipped)
    result['raw'] = get_eval_result(y_true, pred_raw)
    result['raw_prediction'] = get_raw_prediction_diagnostics(pred_raw)
    result['protocol'] = (
        'mean_train_theta_implicit + new_response_theta_explicit '
        '+ nearest_Q_train_psi'
    )
    result['theta_diagnostics'] = {
        'n_users': len(user_ids),
        'n_unique_theta_implicit': int(np.unique(
            np.round(theta_imp_users.detach().cpu().numpy(), 8), axis=0
        ).shape[0]),
        'n_unique_theta_explicit': int(np.unique(
            np.round(theta_exp_users.detach().cpu().numpy(), 8), axis=0
        ).shape[0]),
        'n_unique_theta_base': int(np.unique(
            np.round(theta_base_users.detach().cpu().numpy(), 8), axis=0
        ).shape[0]),
    }
    result['psi_neighbor_audit'] = [psi_audit[i] for i in item_ids]

    if save_pred_path is not None:
        out = data.copy()
        out['pred'] = pred_clipped
        out['pred_raw'] = pred_raw
        out['pred_clipped'] = pred_clipped
        out.to_csv(save_pred_path, index=False)
        print(f'Saved predictions to: {save_pred_path}')

        audit_path = os.path.splitext(save_pred_path)[0] + '_psi_neighbors.csv'
        pd.DataFrame(result['psi_neighbor_audit']).to_csv(audit_path, index=False)
        print(f'Saved Q-neighbor audit to: {audit_path}')

    return result


def eval(model: GNCDM, data: pd.DataFrame, batch_size, save_pred_path=None,
         split_name='eval', eval_setting='S1', full_Q_mat=None,
         train_original_item_ids=None, s3_mean_theta_imp=None):
    """
    CHANGED: use a protocol-specific inference path.

    S1: retains the original diagnostic behavior for backward compatibility.
    S2: theta is diagnosed from new-user responses; psi comes from train Psi_buf.
    S3: mean train implicit prior + explicit diagnosis from unseen responses;
        nearest-Q train psi; no new slots and no retraining.
    """
    eval_setting = str(eval_setting).upper()
    if eval_setting == 'S3':
        return _eval_s3_hybrid_theta(
            model, data, full_Q_mat=full_Q_mat,
            s3_mean_theta_imp=s3_mean_theta_imp,
            save_pred_path=save_pred_path, split_name=split_name,
            train_original_item_ids=train_original_item_ids
        )

    model.eval()
    device = model.device

    # CHANGED (S2): valid/test users are remapped locally, so the response matrix
    # uses only the number of users in the current split. It does not reserve
    # Theta_buf rows for them. The S2 path never sends item_log into g_nn.
    eval_n_user = model.n_user if eval_setting == 'S1' else int(data['user_id'].max()) + 1
    dataset = IDCDataset(data, eval_n_user, model.n_item)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    y_pred_raw = []
    y_pred_clipped = []
    y_true = []
    rows = []

    with torch.no_grad():
        for user_log, item_log, user_mask, user_id, item_id, score in dataloader:
            user_log = user_log.to(device)
            item_log = item_log.to(device)
            user_mask = user_mask.to(device)
            user_id = user_id.to(device)
            item_id = item_id.to(device)

            if eval_setting == 'S2':
                # CHANGED (S2): fresh theta from new learners + frozen train-item psi.
                # forward() is intentionally not used because it would call g_nn on
                # valid/test users and regenerate psi from their responses.
                pred_batch = model.forward_theta_log_psi_buf(
                    user_log, user_mask, user_id, item_id
                ).detach().cpu().numpy().reshape(-1,)
            else:
                # Original S1 behavior: recompute theta and psi from this split.
                pred_batch = model.forward(
                    user_log, item_log, user_mask, user_id, item_id
                ).detach().cpu().numpy().reshape(-1,)

            pred_clipped_batch = np.clip(pred_batch, 0.0, 10.0)
            score_batch = score.detach().cpu().numpy().reshape(-1,)
            y_pred_raw += pred_batch.tolist()
            y_pred_clipped += pred_clipped_batch.tolist()
            y_true += score_batch.tolist()

            if save_pred_path is not None:
                user_ids = user_id.detach().cpu().numpy().reshape(-1,)
                item_ids = item_id.detach().cpu().numpy().reshape(-1,)
                for u, q, s, p_raw, p_clip in zip(
                        user_ids, item_ids, score_batch,
                        pred_batch, pred_clipped_batch):
                    rows.append({
                        'user_id': int(u),
                        'item_id': int(q),
                        'score': float(s),
                        'pred': float(p_clip),
                        'pred_raw': float(p_raw),
                        'pred_clipped': float(p_clip),
                        'eval_setting': eval_setting,
                    })

    print(f'{split_name} {eval_setting} score reconstruction (clipped):')
    result = get_eval_result(y_true, y_pred_clipped)
    result['raw'] = get_eval_result(y_true, y_pred_raw)
    result['raw_prediction'] = get_raw_prediction_diagnostics(y_pred_raw)
    result['protocol'] = (
        'fresh_theta + train_Psi_buf' if eval_setting == 'S2'
        else 'fresh_theta + fresh_psi'
    )

    if save_pred_path is not None and rows:
        pd.DataFrame(rows).to_csv(save_pred_path, index=False)
        print(f'Saved predictions to: {save_pred_path}')
    return result


def train_AE(model: UAutoRec, train_data: pd.DataFrame, valid_data: pd.DataFrame,
             batch_size, lr, n_epoch):
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dataset = AEDataset(train_data, model.n_user, model.n_item)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(n_epoch):
        pbar = tqdm(total=len(dataloader), desc='Epoch %d' % epoch)
        avg_loss = 0
        count = 0
        for idx, (log_vec, obs_vec, user_id) in enumerate(dataloader):
            log_vec = log_vec.to(model.device)
            obs_vec = obs_vec.to(model.device)
            user_id = user_id.to(model.device)
            y_pred = model(log_vec, user_id).reshape(-1,)
            y_true = log_vec.reshape(-1,)
            obs = obs_vec.reshape(-1,)

            # loss
            loss = loss_func(y_pred * obs, y_true * obs)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            count += 1
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)
        print('Epoch %d done. loss = %.6f' % (epoch, avg_loss))
        pbar.close()
        eval_AE(model, train_data, valid_data)


def eval_AE(model: UAutoRec, train_data: pd.DataFrame, test_data: pd.DataFrame):
    y_pred_mat = []
    y_pred_all = []
    y_true_all = []
    train_dataset = AEDataset(train_data, model.n_user, model.n_item)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    test_dataset = AEDataset(test_data, model.n_user, model.n_item)
    model.eval()
    with torch.no_grad():
        for i, (log_vec, obs_vec, user_id) in enumerate(train_dataloader):
            log_vec = log_vec.to(model.device)
            user_id = user_id.to(model.device)
            y_pred = model(log_vec, user_id).detach().cpu().numpy()
            y_pred_mat += [elem for elem in y_pred]
    y_pred_mat = np.stack(y_pred_mat)
    xs, ys = np.where(test_dataset.obs_mat > 0)
    for i, j in zip(xs, ys):
        y_pred_all.append(y_pred_mat[i, j])
        y_true_all.append(test_dataset.log_mat[i, j])
    return get_eval_result(y_true_all, y_pred_all)
