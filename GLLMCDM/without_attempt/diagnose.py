"""Export deterministic no-attempt GLLMCDM mastery profiles.

This is the original post-training diagnosis workflow used to create the CP2
and CP4 theta files consumed by the RQ2 and RQ3 notebooks. Diagnosis uses the
same skill-evidence and model functions as training.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import GLLMCDM

# ============================================================
# CHANGED:
# reuse QMatrixBank + SkillEvidenceBuilder from train.py directly
# ============================================================
from train import QMatrixBank, SkillEvidenceBuilder


# ============================================================
# ============================================================
def load_run_args_json(path: Optional[str]) -> dict:
    """Load the saved training configuration when a path is provided."""
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"run_args_json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# ============================================================
def build_model_from_ckpt(
    model_path: str,
    device: torch.device,
    n_know: int,
    hidden: int,
    alpha: float,
) -> GLLMCDM:
    """Rebuild GLLMCDM, load a supported checkpoint, and enter evaluation mode."""
    net = GLLMCDM(
        n_know=n_know,
        hidden=hidden,
        alpha=alpha,
        monotonicity_assumption=True,
        device=device,
    )

    # FIX: use weights_only=False because this helper may also read the
    # full training checkpoint, not only a plain model state_dict.
    state = torch.load(model_path, map_location=device, weights_only=False)

    # FIX: distinguish a plain state_dict from wrapper/training checkpoints
    # by their CONTENTS. The old check "all keys are strings" was unsafe
    # because latest_checkpoint.pt also has string keys.
    if isinstance(state, dict) and state.get("best_state") is not None:
        # FIX: prefer best_state so diagnosis matches run.py model selection.
        state_dict = state["best_state"]
        source = "training checkpoint: best_state"
    elif isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
        source = "training checkpoint: model_state_dict"
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
        source = "dict['state_dict']"
    elif (
        isinstance(state, dict)
        and len(state) > 0
        and all(isinstance(k, str) for k in state.keys())
        and all(torch.is_tensor(v) for v in state.values())
    ):
        state_dict = state
        source = "plain state_dict"
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected a plain state_dict, "
            "a dict containing 'state_dict', or a training checkpoint containing "
            "'model_state_dict'/'best_state'."
        )

    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    net.eval()
    print(f">>> Loaded model weights from {source}")
    return net


# ============================================================
# ============================================================
def resolve_qbank(
    q_matrix_path: Optional[str],
    net: GLLMCDM,
    n_know: int,
    output_path: str,
) -> Tuple[QMatrixBank, str]:
    """Resolve the Q-matrix used to construct skill-level evidence."""
    # FIX: the current GLLMCDM checkpoint does not contain Q_mat.
    # Requiring the external Q file avoids an invalid fallback path and
    # guarantees that evidence is built with the intended item-skill mapping.
    if q_matrix_path is None or str(q_matrix_path).strip() == "":
        raise ValueError(
            "--q_matrix_path is required for the current GLLMCDM model because "
            "the checkpoint does not contain Q_mat."
        )

    qbank = QMatrixBank(q_matrix_path, n_know=n_know)
    return qbank, f">>> Using q-matrix: {q_matrix_path}"


# ============================================================
# ============================================================
def get_implicit_module(net: GLLMCDM):
    """Return the implicit diagnosis network used by this model version."""
    if hasattr(net, "skill_theta_nn"):
        return net.skill_theta_nn
    if hasattr(net, "f_nn"):
        return net.f_nn

    raise ValueError(
        "Cannot find implicit diagnosis network in model. "
        "Expected attribute 'skill_theta_nn' or 'f_nn'."
    )


# ============================================================
# ============================================================
def infer_implicit_mode(implicit_net, n_know: int) -> str:
    """Infer whether the implicit network expects a 2K or 4K feature vector."""
    first_layer = None
    for m in implicit_net.modules():
        if hasattr(m, "in_features"):
            first_layer = m
            break

    if first_layer is None:
        raise ValueError("Cannot infer implicit input dimension from the model.")

    in_features = int(first_layer.in_features)

    if in_features == 2 * int(n_know):
        return "2K"
    if in_features == 4 * int(n_know):
        return "4K"

    raise ValueError(
        f"Unsupported implicit input dimension: {in_features}. "
        f"Expected {2*n_know} (2K) or {4*n_know} (4K)."
    )


# ============================================================
# 5.1) helper: ensure tensor on device
# ============================================================
def ensure_tensor(x, device: torch.device) -> torch.Tensor:
    """Return a float tensor on the requested device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


# ============================================================
# 6) diagnose_user_level
# ============================================================
@torch.no_grad()
def diagnose_user_level(
    net: GLLMCDM,
    evidence_builder: SkillEvidenceBuilder,
    user_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagnose base, implicit, and explicit mastery for the requested users."""
    n_know = int(net.n_know)
    implicit_net = get_implicit_module(net)
    mode = infer_implicit_mode(implicit_net, n_know)

    theta_list = []
    theta_imp_list = []
    theta_exp_list = []

    for start in tqdm(range(0, len(user_ids), batch_size), desc="Diagnosing theta"):
        batch_user_ids = user_ids[start:start + batch_size]

        # ----------------------------------------------------
        # CHANGED:
        # ----------------------------------------------------
        skill_mean, skill_sum, skill_count, skill_max, skill_min, skill_std, _ = \
            evidence_builder.get_all_user_evidence(
                user_ids=batch_user_ids,
                exclude_item_ids=None,
                device=device,
            )

        # ----------------------------------------------------
        # CHANGED:
        # ----------------------------------------------------
        skill_mean = ensure_tensor(skill_mean, device)
        skill_sum = ensure_tensor(skill_sum, device)
        skill_count = ensure_tensor(skill_count, device)
        skill_max = ensure_tensor(skill_max, device)
        skill_min = ensure_tensor(skill_min, device)
        skill_std = ensure_tensor(skill_std, device)

        # ----------------------------------------------------
        # ----------------------------------------------------
        if mode == "2K":
            feat_imp = torch.cat([skill_mean, torch.tanh(skill_sum)], dim=1)   # (B, 2K)
        else:
            feat_imp = torch.cat([skill_mean, skill_max, skill_min, skill_std], dim=1)  # (B, 4K)

        # ----------------------------------------------------
        # implicit branch
        # ----------------------------------------------------
        theta_imp_batch = implicit_net(feat_imp)           # (B, K)

        # ----------------------------------------------------
        # explicit branch
        # theta_exp = sigmoid(skill_mean)
        # ----------------------------------------------------
        theta_exp_batch = torch.sigmoid(skill_mean)        # (B, K)

        # ----------------------------------------------------
        # combined theta
        # ----------------------------------------------------
        # FIX: for the current 4K model, call the model's own diagnosis
        # function so exported theta_base cannot drift from train/eval logic.
        if mode == "4K":
            theta_batch = net.diagnose_theta_skill(
                skill_mean,
                skill_max,
                skill_min,
                skill_std,
            )
        else:
            # Legacy 2K compatibility only.
            theta_batch = theta_imp_batch * (1 - net.alpha) + theta_exp_batch * net.alpha

        theta_list.append(theta_batch.detach().cpu().numpy())
        theta_imp_list.append(theta_imp_batch.detach().cpu().numpy())
        theta_exp_list.append(theta_exp_batch.detach().cpu().numpy())

    theta_mat = np.concatenate(theta_list, axis=0)
    theta_imp_mat = np.concatenate(theta_imp_list, axis=0)
    theta_exp_mat = np.concatenate(theta_exp_list, axis=0)

    return theta_mat, theta_imp_mat, theta_exp_mat


# ============================================================
# 7) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # input files
    # --------------------------------------------------------
    parser.add_argument("--evidence_file", type=str, required=True,
                        help="CSV with columns: user_id,item_id,score")

    # FIX: current GLLMCDM checkpoints do not store Q_mat, so diagnosis
    # requires the external q-matrix corresponding to the supplied evidence.
    parser.add_argument("--q_matrix_path", type=str, required=True,
                        help="q-matrix path corresponding to evidence_file")

    parser.add_argument("--model_path", type=str, required=True,
                        help="path to gllmcdm_stage1_policy.pt")
    parser.add_argument("--run_args_json", type=str, default=None,
                        help="optional path to run_args.json saved from training")
    parser.add_argument("--output_path", type=str, required=True)

    # optional override model config
    parser.add_argument("--n_know", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # --------------------------------------------------------
    # 7.1 load evidence file
    # --------------------------------------------------------
    df_log = pd.read_csv(args.evidence_file)

    need = {"user_id", "item_id", "score"}
    if not need.issubset(df_log.columns):
        raise ValueError(f"evidence_file must contain columns: {sorted(list(need))}")

    df_log = df_log.copy()
    df_log["user_id"] = pd.to_numeric(df_log["user_id"], errors="raise").astype(int)
    df_log["item_id"] = pd.to_numeric(df_log["item_id"], errors="raise").astype(int)
    df_log["score"] = pd.to_numeric(df_log["score"], errors="coerce").astype(float)

    user_ids = np.sort(df_log["user_id"].unique().astype(int))

    # --------------------------------------------------------
    # --------------------------------------------------------
    train_args = load_run_args_json(args.run_args_json)

    n_know = args.n_know if args.n_know is not None else int(train_args.get("n_know", 18))
    hidden = args.hidden if args.hidden is not None else int(train_args.get("hidden", 128))
    alpha = args.alpha if args.alpha is not None else float(train_args.get("alpha", 0.5))

    print(f"Resolved model config: n_know={n_know}, hidden={hidden}, alpha={alpha}")
    print(f"Rows in evidence: {len(df_log)}")
    print(f"Unique users: {len(user_ids)}")

    # --------------------------------------------------------
    # 7.3 load model
    # --------------------------------------------------------
    device = torch.device(args.device)
    net = build_model_from_ckpt(
        model_path=args.model_path,
        device=device,
        n_know=n_know,
        hidden=hidden,
        alpha=alpha,
    )
    print(f"Using device: {device}")

    # --------------------------------------------------------
    # 7.4 build qbank
    # FIX: use the external q-matrix that corresponds to this evidence file.
    # --------------------------------------------------------
    qbank, q_msg = resolve_qbank(
        q_matrix_path=args.q_matrix_path,
        net=net,
        n_know=n_know,
        output_path=args.output_path,
    )
    print(q_msg)

    # --------------------------------------------------------
    # 7.5 build evidence builder
    # --------------------------------------------------------
    evidence_builder = SkillEvidenceBuilder(
        df_log,
        qbank=qbank,
        max_score=10.0,
        device=device
    )

    # --------------------------------------------------------
    # 7.6 diagnose theta
    # --------------------------------------------------------
    theta, theta_implicit, theta_explicit = diagnose_user_level(
        net=net,
        evidence_builder=evidence_builder,
        user_ids=user_ids,
        batch_size=args.batch_size,
        device=device,
    )

    # --------------------------------------------------------
    # 7.7 prepare output tables
    # --------------------------------------------------------
    col_theta = [f"K{i}" for i in range(n_know)]

    # combined theta
    df_theta = pd.DataFrame(theta, columns=col_theta)
    df_theta.insert(0, "user_id", user_ids.astype(int))

    # implicit theta
    df_theta_imp = pd.DataFrame(theta_implicit, columns=col_theta)
    df_theta_imp.insert(0, "user_id", user_ids.astype(int))

    # explicit theta
    df_theta_exp = pd.DataFrame(theta_explicit, columns=col_theta)
    df_theta_exp.insert(0, "user_id", user_ids.astype(int))

    # --------------------------------------------------------
    # 7.8 save outputs
    # --------------------------------------------------------
    theta_csv = os.path.join(args.output_path, "theta_new.csv")
    theta_npy = os.path.join(args.output_path, "theta_new.npy")

    theta_imp_csv = os.path.join(args.output_path, "theta_implicit.csv")
    theta_imp_npy = os.path.join(args.output_path, "theta_implicit.npy")

    theta_exp_csv = os.path.join(args.output_path, "theta_explicit.csv")
    theta_exp_npy = os.path.join(args.output_path, "theta_explicit.npy")

    df_theta.to_csv(theta_csv, index=False)
    np.save(theta_npy, theta.astype(np.float32))

    df_theta_imp.to_csv(theta_imp_csv, index=False)
    np.save(theta_imp_npy, theta_implicit.astype(np.float32))

    df_theta_exp.to_csv(theta_exp_csv, index=False)
    np.save(theta_exp_npy, theta_explicit.astype(np.float32))

    print(">>> Saved combined theta:")
    print("   ", theta_csv)
    print("   ", theta_npy)

    print(">>> Saved implicit theta:")
    print("   ", theta_imp_csv)
    print("   ", theta_imp_npy)

    print(">>> Saved explicit theta:")
    print("   ", theta_exp_csv)
    print("   ", theta_exp_npy)


if __name__ == "__main__":
    main()


