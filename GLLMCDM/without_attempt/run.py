# Purpose: Parse arguments and run the no-attempt training/evaluation workflow.
# Provenance: Original research file; workflow logic is unchanged.
# run.py
# -*- coding: utf-8 -*-
"""
============================================================
run.py
============================================================
   - gllmcdm_stage1_policy.pt
   - train_history.json
   - test_result.json
   - pred_test.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random  # FIX: reproducible local initialization/shuffling/policy sampling

import numpy as np  # FIX: seed + finite-data/Q validation
import pandas as pd
import torch

from pathlib import Path
from dotenv import load_dotenv

from model import GLLMCDM
from train import (
    ItemTextBank,
    QMatrixBank,
    DummyScorer,
    TrainConfig,
    train_reinforce,
    evaluate,
    export_theta,
)


def parse_args():
    p = argparse.ArgumentParser()

    # long-form logs
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--valid_csv", type=str, default="")
    p.add_argument("--test_csv", type=str, default="")

    # banks
    p.add_argument("--item_text_csv", type=str, required=True)
    p.add_argument("--q_matrix_path", type=str, required=True)

    # output
    p.add_argument("--save_dir", type=str, required=True)

    # ADDED: optionally export deterministic theta after best_state is restored
    p.add_argument(
        "--export_theta",
        action="store_true",
        help="Export theta_base from the restored best validation epoch",
    )
    p.add_argument(
        "--theta_evidence",
        type=str,
        choices=["train", "valid"],
        default="train",
        help="Evidence split used to build best-epoch theta (default: train)",
    )
    p.add_argument(
        "--theta_export_batch_size",
        type=int,
        default=256,
        help="Batch size used only while exporting theta",
    )

    # ADDED: continue from latest_checkpoint.pt in save_dir
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint in save_dir",
    )

    # model hyperparams
    p.add_argument("--n_know", type=int, default=18)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.5)

    p.add_argument("--use_openai", action="store_true", help="Use OpenAI scorer instead of DummyScorer")
    p.add_argument("--openai_model", type=str, default="gpt-5.4-nano", help="OpenAI model name")

    # training hyperparams
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n_epoch", type=int, default=10)

    # reward/baseline
    p.add_argument("--reward_scale_c", type=float, default=100.0)
    p.add_argument("--baseline_beta", type=float, default=0.05)

    p.add_argument("--use_early_stopping", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--early_stop_min_delta", type=float, default=0.01)
    p.add_argument("--monitor_metric", type=str, default="rmse")

    # CHANGED:
    # default = False because this project now wants to use all evidence to build theta
    p.add_argument("--use_leave_one_out", action="store_true",
               help="If set, exclude the target item from user evidence (old behavior). Default: False")

    p.add_argument("--openai_reasoning_effort", type=str, default="high",
               choices=["none", "low", "medium", "high", "xhigh"])
    p.add_argument("--openai_verbosity", type=str, default="low",
               choices=["low", "medium", "high"])
    p.add_argument("--openai_reasoning_summary", type=str, default="none",
               choices=["none", "auto", "concise", "detailed"])
    p.add_argument("--openai_store", action="store_true",
               help="Store Responses API responses")
    p.add_argument("--openai_temperature", type=float, default=0.0,
               help="Only used when reasoning_effort='none'")

    # FIX: make model initialization, DataLoader shuffling, and Gaussian sampling
    # reproducible on the local PyTorch/NumPy/Python side.
    p.add_argument("--seed", type=int, default=2026)

    # device
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()



def main():
    args = parse_args()

    # FIX: set seeds before model construction and DataLoader iteration.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Random seed: {args.seed}")

    # ------------------------
    # Load .env
    # ------------------------
    BASE_DIR = Path(__file__).resolve().parent
    ENV_PATH = BASE_DIR / ".env"
    load_dotenv(ENV_PATH, override=True)

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_APIKEY")
        or os.getenv("OPENAI_KEY")
    )

    if args.use_openai and not api_key:
        raise ValueError("Cannot find OpenAI API key in .env or environment.")

    # FIX: Dummy/offline runs must still work when no API key is configured.
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # ADDED: fixed checkpoint filename inside save_dir
    checkpoint_path = os.path.join(
        args.save_dir,
        "latest_checkpoint.pt",
    )

    print("Checkpoint path:", checkpoint_path)
    print("Resume requested:", args.resume)

    # ------------------------
    # Load data
    # ------------------------
    df_train = pd.read_csv(args.train_csv)
    df_valid = pd.read_csv(args.valid_csv) if args.valid_csv else None
    df_test = pd.read_csv(args.test_csv) if args.test_csv else None

    # sanity check columns
    need = {"user_id", "item_id", "score"}
    for name, df in [("train", df_train), ("valid", df_valid), ("test", df_test)]:
        if df is None:
            continue
        if not need.issubset(df.columns):
            raise ValueError(f"{name}_csv must contain columns {sorted(list(need))}")

        # FIX: fail before expensive LLM calls if score data are invalid.
        score_num = pd.to_numeric(df["score"], errors="coerce")
        if score_num.isna().any():
            raise ValueError(f"{name}_csv contains missing/non-numeric score values.")
        if not np.isfinite(score_num.to_numpy(dtype=float)).all():
            raise ValueError(f"{name}_csv contains non-finite score values.")
        if ((score_num < 0.0) | (score_num > 10.0)).any():
            raise ValueError(f"{name}_csv contains score values outside [0,10].")

    # ------------------------
    # Build banks
    # ------------------------
    text_bank = ItemTextBank(args.item_text_csv)
    q_bank = QMatrixBank(args.q_matrix_path, n_know=args.n_know)

    # FIX: validate item-text and Q coverage before any training/API calls.
    used_item_ids = set()
    for df in (df_train, df_valid, df_test):
        if df is not None:
            used_item_ids.update(df["item_id"].astype(int).unique().tolist())
    for iid in sorted(used_item_ids):
        _ = text_bank.get_text(iid)
        q = q_bank.get_q(iid)
        if not np.isfinite(q).all():
            raise ValueError(f"Q-vector for item_id={iid} contains non-finite values.")
        if not np.any(q > 0):
            raise ValueError(f"Q-vector for item_id={iid} has no positive required skill.")

    # ------------------------
    # Build model
    # ------------------------
    model = GLLMCDM(
        n_know=args.n_know,
        hidden=args.hidden,
        alpha=args.alpha,
        monotonicity_assumption=True,  # we want FC+ monotonic diagnosis
        device=device,
    )

    # ------------------------
    # Scorer (choose)
    # ------------------------
    if args.use_openai:
        from train import OpenAIScorer
        scorer = OpenAIScorer(
            model_name=args.openai_model,
            temperature=args.openai_temperature,
            reasoning_effort=args.openai_reasoning_effort,
            verbosity=args.openai_verbosity,
            store=args.openai_store,
        )
    else:
        scorer = DummyScorer()

    # ------------------------
    # Training config
    # ------------------------
    config = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        n_epoch=args.n_epoch,
        reward_scale_c=args.reward_scale_c,
        baseline_beta=args.baseline_beta,
        clip_pred_0_10=True,
        use_early_stopping=args.use_early_stopping,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        monitor_metric=args.monitor_metric,
        use_leave_one_out=args.use_leave_one_out,  # CHANGED

        # ADDED: pass checkpoint settings to train.py
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )

    # ------------------------
    # Train
    # ------------------------
    history = train_reinforce(
        model=model,
        train_df=df_train,
        valid_df=df_valid,
        text_bank=text_bank,
        q_bank=q_bank,
        scorer=scorer,
        config=config,
        device=device,
    )

    # ------------------------
    # Save artifacts
    # ------------------------
    best_model_path = os.path.join(args.save_dir, "gllmcdm_stage1_policy.pt")
    torch.save(model.state_dict(), best_model_path)

    # ADDED: export theta only after train_reinforce() has restored best_state.
    # This keeps the existing training, logit/policy, and checkpoint logic
    # unchanged. Validation chooses the best epoch; --theta_evidence chooses
    # which learner responses are used to derive theta from those best weights.
    if args.export_theta:
        best_records = [h for h in history if h.get("is_best") is True]
        if not best_records:
            raise ValueError(
                "--export_theta requires validation data so a best epoch can be selected."
            )

        best_record = best_records[-1]
        if args.theta_evidence == "train":
            theta_evidence_df = df_train
            theta_evidence_path = args.train_csv
        else:
            if df_valid is None:
                raise ValueError(
                    "--theta_evidence valid requires --valid_csv."
                )
            theta_evidence_df = df_valid
            theta_evidence_path = args.valid_csv

        theta_csv_path = os.path.join(args.save_dir, "theta_base_best.csv")
        theta_npy_path = os.path.join(args.save_dir, "theta_base_best.npy")
        theta_implicit_csv_path = os.path.join(
            args.save_dir,
            "theta_implicit_best.csv",
        )
        theta_implicit_npy_path = os.path.join(
            args.save_dir,
            "theta_implicit_best.npy",
        )
        theta_explicit_csv_path = os.path.join(
            args.save_dir,
            "theta_explicit_best.csv",
        )
        theta_explicit_npy_path = os.path.join(
            args.save_dir,
            "theta_explicit_best.npy",
        )
        theta_export_result = export_theta(
            model=model,
            evidence_df=theta_evidence_df,
            q_bank=q_bank,
            output_csv_path=theta_csv_path,
            output_npy_path=theta_npy_path,
            output_implicit_csv_path=theta_implicit_csv_path,
            output_implicit_npy_path=theta_implicit_npy_path,
            output_explicit_csv_path=theta_explicit_csv_path,
            output_explicit_npy_path=theta_explicit_npy_path,
            batch_size=args.theta_export_batch_size,
            device=device,
        )

        # ADDED: save provenance so the exported best-epoch theta files can be
        # reproduced with the same model, evidence, Q-matrix, and settings.
        theta_metadata = {
            "theta_type": "theta_base_deterministic",
            "best_epoch_zero_based": int(best_record["epoch"]),
            "monitor_metric": args.monitor_metric,
            "best_metric": float(best_record["valid"][args.monitor_metric]),
            "evidence_split": args.theta_evidence,
            "evidence_file": theta_evidence_path,
            "q_matrix_path": args.q_matrix_path,
            "model_file": best_model_path,
            "n_know": int(args.n_know),
            "hidden": int(args.hidden),
            "alpha": float(args.alpha),
            **theta_export_result,
        }
        with open(
            os.path.join(args.save_dir, "theta_best_metadata.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(theta_metadata, f, ensure_ascii=False, indent=2)

        print(
            f"Exported theta from best epoch {best_record['epoch']} "
            f"using {args.theta_evidence} evidence."
        )

    with open(os.path.join(args.save_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.save_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    rows = []
    for h in history:
        row = {
            "epoch": h.get("epoch"),
            "train_loss": h.get("train_loss"),
            "train_reward": h.get("train_reward"),
            "train_rmse": h.get("train_rmse"),
            "baseline": h.get("baseline"),
            "valid_rmse": None,
            "valid_mae": None,
            "valid_mse": None,
            "is_best": h.get("is_best"),
            "patience_count": h.get("patience_count"),
        }

        valid = h.get("valid")
        if isinstance(valid, dict):
            row["valid_rmse"] = valid.get("rmse")
            row["valid_mae"] = valid.get("mae")
            row["valid_mse"] = valid.get("mse")

        for k, v in h.items():
            if k.startswith("best_"):
                row[k] = v

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(args.save_dir, "train_history_flat.csv"),
        index=False
    )

    # ------------------------
    # Test + pred_test.csv
    # ------------------------
    if df_test is not None:
        pred_path = os.path.join(args.save_dir, "pred_test.csv")
        res = evaluate(model, df_test, text_bank, q_bank, scorer, config, device, save_pred_path=pred_path)

        with open(os.path.join(args.save_dir, "test_result.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        print("Test:", res)

    print("Saved to:", args.save_dir)


if __name__ == "__main__":
    main()
