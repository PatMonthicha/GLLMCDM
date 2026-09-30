# Purpose: Train and evaluate full GLLMCDM using the prompt with learner attempts.
# Provenance: Original research file; training and prompt logic are unchanged.
# train.py
# -*- coding: utf-8 -*-
"""
============================================================
train.py (REINFORCE training + evaluation + pred_test.csv)
============================================================

------------------------------

DATA INPUTS
-----------
1) train/valid/test csv (long format):
      user_id, item_id, score   (score ∈ [0,10])

2) item_text_csv:
      item_id, item_text

3) q_matrix_path:
   - .csv: item_id,K0..K17

4) NEW: attempt master csv (optional but supported now):
      user_id, item_id, attempt_clean

CORE IDEA: UNLOCK n_item
------------------------
   skill_mean  (K,)
   skill_sum   (K,)
   skill_count (K,)
   skill_max   (K,)
   skill_min   (K,)
   skill_std   (K,)

RL ELEMENTS
-----------
context x(u,i):

action a:
  - theta_action ∈ (0,1)^K  (sampled by policy)

reward r:
  - r = -(y_hat - y_true)^2 / c   (c=100)

done:
  - True (one-step bandit)

OUTPUTS
-------
- test_result.json
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import os
import re
import hashlib
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import openai
from openai import OpenAI
from dotenv import load_dotenv


import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error

from model import GLLMCDM


# ============================================================
# 1) ItemTextBank: item_id -> item_text
# ============================================================
class ItemTextBank:
    """
    Purpose:

    File format:
      item_id,item_text
    """

    def __init__(self, item_text_csv: str):
        df = pd.read_csv(item_text_csv)
        if not {"item_id", "item_text"}.issubset(df.columns):
            raise ValueError("item_text_csv must contain columns: item_id,item_text")
        if df["item_id"].duplicated().any():
            dup = df[df["item_id"].duplicated()]["item_id"].tolist()
            raise ValueError(f"Duplicate item_id in item_text_csv: {dup[:10]}")
        self.text_map: Dict[int, str] = {int(r["item_id"]): str(r["item_text"]) for _, r in df.iterrows()}

    def get_text(self, item_id: int) -> str:
        """Input: item_id (int) -> Output: item_text (str)"""
        item_id = int(item_id)
        if item_id not in self.text_map:
            raise KeyError(f"item_id={item_id} not found in item_text_csv")
        return self.text_map[item_id]


# ============================================================
# 1.5) NEW: AttemptBank: (user_id, item_id) -> attempt
# ============================================================
class AttemptBank:
    """
    Purpose:

    Expected file format:
      user_id,item_id,attempt_clean
      - user_id
      - item_id
      - attempt_clean

    Logic:
    """

    def __init__(self, attempt_csv: str):
        df = pd.read_csv(attempt_csv)

        need = {"user_id", "item_id", "attempt_clean"}
        if not need.issubset(df.columns):
            raise ValueError(f"attempt_csv must contain columns: {sorted(list(need))}")

        df = df.copy()
        df["user_id"] = pd.to_numeric(df["user_id"], errors="raise").astype(int)
        df["item_id"] = pd.to_numeric(df["item_id"], errors="raise").astype(int)
        df["attempt_clean"] = df["attempt_clean"].fillna("").astype(str)

        if df.duplicated(subset=["user_id", "item_id"]).any():
            dup = df[df.duplicated(subset=["user_id", "item_id"], keep=False)].copy()
            raise ValueError(
                "Duplicate (user_id, item_id) found in attempt_csv. "
                f"Examples:\n{dup[['user_id','item_id']].head(10)}"
            )

        self.attempt_map = {
            (int(r["user_id"]), int(r["item_id"])): str(r["attempt_clean"])
            for _, r in df.iterrows()
        }

    def get_attempt(self, user_id: int, item_id: int) -> str:
        key = (int(user_id), int(item_id))

        if key not in self.attempt_map:
            return "[NO SUBMISSION]"

        attempt = str(self.attempt_map[key]).strip()
        if attempt == "":
            return "[EMPTY SUBMISSION]"

        return attempt


# ============================================================
# 2) QMatrixBank: item_id -> q_vector (K,)
# ============================================================
class QMatrixBank:
    """
    Purpose:

    Two modes:
    """

    def __init__(self, q_path: str, n_know: int = 18):
        self.n_know = int(n_know)

        if q_path.endswith(".npy"):
            Q = np.load(q_path)
            if Q.ndim != 2 or Q.shape[1] != self.n_know:
                raise ValueError(f"Q .npy must have shape (n_items,{self.n_know}), got {Q.shape}")
            self.mode = "npy"
            self.Q = Q.astype(np.float32)  # (n_items,K)
            self.map = None
        else:
            df = pd.read_csv(q_path)
            required = ["item_id"] + [f"K{i}" for i in range(self.n_know)]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Q .csv missing columns: {missing}")
            if df["item_id"].duplicated().any():
                dup = df[df["item_id"].duplicated()]["item_id"].tolist()
                raise ValueError(f"Duplicate item_id in Q csv: {dup[:10]}")

            self.mode = "csv"
            self.Q = None
            self.map = {
                int(r["item_id"]): r[[f"K{i}" for i in range(self.n_know)]].astype(float).values.astype(np.float32)
                for _, r in df.iterrows()
            }

    def get_q(self, item_id: int) -> np.ndarray:
        """
        Output:
          q_vector: (K,) float32
        """
        item_id = int(item_id)
        if self.mode == "npy":
            if item_id < 0 or item_id >= self.Q.shape[0]:
                raise KeyError(f"item_id={item_id} out of range for Q npy (0..{self.Q.shape[0]-1})")
            return self.Q[item_id]  # (K,)
        else:
            assert self.map is not None
            if item_id not in self.map:
                raise KeyError(f"item_id={item_id} not found in Q csv")
            return self.map[item_id]  # (K,)


# ============================================================
# 3) SkillEvidenceBuilder: build fixed-size evidence per user
# ============================================================
class SkillEvidenceBuilder:
    """

    For each user u, for each skill k:
      signed_score: score ∈ [0,10] -> signed ∈ [-1,1]
        signed = (score/10)*2 - 1

      skill_mean
      skill_sum
      skill_count
      skill_max
      skill_min
      skill_std

    # CHANGED:
    """

    def __init__(
        self,
        df_log: pd.DataFrame,
        qbank: QMatrixBank,
        max_score: float = 10.0,
        device: torch.device = torch.device("cpu"),
    ):
        need = {"user_id", "item_id", "score"}
        if not need.issubset(df_log.columns):
            raise ValueError(f"df_log must contain columns: {sorted(list(need))}")

        self.df = df_log.copy().reset_index(drop=True)
        self.df["user_id"] = pd.to_numeric(self.df["user_id"], errors="raise").astype(int)
        self.df["item_id"] = pd.to_numeric(self.df["item_id"], errors="raise").astype(int)
        self.df["score"] = pd.to_numeric(self.df["score"], errors="coerce").astype(float)

        self.qbank = qbank
        self.max_score = float(max_score)
        self.device = device
        self.n_know = qbank.n_know

        # group by user for faster evidence building
        grouped = {}
        for _, r in self.df.iterrows():
            uid = int(r["user_id"])
            iid = int(r["item_id"])
            sc = float(r["score"]) if not pd.isna(r["score"]) else np.nan
            grouped.setdefault(uid, []).append((iid, sc))
        self.user_rows: Dict[int, List[Tuple[int, float]]] = grouped
        self.user_ids = np.sort(np.array(list(self.user_rows.keys()), dtype=int))

        # cache only for no-exclude case
        self._cache_no_exclude: Dict[int, Tuple[np.ndarray, ...]] = {}

    def score_to_signed(self, score: float) -> float:
        """0..10 -> -1..1"""
        return (float(score) / self.max_score) * 2.0 - 1.0

    def get_user_evidence(
        self,
        user_id: int,
        exclude_item_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Output:
          skill_mean  (K,)
          skill_sum   (K,)
          skill_count (K,)
          skill_max   (K,)
          skill_min   (K,)
          skill_std   (K,)
        """
        user_id = int(user_id)

        if exclude_item_id is None and user_id in self._cache_no_exclude:
            return self._cache_no_exclude[user_id]

        rows = self.user_rows.get(user_id, [])
        K = self.n_know

        vals_per_skill = [[] for _ in range(K)]

        for iid, sc in rows:
            if exclude_item_id is not None and int(iid) == int(exclude_item_id):
                continue
            if pd.isna(sc):
                continue

            q = self.qbank.get_q(iid)   # (K,)
            signed = self.score_to_signed(float(sc))

            for k in range(K):
                if q[k] > 0:
                    vals_per_skill[k].append(float(signed))

        skill_sum = np.zeros(K, dtype=np.float32)
        skill_count = np.zeros(K, dtype=np.float32)
        skill_mean = np.zeros(K, dtype=np.float32)
        skill_max = np.zeros(K, dtype=np.float32)
        skill_min = np.zeros(K, dtype=np.float32)
        skill_std = np.zeros(K, dtype=np.float32)

        for k in range(K):
            vals = vals_per_skill[k]
            if len(vals) == 0:
                continue

            arr = np.asarray(vals, dtype=np.float32)
            skill_sum[k] = float(arr.sum())
            skill_count[k] = float(len(arr))
            skill_mean[k] = float(arr.mean())
            skill_max[k] = float(arr.max())
            skill_min[k] = float(arr.min())
            skill_std[k] = float(arr.std(ddof=0))

        out = (
            skill_mean.astype(np.float32),
            skill_sum.astype(np.float32),
            skill_count.astype(np.float32),
            skill_max.astype(np.float32),
            skill_min.astype(np.float32),
            skill_std.astype(np.float32),
        )

        if exclude_item_id is None:
            self._cache_no_exclude[user_id] = out

        return out

    # CHANGED:
    def get_all_user_evidence(self, user_ids=None, exclude_item_ids=None, device: Optional[torch.device] = None):
        """
        Return stacked evidence for many users.

        Args:
            user_ids: iterable of user ids; if None use all users in builder
            exclude_item_ids:
                - None
                - scalar
                - iterable with same length as user_ids
            device: if provided, return torch tensors on this device

        Return:
            skill_mean:  (B, K)
            skill_sum:   (B, K)
            skill_count: (B, K)
            skill_max:   (B, K)
            skill_min:   (B, K)
            skill_std:   (B, K)
            user_ids_np: (B,)
        """
        if user_ids is None:
            user_ids = self.user_ids
        user_ids = np.asarray(list(user_ids), dtype=int)

        if exclude_item_ids is None:
            exclude_item_ids = [None] * len(user_ids)
        elif np.isscalar(exclude_item_ids):
            exclude_item_ids = [exclude_item_ids] * len(user_ids)
        else:
            exclude_item_ids = list(exclude_item_ids)
            if len(exclude_item_ids) != len(user_ids):
                raise ValueError("exclude_item_ids must have same length as user_ids")

        mean_list, sum_list, cnt_list = [], [], []
        max_list, min_list, std_list = [], [], []

        for uid, ex_iid in zip(user_ids, exclude_item_ids):
            s_mean, s_sum, s_cnt, s_max, s_min, s_std = self.get_user_evidence(
                user_id=int(uid),
                exclude_item_id=ex_iid,
            )
            mean_list.append(s_mean)
            sum_list.append(s_sum)
            cnt_list.append(s_cnt)
            max_list.append(s_max)
            min_list.append(s_min)
            std_list.append(s_std)

        skill_mean = np.stack(mean_list, axis=0).astype(np.float32)
        skill_sum = np.stack(sum_list, axis=0).astype(np.float32)
        skill_count = np.stack(cnt_list, axis=0).astype(np.float32)
        skill_max = np.stack(max_list, axis=0).astype(np.float32)
        skill_min = np.stack(min_list, axis=0).astype(np.float32)
        skill_std = np.stack(std_list, axis=0).astype(np.float32)

        if device is not None:
            skill_mean = torch.tensor(skill_mean, dtype=torch.float32, device=device)
            skill_sum = torch.tensor(skill_sum, dtype=torch.float32, device=device)
            skill_count = torch.tensor(skill_count, dtype=torch.float32, device=device)
            skill_max = torch.tensor(skill_max, dtype=torch.float32, device=device)
            skill_min = torch.tensor(skill_min, dtype=torch.float32, device=device)
            skill_std = torch.tensor(skill_std, dtype=torch.float32, device=device)

        return skill_mean, skill_sum, skill_count, skill_max, skill_min, skill_std, user_ids


# ============================================================
# 4) Dataset: one row = one (u,i,y) interaction
# ============================================================
class BanditDataset(Dataset):
    """
    For each row idx in df_log (u,i,y):
      - item_text = text_bank[item_id]
      - q_vector  = q_bank[item_id]
      - NEW: attempt = attempt_bank[(u,i)] if available
      - evidence  = builder.get_user_evidence(...)

    # CHANGED:

    Returns dict:
      user_id, item_id, y_true,
      item_text,
      attempt,                 # NEW
      q_vector (K,),
      skill_mean (K,), skill_sum (K,), skill_count (K,),
      skill_max (K,), skill_min (K,), skill_std (K,)
    """

    def __init__(
        self,
        df_log: pd.DataFrame,
        text_bank: ItemTextBank,
        q_bank: QMatrixBank,
        attempt_bank: Optional[AttemptBank] = None,   # NEW
        max_score: float = 10.0,
        use_leave_one_out: bool = False,   # CHANGED
        device: torch.device = torch.device("cpu"),
    ):
        self.df = df_log.copy().reset_index(drop=True)
        self.text_bank = text_bank
        self.q_bank = q_bank
        self.attempt_bank = attempt_bank   # NEW
        self.use_leave_one_out = bool(use_leave_one_out)
        self.builder = SkillEvidenceBuilder(self.df, qbank=q_bank, max_score=max_score, device=device)

        self.user_ids = self.df["user_id"].astype(int).values
        self.item_ids = self.df["item_id"].astype(int).values
        self.scores = self.df["score"].astype(float).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        u = int(self.user_ids[idx])
        iid = int(self.item_ids[idx])
        y = float(self.scores[idx])

        item_text = self.text_bank.get_text(iid)
        q_vec = self.q_bank.get_q(iid)  # (K,)

        attempt = ""
        if self.attempt_bank is not None:
            attempt = self.attempt_bank.get_attempt(u, iid)

        # CHANGED:
        # default = no LOO, because this project wants "use all evidence to build theta"
        exclude_item_id = iid if self.use_leave_one_out else None

        skill_mean, skill_sum, skill_count, skill_max, skill_min, skill_std = \
            self.builder.get_user_evidence(u, exclude_item_id=exclude_item_id)

        return {
            "user_id": u,
            "item_id": iid,
            "y_true": y,
            "item_text": item_text,
            "attempt": attempt,            # NEW
            "q_vector": q_vec,               # (K,)
            "skill_mean": skill_mean,        # (K,)
            "skill_sum": skill_sum,          # (K,)
            "skill_count": skill_count,      # (K,)
            "skill_max": skill_max,          # (K,)
            "skill_min": skill_min,          # (K,)
            "skill_std": skill_std,          # (K,)
        }


def collate_bandit(batch: List[Dict], device: torch.device):
    """
    Turn list of samples into one batch of tensors.

    Outputs:
      user_ids   : list[int] length B
      item_ids   : list[int] length B
      item_texts : list[str] length B
      attempts   : list[str] length B   # NEW
      q_vec      : (B,K)
      s_mean     : (B,K)
      s_sum      : (B,K)
      s_cnt      : (B,K)
      s_max      : (B,K)
      s_min      : (B,K)
      s_std      : (B,K)
      y_true     : (B,)
    """
    user_ids = [b["user_id"] for b in batch]
    item_ids = [b["item_id"] for b in batch]
    item_texts = [b["item_text"] for b in batch]
    attempts = [b["attempt"] for b in batch]   # NEW

    y_true = torch.tensor([b["y_true"] for b in batch], dtype=torch.float32, device=device)                 # (B,)
    q_vec  = torch.tensor(np.stack([b["q_vector"] for b in batch]), dtype=torch.float32, device=device)     # (B,K)
    s_mean = torch.tensor(np.stack([b["skill_mean"] for b in batch]), dtype=torch.float32, device=device)   # (B,K)
    s_sum  = torch.tensor(np.stack([b["skill_sum"] for b in batch]), dtype=torch.float32, device=device)    # (B,K)
    s_cnt  = torch.tensor(np.stack([b["skill_count"] for b in batch]), dtype=torch.float32, device=device)  # (B,K)
    s_max  = torch.tensor(np.stack([b["skill_max"] for b in batch]), dtype=torch.float32, device=device)    # (B,K)
    s_min  = torch.tensor(np.stack([b["skill_min"] for b in batch]), dtype=torch.float32, device=device)    # (B,K)
    s_std  = torch.tensor(np.stack([b["skill_std"] for b in batch]), dtype=torch.float32, device=device)    # (B,K)

    return user_ids, item_ids, item_texts, attempts, q_vec, s_mean, s_sum, s_cnt, s_max, s_min, s_std, y_true


# ============================================================
# 5) Predictor: Dummy now, LLM later
# ============================================================

SKILL_DEFS_EN = [
    "Basic Data Structure Selection",
    "Object-Oriented Programming Concepts",
    "Basic Syntax and Formatting",
    "Variables",
    "Data types",
    "Lists",
    "Tuples",
    "Conditional Statements",
    "Dictionaries",
    "Sets",
    "Input handling",
    "Loops",
    "Functions",
    "Classes",
    "File Handling",
    "Exception Handling",
    "Logical reasoning",
    "Algorithmic thinking",
]

SKILLS_EN_TEXT = "\n".join(SKILL_DEFS_EN)


class LLMScorer:
    """
    Interface for predictor (Stage2).

    Inputs:
      item_text: str
      q_vector: (K,) np.ndarray
      theta_action: (K,) np.ndarray in (0,1)
      student_attempt_code: str  # NEW

    Output:
      y_hat: float in [0,10]
    """
    def score(
        self,
        item_text: str,
        q_vector: np.ndarray,
        theta_action: np.ndarray,
        student_attempt_code: str = "",   # NEW
    ) -> float:
        raise NotImplementedError


class DummyScorer(LLMScorer):
    """
    DummyPredictor is only for pipeline debugging:
      - does NOT read item_text
      - does NOT call any API
      - NEW: accepts student_attempt_code but ignores it

    prediction = 10 * mean(theta_action over skills that q_vector=1)
    """
    def score(
        self,
        item_text: str,
        q_vector: np.ndarray,
        theta_action: np.ndarray,
        student_attempt_code: str = "",   # NEW
    ) -> float:
        q = (q_vector > 0).astype(np.float32)          # (K,)
        denom = float(max(q.sum(), 1.0))
        val = float((theta_action * q).sum() / denom)  # scalar
        return float(np.clip(val * 10.0, 0.0, 10.0))


class OpenAIScorer(LLMScorer):
    """
    Call OpenAI model to predict score 0-10 given:
      item_text, q_vector, theta_action, student_attempt_code

    Design goals:
      - Output ONLY one number
      - Force the model to consider item difficulty from item_text + required skills
      - Keep all reasoning internal
      - Return a single numeric prediction in [0,10]

    NOTE:
      - This costs API usage.
      - Requires OPENAI_API_KEY in environment.
    """
    def __init__(
        self,
        model_name: str = "gpt-5.4-nano",
        temperature: float = 0.0,
        reasoning_effort: str = "high",
        verbosity: str = "low",
        store: bool = False,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.store = store

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        # ADDED: keep SDK retry small and let our outer retry loop control long waits more predictably.
        self.client = OpenAI(
            api_key=api_key,
            max_retries=2,
            timeout=180.0,
        )

        # ADDED: outer retry settings for temporary network/API failures.
        # With the waiting schedule below, the program can keep trying for roughly 1 hour or more.
        self.max_api_retries = 12
        self.max_retry_wait_seconds = 300

        # FIX: if the API call succeeds but the response contains no numeric score,
        # retry instead of silently fabricating a score of 5.0 for training.
        self.max_format_retries = 2

    def build_prompt(
        self,
        item_text: str,
        q_vector: np.ndarray,
        theta_action: np.ndarray,
        student_attempt_code: str = "",   # NEW
    ) -> str:
        q_idx = [i for i, v in enumerate(q_vector.tolist()) if float(v) > 0.0]
        theta_list = [float(x) for x in theta_action.tolist()]

        required_skills = [
            SKILL_DEFS_EN[i]
            for i in q_idx
            if 0 <= i < len(SKILL_DEFS_EN)
        ]
        required_skills_text = "\n".join([f"- {skill}" for skill in required_skills])
        if not required_skills_text.strip():
            required_skills_text = "- None explicitly provided"

        student_skill_text = "\n".join(
            [
                f"- {SKILL_DEFS_EN[i]}: {float(theta_list[i]):.4f}"
                for i in range(min(len(theta_list), len(SKILL_DEFS_EN)))
            ]
        )
        if not student_skill_text.strip():
            student_skill_text = "- No student skill levels provided"

        # NEW: support attempt input
        student_attempt_code = str(student_attempt_code).strip()
        if student_attempt_code == "":
            student_attempt_code = "[NO ATTEMPT PROVIDED]"

        # CHANGED:
        prompt = f"""
You are an expert predictor for a Python programming assessment.

Your task is to predict the most likely score a student would obtain on this item.

You are estimating the likely score, not verifying correctness by executing test cases.
Use:
1. the item text,
2. the required skills,
3. the student's diagnosed skill profile,
4. the student's attempt code as additional supporting evidence.

Output rule:
- Return ONLY one number in [0,10]
- Use exactly one decimal place
- No words, no explanation, no JSON

Reason internally as follows:
1. Read the item text carefully.
2. Use Required Skills to identify the core skills for this item.
3. Infer how demanding those core skills are in this item.
4. In Student Skill Levels, use the required skills as the main evidence.
5. Compare the student's required-skill levels with the item's demands.
6. Read the Student Attempt Code and judge whether its core logic is consistent with the item.
7. Use the attempt only to refine the prediction from the diagnosed skill profile.
8. Convert that into the most likely score.

Score meaning:
- 0.0-1.9 = the student is unlikely to carry the essential core logic of the item
- 2.0-3.9 = only limited fragments are likely; the core solution is still unlikely to work
- 4.0-5.9 = some meaningful progress is likely, but an important part of the core logic is still missing
- 6.0-7.9 = the core solution is likely to work in broad outline, but important weaknesses remain
- 8.0-8.9 = the core solution is likely to work well, with only minor weaknesses
- 9.0-10.0 = the student is very likely to succeed on the core required skills, with no major bottleneck

Calibration rules:
- Put primary weight on the required skills.
- Do not simply average all skill levels.
- Skill levels around 0.40-0.60 are limited or unstable, not strong.
- Skill levels around 0.80 or above on the required skills indicate strong mastery and can support high scores.
- If one core required skill is severely below the item's demand, lower the score sharply.
- If the required skills are all strong, especially around 0.80 or above, and no major bottleneck is apparent, scores in the 9.0-10.0 range are appropriate.
- A minor weakness in one required skill does not by itself rule out a very high score if the core logic is still likely to succeed.
- Prefer decisive low or high scores when justified; do not overuse the middle range.
- Use Student Attempt Code only to confirm, refine, or moderately adjust the prediction.
- Focus on the core logic of the attempt, not comments, metadata, or superficial structure.

Item Text:
{item_text}

Required Skills:
{required_skills_text}

Student Skill Levels:
{student_skill_text}

Student Attempt Code:
{student_attempt_code}

Now output ONLY the final predicted score as a single number in [0,10], with exactly one decimal place.
""".strip()

        return prompt

    # ADDED: retry only temporary errors instead of stopping the whole training run.
    def _call_openai_with_retry(self, prompt: str):
        retryable_errors = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        )

        for attempt in range(self.max_api_retries + 1):
            try:
                return self.client.responses.create(
                    model=self.model_name,
                    input=prompt,
                    reasoning={
                        "effort": self.reasoning_effort,
                    },
                    text={"verbosity": self.verbosity},
                    store=self.store,
                )

            except retryable_errors as exc:
                if attempt >= self.max_api_retries:
                    print()
                    print("OpenAI request failed after all retry attempts.")
                    raise

                # ADDED: exponential backoff capped at 5 minutes.
                # Wait sequence: 5, 10, 20, 40, 80, 160, 300, 300, ... seconds.
                wait_seconds = min(5 * (2 ** attempt), self.max_retry_wait_seconds)

                print()
                print(f"[OpenAI retry] {type(exc).__name__}: {exc}")
                print(
                    f"[OpenAI retry] Waiting {wait_seconds} seconds before "
                    f"attempt {attempt + 2}/{self.max_api_retries + 1}"
                )
                time.sleep(wait_seconds)

    def score(
        self,
        item_text: str,
        q_vector: np.ndarray,
        theta_action: np.ndarray,
        student_attempt_code: str = "",   # NEW
    ) -> float:
        """
        Inputs:
          item_text: str
          q_vector: (K,) numpy array
          theta_action: (K,) numpy array in (0,1)
          student_attempt_code: str

        Output:
          y_hat: float in [0,10]
        """
        prompt = self.build_prompt(
            item_text=item_text,
            q_vector=q_vector,
            theta_action=theta_action,
            student_attempt_code=student_attempt_code,
        )

        # FIX: a successful API request can still return an unparsable response.
        # Retry a small number of times; never turn a parse failure into a fake 5.0
        # because that fake value would create a real REINFORCE reward.
        last_text = ""
        for format_attempt in range(self.max_format_retries + 1):
            resp = self._call_openai_with_retry(prompt)
            text = (resp.output_text or "").strip()
            last_text = text

            # Extract first number robustly. Extra prose is tolerated, although the
            # prompt still asks the model to return only one numeric score.
            m = re.search(r"-?\d+(?:\.\d+)?", text)
            if m is not None:
                y = float(m.group(0))
                if np.isfinite(y):
                    return float(np.clip(y, 0.0, 10.0))

            if format_attempt < self.max_format_retries:
                print()
                print(
                    f"[OpenAI format retry] Could not parse a finite score from: {text!r}. "
                    f"Retrying ({format_attempt + 2}/{self.max_format_retries + 1})."
                )

        raise RuntimeError(
            "OpenAI response could not be parsed as a finite numeric score after "
            f"{self.max_format_retries + 1} attempts. Last response: {last_text!r}"
        )


# ============================================================
# 6) Training config
# ============================================================
@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 1e-4
    n_epoch: int = 10

    reward_scale_c: float = 100.0      # r = -(err^2)/c
    baseline_beta: float = 0.05        # EMA baseline update speed
    clip_pred_0_10: bool = True        # clamp y_hat

    # CHANGED:
    # default = False because this project now wants to use all evidence to build theta
    use_leave_one_out: bool = False

    use_early_stopping: bool = False
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.00
    monitor_metric: str = "rmse"

    # ADDED: checkpoint settings. They do not change the model logic.
    checkpoint_path: Optional[str] = None
    resume: bool = False


# ============================================================
# 7) Training: REINFORCE with baseline (NO entropy term)
# ============================================================

# ADDED: save the latest completed epoch so training can resume after a crash.
def save_checkpoint(
    path: str,
    model: GLLMCDM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    baseline: float,
    history: List[Dict],
    best_metric: float,
    best_state,
    patience_count: int,
):
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "baseline": float(baseline),
        "history": history,
        "best_metric": float(best_metric),
        "best_state": best_state,
        "patience_count": int(patience_count),

        # FIX: preserve random-number states so --resume continues the same local
        # PyTorch/NumPy/Python stochastic stream instead of silently changing it.
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            [x.cpu() for x in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else None
        ),
    }

    # ADDED: write to a temporary file first to reduce the chance of a corrupted checkpoint.
    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)
    print(f"Checkpoint saved after epoch {epoch}: {path}")


# ADDED: restore the latest completed epoch and continue from the next epoch.
def load_checkpoint(
    path: str,
    model: GLLMCDM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # ADDED: ensure optimizer tensors are on the selected device after loading.
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

    # FIX: restore RNG states saved at the end of the completed epoch.
    # This affects only reproducibility after --resume; it does not change the model math.
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng_state_all"]])

    return {
        "start_epoch": int(checkpoint["epoch"]) + 1,
        "baseline": float(checkpoint.get("baseline", 0.0)),
        "history": checkpoint.get("history", []),
        "best_metric": float(checkpoint.get("best_metric", float("inf"))),
        "best_state": checkpoint.get("best_state"),
        "patience_count": int(checkpoint.get("patience_count", 0)),
    }


def train_reinforce(
    model: GLLMCDM,
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    text_bank: ItemTextBank,
    q_bank: QMatrixBank,
    scorer: LLMScorer,
    config: TrainConfig,
    device: torch.device,
    attempt_bank: Optional[AttemptBank] = None,   # NEW
):
    """
    --------------------------
    Training loop (per batch)
    --------------------------
    Inputs from DataLoader:
      item_texts : list[str]
      attempts   : list[str]    # NEW
      q_vec   : (B,K)
      s_mean  : (B,K)
      s_sum   : (B,K)
      s_cnt   : (B,K)
      s_max   : (B,K)
      s_min   : (B,K)
      s_std   : (B,K)
      y_true  : (B,)

    Stage1:
      theta_base = diagnose_theta_skill(s_mean, s_max, s_min, s_std)  # (B,K)

    Policy:
      sample = sample_theta_action(theta_base, q_vec, s_cnt)         # theta_action (B,K), log_prob (B,)

    Stage2 scorer:
      y_hat = scorer(item_text, q_vector, theta_action, attempt)     # (B,)

    Reward:
      r = -(y_hat - y_true)^2 / c                                     # (B,)

    Advantage:
      A = r - baseline_previous

    Baseline:
      baseline <- (1-beta)baseline_previous + beta*mean(r)
      # FIX: updated baseline is used from the next batch onward

    Loss (REINFORCE):
      loss = -mean( A.detach() * log_prob )

    Note:
    """
    model.to(device)
    model.train()

    train_ds = BanditDataset(
        train_df,
        text_bank=text_bank,
        q_bank=q_bank,
        attempt_bank=attempt_bank,              # NEW
        max_score=10.0,
        use_leave_one_out=config.use_leave_one_out,  # CHANGED
        device=device,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_bandit(b, device),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    baseline = 0.0
    history = []

    # ---------- NEW: early stopping state ----------
    best_metric = float("inf")
    best_state = None
    patience_count = 0
    # ----------------------------------------------

    # ADDED: default to a fresh run; when --resume is used, restore the last completed epoch.
    start_epoch = 0

    if config.resume:
        if not config.checkpoint_path:
            raise ValueError("resume=True but checkpoint_path is not set.")
        if not os.path.isfile(config.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")

        restored = load_checkpoint(
            path=config.checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        start_epoch = restored["start_epoch"]
        baseline = restored["baseline"]
        history = restored["history"]
        best_metric = restored["best_metric"]
        best_state = restored["best_state"]
        patience_count = restored["patience_count"]

        print("=" * 70)
        print(f"Resumed from checkpoint: {config.checkpoint_path}")
        print(f"Training will continue from epoch {start_epoch}")
        print("=" * 70)

    # ADDED: n_epoch still means the total number of epochs, not extra epochs.
    for ep in range(start_epoch, config.n_epoch):
        pbar = tqdm(train_loader, desc=f"Epoch {ep}")
        losses, rewards, rmses = [], [], []

        for user_ids, item_ids, item_texts, attempts, q_vec, s_mean, s_sum, s_cnt, s_max, s_min, s_std, y_true in pbar:
            # ---- Stage1 diagnosis (monotone FC+; no count in mastery) ----
            theta_base = model.diagnose_theta_skill(s_mean, s_max, s_min, s_std)  # (B,K)

            # ---- Policy sample: count influences log_sigma (uncertainty), not mastery ----
            sample = model.sample_theta_action(theta_base, q_vec, s_cnt, deterministic=False)
            theta_action = sample.theta_action  # (B,K)

            # FIX: fail fast if the REINFORCE graph is accidentally changed again.
            # The sampled action is fixed, but log_prob must still carry gradients
            # through mu/sigma back to Policy and Diagnosis parameters.
            if theta_action.requires_grad:
                raise RuntimeError(
                    "REINFORCE graph error: theta_action unexpectedly requires grad. "
                    "The sampled action must be fixed during score-function backward."
                )
            if not sample.log_prob.requires_grad:
                raise RuntimeError(
                    "REINFORCE graph error: log_prob has no gradient path to model parameters."
                )

            # ---- Non-differentiable scoring (Dummy now / OpenAI later) ----
            y_hat_list = []
            for t, a, q, th in zip(
                item_texts,
                attempts,   # NEW
                q_vec.detach().cpu().numpy(),
                theta_action.detach().cpu().numpy(),
            ):
                yh = scorer.score(t, q, th, a)   # NEW
                if config.clip_pred_0_10:
                    yh = float(np.clip(yh, 0.0, 10.0))
                y_hat_list.append(yh)
            y_hat = torch.tensor(y_hat_list, dtype=torch.float32, device=device)  # (B,)

            # ---- Reward ----
            err = y_hat - y_true                      # (B,)
            r = -(err * err) / float(config.reward_scale_c)  # (B,)

            # ---- Baseline EMA ----
            # FIX: advantage for the current batch uses the baseline accumulated
            # BEFORE seeing the current sampled actions/rewards. This keeps the
            # simple EMA baseline independent of the current action sample.
            adv = r - baseline  # (B,)

            # Update the EMA only for use by the NEXT batch.
            r_mean = float(r.detach().mean().cpu().item())
            baseline = (1.0 - config.baseline_beta) * baseline + config.baseline_beta * r_mean

            # ---- REINFORCE loss (no entropy) ----
            loss = -(adv.detach() * sample.log_prob).mean()

            # FIX: fail early instead of spending more LLM calls after numerical failure.
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite REINFORCE loss detected: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()

            # FIX: check gradients before optimizer.step(); this is a safety check only.
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"Non-finite gradient detected in parameter: {name}")

            optimizer.step()

            rmse = float(torch.sqrt(torch.mean(err * err)).detach().cpu().item())
            losses.append(float(loss.detach().cpu().item()))
            rewards.append(r_mean)
            rmses.append(rmse)
            pbar.set_postfix({"loss": np.mean(losses), "reward": np.mean(rewards), "rmse": np.mean(rmses)})

        ep_info = {
            "epoch": ep,
            "train_loss": float(np.mean(losses)) if losses else None,
            "train_reward": float(np.mean(rewards)) if rewards else None,
            "train_rmse": float(np.mean(rmses)) if rmses else None,
            "baseline": float(baseline),
        }

        if valid_df is not None:
            ep_info["valid"] = evaluate(
                model, valid_df, text_bank, q_bank, scorer, config, device,
                save_pred_path=None,
                attempt_bank=attempt_bank,   # NEW
            )

            # ---------- NEW: early stopping logic ----------
            current_metric = ep_info["valid"][config.monitor_metric]

            if current_metric < best_metric - config.early_stop_min_delta:
                best_metric = current_metric
                patience_count = 0

                # save best weights in memory
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                ep_info["is_best"] = True
            else:
                patience_count += 1
                ep_info["is_best"] = False

            ep_info["best_" + config.monitor_metric] = float(best_metric)
            ep_info["patience_count"] = int(patience_count)
            # -----------------------------------------------

        history.append(ep_info)

        # ADDED: save after every fully completed epoch.
        # If the program stops during the next epoch, resume restarts from this saved point.
        if config.checkpoint_path:
            save_checkpoint(
                path=config.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=ep,
                baseline=baseline,
                history=history,
                best_metric=best_metric,
                best_state=best_state,
                patience_count=patience_count,
            )

        # ---------- NEW: stop after appending history ----------
        if valid_df is not None and config.use_early_stopping:
            if patience_count >= config.early_stop_patience:
                print(f"Early stopping triggered at epoch {ep}")
                break
        # ------------------------------------------------------

    # ---------- NEW: restore best checkpoint ----------
    if best_state is not None:
        model.load_state_dict(best_state)
    # -----------------------------------------------

    return history


# ============================================================
# 8) Evaluation + save pred_test.csv
# ============================================================
@torch.no_grad()
def evaluate(
    model: GLLMCDM,
    df: pd.DataFrame,
    text_bank: ItemTextBank,
    q_bank: QMatrixBank,
    scorer: LLMScorer,
    config: TrainConfig,
    device: torch.device,
    save_pred_path: Optional[str] = None,
    attempt_bank: Optional[AttemptBank] = None,   # NEW
) -> Dict[str, float]:
    """
    Evaluation uses deterministic action:
      deterministic=True => z=mu (no sampling noise)
      theta_action = sigmoid(mu)

    Evaluation (main result) uses theta_base directly:
      theta_base = diagnose_theta_skill(s_mean, s_max, s_min, s_std)
      y_hat = scorer(item_text, q_vector, theta_base, attempt)

    If save_pred_path is provided, writes:
      user_id,item_id,score,pred
    """
    model.eval()

    ds = BanditDataset(
        df,
        text_bank=text_bank,
        q_bank=q_bank,
        attempt_bank=attempt_bank,               # NEW
        max_score=10.0,
        use_leave_one_out=config.use_leave_one_out,  # CHANGED
        device=device,
    )
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_bandit(b, device),
    )

    y_true_all, y_hat_all = [], []
    rows = []

    for user_ids, item_ids, item_texts, attempts, q_vec, s_mean, s_sum, s_cnt, s_max, s_min, s_std, y_true in loader:
        theta_base = model.diagnose_theta_skill(s_mean, s_max, s_min, s_std)  # (B,K)

        for u, iid, t, a, q, th, yt in zip(
            user_ids,
            item_ids,
            item_texts,
            attempts,   # NEW
            q_vec.detach().cpu().numpy(),
            theta_base.detach().cpu().numpy(),
            y_true.detach().cpu().numpy(),
        ):
            yh = scorer.score(t, q, th, a)   # NEW
            if config.clip_pred_0_10:
                yh = float(np.clip(yh, 0.0, 10.0))

            y_true_all.append(float(yt))
            y_hat_all.append(float(yh))

            if save_pred_path is not None:
                rows.append({"user_id": int(u), "item_id": int(iid), "score": float(yt), "pred": float(yh)})

    y_true_all = np.array(y_true_all, dtype=np.float32)
    y_hat_all = np.array(y_hat_all, dtype=np.float32)

    mse = float(mean_squared_error(y_true_all, y_hat_all))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_all, y_hat_all))

    if save_pred_path is not None:
        dirpath = os.path.dirname(save_pred_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        pd.DataFrame(rows).to_csv(save_pred_path, index=False)
        print(f"Saved pred_test.csv to: {save_pred_path}")

    return {"rmse": rmse, "mae": mae, "mse": mse}
