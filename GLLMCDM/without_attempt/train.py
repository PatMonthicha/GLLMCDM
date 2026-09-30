# Purpose: Train and evaluate GLLMCDM using the prompt without learner attempts.
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
import random  # FIX: save/restore Python RNG state for reproducible --resume
import time  # ADDED: pause before retry when network/API is temporarily unavailable
import numpy as np
import pandas as pd
from tqdm import tqdm
import openai  # ADDED: OpenAI exception classes for retry handling
from openai import OpenAI
import hashlib
from pathlib import Path
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
      - evidence  = builder.get_user_evidence(...)

    # CHANGED:

    Returns dict:
      user_id, item_id, y_true,
      item_text,
      q_vector (K,),
      skill_mean (K,), skill_sum (K,), skill_count (K,),
      skill_max (K,), skill_min (K,), skill_std (K,)
    """

    def __init__(
        self,
        df_log: pd.DataFrame,
        text_bank: ItemTextBank,
        q_bank: QMatrixBank,
        max_score: float = 10.0,
        use_leave_one_out: bool = False,   # CHANGED
        device: torch.device = torch.device("cpu"),
    ):
        self.df = df_log.copy().reset_index(drop=True)
        self.text_bank = text_bank
        self.q_bank = q_bank
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

    y_true = torch.tensor([b["y_true"] for b in batch], dtype=torch.float32, device=device)                 # (B,)
    q_vec  = torch.tensor(np.stack([b["q_vector"] for b in batch]), dtype=torch.float32, device=device)     # (B,K)
    s_mean = torch.tensor(np.stack([b["skill_mean"] for b in batch]), dtype=torch.float32, device=device)   # (B,K)
    s_sum  = torch.tensor(np.stack([b["skill_sum"] for b in batch]), dtype=torch.float32, device=device)    # (B,K)
    s_cnt  = torch.tensor(np.stack([b["skill_count"] for b in batch]), dtype=torch.float32, device=device)  # (B,K)
    s_max  = torch.tensor(np.stack([b["skill_max"] for b in batch]), dtype=torch.float32, device=device)    # (B,K)
    s_min  = torch.tensor(np.stack([b["skill_min"] for b in batch]), dtype=torch.float32, device=device)    # (B,K)
    s_std  = torch.tensor(np.stack([b["skill_std"] for b in batch]), dtype=torch.float32, device=device)    # (B,K)

    return user_ids, item_ids, item_texts, q_vec, s_mean, s_sum, s_cnt, s_max, s_min, s_std, y_true


# ============================================================
# 5) Predictor: Dummy now, LLM later
# ============================================================

import os
import re
import numpy as np
from openai import OpenAI

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

    Output:
      y_hat: float in [0,10]
    """
    def score(self, item_text: str, q_vector: np.ndarray, theta_action: np.ndarray) -> float:
        raise NotImplementedError


class DummyScorer(LLMScorer):
    """
    DummyPredictor is only for pipeline debugging:
      - does NOT read item_text
      - does NOT call any API

    prediction = 10 * mean(theta_action over skills that q_vector=1)
    """
    def score(self, item_text: str, q_vector: np.ndarray, theta_action: np.ndarray) -> float:
        q = (q_vector > 0).astype(np.float32)          # (K,)
        denom = float(max(q.sum(), 1.0))
        val = float((theta_action * q).sum() / denom)  # scalar
        return float(np.clip(val * 10.0, 0.0, 10.0))


class OpenAIScorer(LLMScorer):
    """
    Call OpenAI model to predict score 0-10 given:
      item_text, q_vector, theta_action

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
        # ADDED: small SDK-level retry and a per-request timeout
        self.client = OpenAI(
            api_key=api_key,
            max_retries=2,
            timeout=180.0,
        )

        # ADDED: outer retry loop for longer network interruptions
        self.max_api_retries = 12

        # FIX: if the API call succeeds but the response contains no numeric score,
        # retry instead of silently fabricating a score of 5.0 for training.
        self.max_format_retries = 2

    def build_prompt(self, item_text: str, q_vector: np.ndarray, theta_action: np.ndarray) -> str:
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

        prompt = f"""

You are an expert predictor for a Python programming assessment.

Your task is to predict the most likely score a student would obtain on this item.
You are NOT grading an actual student answer.
Use:
1. the item text,
2. the required skills,
3. the student's diagnosed skill profile.

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
6. Decide whether the student is likely to carry the core logic of the item.
7. Convert that into the most likely score.

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

Item Text:
{item_text}

Required Skills:
{required_skills_text}

Student Skill Levels:
{student_skill_text}

Now output ONLY the final predicted score as a single number in [0,10], with exactly one decimal place.

""".strip()

        return prompt

    # ADDED: call OpenAI with retry for transient network/API errors
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
                    print("OpenAI request failed after all retries.")
                    raise

                # ADDED: exponential backoff, capped at 5 minutes
                wait_seconds = min(5 * (2 ** attempt), 300)

                print()
                print(f"[OpenAI retry] {type(exc).__name__}: {exc}")
                print(
                    f"[OpenAI retry] Waiting {wait_seconds} seconds "
                    f"before retry {attempt + 1}/{self.max_api_retries}"
                )

                time.sleep(wait_seconds)

    def score(self, item_text: str, q_vector: np.ndarray, theta_action: np.ndarray) -> float:
        """
        Inputs:
          item_text: str
          q_vector: (K,) numpy array
          theta_action: (K,) numpy array in (0,1)

        Output:
          y_hat: float in [0,10]
        """
        prompt = self.build_prompt(item_text, q_vector, theta_action)

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

    # ADDED: save the latest checkpoint after every completed epoch
    checkpoint_path: Optional[str] = None

    # ADDED: load the checkpoint and continue training when True
    resume: bool = False

# ============================================================
# ADDED: Checkpoint helpers
# ============================================================
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
    """Save state after each completed epoch."""
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

    # ADDED: atomic-style save to reduce checkpoint corruption risk
    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)

    print(f"Checkpoint saved after epoch {epoch}: {path}")


def load_checkpoint(
    path: str,
    model: GLLMCDM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """Restore model, optimizer, history, baseline, and early-stopping state."""
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # FIX: keep optimizer state tensors on the selected device after --resume.
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

    # FIX: restore RNG states saved at the end of the completed epoch.
    # This changes only reproducibility after --resume, not the model equations.
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


# ============================================================
# ADDED: Export deterministic theta from the restored best model
# ============================================================
@torch.no_grad()
def export_theta(
    model: GLLMCDM,
    evidence_df: pd.DataFrame,
    q_bank: QMatrixBank,
    output_csv_path: str,
    output_npy_path: str,
    output_implicit_csv_path: str,
    output_implicit_npy_path: str,
    output_explicit_csv_path: str,
    output_explicit_npy_path: str,
    batch_size: int,
    device: torch.device,
) -> Dict:
    """
    Export one deterministic theta_base vector per learner.

    ADDED because theta is derived from model weights + learner evidence and is
    not stored as a persistent buffer inside GLLMCDM. This function must be
    called after train_reinforce(), which restores best_state before returning.

    Important:
    - Reuses SkillEvidenceBuilder and model.diagnose_theta_skill() unchanged.
    - Does not call sample_theta_action(), so no random policy noise is added.
    - Does not change the model's logit, policy, loss, or checkpoint logic.
    """
    if batch_size <= 0:
        raise ValueError("export_theta batch_size must be greater than 0.")

    evidence_builder = SkillEvidenceBuilder(
        evidence_df,
        qbank=q_bank,
        max_score=10.0,
        device=device,
    )
    user_ids = evidence_builder.user_ids
    if len(user_ids) == 0:
        raise ValueError("Cannot export theta because the evidence data has no users.")

    was_training = model.training
    model.eval()
    theta_base_batches = []
    theta_implicit_batches = []
    theta_explicit_batches = []

    for start in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[start:start + batch_size]
        (
            skill_mean,
            _skill_sum,
            _skill_count,
            skill_max,
            skill_min,
            skill_std,
            _,
        ) = evidence_builder.get_all_user_evidence(
            user_ids=batch_user_ids,
            exclude_item_ids=None,
            device=device,
        )

        # ADDED: export the two diagnosis branches for analysis only. These are
        # the same operations already used inside diagnose_theta_skill(); no
        # model parameters, logit, policy, or training behavior are modified.
        feat_implicit = torch.cat(
            [skill_mean, skill_max, skill_min, skill_std],
            dim=-1,
        )
        theta_implicit_batch = model.skill_theta_nn(feat_implicit)
        theta_explicit_batch = torch.sigmoid(skill_mean)

        # ADDED: keep theta_base from the model's existing diagnosis method
        # itself, rather than replacing or altering the model's original logic.
        theta_base_batch = model.diagnose_theta_skill(
            skill_mean,
            skill_max,
            skill_min,
            skill_std,
        )
        theta_base_batches.append(theta_base_batch.detach().cpu().numpy())
        theta_implicit_batches.append(theta_implicit_batch.detach().cpu().numpy())
        theta_explicit_batches.append(theta_explicit_batch.detach().cpu().numpy())

    theta_base = np.concatenate(theta_base_batches, axis=0).astype(np.float32)
    theta_implicit = np.concatenate(theta_implicit_batches, axis=0).astype(np.float32)
    theta_explicit = np.concatenate(theta_explicit_batches, axis=0).astype(np.float32)
    theta_columns = [f"K{i}" for i in range(theta_base.shape[1])]

    def save_theta_artifacts(theta: np.ndarray, csv_path: str, npy_path: str):
        """ADDED: save one theta variant with the same user ordering."""
        csv_dir = os.path.dirname(csv_path)
        npy_dir = os.path.dirname(npy_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        if npy_dir:
            os.makedirs(npy_dir, exist_ok=True)

        theta_df = pd.DataFrame(theta, columns=theta_columns)
        theta_df.insert(0, "user_id", user_ids.astype(int))
        theta_df.to_csv(csv_path, index=False)
        np.save(npy_path, theta)

    save_theta_artifacts(theta_base, output_csv_path, output_npy_path)
    save_theta_artifacts(
        theta_implicit,
        output_implicit_csv_path,
        output_implicit_npy_path,
    )
    save_theta_artifacts(
        theta_explicit,
        output_explicit_csv_path,
        output_explicit_npy_path,
    )

    if was_training:
        model.train()

    return {
        "n_users": int(theta_base.shape[0]),
        "n_know": int(theta_base.shape[1]),
        "theta_base_csv_path": output_csv_path,
        "theta_base_npy_path": output_npy_path,
        "theta_implicit_csv_path": output_implicit_csv_path,
        "theta_implicit_npy_path": output_implicit_npy_path,
        "theta_explicit_csv_path": output_explicit_csv_path,
        "theta_explicit_npy_path": output_explicit_npy_path,
    }


# ============================================================
# 7) Training: REINFORCE with baseline (NO entropy term)
# ============================================================
def train_reinforce(
    model: GLLMCDM,
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    text_bank: ItemTextBank,
    q_bank: QMatrixBank,
    scorer: LLMScorer,
    config: TrainConfig,
    device: torch.device,
):
    """
    --------------------------
    Training loop (per batch)
    --------------------------
    Inputs from DataLoader:
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
      y_hat = scorer(item_text, q_vector, theta_action)              # (B,)

    Reward:
      r = -(y_hat - y_true)^2 / c                                     # (B,)

    Advantage / baseline:
      A_t = r_t - baseline_{t-1}
      baseline_t <- (1-beta)baseline_{t-1} + beta*mean(r_t)

    # FIX: the current batch uses the baseline accumulated before that batch;
    # the EMA update is used by the next batch.

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

    # ADDED: default training starts from epoch 0
    start_epoch = 0

    # ADDED: restore checkpoint when run.py is called with --resume
    if config.resume:
        if not config.checkpoint_path:
            raise ValueError("resume=True but checkpoint_path is not set.")

        if not os.path.isfile(config.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {config.checkpoint_path}"
            )

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

    # CHANGED: continue from start_epoch instead of always starting at 0
    for ep in range(start_epoch, config.n_epoch):
        pbar = tqdm(train_loader, desc=f"Epoch {ep}")
        losses, rewards, rmses = [], [], []

        for user_ids, item_ids, item_texts, q_vec, s_mean, s_sum, s_cnt, s_max, s_min, s_std, y_true in pbar:
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

            # ---- Non-differentiable scoring (Dummy now) ----
            y_hat_list = []
            for t, q, th in zip(item_texts, q_vec.detach().cpu().numpy(), theta_action.detach().cpu().numpy()):
                yh = scorer.score(t, q, th)
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

            # FIX: update the EMA only for use by the NEXT batch.
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
            ep_info["valid"] = evaluate(model, valid_df, text_bank, q_bank, scorer, config, device, save_pred_path=None)
        
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

        # ADDED: save checkpoint after every completed epoch
        # If interrupted mid-epoch, resume starts that epoch again.
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
) -> Dict[str, float]:
    """
    Evaluation uses deterministic action:
      deterministic=True => z=mu (no sampling noise)
      theta_action = sigmoid(mu)

    Evaluation (main result) uses theta_base directly:
      theta_base = diagnose_theta_skill(s_mean, s_max, s_min, s_std)
      y_hat = scorer(item_text, q_vector, theta_base)

    If save_pred_path is provided, writes:
      user_id,item_id,score,pred
    """
    model.eval()

    ds = BanditDataset(
        df,
        text_bank=text_bank,
        q_bank=q_bank,
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

    for user_ids, item_ids, item_texts, q_vec, s_mean, s_sum, s_cnt, s_max, s_min, s_std, y_true in loader:
        theta_base = model.diagnose_theta_skill(s_mean, s_max, s_min, s_std)  # (B,K)
        #sample = model.sample_theta_action(theta_base, q_vec, s_cnt, deterministic=True)
        #theta_action = sample.theta_action  # (B,K)

        for u, iid, t, q, th, yt in zip(
            user_ids,
            item_ids,
            item_texts,
            q_vec.detach().cpu().numpy(),
            theta_base.detach().cpu().numpy(),
            y_true.detach().cpu().numpy(),
        ):
            yh = scorer.score(t, q, th)
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
