# Purpose: Define the no-attempt GLLMCDM diagnosis network and policy head.
# Provenance: Original research file; executable model logic is unchanged.
# model.py
# -*- coding: utf-8 -*-
"""
============================================================
GLLMCDM: Stage1 (Diagnosis) + Policy Head (REINFORCE)
============================================================

----------------------------------
    theta_base ∈ (0,1)^K   (K = 18 skills)



----------------------------------

- effective weight >= 0
- + activation monotone (Sigmoid)

(3) Policy (Contextual Bandit)
------------------------------
REINFORCE (policy gradient)

    action a = theta_action ∈ (0,1)^K

    z ∈ R^K
    theta_action = sigmoid(z)

    z ~ N(mu, sigma^2)

    z_base = logit(theta_base)
    mu     = z_base + delta_mu(x)
    sigma  = exp(log_sigma(x))

    x = [theta_base, q_vector_target, log1p(skill_count)] ∈ R^(3K)


------------------------------------------------

    log pi(theta|x) = log N(z|mu,sigma^2) + log|dz/dtheta|

    dz/dtheta = 1 / (theta(1-theta))
    log|dz/dtheta| = -log(theta) - log(1-theta)

    -log(sigmoid(z)) = softplus(-z)
    -log(1-sigmoid(z)) = softplus(z)
    log_det = sum_k [softplus(-z_k) + softplus(z_k)]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) PosLinear = FC+ (non-negative effective weights)
# ============================================================
class PosLinear(nn.Linear):
    """
    Linear layer with non-negative "effective weights".

    Why?
    - If weights are constrained to be >= 0 and activation is monotone (sigmoid),
      then increasing an input feature cannot decrease the output (monotonic).

    Implementation trick:
    - Convert raw weight W into |W| (absolute value) as effective weight.
      Using: abs(W) = W + 2*ReLU(-W)
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # weight_eff = abs(weight) >= 0
        weight_eff = self.weight + 2.0 * F.relu(torch.neg(self.weight))
        return F.linear(input, weight_eff, self.bias)


def safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Numerically stable logit for values in (0,1).
    Input:  p shape (...,K) in (0,1)
    Output: logit(p) shape (...,K) in R
    """
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log(1.0 - p)


# ============================================================
# 2) Output container for policy sampling
# ============================================================
@dataclass
class PolicySample:
    """
    PolicySample is just a "bundle" of outputs from one policy sampling step.
    We keep extra fields for debugging/analysis.

    Shapes:
      B = batch_size
      K = n_know (18)

    Important fields used in training:
      - theta_action : (B,K)  action sampled
      - log_prob     : (B,)   log pi(theta_action|x) used in REINFORCE loss
    """
    theta_base: torch.Tensor      # (B,K)
    theta_action: torch.Tensor    # (B,K)
    z_base: torch.Tensor          # (B,K)
    mu: torch.Tensor              # (B,K)
    log_sigma: torch.Tensor       # (B,K)
    z: torch.Tensor               # (B,K)
    log_prob: torch.Tensor        # (B,)
    entropy: torch.Tensor         # (B,)  (we keep it but DO NOT use in loss by default)


# ============================================================
# 3) Main model
# ============================================================
class GLLMCDM(nn.Module):
    """
    ----------------------------
    Stage1 (Diagnosis)
    ----------------------------
    Inputs (fixed-size, unlock n_item):
      skill_mean: (B,K)  signed mean per skill (~[-1,1])
      skill_max : (B,K)  signed max per skill
      skill_min : (B,K)  signed min per skill
      skill_std : (B,K)  signed std per skill
      *We intentionally do NOT use count in mastery to avoid:
        "attempts more => mastery higher" bias (especially under monotonic constraints).

    Implicit branch (aligned with GNCDM Version B):
      feat_imp  = [skill_mean, skill_max, skill_min, skill_std]   # (B,4K)
      theta_imp = FC+×2(feat_imp)                                 # (B,K) in (0,1)

    Explicit branch (simple, Q-based spirit):
      theta_exp = sigmoid(skill_mean)                             # (B,K)

    Mix:
      theta_base = (1-alpha)*theta_imp + alpha*theta_exp

    ----------------------------
    Policy (Contextual Bandit)
    ----------------------------
    Context to policy head:
      x = [theta_base, q_vector, log1p(skill_count)]   # (B,3K)

    Outputs:
      delta_mu  : (B,K)
      log_sigma : (B,K)

    Sampling:
      z_base = logit(theta_base)
      mu = z_base + delta_mu
      z ~ N(mu, sigma^2)
      theta_action = sigmoid(z)

    log_prob (action=theta):
      log pi(theta|x) = log N(z|mu,sigma^2) + log|dz/dtheta|
    """

    def __init__(
        self,
        n_know: int = 18,
        hidden: int = 128,
        alpha: float = 0.5,
        monotonicity_assumption: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_know = int(n_know)        # K
        self.alpha = float(alpha)
        self.device = device or torch.device("cpu")

        # ---------- (A) Implicit diagnosis network: FC+×2 + Sigmoid ----------
        # CHANGED: align Stage1 implicit input with GNCDM Version B
        # Input:  feat_imp (B,4K) = [mean, max, min, std]
        # Output: theta_imp (B,K)
        f_linear = PosLinear if monotonicity_assumption else nn.Linear

        self.skill_theta_nn = nn.Sequential(
            # CHANGED: use 4K features instead of 2K
            f_linear(4 * self.n_know, self.n_know),  # (B,4K)->(B,K)
            nn.Sigmoid(),
            f_linear(self.n_know, self.n_know),      # (B,K)->(B,K)
            nn.Sigmoid(),
        ).to(self.device)

        # ---------- (B) Policy head ----------
        # Input:  x = [theta_base, q_vector, log1p(count)] (B,3K)
        # Output: [delta_mu(K), log_sigma(K)]             (B,2K)
        self.policy_head = nn.Sequential(
            nn.Linear(3 * self.n_know, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2 * self.n_know),
        ).to(self.device)

        # weight init
        for name, p in self.named_parameters():
            if p.ndim >= 2 and "weight" in name:
                nn.init.xavier_normal_(p)

    # ============================================================
    # Stage1: diagnosis function
    # ============================================================
    def diagnose_theta_skill(
        self,
        skill_mean: torch.Tensor,  # (B,K)
        skill_max: torch.Tensor,   # (B,K)
        skill_min: torch.Tensor,   # (B,K)
        skill_std: torch.Tensor,   # (B,K)
    ) -> torch.Tensor:
        """
        Inputs:
          skill_mean: signed mean per skill (~[-1,1]), shape (B,K)
          skill_max : signed max per skill, shape (B,K)
          skill_min : signed min per skill, shape (B,K)
          skill_std : signed std per skill, shape (B,K)

        Output:
          theta_base: mastery profile in (0,1), shape (B,K)
        """
        skill_mean = skill_mean.to(self.device)
        skill_max = skill_max.to(self.device)
        skill_min = skill_min.to(self.device)
        skill_std = skill_std.to(self.device)

        # CHANGED: align implicit diagnosis with GNCDM Version B
        feat_imp = torch.cat([skill_mean, skill_max, skill_min, skill_std], dim=-1)  # (B,4K)

        theta_imp = self.skill_theta_nn(feat_imp)             # (B,K)
        theta_exp = torch.sigmoid(skill_mean)                 # (B,K)
        theta_base = (1.0 - self.alpha) * theta_imp + self.alpha * theta_exp  # (B,K)
        return theta_base

    # ============================================================
    # Policy head forward
    # ============================================================
    def policy_forward(
        self,
        theta_base: torch.Tensor,        # (B,K)
        q_vector_target: torch.Tensor,   # (B,K)
        skill_count: torch.Tensor,       # (B,K)
        log_sigma_min: float = -4.0,
        log_sigma_max: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
          theta_base      : (B,K)
          q_vector_target : (B,K)
          skill_count     : (B,K)

        Outputs:
          delta_mu  : (B,K)
          log_sigma : (B,K) (clipped)
        """
        theta_base = theta_base.to(self.device)
        q_vector_target = q_vector_target.to(self.device)
        skill_count = skill_count.to(self.device)

        log_count = torch.log1p(skill_count)                                # (B,K)
        x = torch.cat([theta_base, q_vector_target, log_count], dim=-1)     # (B,3K)
        out = self.policy_head(x)                                           # (B,2K)

        delta_mu = out[..., : self.n_know]                                  # (B,K)
        log_sigma = out[..., self.n_know :]                                 # (B,K)
        log_sigma = log_sigma.clamp(min=log_sigma_min, max=log_sigma_max)
        return delta_mu, log_sigma

    # ============================================================
    # Sample action theta + compute log_prob(theta|x)
    # ============================================================
    def sample_theta_action(
        self,
        theta_base: torch.Tensor,        # (B,K)
        q_vector_target: torch.Tensor,   # (B,K)
        skill_count: torch.Tensor,       # (B,K)
        deterministic: bool = False,
    ) -> PolicySample:
        """
        Inputs:
          theta_base      : (B,K) from diagnosis
          q_vector_target : (B,K) indicates relevant skills for target item
          skill_count     : (B,K) amount of evidence per skill
          deterministic   : if True, do not sample noise (z=mu)

        Outputs (PolicySample):
          theta_action : (B,K) action sampled in (0,1)
          log_prob     : (B,)  log pi(theta_action|x) used in REINFORCE
          entropy      : (B,)  Gaussian entropy in z-space (kept for analysis; not used by default)
        """
        theta_base = theta_base.to(self.device)
        q_vector_target = q_vector_target.to(self.device)
        skill_count = skill_count.to(self.device)

        # z_base = logit(theta_base)
        z_base = safe_logit(theta_base)                            # (B,K)

        # policy parameters
        delta_mu, log_sigma = self.policy_forward(theta_base, q_vector_target, skill_count)
        mu = z_base + delta_mu                                     # (B,K)
        sigma = torch.exp(log_sigma)                               # (B,K)

        # sample z
        if deterministic:
            # FIX: even in deterministic/debug mode, treat the chosen action as fixed
            # when evaluating the score-function log-density.
            z = mu.detach()                                        # (B,K)
        else:
            eps = torch.randn_like(mu)                             # (B,K)
            z = mu + eps * sigma                                   # (B,K)

            # FIX: REINFORCE remains stochastic in the FORWARD pass, but the
            # sampled action must be treated as fixed in the score-function
            # BACKWARD pass. Detaching only z keeps gradients through mu/sigma
            # (and therefore theta_base + Diagnosis), while removing the
            # unintended pathwise gradient through the random sample itself.
            z = z.detach()                                         # (B,K)

        # map to theta
        theta_action = torch.sigmoid(z)                            # (B,K)

        # log prob in z-space: log N(z; mu, sigma^2)
        log2pi = math.log(2.0 * math.pi)
        log_prob_z_dim = -0.5 * ((z - mu) / sigma).pow(2) - log_sigma - 0.5 * log2pi  # (B,K)
        log_prob_z = log_prob_z_dim.sum(dim=-1)                    # (B,)

        # Jacobian term for theta = sigmoid(z):
        # log|dz/dtheta| = -log(theta) - log(1-theta)
        # stable:
        #   -log(sigmoid(z)) = softplus(-z)
        #   -log(1-sigmoid(z)) = softplus(z)
        log_det = (F.softplus(-z) + F.softplus(z)).sum(dim=-1)     # (B,)

        # logistic-normal log prob for action=theta
        log_prob_theta = log_prob_z + log_det                      # (B,)

        # entropy (optional, not used in loss unless you add it)
        entropy_dim = 0.5 * (1.0 + log2pi) + log_sigma             # (B,K)
        entropy = entropy_dim.sum(dim=-1)                          # (B,)

        return PolicySample(
            theta_base=theta_base,
            theta_action=theta_action,
            z_base=z_base,
            mu=mu,
            log_sigma=log_sigma,
            z=z,
            log_prob=log_prob_theta,
            entropy=entropy,
        )
