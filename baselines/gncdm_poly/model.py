# Purpose: Define the continuous-score G-NCDM model and diagnostic branches.
# Provenance: Original research baseline; executable model logic is unchanged.
# model.py
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Jiatong Li
# All rights reserved.
# 
# This software is the confidential and proprietary information
# of Jiatong Li. You shall not disclose such confidential
# information and shall use it only in accordance with the terms of
# the license agreement.


from collections import OrderedDict
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn 
import torch.nn.functional as F
 
class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)
        
class GNCDM(nn.Module):
    def __init__(self, n_user:int, n_item:int, n_know:int, \
        user_dim:int, item_dim:int, alpha:float, \
        Q_mat: np.array = None, \
        monotonicity_assumption: bool = True,\
        device = torch.device('cpu')):
        '''
        Args:
            n_user:int, the number of learners
            n_item:int, the number of test items
            n_know:int, the number of knowledge concepts,
                which equals to the dimension of diagnostic results.
            user_dim:int, the dimension of aggregated user representations.
            item_dim:int, the dimension of aggregated item representations.
            Q_mat:np.array((n_item,n_know)), the binary Q-matrix.
            monotonicity_assumption:bool (default False) whether to apply
                the monotonicity assumption to the diagnostic module. If True,
                the monotonicity assumption is applied.
            device:torch.device
        '''
        super(GNCDM,self).__init__()
        self.n_user = n_user 
        self.n_item = n_item 
        self.n_know = n_know
        self.user_dim = user_dim 
        self.item_dim = item_dim
        self.itf = self.ncd_func
        self.device = device

        self.Q_mat = torch.Tensor(Q_mat) \
            if Q_mat is not None else torch.ones((n_item, n_know))

        self.K_diff_mat = nn.Parameter(torch.zeros((n_know, user_dim)),\
            requires_grad=False).to(device)
        self.K_diff_mat.requires_grad = True

        self.Q_mat = self.Q_mat.to(device)

        self.alpha = alpha

        # Buffer of examinee traits
        self.Theta_buf = nn.Parameter(torch.zeros((n_user, n_know))\
            , requires_grad=False).to(device)

        # Buffer of question feature traits
        self.Psi_buf = nn.Parameter(torch.zeros((n_item, n_know))\
            , requires_grad=False).to(device)
        
        f_linear = nn.Linear if monotonicity_assumption is False else PosLinear


        self.f_nn = nn.Sequential(
            OrderedDict(
                [
                    ('f_layer_1', f_linear(n_item, n_know)),
                    ('f_activate_1', nn.Sigmoid()),
                    ('f_layer_2', f_linear(n_know, n_know)),
                    ('f_activate_2', nn.Sigmoid())
                ]
            )
        ).to(device)

        self.g_nn = nn.Sequential(
            OrderedDict(
                [
                    ('g_layer_1', nn.Linear(n_user, n_know)),
                    ('g_activate_1', nn.Sigmoid()),
                    ('g_layer_2', nn.Linear(n_know, n_know)),
                    ('g_activate_2', nn.Sigmoid()),
                    ('g_layer_3', nn.Linear(n_know, n_know)),
                    ('g_activate_3', nn.Sigmoid())
                ]
            )
        ).to(device)

        self.theta_agg_mat = f_linear(n_know, user_dim).to(device)      # input = (B,18) output theta_agg = (B, 32)
        self.psi_agg_mat = nn.Linear(n_know, item_dim).to(device)

        '''
        user_dim=32: weight shape = (32, 18), bias shape = (32, )
        '''

        self.ncd = nn.Sequential(
            OrderedDict([
                ('pred_layer_1', nn.Linear(user_dim, 64)),
                ('pred_activate_1', nn.Sigmoid()),
                ('pred_dropout_1', nn.Dropout(p=0.5)),
                ('pred_layer_2', nn.Linear(64, 32)),
                ('pred_activate_2', nn.Sigmoid()),
                ('pred_dropout_2', nn.Dropout(p=0.5)),
                ('pred_layer_3', nn.Linear(32, 1)),
                #('pred_activate_3', nn.Sigmoid()),

            ])
        ).to(device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def ncd_func(self, theta, psi):
        """
        - input: theta_agg, psi_agg (batch_size, user_dim)
        """
        assert(self.user_dim == self.item_dim)
        y_score = self.ncd(theta-psi)     # real-valued output
        #print("xxxxxxxxxxxxxxxxxxxxxxxx")
        #print(torch.min(theta),torch.max(theta))
        #print(theta)
        #print(len(y_score))
        return y_score

    def diagnose_theta_implicit(self, user_log: torch.Tensor):
        """
        Return the implicit mastery branch for logs expressed in the
        training-item coordinate system.

        CHANGED FOR OPEN-WORLD S3:
        The original f_nn is intentionally kept unchanged. Therefore this
        branch can only consume vectors whose last dimension equals n_item
        used during training. New-item responses are not inserted into these
        coordinates.
        """
        if user_log.shape[-1] != self.n_item:
            raise ValueError(
                f"Implicit branch expects {self.n_item} training-item "
                f"coordinates, but received {user_log.shape[-1]}."
            )
        return self.f_nn(user_log)

    def diagnose_theta_explicit(self,
                                user_log: torch.Tensor,
                                user_mask: torch.Tensor = None,
                                q_mat: torch.Tensor = None):
        """
        Return the explicit mastery branch from response evidence and a
        Q-matrix.

        The optional q_mat makes the explicit branch usable with unseen
        items at inference. This is deterministic and introduces no learned
        parameter for the new items.

        Args:
            user_log: (B, M), signed scores in [-1, 1].
            user_mask: (B, M), 1 for observed responses and 0 otherwise.
            q_mat: (M, K). If omitted, the training Q-matrix is used.
        """
        if user_mask is None:
            user_mask = (user_log != 0).float()

        q_used = self.Q_mat if q_mat is None else q_mat
        q_used = q_used.to(device=user_log.device, dtype=user_log.dtype)

        if user_log.shape[-1] != q_used.shape[0]:
            raise ValueError(
                "Explicit branch requires user_log and q_mat to describe "
                f"the same number of items; got {user_log.shape[-1]} and "
                f"{q_used.shape[0]}."
            )
        if q_used.shape[1] != self.n_know:
            raise ValueError(
                f"q_mat must have {self.n_know} skill columns; "
                f"received {q_used.shape[1]}."
            )

        evid_count_raw = user_mask @ q_used
        evid_sum = (user_log * user_mask) @ q_used

        # Skills with no evidence use neutral signed evidence 0, so sigmoid(0)=0.5.
        skill_score = torch.where(
            evid_count_raw > 0,
            evid_sum / evid_count_raw.clamp(min=1e-6),
            torch.zeros_like(evid_sum),
        )
        return torch.sigmoid(skill_score)

    def combine_theta(self,
                      theta_imp: torch.Tensor,
                      theta_exp: torch.Tensor):
        """Combine implicit and explicit branches using the trained alpha."""
        if theta_imp.shape != theta_exp.shape:
            raise ValueError(
                f"theta_imp and theta_exp must have identical shapes; "
                f"got {tuple(theta_imp.shape)} and {tuple(theta_exp.shape)}."
            )
        return theta_imp * (1 - self.alpha) + theta_exp * self.alpha

    def diagnose_theta_components(self,
                                  user_log: torch.Tensor,
                                  user_mask: torch.Tensor = None):
        """Return (theta_implicit, theta_explicit, theta_base) for seen items."""
        theta_imp = self.diagnose_theta_implicit(user_log)
        theta_exp = self.diagnose_theta_explicit(
            user_log, user_mask=user_mask, q_mat=self.Q_mat
        )
        theta = self.combine_theta(theta_imp, theta_exp)
        return theta_imp, theta_exp, theta

    def diagnose_theta(self,
                       user_log: torch.Tensor,
                       user_mask: torch.Tensor = None):
        """
        Diagnose learner cognitive states from logs in the training-item
        coordinate system. The architecture and output are unchanged.
        """
        _, _, theta = self.diagnose_theta_components(user_log, user_mask)
        return theta

    def diagnose_theta_s3(self,
                          user_log_new: torch.Tensor,
                          user_mask_new: torch.Tensor,
                          q_mat_new: torch.Tensor,
                          mean_train_theta_imp: torch.Tensor):
        """
        Diagnose S3 learners from responses on post-training unseen items.

        CHANGED FOR S3:
        - implicit branch: frozen population prior = mean implicit mastery
          computed from training learners only;
        - explicit branch: learner-specific evidence computed from the unseen
          responses and their external Q-vectors;
        - no f_nn call is made on new-item coordinates and no parameter is fit.

        Returns:
            theta_imp, theta_exp, theta_base, each with shape (B, K).
        """
        theta_exp = self.diagnose_theta_explicit(
            user_log_new, user_mask=user_mask_new, q_mat=q_mat_new
        )

        mean_imp = mean_train_theta_imp.to(
            device=user_log_new.device, dtype=user_log_new.dtype
        )
        if mean_imp.ndim == 1:
            mean_imp = mean_imp.unsqueeze(0)
        if mean_imp.shape[-1] != self.n_know:
            raise ValueError(
                f"mean_train_theta_imp must have {self.n_know} dimensions."
            )
        theta_imp = mean_imp.expand(user_log_new.shape[0], -1)
        theta = self.combine_theta(theta_imp, theta_exp)
        return theta_imp, theta_exp, theta

    def diagnose_psi(self, item_log: torch.Tensor):
        '''
        Args:
            item_log:torch.Tensor((batch_size, n_items)), the user logs.
                
        Return:
            psi:torch.Tensor((batch_size, n_know)), diagnostic results
                of each item.
        '''
        psi = self.g_nn(item_log)
        return psi

    def diagnose_theta_psi(self,
                           user_log: torch.Tensor,
                           item_log: torch.Tensor,
                           user_mask: torch.Tensor = None):
        '''
        Diagnose learners' and items' traits simultaneously.

        Args:
            user_log:  torch.Tensor((batch_size, n_items)), signed [-1,1]
            item_log:  torch.Tensor((batch_size, n_users)), signed [-1,1]
        Return:
            theta: torch.Tensor((batch_size, n_know))
            psi:   torch.Tensor((batch_size, n_know))
        '''
        theta = self.diagnose_theta(user_log, user_mask)
        psi = self.diagnose_psi(item_log)
        return theta, psi
    
    def update_Theta_buf(self, theta_new, user_id):
        self.Theta_buf[user_id] = theta_new
    
    def update_Psi_buf(self, psi_new, item_id):
        self.Psi_buf[item_id] = psi_new

    def predict_response(self, theta, psi, Q_batch):
        '''
        Predict response scores given a batch of theta (learners' cognitive states),
        psi (items' features), and Q-vectors of these items
        Args:
            theta:torch.Tensor((batch_size, n_know)), learners' cognitive states
            psi:torch.Tensor((batch_size, n_know)), items' cognitive states
            Q_batch:torch.Tensor((batch_size, n_know)), Q-vectors. Q_batch[i] is
                the Q-vector of the item with feature psi[i]
        Return:
            output:torch.Tensor((batch_size,1)), the predicted score (0–10 scale)
                of each pair of learner and item.
        '''
        # theta: (B, n_know) = (B, 18), psi: (B, n_know) = (B, 18), self.Q_mat shape = (n_item, n_know) = (16, 18)
        theta_agg = self.theta_agg_mat(theta * Q_batch)    # (B, user_dim)= (B, 32)
        psi_agg = self.psi_agg_mat(psi * Q_batch)          # (B, item_dim)
        #print("xxxxxxxxxxxxxxxxxxxxxx")
        #print(theta_agg) 
        output = self.itf(theta_agg, psi_agg)
        return output

    def forward(self,
                user_log: torch.Tensor,
                item_log: torch.Tensor,
                user_mask: torch.Tensor,
                user_id: torch.LongTensor,
                item_id: torch.LongTensor):
        theta, psi = self.diagnose_theta_psi(user_log, item_log, user_mask)   # (B, 18)
        Q_batch = self.Q_mat[item_id].squeeze(dim=1)   # (B, 18)
        output = self.predict_response(theta, psi, Q_batch)
        return output

    def forward_using_buf(self, user_id: torch.LongTensor, \
        item_id: torch.LongTensor):
        ''' 
        Unlike forward(), this method predict response using thetas
        and psis from bufferes rather than from outputs of diagnostic modules
        given response logs.
        '''
        theta = self.Theta_buf[user_id].squeeze(dim=1)
        psi = self.Psi_buf[item_id].squeeze(dim=1)
        Q_batch = self.Q_mat[item_id].squeeze(dim=1)
        output = self.predict_response(theta, psi, Q_batch)
        return output

    def get_Theta_buf(self):
        return self.Theta_buf.detach().cpu()

    def get_Psi_buf(self):
        return self.Psi_buf.detach().cpu()
    
    def forward_theta_log_psi_buf(self,
                              user_log: torch.Tensor,
                              user_mask: torch.Tensor,
                              user_id: torch.LongTensor,
                              item_id: torch.LongTensor):
        """
        Predict using theta computed from the current new-user log,
        and psi taken from Psi_buf learned from training items.

        CHANGED NOTE: user_id is retained only for call compatibility and is
        intentionally not used, so new learners need no Theta_buf slot.
        """
        theta = self.diagnose_theta(user_log, user_mask)          # (B, K)
        psi   = self.Psi_buf[item_id].squeeze(dim=1)              # (B, K)
        Q_batch = self.Q_mat[item_id].squeeze(dim=1)              # (B, K)
        return self.predict_response(theta, psi, Q_batch)



# 2025.04.21. Add UAutoRec and CDAE
class UAutoRec(nn.Module):
    def __init__(self, n_user: int, n_item: int, \
        hidden_dim: int, device = torch.device('cpu')):
        super(UAutoRec, self).__init__()
        self.n_user = n_user 
        self.n_item = n_item 
        self.hidden_dim = hidden_dim 
        self.device = device 
        self.f_enc = nn.Linear(n_item, \
            hidden_dim).to(device)
        self.f_dec = nn.Linear(hidden_dim, \
            n_item).to(device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def forward(self, x_input: torch.Tensor, \
        user_id: torch.LongTensor):
        h = torch.sigmoid(self.f_enc(x_input))
        x_output = torch.sigmoid(self.f_dec(h))
        return x_output

class CDAE(nn.Module):
    def __init__(self, n_user: int, n_item: int, \
        hidden_dim: int, device = torch.device('cpu')):
        super(CDAE, self).__init__()
        self.n_user = n_user 
        self.n_item = n_item 
        self.hidden_dim = hidden_dim 
        self.device = device 
        self.f_enc = nn.Linear(n_item, \
            hidden_dim).to(device)
        self.user_emb = nn.Embedding(n_user, \
            n_item).to(device)
        self.dropout = nn.Dropout(p=0.5).to(device)
        self.f_dec = nn.Linear(hidden_dim, \
            n_item).to(device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def forward(self, x_input: torch.Tensor, \
        user_id: torch.LongTensor):
        # print(x_input.size(),self.user_emb(user_id).size())
        h = torch.sigmoid(self.dropout(\
            self.f_enc(x_input+self.user_emb(user_id).squeeze(dim=1))))
        x_output = torch.sigmoid(self.f_dec(h))
        return x_output
