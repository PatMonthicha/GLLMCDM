"""Load continuous-score NCDM response logs and Q-matrices."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {'user_id', 'item_id', 'score'}


class ResponseDataset(Dataset):
    """NCDM responses with zero-based IDs and continuous scores in [0, 10]."""

    def __init__(self, data_file, q_matrix):
        self.data_file = Path(data_file)
        self.data = pd.read_csv(self.data_file)
        missing = REQUIRED_COLUMNS.difference(self.data.columns)
        if missing:
            raise ValueError(
                f'{self.data_file} is missing columns: {sorted(missing)}'
            )

        self.q_matrix = np.asarray(q_matrix, dtype=np.float32)
        if self.q_matrix.ndim != 2:
            raise ValueError('Q-matrix must be a 2-D array.')

        self.user_ids = self.data['user_id'].to_numpy(dtype=np.int64)
        self.item_ids = self.data['item_id'].to_numpy(dtype=np.int64)
        self.scores = self.data['score'].to_numpy(dtype=np.float32)

        if len(self.data) == 0:
            raise ValueError(f'{self.data_file} contains no responses.')
        if self.user_ids.min() < 0 or self.item_ids.min() < 0:
            raise ValueError('user_id and item_id must be zero-based non-negative IDs.')
        if self.item_ids.max() >= self.q_matrix.shape[0]:
            raise ValueError(
                f'item_id {self.item_ids.max()} exceeds Q-matrix rows '
                f'({self.q_matrix.shape[0]}).'
            )
        if not np.isfinite(self.scores).all():
            raise ValueError(f'{self.data_file} contains non-finite scores.')
        if (self.scores < 0.0).any() or (self.scores > 10.0).any():
            raise ValueError(f'{self.data_file} contains scores outside [0, 10].')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        item_id = self.item_ids[index]
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(item_id, dtype=torch.long),
            torch.from_numpy(self.q_matrix[item_id]),
            torch.tensor(self.scores[index], dtype=torch.float32),
        )


def load_q_matrix(q_matrix_file):
    q_matrix = np.load(q_matrix_file)
    if q_matrix.ndim != 2:
        raise ValueError('Q-matrix must be a 2-D array.')
    return q_matrix.astype(np.float32, copy=False)
