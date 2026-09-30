# Purpose: Define command-line arguments for the G-NCDM S1/S2/S3 workflow.
# Provenance: Original research baseline; argument behavior is unchanged.
# -*- coding: utf-8 -*-
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', help='the path of the train file')
    parser.add_argument('--test_file', help='the path of the test file')
    parser.add_argument('--valid_file', help='the path of the valid file', default='')
    parser.add_argument('--Q_matrix', help='the path of the full Q-matrix', default='')
    parser.add_argument('--save_path', help='the save path of all results')

    # Kept for backward compatibility. In strict S2/S3 these values are
    # recomputed from the training split only, so no valid/test slots are reserved.
    parser.add_argument('--n_user', help='legacy total user count', default=None)
    parser.add_argument('--n_item', help='legacy total item count', default=None)
    parser.add_argument('--n_know', help='the number of knowledge points')
    parser.add_argument('--user_dim', help='the dimension of user vector', default=64)
    parser.add_argument('--item_dim', help='the dimension of item vector', default=2)
    parser.add_argument('--alpha', help='the hyperparameter for mingled learner GDF', default=0.99)
    parser.add_argument('--training_config', help='the path to the config json file')
    parser.add_argument('--n_epoch', type=int, default=None)

    parser.add_argument('--use_early_stopping', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0)
    parser.add_argument('--monitor_metric', choices=['rmse', 'mae', 'mse'], default='rmse')

    # ADDED: explicitly choose the evaluation protocol.
    # S2 = new learners + seen items.
    # S3 = new learners + new items, using train-only cold-start fallback.
    parser.add_argument('--eval_setting', choices=['S1', 'S2', 'S3'], default='S1')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
