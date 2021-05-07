import argparse
import json
import os
from datetime import datetime

import pandas as pd
import torch
import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

from ours.checkpoint import scikit_model_save


def parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, help='policy name')
    parser.add_argument('--seed', type=int, help='random seed for dataloader')
    parser.add_argument('--save-dir', type=str, help='saved model path')
    parser.add_argument('--features-dir', type=str, help='masks set')
    args = parser.parse_args()
    return args


def main():
    args = parse()
    save_dir = args.save_dir
    features_dir = args.features_dir
    seed = args.seed
    with open(args.config, 'r') as f:
        config_json = json.load(f)
    n_features = config_json['policy']['gen']['method']['n_bins']
    target = config_json['policy']['learner']['target']
    model_name = config_json['policy']['learner']['name']
    df_path = os.path.join(features_dir, 'gt.csv')
    df = pd.read_csv(df_path)

    def read_point(x_path):
        return torch.load(x_path).detach().cpu().numpy()

    df_feat = pd.DataFrame(df['feature_path'].apply(read_point).tolist(),
                           columns=[f'feat_{i}' for i in range(n_features)])

    df_feat['score'] = df[target]
    ss = StandardScaler()

    X = df_feat.drop(columns='score')
    y = df_feat['score']

    X = ss.fit_transform(X)

    lin_mod = LinearRegression()
    scores = cross_validate(lin_mod, X, y, cv=5,
                            scoring=('r2', 'neg_mean_absolute_percentage_error'),
                            return_train_score=True
                            )
    print(X.shape, y.shape)
    print(scores['test_neg_mean_absolute_percentage_error'])

    clf = LinearRegression().fit(X, y)
    clf.score(X, y)

    cur_date = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = os.path.join(save_dir, f'{model_name}_{cur_date}.dill')
    scikit_model_save({'clf': clf, 'scaler': ss}, file_name)


if __name__ == '__main__':
    main()
