import argparse
import json
import os
from ntpath import basename

import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy
from torch.distributions import Categorical
from torch.nn import AvgPool2d, BCELoss, MaxPool2d
from tqdm import tqdm

tqdm.pandas()


def feature_hist_log(x, n_bins, min, max, *args, **kwargs):
    """creates histogram from the probabiliy / entropy map"""
    # hist = torch.histc(torch.log(x.flatten()), bins=n_bins, min=min, max=max)
    # return hist

    hist = torch.histc(torch.log(x.flatten()), bins=n_bins//2, min=min, max=max)
    entr = - x * torch.log(x)
    hist_entr = torch.histc(torch.log(entr.flatten()), bins=n_bins//2, min=min, max=max)
    return torch.cat([hist, hist_entr])


def feature_hist(x, n_bins, min, max, *args, **kwargs):
    """creates histogram from the probabiliy / entropy map"""
    hist = torch.histc(x.flatten(), bins=n_bins, min=min, max=max)
    return hist


def feature_pool(pred, k=16):
    """creates histogram from the probabiliy / entropy map"""
    reduced = None
    return reduced


def feature_pca(pred):
    """creates histogram from the probabiliy / entropy map"""
    reduced = None
    return reduced


def feature_tsne(pred):
    """creates histogram from the probabiliy / entropy map"""
    hist = None
    return hist


def map_mam_pool_hist(pred, k=16, bins=15):
    with torch.no_grad():
        max_p = MaxPool2d(k)
        avg_p = AvgPool2d(k)
        pool_max = max_p(pred).max(axis=0).values
        pool_min = (-max_p(-pred)).min(axis=0).values
        pool_avg = avg_p(pred).mean(axis=0)

        uncertainty_map = get_entropy(pred)
        u_pool_max = max_p(uncertainty_map).max(axis=0).values
        u_pool_min = (- max_p(-uncertainty_map)).min(axis=0).values
        u_pool_avg = avg_p(uncertainty_map).mean(axis=0)
        maps = list(map(lambda x: torch.histc(input=x, bins=bins, min=0, max=1), [pool_max, pool_min, pool_avg]))
        u_maps = list(map(lambda x: torch.histc(input=torch.log(x), bins=bins,
                      min=-20, max=0), [u_pool_max, u_pool_min, u_pool_avg]))
        f_maps = maps + u_maps
        maps = torch.cat(f_maps)
        return maps


def get_entropy(pred):
    min_real = torch.finfo(pred.dtype).min
    log_pred = torch.clamp(torch.log(pred), min=min_real)
    p_log_p = - pred * log_pred
    return p_log_p


def feature_extracor(pred_map, pool_size, hist_bin_size):

    return pred_map.max()


def parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, help='policy name')
    parser.add_argument('--in-dir', type=str, help='pred_map input dir')
    parser.add_argument('--out-dir', type=str, help='feature output dir')
    args = parser.parse_args()
    return args


feature_mapper = {
    'hist': feature_hist,
    'hist_log': feature_hist_log,
    'pca': feature_pca,
    'tsne': feature_tsne,
    'pool': feature_pool,
}


def main():
    args = parse()
    in_dir = args.in_dir
    out_dir = args.out_dir
    with open(args.config, 'r') as f:
        config_json = json.load(f)
    feature_mapper_name = config_json['policy']['gen']['method']['name']

    csv_dir = os.path.join(in_dir, 'gt.csv')
    df_res = pd.read_csv(csv_dir)

    assert feature_mapper_name in feature_mapper.keys()

    def feature_gen(x):
        return feature_mapper[feature_mapper_name](x, **config_json['policy']['gen']['method'])

    def create_map_feature(tensor_path):
        batch = torch.load(tensor_path)
        batch_feature = feature_gen(batch)
        batch_new_dir = os.path.join(out_dir, basename(tensor_path))
        torch.save(batch_feature, batch_new_dir)
        return batch_new_dir

    df_res['feature_path'] = df_res['path'].progress_apply(create_map_feature)
    df_res.to_csv(os.path.join(out_dir, basename(csv_dir)), index=False)


if __name__ == '__main__':
    main()
