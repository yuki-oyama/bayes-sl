# %%
import os
import numpy as np
import pandas as pd
import time
from model import spLogit
from dataset import spDataset
from icecream import ic
import json
import argparse

#### argparse ####
parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
  return v.lower() in ('true', '1')

def float_or_none(value):
    try:
        return float(value)
    except:
        return None

# Model parameters
model_arg = add_argument_group('Model')
model_arg.add_argument('--seed', type=int, default=123, help='random seed')
model_arg.add_argument('--root', type=str, default=None, help='root directory to save results')
model_arg.add_argument('--out_dir', type=str, default='test', help='output directory to be created')
model_arg.add_argument('--nIter', type=int, default=1000, help='number of iterations')
model_arg.add_argument('--nIterBurn', type=int, default=500, help='number of the first iterations for burn-in')
model_arg.add_argument('--nGrid', type=int, default=100, help='number of grids for griddy Gibbs sampler')
model_arg.add_argument('--areas', nargs='+', type=str, default=['kiyosumi', 'kiba'], help='areas to read data')
model_arg.add_argument('--areas_est', nargs='+', type=str, default=['kiyosumi', 'kiba'], help='areas to be analyzed ("All" is also possible)')
model_arg.add_argument('--mixedVars', nargs='+', type=str, default=['Tree', 'Bldg', 'Road', 'Sky', 'Dist'], help='variable names for random coefficients')
model_arg.add_argument('--spatialLag', type=str2bool, default=True, help='whether to consider spatial lag')
model_arg.add_argument('--sp_key', type=str, default='adjacency', help='spatial weight matrix')
model_arg.add_argument('--nNeighbor', type=int, default=3, help='number of neighbors to define spatial weight matrix')
model_arg.add_argument('--dist_neighbor', type=float, default=200, help='distance within which neighbors exist')



def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed

#### Main Codes ####
if __name__ == "__main__":
    config, _ = get_config()
    np.random.seed(config.seed)
    
    # output directory
    if config.root is not None:
        out_dir = os.path.join(config.root, "results", "real", config.out_dir)
    else:
        out_dir = os.path.join("results", "real", config.out_dir)
    
    try:
        os.makedirs(out_dir, exist_ok = False)
    except:
        out_dir += '_' + time.strftime("%Y%m%dT%H%M")
        os.makedirs(out_dir, exist_ok = False)

    ## define dataset
    sp_data = spDataset(sp_key=config.sp_key, k=config.nNeighbor, d_lim=config.dist_neighbor)
    for area in config.areas:
        user_df = pd.read_csv(f'data/users_{area}.csv').set_index('user')
        street_df = pd.read_csv(f'data/streets_{area}.csv').set_index('street_id')
        angle_df = pd.read_csv(f'data/incidence_angle_{area}.csv')
        choice_df = pd.read_csv(f'data/choice_{area}.csv', index_col=0)
        dist_df = pd.read_csv(f'data/distance_matrix_{area}.csv', index_col=0)
        sp_data.set_data(area, user_df, street_df, choice_df, dist_df, angle_df)

    # %%
    features = [
        ['intercept', None, None, 1],
        ['Tree', 'tree', None, 1],
        ['Bldg', 'building', None, 1],
        ['Road', 'road', None, 1],
        ['Sky', 'sky', None, 1],
        ['InRiver', 'inner_river_0', None, 1],
        ['InAvenue', 'inner_avenue_0', None, 1],
        ['InArea', 'inner_area', 'resident', 1],
        ['Dist', 'distance_km', None, 1],
        ['DistRes10under', 'distance_km', 'Under_10_years', 1],
        ['DistFreq1under', 'distance_km', 'Under_1_day', 1],
        ['DistFemale', 'distance_km', 'female', 1],
        ['Dist40up', 'distance_km', 'Upper_40', 1],
        ['DistEast', 'dist_east', None, 1],
        ['DistNorth', 'dist_north', None, 1],
        ['DistSouth', 'dist_south', None, 1],
    ]
    nRnd = 0
    for i, [vname, v, s, is_fixed] in enumerate(features):
        if vname in config.mixedVars:
            features[i] = [vname, v, s, 0]
            nRnd += 1
        else:
            features[i] = [vname, v, s, 1]
    # print(features)
    sp_data.set_features(features)
    sp_data.merge_data()

    # model name
    if nRnd == 0:
        if not config.spatialLag:
            model_name = 'logit'
        else:
            model_name = 'logitLag'
    else:
        if not config.spatialLag:
            model_name = 'mixed'
        else:
            model_name = 'mixedLag' 
    
    # parameters
    nIter = config.nIter #20000
    nIterBurn = config.nIterBurn #10000
    nGrid = config.nGrid #100
    seed = config.seed #111
    rho_a = 1.01
    A = 1.04
    nu = 2
    estDfs = {}
    fits = {}
    for area in config.areas_est:
        d = sp_data.datasets[area]
        splogit = spLogit(seed=seed, A=A, nu=nu, rho_a=rho_a, spatialLag=config.spatialLag)
        splogit.load_data_from_spData(d)
        postRes, modelFits, postParams, elasRes, meRes = splogit.estimate(nIter=nIter, nIterBurn=nIterBurn, nGrid=nGrid)
        dfRes = pd.DataFrame(postRes).T
        print(dfRes)
        estDfs[area] = dfRes
        fits[area] = modelFits

    # %%
    for area in config.areas_est:
        estDfs[area].to_csv(f'{out_dir}/{model_name}_Est_{area}.csv', index=True)

    # %%
    fitDf = pd.DataFrame(fits)
    fitDf.to_csv(f'{out_dir}/{model_name}_Fit.csv', index=True)

    with open(os.path.join(out_dir, "config.json"), mode="w") as f:
        json.dump(config.__dict__, f, indent=4)
