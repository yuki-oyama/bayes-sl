import numpy as np
import pandas as pd
from model import spLogit
from dataset import spDataset
from icecream import ic

# %%
sp_data = spDataset()
areas = ['kiyosumi', 'kiba']
for area in areas:
    user_df = pd.read_csv(f'dataset/users_{area}.csv').set_index('user')
    street_df = pd.read_csv(f'dataset/streets_{area}.csv').set_index('street_id')
    angle_df = pd.read_csv(f'dataset/incidence_angle_{area}.csv')
    choice_df = pd.read_csv(f'dataset/choice_{area}.csv', index_col=0)
    dist_df = pd.read_csv(f'dataset/distance_matrix_{area}.csv', index_col=0)
    sp_data.set_data(area, user_df, street_df, choice_df, dist_df, angle_df)

# %%
features = [
    ('intercept', None, None, 0),
    ('Tree', 'tree', None, 0),
    ('Bldg', 'building', None, 0),
    ('Road', 'road', None, 0),
    ('Sky', 'sky', None, 0),
    ('InRiver', 'inner_river_0', None, 1),
    ('InAvenue', 'inner_avenue_0', None, 1),
    ('InArea', 'inner_area', 'resident', 1),
    ('Dist', 'distance_km', None, 0),
    ('DistRes10under', 'distance_km', 'Under_10_years', 1),
    ('DistFreq1under', 'distance_km', 'Under_1_day', 1),
    ('DistFemale', 'distance_km', 'female', 1),
    ('Dist40up', 'distance_km', 'Upper_40', 1)
]
sp_data.set_features(features)
sp_data.merge_data()

# %%
seed = 111
rho_a = 1.01
A = 1.04
nu = 2
dfs = []
areas_est = ['All']
for area in areas_est:
    d = sp_data.datasets[area]
    splogit = spLogit(seed=seed, A=A, nu=nu, rho_a=rho_a)
    splogit.load_data_from_spData(d)
    postRes, modelFits, postParams = splogit.estimate(nIter=2000, nIterBurn=1000, nGrid=100)
    dfRes = pd.DataFrame(postRes).T
    print(dfRes)
    dfs.append(dfRes)

# %%
dfs[0].to_csv('all_iter2000_burn1000_grid100.csv', index=True)
# dfs[1].to_csv('kiba_iter20000_burn10000_grid100.csv', index=True)
