import numpy as np
import pandas as pd
from model import spLogit
from dataset import spDataset
from icecream import ic

# %%
sp_data = spDataset()
areas = ['kiyosumi', 'kiba']
for area in areas:
    user_df = pd.read_csv(f'data/users_{area}.csv').set_index('user')
    street_df = pd.read_csv(f'data/streets_{area}.csv').set_index('street_id')
    angle_df = pd.read_csv(f'data/incidence_angle_{area}.csv')
    choice_df = pd.read_csv(f'data/choice_{area}.csv', index_col=0)
    dist_df = pd.read_csv(f'data/distance_matrix_{area}.csv', index_col=0)
    sp_data.set_data(area, user_df, street_df, choice_df, dist_df, angle_df)

# %%
features = [
    ('intercept', None, None, 1),
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
    ('Dist40up', 'distance_km', 'Upper_40', 1),
    ('DistEast', 'dist_east', None, 1),
    ('DistNorth', 'dist_north', None, 1),
    ('DistSouth', 'dist_south', None, 1),
]
sp_data.set_features(features)
sp_data.merge_data()

# %%
model_name = 'mixedLag'
nIter = 20000
nIterBurn = 10000
nGrid = 100
seed = 111
rho_a = 1.01
A = 1.04
nu = 2
estDfs = {}
fits = {}
areas_est = ['kiyosumi', 'kiba'] # 'All' is also a possible option
for area in areas_est:
    d = sp_data.datasets[area]
    splogit = spLogit(seed=seed, A=A, nu=nu, rho_a=rho_a, spatialLag=True)
    splogit.load_data_from_spData(d)
    postRes, modelFits, postParams = splogit.estimate(nIter=nIter, nIterBurn=nIterBurn, nGrid=nGrid)
    dfRes = pd.DataFrame(postRes).T
    print(dfRes)
    estDfs[area] = dfRes
    fits[area] = modelFits

# %%
for area in areas_est:
    estDfs[area].to_csv(f'{model_name}Est_{area}_iter{nIter}_burn{nIterBurn}_grid{nGrid}.csv', index=True)

# %%
fitDf = pd.DataFrame(fits)
fitDf.to_csv(f'{model_name}Fit_iter{nIter}_burn{nIterBurn}_grid{nGrid}.csv', index=True)
