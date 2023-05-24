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
    ('c_tree', 'tree', None, 1),
    ('c_bldg', 'building', None, 1),
    ('c_road', 'road', None, 1),
    ('c_sky', 'sky', None, 1),
    ('b_river', 'inner_river_0', None, 1),
    ('b_avenue', 'inner_avenue_0', None, 1),
    ('b_area', 'inner_area', 'resident', 1),
    ('b_dist_base', 'distance_km', None, 0),
    ('a_10under', 'distance_km', 'Under_10_years', 1),
    ('a_1under', 'distance_km', 'Under_1_day', 1),
    ('a_female', 'distance_km', 'female', 1),
    ('a_40up', 'distance_km', 'Upper_40', 1)
]
sp_data.set_features(features)

# %%
seed = 111
rho_a = 1.01
A = 1.04
nu = 2
d = sp_data.datasets['kiyosumi']
splogit = spLogit(seed,
            d['nInd'], d['nSpc'], d['nFix'], d['nRnd'], d['x'], d['y'], d['W'],
            A, nu, rho_a)
post_params = splogit.estimate(nIter=2000, nIterBurn=1000, nGrid=100)

# %%
post_paramFix, post_paramRnd, post_zeta, post_iwDiagA, post_Sigma, post_rho, post_y, post_omega = post_params
### calculate posterior mean of beta and sigma
alpha_mean_hat = np.mean(post_paramFix, axis=0)
zeta_mean_hat = np.mean(post_zeta, axis=0)
sigma_mean_hat = np.mean(post_Sigma, axis=0)
y_mean_hat = np.mean(post_y, axis=0)
rho_post_mean = np.mean(post_rho)
# error = np.mean(Y - y_mean_hat)
ic(alpha_mean_hat)
ic(zeta_mean_hat)
ic(rho_post_mean)
ic(sigma_mean_hat)

sp_data.xRnd_names
