"""
Add directional variables to street data
Update: 2023/01/19
@author: yuki oyama
"""

import pandas as pd

# %%
# read data
st_kiyo = pd.read_csv('dataset/streets_kiyosumi.csv')
st_kiba = pd.read_csv('dataset/streets_kiba.csv')
df_kiyo = pd.read_csv('dataset/apollo_data_w_angle/data_kiyo.csv')
df_kiba = pd.read_csv('dataset/apollo_data_w_angle/data_kiba.csv')

# %%
direc_kiyo = {}
direc_kiba = {}
for si in st_kiyo.index:
    s = st_kiyo.loc[si]['street']
    direc_kiyo[si] = {**df_kiyo.query(f'street == {s}').iloc[0][['east', 'south', 'west', 'north']]}
for si in st_kiba.index:
    s = st_kiba.loc[si]['street']
    direc_kiba[si] = {**df_kiba.query(f'street == {s}').iloc[0][['east', 'south', 'west', 'north']]}

# %%
direcdf_kiyo = pd.DataFrame(direc_kiyo).T
direcdf_kiba = pd.DataFrame(direc_kiba).T
st_kiyo = st_kiyo.join(direcdf_kiyo)
st_kiba = st_kiba.join(direcdf_kiba)

# %%
st_kiyo.to_csv('dataset/streets_kiyosumi.csv', index=False)
st_kiba.to_csv('dataset/streets_kiba.csv', index=False)

# %%
st_kiyo['dist_east'] = st_kiyo['distance_km'] * st_kiyo['east']
st_kiyo['dist_west'] = st_kiyo['distance_km'] * st_kiyo['west']
st_kiyo['dist_north'] = st_kiyo['distance_km'] * st_kiyo['north']
st_kiyo['dist_south'] = st_kiyo['distance_km'] * st_kiyo['south']

# %%
st_kiba['dist_east'] = st_kiba['distance_km'] * st_kiba['east']
st_kiba['dist_west'] = st_kiba['distance_km'] * st_kiba['west']
st_kiba['dist_north'] = st_kiba['distance_km'] * st_kiba['north']
st_kiba['dist_south'] = st_kiba['distance_km'] * st_kiba['south']
