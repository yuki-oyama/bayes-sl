"""
Get estimation data with index
@author: yuki oyama

- change user ids: residents from 0-24 and visitors from 25-49
- change street number to index
"""
import pandas as pd

# %%
areas = ['kiyosumi', 'kiba']

# %%
# streets
street_idxs = {}
n_streets = {}
for area in areas:
    df = pd.read_csv(f'data/network/street_data_{area}.csv')
    idxs = {}
    for i in df.index: idxs[df.loc[i, 'street']] = i
    street_idxs[area] = idxs
    n_streets[area] = len(df)

# %%
# users
# change resident ids: 0-24
# change visitor ids: 25-49
resident_id_idx = {
    n: (n-1) for n in range(1,26)
}
visitor_id_idx = {
    n: 25 + (n-1) for n in range(1,26)
}

# %%
database = pd.read_csv('data/cleaned_data_all_new.csv')

# %%
df_kiyo = database[database['city'] == 'Kiyosumi']
df_kiba = database[database['city'] == 'Kiba']
dfs = {'kiyosumi': df_kiyo, 'kiba': df_kiba}

df_kiyo.head()

# %%
new_users = {}
new_streets = {}
for area in areas:
    new_user, new_street = [], []
    df = dfs[area]
    idx_ = street_idxs[area]
    for i in df.index:
        user, street, visitor = df.loc[i, ['person', 'street', 'visitor']]
        new_street.append(idx_[street])
        if visitor == 0:
            new_user.append(resident_id_idx[user])
        elif visitor == 1:
            new_user.append(visitor_id_idx[user])
    new_users[area] = new_user
    new_streets[area] = new_street
    # df['person_idx'] = new_user
    # df['street_idx'] = new_street

# %%
df_new = dfs['kiyosumi'].copy()
len(df_new)
len(new_users['kiyosumi'])
len(new_streets['kiyosumi'])

# %%
for area in areas:
    df = dfs[area].copy()
    users, streets = new_users[area], new_streets[area]
    df['person_idx'] = users
    df['street_idx'] = streets
    dfs[area] = df

# %%
new_data = pd.concat([dfs['kiyosumi'], dfs['kiba']])

# %%
new_data.to_csv('data/estimation_idx/choice_all.csv', index=False)

# %%
incidences = {}
for area in areas:
    data = []
    df = pd.read_csv(f'data/network/incidence_{area}.csv')
    for one, other in zip(df['one'], df['other']):
        if one in street_idxs[area].keys() and other in street_idxs[area].keys():
            data.append(
                {'one': street_idxs[area][one],
                 'other': street_idxs[area][other]}
            )
    incidences[area] = pd.DataFrame(data)
    incidences[area].to_csv(f'data/estimation_idx/incidence_{area}.csv', index=False)
