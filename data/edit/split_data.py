"""
Get estimation data with index
@author: yuki oyama

Outputs:
- {area}_users.csv: N data in total including socioeconomic characteristics
- {area}_streets.csv: S data in total including street attributes
- {area}_choice.csv: N x S data in total
- {area}_incidence.csv: incidence with indexes
- {area}_angle.csv: angle with indexes
"""

import pandas as pd

# %%
areas = ['kiyosumi', 'kiba']

# %%
# streets
street_data = {}
street_idxs = {}
n_streets = {}
for area in areas:
    df = pd.read_csv(f'data/network/street_data_{area}.csv')
    street_data[area] = df
    idxs = {}
    for i in df.index: idxs[df.loc[i, 'street']] = i
    street_idxs[area] = idxs
    n_streets[area] = len(df)


# %%
street_data['kiyosumi'].to_csv('data/estimation_idx/streets_kiyosumi.csv', index=True, index_label='street_id')
street_data['kiba'].to_csv('data/estimation_idx/streets_kiba.csv', index=True, index_label='street_id')

# %%
# users
database = pd.read_csv('data/estimation_idx/choice_all.csv')
user_chars = ['male', 'female', 'age_0_20', 'age_30', 'age_40', 'age_50',
       'age_60+', 'under_year', 'year_1_5', 'year_5_10', 'year_10_20',
       'year_20+', 'everyday', '2_3_days_per_week', '1_day_per_week',
       '2_3_days_per_month', '1_month', 'resident', 'visitor', 'Lower_40',
       'Upper_40', 'Under_10_years', 'Upper_10_years', 'Under_1_day',
       'Upper_1_day']

# %%
df_kiyo = database[database['city'] == 'Kiyosumi']
df_kiba = database[database['city'] == 'Kiba']
dfs = {'kiyosumi': df_kiyo, 'kiba': df_kiba}

df_kiyo.head()

# %%
user_dfs = {}
choice_dfs = {}
for area in areas:
    df = dfs[area]
    users = {}
    choices = {}
    for i in df.index:
        user_id, street_id, choice = df.loc[i, ['person_idx', 'street_idx', 'included']]
        if user_id not in users.keys():
            users.update({user_id: {**df.loc[i, user_chars]}})
            choices[user_id] = {}
        choices[user_id].update({street_id: choice})
    user_dfs[area] = pd.DataFrame(users).T
    choice_dfs[area] = pd.DataFrame(choices).T

# %%
choice_dfs['kiyosumi'].to_csv('data/estimation_idx/choice_kiyosumi.csv', index=True)
choice_dfs['kiba'].to_csv('data/estimation_idx/choice_kiba.csv', index=True)

# %%
user_dfs['kiyosumi'].to_csv('data/estimation_idx/users_kiyosumi.csv', index=True, index_label='user')
user_dfs['kiba'].to_csv('data/estimation_idx/users_kiba.csv', index=True, index_label='user')

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

# %%
angles = {}
for area in areas:
    data = []
    df = pd.read_csv(f'data/network/incidence_angle_{area}.csv')
    for one, other, angle in zip(df['one'], df['other'], df['angle']):
        if one in street_idxs[area].keys() and other in street_idxs[area].keys():
            data.append(
                {'one': street_idxs[area][one],
                 'other': street_idxs[area][other],
                 'angle': angle
                 }
            )
    angles[area] = pd.DataFrame(data)
    angles[area].to_csv(f'data/estimation_idx/incidence_angle_{area}.csv', index=False)
