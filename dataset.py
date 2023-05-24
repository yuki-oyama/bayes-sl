import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix

def get_incidence_matrix(nSpc, angle_df, angle=True):
    """
    Arguments:
        nSpc: number of spatial units
        angle_df: dataframe including rows of "one", "other", "angle"
    Outpus:
        A: incidence matrix in csr_matrix format
    """
    if angle:
        ele = (180. - angle_df[key]) / 180.
    else:
        ele = np.ones(len(angle_df), dtype=np.float64) # basic incidence
    A = csr_matrix(
        (ele, (angle_df['one'], angle_df['other'])), shape=(nSpc, nSpc)
        )
    A /= np.sum(A, axis=1) # normalization
    return csr_matrix(A)

def get_spatial_weight_matrix(dist_mtrx, key='exp', scale=1., k=3, d_lim=100):
    """
    Arguments:
        dist_mtrx: distance matrix with size S x S
        key:
            - adjacency: take 1 for only connected edges
            - nearest: take 1 for k nearest edges
            - within: take 1 for edges within d_lim
            - power: take 1/d^scale, and 1 for connected edges
            - exp: take exp(-scale*d), and 1 for connected edges
    Outpus:
        W: spatial weight matrix in csr_matrix format
    """
    nSpc, _ = dist_mtrx.shape

    if key == 'adjacency':
        W = (dist_mtrx == 0)
    elif key == 'nearest':
        W = np.zeros_like(dist_mtrx)
        for i in range(nSpc):
            ds = dist_mtrx[i]
            ds_sorted = np.sort(ds)
            hot_idxs = np.where(ds <= ds_sorted[k+1])
            W[i, hot_idxs] = 1
    elif key == 'nearest_dist':
        W = np.zeros_like(dist_mtrx)
        for i in range(nSpc):
            ds = dist_mtrx[i]
            ds_sorted = np.sort(ds)
            hot_idxs = np.where(ds <= ds_sorted[k+1])
            W[i, hot_idxs] = ds[hot_idxs]
    elif key == 'within':
        W = (dist_mtrx <= d_lim)
    elif key == 'power':
        d = np.clip(dist_mtrx/100, 1., None)
        W = 1./(d ** scale)
    elif key == 'exp':
        W = np.exp(-scale * (dist_mtrx/1000))

    # normalization
    W *= (np.ones_like(W) - np.eye(nSpc))
    W /= W.sum(axis=1)
    return csr_matrix(W)


class spDataset(object):

    def __init__(self, sp_key='adjacency', scale=1., k=3, d_lim=100):
        self.areas = []
        self.datasets = {}
        self.sp_key = sp_key
        self.scale = scale
        self.k = k
        self.d_lim = d_lim

    def set_data(self,
            area,
            user_df, street_df, choice_df,
            dist_df, angle_df):
        self.areas.append(area)
        nInd = len(user_df)
        nSpc = len(street_df)
        xInd = user_df
        xSpc = street_df
        y = choice_df.values
        W = self.get_W(nSpc, dist_df, angle_df)
        dataset = {
            'nInd': nInd,
            'nSpc': nSpc,
            'xInd': xInd,
            'xSpc': xSpc,
            'y': y,
            'W': W,
        }
        self.datasets[area] = dataset

    def get_W(self, nSpc, dist_df, angle_df):
        if self.sp_key in ['adjacency', 'angle']:
            W = get_incidence_matrix(nSpc, angle_df, (self.sp_key == 'angle'))
        else:
            dist_mtrx = np.zeros(dist_df.shape) # S x S
            for col in dist_df.columns: dist_mtrx[:,int(col)] = dist_df[col].values
            W = get_spatial_weight_matrix(
                    dist_mtrx, key=self.sp_key, scale=self.scale, k=self.k, d_lim=self.d_lim)
        return W

    def set_features(self, features):
        """Auguments:
        f_keys: list of (feature_name, spc_attribute_name, ind_char_name, isFixVar)
        """
        self.xFix_names = []
        self.xRnd_names = []
        nFix = sum([isFix for (_, _, _, isFix) in features])
        nRnd = len(features) - nFix
        for area in self.areas:
            d = self.datasets[area]
            nInd, nSpc = d['nInd'], d['nSpc']
            xFix, xRnd = [], []
            for f_name, satt, ichar, isFix in features:
                x_sp = np.ones(nSpc) if satt is None else d['xSpc'][satt].values # (S,1)
                x_ind = np.ones(nInd) if ichar is None else d['xInd'][ichar].values # (N,1)
                if isFix:
                    xFix.append(x_ind[:, np.newaxis] * x_sp[np.newaxis, :]) # (N, S)
                    self.xFix_names.append(f_name)
                else:
                    xRnd.append(x_ind[:, np.newaxis] * x_sp[np.newaxis, :]) # (N, S)
                    self.xRnd_names.append(f_name)
            xFix = np.stack(xFix).transpose(1,2,0) # (N, S, nFix)
            xRnd = np.stack(xRnd).transpose(1,2,0) # (N, S, nRnd)
            x = np.concatenate([xFix, xRnd], axis=2) # (N, S, nFix+nRnd)
            d.update({'x': x, 'xFix': xFix, 'xRnd': xRnd, 'nFix': nFix, 'nRnd': nRnd})
            self.datasets[area] = d

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
    ('intercept', None, None, 1),
    ('tree', 'tree', None, 1),
    ('dist', 'distance_km', None, 0),
]
sp_data.set_features(features)
sp_data.datasets['kiba']['xRnd'].shape
