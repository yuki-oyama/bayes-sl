import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import linalg as splinalg
from scipy.optimize import minimize
from numdifftools import Hessian
from sklearn import linear_model
# from optimparallel import minimize_parallel
from polyagamma import random_polyagamma
from sklearn.preprocessing import scale
from scipy.special import beta
from tqdm import tqdm
from icecream import ic
from copy import copy


def get_incidence_matrix(S, df, key='angle'):
    """
    Arguments:
        S: number of streets
        df: incidence_df
    Outpus:
        A: incidence matrix in csr_matrix format
    """

    angles = (180 - df[key]) / 180 if key == 'angle' else np.ones(len(df), dtype=np.float)
    A = csr_matrix(
        (angles, (df['one'], df['other'])), shape=(S,S)
        ) # S x S
    A /= np.sum(A, axis=1)

    return csr_matrix(A)

def get_spatial_weight_matrix(dist_mtrx, key='exp', alpha=1., k=8, d_lim=100):
    """
    Arguments:
        dist_mtrx: distance matrix with size S x S
        key:
            - adjacency: take 1 for only connected edges
            - nearest: take 1 for k nearest edges
            - within: take 1 for edges within d_lim
            - power: take 1/d^alpha, and 1 for connected edges
            - exp: take exp(-alpha*d), and 1 for connected edges
    Outpus:
        A: spatial weight matrix in csr_matrix format
    """
    S, _ = dist_mtrx.shape

    if key == 'adjacency':
        A = (dist_mtrx == 0)
        A *= (np.ones((S,S)) - np.eye(S))
        A /= A.sum(axis=1)
        return csr_matrix(A)
    elif key == 'nearest':
        A = np.zeros_like(dist_mtrx)
        for i in range(S):
            ds = dist_mtrx[i]
            ds_sorted = np.sort(ds)
            #ds_sorted = ds_sorted[np.where(ds_sorted > 0)] # not to consider self and connected in this stage
            hot_idxs = np.where(ds <= ds_sorted[k+1])
            A[i, hot_idxs] = 1.
        A *= (np.ones((S,S)) - np.eye(S))
        A /= A.sum(axis=1)
        return csr_matrix(A)
    elif key == 'within':
        A = (dist_mtrx <= d_lim) * (np.ones((S,S)) - np.eye(S))
        A /= A.sum(axis=1)
        return csr_matrix(A)
    elif key == 'power':
        d = np.clip(dist_mtrx/100, 1., None)
        A = 1./(d ** alpha)
        A *= (np.ones((S,S)) - np.eye(S)) # wii = 0
        A /= A.sum(axis=1)
        return csr_matrix(A)
    elif key == 'exp':
        A = np.exp(-alpha * (dist_mtrx/1000))
        A *= (np.ones((S,S)) - np.eye(S)) # wii = 0
        A /= A.sum(axis=1)
        return csr_matrix(A)
