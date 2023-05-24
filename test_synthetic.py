import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
from icecream import ic
from model import spLogit

def generate_data(
        nInd, nSpc,
        paramFix=np.array([0.2, -0.5, 0.4, 0.09]),
        paramRnd_mean=np.array([0.1, -2.0]),
        paramRnd_std=np.array([0.2, 0.5]),
        rho=0.4
        ):
    nFix = paramFix.shape[0]
    nRnd = paramRnd_mean.shape[0]
    x = np.random.normal(size=(nInd,nSpc,nFix+nRnd))
    x -= np.mean(x, axis=1, keepdims=True)
    x /= np.std(x, axis=1, keepdims=True)
    xFix = x[:,:,:nFix]
    xRnd = x[:,:,nFix:]

    # define spatial weight matrix
    xy = np.random.uniform(size=(nSpc,2))
    nbrs = NearestNeighbors(n_neighbors=4).fit(xy)
    spW = nbrs.kneighbors_graph(xy)
    I = sp.identity(nSpc, format='csc')
    spW -= I
    spW /= spW.sum(axis=1)
    spW = sp.csc_matrix(spW)
    invA = sp.linalg.inv(I - rho * spW)

    # observation
    paramRnd = paramRnd_mean + (np.diag(paramRnd_std) @ np.random.randn(nRnd,nInd)).T
    paramAll = np.concatenate([np.tile(paramFix, (nInd,1)), paramRnd], axis=1)
    mu = np.einsum('nsk,nk->ns', x, paramAll) + np.random.randn(nInd, nSpc)
    mu = np.einsum('ij,nj->ni', invA.toarray(), mu)
    prob = 1 / (1 + np.exp(-mu))
    y = np.random.binomial(1, prob)

    return x, xFix, xRnd, y, spW

# %%
if __name__ == '__main__':
    # generate synthetic data
    nInd, nSpc = 50, 200
    nFix, nRnd = 4, 2
    x, xFix, xRnd, y, W = generate_data(nInd, nSpc)

    # %%
    seed = 111
    rho_a = 1.01
    A = 1.04
    nu = 2
    splogit = spLogit(seed, nInd, nSpc, nFix, nRnd, x, y, W)

    # %%
    postRes, modelFits = splogit.estimate(nIter=500, nIterBurn=250, nGrid=20)

    # %%
    for paramName, stats in postRes.items():
        print(paramName)
        for statName, stat in stats.items():
            print(statName, stat)