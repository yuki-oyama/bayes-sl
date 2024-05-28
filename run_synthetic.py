# %%
import os
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from icecream import ic
from model import spLogit
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
model_arg.add_argument('--nInd', type=int, default=50, help='number of individuals')
model_arg.add_argument('--nSpc', type=int, default=200, help='number of spatial units')
model_arg.add_argument('--paramFix', nargs='+', type=float, default=[0.2, -0.5, 0.4, 0.09], help='true parameters for fixed effects')
model_arg.add_argument('--paramRnd_mean', nargs='+', type=float, default=[0.1, -2.0], help='true mean parameters for random effects')
model_arg.add_argument('--paramRnd_std', nargs='+', type=float, default=[0.2, 0.5], help='true std parameters for random effects')
model_arg.add_argument('--rhos', nargs='+', type=float, default=[0.4], help='true spatial parameter (for multiple times of tests)')
model_arg.add_argument('--nIter', type=int, default=1000, help='number of iterations')
model_arg.add_argument('--nIterBurn', type=int, default=500, help='number of the first iterations for burn-in')
model_arg.add_argument('--nGrid', type=int, default=100, help='number of grids for griddy Gibbs sampler')
model_arg.add_argument('--iterThin', type=int, default=1, help='retain posterior draw for every X draws after burn-in')
model_arg.add_argument('--eval_effect', type=str2bool, default=False, help='if evaluating effect sizes or not (slow)')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed

#### Main Codes ####
def generate_data(
        nInd, nSpc,
        paramFix=np.array([0.2, -0.5, 0.4, 0.09]),
        paramRnd_zeta=np.array([0.1, -2.0]),
        paramRnd_sigma=np.array([[1.0, -0.5],[-0.5, 1.0]]),
        rho=0.4
        ):
    # print(paramRnd_sigma)
    nFix = paramFix.shape[0]
    nRnd = paramRnd_zeta.shape[0]
    assert nRnd == paramRnd_sigma.shape[0], "mean and std shapes must be equal to each other!"

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
    # paramRnd = paramRnd_mean + (np.diag(paramRnd_std) @ np.random.randn(nRnd,nInd)).T
    chSigma = np.linalg.cholesky(paramRnd_sigma)
    paramRnd = paramRnd_zeta + (chSigma @ np.random.randn(nRnd, nInd)).T
    paramAll = np.concatenate([np.tile(paramFix, (nInd,1)), paramRnd], axis=1)
    mu = np.einsum('nsk,nk->ns', x, paramAll) #+ np.random.randn(nInd, nSpc)
    mu = np.einsum('ij,nj->ni', invA.toarray(), mu)
    prob = 1 / (1 + np.exp(-mu))
    y = np.random.binomial(1, prob)

    d_elasFix = np.zeros(shape=(nFix,nInd))
    id_elasFix = np.zeros(shape=(nFix,nInd))
    d_meFix = np.zeros(shape=(nFix,nInd))
    id_meFix = np.zeros(shape=(nFix,nInd))
    d_elasRnd = np.zeros(shape=(nRnd,nInd))
    id_elasRnd = np.zeros(shape=(nRnd,nInd))
    d_meRnd = np.zeros(shape=(nRnd,nInd))
    id_meRnd = np.zeros(shape=(nRnd,nInd))
    I = I.toarray()
    for k in range(nFix):
        for n in range(nInd):
            pn = prob[n,:]
            xnk = xFix[n,:,k]
            ak = paramFix[k]
            elas_nk = (1 - pn[:,np.newaxis]) * xnk[np.newaxis,:] * invA.toarray() * ak
            me_nk = pn[:,np.newaxis] * (1 - pn[:,np.newaxis]) * invA.toarray() * ak
            d_elasFix[k,n] = np.diag(elas_nk).mean()
            id_elasFix[k,n] = elas_nk.sum() / nSpc - d_elasFix[k,n]
            d_meFix[k,n] = np.diag(me_nk).mean()
            id_meFix[k,n] = me_nk.sum() / nSpc - d_meFix[k,n]
    for k in range(nRnd):
        for n in range(nInd):
            pn = prob[n,:]
            xnk = xRnd[n,:,k]
            bnk = paramRnd[n,k]
            elas_nk = (1 - pn[:,np.newaxis]) * xnk[np.newaxis,:] * invA.toarray() * bnk
            me_nk = pn[:,np.newaxis] * (1 - pn[:,np.newaxis]) * invA.toarray() * bnk
            d_elasRnd[k,n] = np.diag(elas_nk).mean()
            id_elasRnd[k,n] = elas_nk.sum() / nSpc - d_elasRnd[k,n]
            d_meRnd[k,n] = np.diag(me_nk).mean()
            id_meRnd[k,n] = me_nk.sum() / nSpc - d_meRnd[k,n]
    d_elasFix = d_elasFix.mean(axis=1)
    id_elasFix = id_elasFix.mean(axis=1)
    d_elasRnd = d_elasRnd.mean(axis=1)
    id_elasRnd = id_elasRnd.mean(axis=1)
    d_meFix = d_meFix.mean(axis=1)
    id_meFix = id_meFix.mean(axis=1)
    d_meRnd = d_meRnd.mean(axis=1)
    id_meRnd = id_meRnd.mean(axis=1)
    print("Elasticities")
    print(d_elasFix)
    print(id_elasFix)
    print(d_elasRnd)
    print(id_elasRnd)
    print("Marginal effects")
    print(d_meFix)
    print(id_meFix)
    print(d_meRnd)
    print(id_meRnd)
    return x, xFix, xRnd, y, spW

# %%
if __name__ == '__main__':
    config, _ = get_config()
    
    # output directory
    if config.root is not None:
        out_dir = os.path.join(config.root, "results", "synthetic", config.out_dir)
    else:
        out_dir = os.path.join("results", "synthetic", config.out_dir)
    
    try:
        os.makedirs(out_dir, exist_ok = False)
    except:
        out_dir += '_' + time.strftime("%Y%m%dT%H%M")
        os.makedirs(out_dir, exist_ok = False)

    # generate synthetic data
    nInd, nSpc = config.nInd, config.nSpc
    nFix, nRnd = len(config.paramFix), len(config.paramRnd_mean)
    xFixName = [f'alpha{str(i+1)}' for i in range(nFix)]
    xRndName = [f'beta{str(i+1)}' for i in range(nRnd)]
    rhos = config.rhos

    # %%
    res = {}
    for rho in rhos:
        print(f"Run with rho = {rho}")
        Sigma = np.ones((nRnd, nRnd), dtype=np.float64) * -0.2
        for k, v in enumerate(config.paramRnd_std):
            Sigma[k,k] = v
        x, xFix, xRnd, y, W = generate_data(nInd, nSpc, np.array(config.paramFix), 
                                            np.array(config.paramRnd_mean), Sigma, rho)

        rho_a = 1.01
        A = 1.04
        nu = 2
        splogit = spLogit(config.seed, nInd, nSpc, nFix, nRnd, x, y, W, xFixName=xFixName, xRndName=xRndName, eval_effect=config.eval_effect)

        postRes, modelFits, postParams, elasRes, meRes = splogit.estimate(nIter=config.nIter, nIterBurn=config.nIterBurn, nGrid=config.nGrid, iterThin=config.iterThin)

        dfRes = pd.DataFrame(postRes).T
        res[rho] = dfRes['mean']
        print(dfRes)

        if config.eval_effect:
            elas = pd.DataFrame(elasRes).T
            maref = pd.DataFrame(meRes).T
            elas.to_csv(f'{out_dir}/elas_{rho}.csv', index=True)
            maref.to_csv(f'{out_dir}/me_{rho}.csv', index=True)

    # %%
    # write results
    file_path = os.path.join(out_dir, "res.csv")
    
    resDf = pd.DataFrame(res)
    resDf.to_csv(file_path, index=True)

    with open(os.path.join(out_dir, "config.json"), mode="w") as f:
        json.dump(config.__dict__, f, indent=4)