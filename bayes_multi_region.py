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

class SpatialLogit(object):

    def __init__(self, areas, dataset, betas, incidence_key='angle', dist_key='nearest', key='incidence'):
        """
        Arguments:
            areas: list of area names for keys of dataset
            dataset: user, street, choice, incidence data for each area
            betas: list of (name, init_val, lower, upper, to_estimate, variable, interaction)
            incidence_key: how to define the network weight matrix
        """

        self.areas = areas
        self.dataset = dataset

        # prepare variables from betas
        self.init_beta = []
        self.freebetaNames = []
        self.bounds = []
        self.x_st = {area: [] for area in areas}
        self.x_user = {area: [] for area in areas}
        self.x = {area: [] for area in areas}
        self.y = {}
        self.W = {}

        for name, init_val, lower, upper, to_estimate, variable, interaction in betas:
            if to_estimate == 0:
                self.init_beta.append(init_val)
                self.freebetaNames.append(name)
                self.bounds.append((lower, upper))
                for area in areas:
                    street_df, user_df = dataset[area]['street'], dataset[area]['user']
                    xs = np.ones(len(street_df), dtype=np.float) if variable is None else street_df[variable].values
                    xu = np.ones(len(user_df), dtype=np.float) if interaction is None else user_df[interaction].values
                    self.x_st[area].append(xs)
                    self.x_user[area].append(xu)
                    self.x[area].append(xs[np.newaxis,:] * xu[:,np.newaxis]) # N x S

        for area in areas:
            if key == 'incidence':
                choice_df, incidence_df = dataset[area]['choice'], dataset[area]['incidence']
                self.x[area] = np.array(self.x[area]).transpose(2,1,0) # S x N x K
                self.y[area] = np.array(choice_df).T # S x N
                W = get_incidence_matrix(self.x[area].shape[0], incidence_df, key=incidence_key)
            elif key == 'dist_mtrx':
                choice_df, dist_df = dataset[area]['choice'], dataset[area]['distance']
                self.x[area] = np.array(self.x[area]).transpose(2,1,0) # S x N x K
                self.y[area] = np.array(choice_df).T # S x N
                dist_mtrx = np.zeros(shape=dist_df.shape)
                for col in dist_df.columns: dist_mtrx[:,int(col)] = dist_df[col].values
                W = get_spatial_weight_matrix(dist_mtrx, key=dist_key)
            self.W[area] = W # S x S

        # convert to numpy arrays, and obtain incidence matrix
        self.init_beta = np.array(self.init_beta, dtype=np.float)
        self.beta_prior_mean = np.zeros((len(self.init_beta),), dtype=np.float)
        self.beta_prior_var = sp.identity(len(self.init_beta), format='csc') * 10**8
        self.rho_a = 1.01

    def two_by_three(self, M, X):
        MX = np.apply_along_axis(
            lambda x: M @ x, 0, X
        )
        return MX

    def estimate(self, niter=1000, nretain=500, griddy_n=100):
        nn = np.arange(len(self.areas))
        K = len(self.init_beta)

        beta_prior_mean = self.beta_prior_mean
        beta_prior_var = self.beta_prior_var
        rho_a = self.rho_a

        dims = [(*X.shape[:2],) for X in self.x.values()]
        I = [sp.identity(S, format='csc') for S, N in dims]
        ndiscard = niter - nretain

        # pdf of a Beta distribution: https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
        beta_prob = lambda rho, a: 1/beta(a,a) * ((1 + rho)**(a-1) * (1 - rho)**(a-1)) / (2**(2*a - 1))

        # save the posterior draws here
        postb = np.zeros((K, nretain), dtype=np.float)
        postr = np.zeros(nretain, dtype=np.float)
        posty = [np.zeros((S, N, nretain), dtype=np.float) for S, N in dims]
        postom = [np.zeros((S, N, nretain), dtype=np.float) for S, N in dims]

        # pre-calculate some terms for faster draws
        beta_prior_var_inv = splinalg.inv(beta_prior_var)
        kappa = [Y - 1/2 for Y in self.y.values()] # (S,N)

        # set-up for griddy gibbs
        Ais = [np.zeros((S, S, griddy_n), dtype=np.float) for S, N in dims]
        AiXs = [np.zeros((S, N, K, griddy_n), dtype=np.float) for S, N in dims]
        YAiXs = [np.zeros((griddy_n, S, N, K), dtype=np.float) for S, N in dims]
        rrhos = np.linspace(-1, 1, griddy_n + 2)
        rrhos = rrhos[1:-1]

        print("Pre-calculate griddy Gibbs...")
        for ii in tqdm(range(griddy_n)):
            for n, idt, W, X, Y in zip(nn, I, self.W.values(), self.x.values(), self.y.values()):
                tempA = idt - rrhos[ii] * W
                Ai = splinalg.inv(tempA)
                Ais[n][:,:,ii] = Ai.toarray()
                # AiXs[:,:,:,ii] = np.einsum('ij,jkl->ikl', Ai.toarray(), X)
                AiXs[n][:,:,:,ii] = self.two_by_three(Ai, X)
                YAiXs[n][ii,:,:,:] = Y[:,:,np.newaxis] * AiXs[n][:,:,:,ii]

        # starting values (won't matter after sufficient draws)
        curr = lambda: None
        curr.rho = 0
        # start from OLS
        Xflat = [X.reshape(-1,K) for X in self.x.values()]
        kappa_flat = [k.reshape(-1,) for k in kappa]
        # curr.beta = np.linalg.inv(Xflat.T @ Xflat) @ Xflat.T @ kappa_flat
        curr.beta = np.zeros_like(self.init_beta)
        curr.A = [idt - curr.rho * W for idt, W in zip(I, self.W.values())]
        curr.AiX = [
            np.einsum('ij,jkl->ikl', splinalg.inv(A).toarray(), X)
            for A, X in zip(curr.A, self.x.values())
        ]
        curr.mu = [AiX @ curr.beta for AiX in curr.AiX]
        curr.xb = [X @ curr.beta for X in self.x.values()]

        ### Gibbs sampling
        print("Gibbs sampling...")
        for iter in tqdm(range(niter)):
            # sample omega
            curr.om = [random_polyagamma(z=mu, size=(S,N)) for mu, (S,N) in zip(curr.mu, dims)]
            yy = [k / om for k, om in zip(kappa, curr.om)]
            curr.Ay = [A @ y for A, y in zip(curr.A, yy)]

            # draw beta
            tx = [AiX * np.sqrt(om)[:,:,np.newaxis] for AiX, om in zip(curr.AiX, curr.om)]
            ty = [y * np.sqrt(om) for y, om in zip(yy, curr.om)]
            tx_flat = [e.reshape(-1,K) for e in tx]
            ty_flat = [e.reshape(-1,) for e in ty]
            tx = np.concatenate(tx_flat) # (N1S1 + N2S2, K)
            ty = np.concatenate(ty_flat) # (N1S1 + N2S2,)
            V = np.linalg.inv(beta_prior_var_inv + tx.T @ tx)
            b = V @ (beta_prior_var_inv @ beta_prior_mean + tx.T @ ty)
            b = np.array(b).reshape(-1,)
            curr.beta = np.random.multivariate_normal(mean=b, cov=V)
            curr.xb = [X @ curr.beta for X in self.x.values()]
            curr.mu = [AiX @ curr.beta for AiX in curr.AiX]

            # draw rho using griddy Gibbs
            mus = [(YAiX @ curr.beta).sum(axis=(1,2)) for YAiX in YAiXs]
            mus = np.sum(mus, axis=0)
            summu = [AiX.transpose(3,0,1,2) @ curr.beta for AiX in AiXs]
            summu = [np.log(1 + np.exp(s)).sum(axis=(1,2)) for s in summu]
            summu = np.sum(summu, axis=0)
            # summu = np.zeros(griddy_n, dtype=np.float)
            # for i in range(griddy_n):
            #     mui = AiXs[:,:,:,i] @ curr.beta
            #     summu[i] = np.sum(np.log(1 + np.exp(mui)))

            ll = mus - summu + np.log(beta_prob(rrhos, rho_a))
            den = ll
            y = rrhos
            den = den - np.max(den)
            x = np.exp(den)
            isum = np.sum(
                (y[1:] + y[:-1]) * (x[1:] + x[:-1]) / 2
            )
            z = np.abs(x/isum)
            den = np.cumsum(z)
            rnd = np.random.uniform() * np.sum(z)
            idx = np.max(np.append([0], np.where(den <= rnd)))
            curr.rho = rrhos[idx]
            curr.A = [idt - curr.rho * W for idt, W in zip(I, self.W.values())]
            curr.AiX = [AiX[:,:,:,idx] for AiX in AiXs]
            curr.mu = [AiX @ curr.beta for AiX in curr.AiX]

            if iter > ndiscard:
                s = iter - ndiscard
                postb[:,s] = curr.beta
                postr[s] = curr.rho
                for n in nn:
                    mu = np.exp(curr.mu[n])
                    posty[n][:,:,s] = mu / (1 + mu)
                    postom[n][:,:,s] = curr.om[n]

        return postb, postr, posty, postom

if __name__ == '__main__':
    # Read data
    areas = ['kiba'] #, 'kiba'
    dataset = {}
    for area in areas:
        user_df = pd.read_csv(f'dataset/users_{area}.csv').set_index('user')
        street_df = pd.read_csv(f'dataset/streets_{area}.csv').set_index('street_id')
        incidence_df = pd.read_csv(f'dataset/incidence_angle_{area}.csv')
        choice_df = pd.read_csv(f'dataset/choice_{area}.csv', index_col=0)
        dist_df = pd.read_csv(f'dataset/distance_matrix_{area}.csv', index_col=0)
        dataset[area] = {
            'user': user_df,
            'street': street_df,
            'incidence': incidence_df,
            'choice': choice_df,
            'distance': dist_df,
        }

    # %%
    # parameters to be estimated
    betas = [
        ('intercept', 0, None, None, 0, None, None),
        ('c_tree', 0, None, None, 0, 'tree', None),
        ('c_bldg', 0, None, None, 0, 'building', None),
        ('c_road', 0, None, None, 0, 'road', None),
        ('c_sky', 0, None, None, 0, 'sky', None),
        ('b_river', 0, None, None, 0, 'inner_river_0', None),
        ('b_avenue', 0, None, None, 0, 'inner_avenue_0', 'resident'),
        ('b_area', 0, None, None, 0, 'inner_area', None),
        ('b_dist_base', 0, None, None, 0, 'distance_km', None),
        ('a_10under', 0, None, None, 0, 'distance_km', 'Under_10_years'),
        ('a_1under', 0, None, None, 0, 'distance_km', 'Under_1_day'),
        ('a_female', 0, None, None, 0, 'distance_km', 'female'),
        ('a_40up', 0, None, None, 0, 'distance_km', 'Upper_40'),
        ('g_east', 0, None, None, 0, 'dist_east', None),
        ('g_south', 0, None, None, 0, 'dist_south', None),
        ('g_north', 0, None, None, 0, 'dist_north', None),
    ]

    # %%
    spl = SpatialLogit(areas, dataset, betas, incidence_key='adjacency', dist_key='within', key='dist_mtrx')
    # res = spl.estimate(niter=20000, nretain=5000)
    res = spl.estimate(niter=1000, nretain=500, griddy_n=500)
    postb, postr, posty, postom = res

    summary = ''
    print('name \t mean \t median \t std. \t beta/std.')
    summary += 'name \t mean \t median \t std. \t beta/std. \n'
    rho_post_mean = np.mean(postr)
    rho_post_median = np.median(postr)
    rho_post_std = np.std(postr)
    print(f'rho \t {rho_post_mean} \t {rho_post_median} \t {rho_post_std}  \t {rho_post_mean/rho_post_std}')
    summary += f'rho \t {rho_post_mean} \t {rho_post_median} \t {rho_post_std}  \t {rho_post_mean/rho_post_std} \n'

    beta_mean_hat = np.mean(postb, axis=1)
    beta_median_hat = np.median(postb, axis=1)
    beta_std_hat = np.std(postb, axis=1)

    # %%
    for b, m, s, name in zip(beta_mean_hat, beta_median_hat, beta_std_hat, spl.freebetaNames):
        print(f'{name} \t {b} \t {m} \t {s} \t {b/s}')
        summary += f'{name} \t {b} \t {m} \t {s} \t {b/s} \n'

    # with open('model/sar_logit/results/bayes_20000/summary.csv', 'w') as f:
    #     f.write(summary)
    #
    # names = ['posterior_beta', 'posterior_rho', 'posterior_y', 'posterior_omega']
    # for name, post in zip(names, res):
    #     if type(post) == list:
    #         for area, p in zip(areas, post):
    #             np.save(f'model/sar_logit/results/bayes_20000/{name}_{area}.npy', p)
    #             # df = pd.DataFrame(p)
    #             # df.to_csv(f'model/sar_logit/results/bayes_20000/{name}_{area}.csv', index=False)
    #     else:
    #         np.save(f'model/sar_logit/results/bayes_20000/{name}.npy', post)
    #         # df = pd.DataFrame(post)
    #         # df.to_csv(f'model/sar_logit/results/bayes_20000/{name}.csv', index=False)
