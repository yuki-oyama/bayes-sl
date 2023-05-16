import numpy as np
from polyagamma import random_polyagamma
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import linalg as splinalg
from sklearn.preprocessing import scale
from scipy.special import beta
from tqdm import tqdm
from icecream import ic

# %%
np.random.seed(111)


# %%
def get_stats(a, axis=None):
    print('mean:', np.mean(a, axis=axis))
    print('min:', np.min(a, axis=axis))
    print('max:', np.max(a, axis=axis))
    print('std:', np.std(a, axis=axis))

# %%
## generate synthetic data
N = 50      # number of individuals
S = 200     # number of spatial units
K = 6       # number of parameters
#vars: intercept, dist, dist*male, dist*age, f1, f2

# %%
## define variables
dist_km = np.random.lognormal(size=(S,))
st_features = np.random.normal(size=(S,2))
st_features -= np.mean(st_features)
st_features /= np.std(st_features)
gender  = np.random.binomial(1,0.6,size=(N,))
elderly  = np.random.binomial(1,0.2,size=(N,))
intercept = np.random.uniform(size=(N,S,1))
dist_tile = np.tile(dist_km, (N,1))
dist_tile = np.tile(dist_tile, (3,1,1)).reshape(N,S,3)
person_features = np.vstack([np.ones(N), gender, elderly]).T
interactions = person_features[:,np.newaxis,:] * dist_tile # (N, S, 3)
pst_features = np.concatenate([intercept, interactions], axis=2) # (N, S, 4)

## features
pst_features.shape
st_features.shape

## Xs
X = np.concatenate([pst_features, np.tile(st_features, (N,1,1))], axis=2)

# %%
## define spatial weight matrix
xy = np.random.uniform(size=(S,2))
nbrs = NearestNeighbors(n_neighbors=4).fit(xy)
W = nbrs.kneighbors_graph(xy)
I = sp.identity(S, format='csc')
W -= I
W /= W.sum(axis=1)
W = sp.csc_matrix(W)

# %%
## define true parameters
rho = 0.4
BETA = np.array([0.1, -2.0, 0.2, -0.5, 0.4, 0.09])
Ainv = splinalg.inv(I - rho * W) # (S, S)

# %%
Ys = []
for n in range(N):
    MU = Ainv @ (X[n] @ BETA + np.random.normal(size=(S,))) # (S, )
    pr = 1 / (1 + np.exp(-MU)) # (S, )
    Y = np.random.binomial(1, pr) # (S, )
    Ys.append(Y)
Y = np.vstack(Ys)

# %%
beta_prior_mean = np.zeros((K,), dtype=np.float)
beta_prior_var = sp.identity(K, format='csc') * 10**8
rho_a = 1.01
niter = 1000
nretain = 500
griddy_n = 100
thinning = 10

# %%
def spatial_logit(
    X, Y, W,
    beta_prior_mean = np.zeros((X.shape[-1],), dtype=np.float),
    beta_prior_var = sp.identity(X.shape[-1], format='csc') * 10**8,
    rho_a = 1.01,
    niter = 1000, nretain = 500, griddy_n = 100, thinning = 10
    ):

    # dimensions
    N, S, K = X.shape

    # number of discard in bayesian sampling
    ndiscard = niter - nretain

    # pdf of a Beta distribution: https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
    beta_prob = lambda rho, a: 1/beta(a,a) * ((1 + rho)**(a-1) * (1 - rho)**(a-1)) / (2**(2*a - 1))

    # save the posterior draws here
    postb = np.zeros((K, nretain), dtype=np.float)
    postr = np.zeros(nretain, dtype=np.float)
    posty = np.zeros((N, S, nretain), dtype=np.float)
    postom = np.zeros((N, S, nretain), dtype=np.float)

    # pre-calculate some terms for faster draws
    beta_prior_var_inv = splinalg.inv(beta_prior_var)
    kappa = Y - 1/2 # (N,S)
    I = sp.identity(S, format='csc')
    AiX = np.einsum('ij,njk->nik', Ainv.toarray(), X)

    # set-up for griddy gibbs
    Ais = np.zeros((S, S, griddy_n), dtype=np.float)
    AiXs = np.zeros((N, S, K, griddy_n), dtype=np.float)
    YAiXs = np.zeros((griddy_n, N, S, K), dtype=np.float)
    rrhos = np.linspace(-1, 1, griddy_n + 2)
    rrhos = rrhos[1:-1]

    print("Pre-calculate griddy Gibbs...")
    for ii in tqdm(range(griddy_n)):
        tempA = I - rrhos[ii] * W
        Ai = splinalg.inv(tempA)
        Ais[:,:,ii] = Ai.toarray()
        AiXs[:,:,:,ii] = np.einsum('ij,njk->nik', Ai.toarray(), X)
        YAiXs[ii,:,:,:] = Y[:,:,np.newaxis] * AiXs[:,:,:,ii]

    # starting values (won't matter after sufficient draws)
    curr = lambda: None
    curr.rho = 0
    # starting from OLS estimates
    Xflat = X.reshape(-1,K) #(N*S, K)
    kappa_flat = kappa.reshape(-1,) #(N*S,)
    curr.beta = np.linalg.inv(Xflat.T @ Xflat) @ Xflat.T @ kappa_flat
    curr.A = I - curr.rho * W
    curr.AiX = np.einsum('ij,njk->nik', splinalg.inv(curr.A).toarray(), X)
    curr.mu = curr.AiX @ curr.beta
    curr.xb = X @ curr.beta

    ### Gibbs sampling
    print("Gibbs sampling...")
    for iter in tqdm(range(niter)):
        # draw omega
        curr.om = random_polyagamma(z=curr.mu, size=(N,S))

        # draw beta
        z = kappa / curr.om #(N,S)
        # curr.Az = curr.A @ z
        AiXom = curr.AiX * np.sqrt(curr.om)[:,:,np.newaxis] # (Ainv X)*sqrt(omega) (N, S, K)
        Zom = z * np.sqrt(curr.om) #sqrt(omega)*z (N,S)
        AiX2 = np.zeros((K,K))
        AiXz = np.zeros((K,))
        for n in range(N):
            AiX2 += AiXom[n].T @ AiXom[n] #(K,S)*(S,K)->(K,K)
            AiXz += AiXom[n].T @ Zom[n] #(K,S)*(S,)->(K,)
        post_var = np.linalg.inv(beta_prior_var_inv + AiX2)
        post_mean = post_var @ (beta_prior_var_inv @ beta_prior_mean + AiXz)
        post_mean = np.array(post_mean).reshape(-1,)
        curr.beta = np.random.multivariate_normal(mean=post_mean, cov=post_var)
        curr.xb = X @ curr.beta
        curr.mu = curr.AiX @ curr.beta

        # draw rho using griddy Gibbs
        mus = YAiXs @ curr.beta
        mus = np.sum(mus, axis=(1,2))
        summu = AiXs.transpose(3,0,1,2) @ curr.beta # (griddy_n, N, S)
        summu = np.log(1 + np.exp(summu)).sum(axis=(1,2)) # (griddy_n,)

        ll = mus - summu + np.log(beta_prob(rrhos, rho_a)) # log-odds (or log-likelihood?): y*log(p(y=1)) + log(p(rho))
        den = ll - np.max(ll) # normalization
        x = np.exp(den) # p(y=1)^y * p(rho)
        isum = np.sum(
            (rrhos[1:] - rrhos[:-1]) * (x[1:] + x[:-1]) / 2
        ) # approximate integral by piecewise linear, thus trapezoid: (rho_{i+1} - rho_i) * (p_i + p_{i+1}) / 2
        z = np.abs(x/isum) # not necessary (maybe)
        den = np.cumsum(z)
        rnd = np.random.uniform() * np.sum(z)
        idx = np.max(np.append([0], np.where(den <= rnd)))
        curr.rho = rrhos[idx]
        curr.A = I - curr.rho * W
        curr.AiX = AiXs[:,:,:,idx]
        curr.mu = curr.AiX @ curr.beta

        if iter > ndiscard:
            s = iter - ndiscard
            postb[:,s] = curr.beta
            postr[s] = curr.rho
            mu = np.exp(curr.mu)
            posty[:,:,s] = mu / (1 + mu)
            postom[:,:,s] = curr.om

    return postb, postr, posty, postom

# %%
res = spatial_logit(X, Y, W, niter=1000, nretain=500)
postb, postr, posty, postom = res

### calculate posterior mean of beta and sigma
beta_mean_hat = np.mean(res[0], axis=1)
y_mean_hat = np.mean(res[2], axis=2)
rho_post_mean = np.mean(res[1])
# error = np.mean(Y - y_mean_hat)
ic(beta_mean_hat)
ic(BETA)
# ic(error)
ic(rho_post_mean)
ic(rho)
