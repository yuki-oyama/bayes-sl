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
from scipy.stats import invwishart

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
Kr = 2       # number of random parameters
Kf = 4       # number of fixed parameters
#vars: intercept (random), dist (random), dist*male, dist*age, f1, f2

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
Xr, Xf = X[:,:,:2], X[:,:,2:]

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
BETA_mean = np.array([0.1, -2.0])
BETA_std = np.array([0.2, 0.5])
ALPHA = np.array([0.2, -0.5, 0.4, 0.09])
Ainv = splinalg.inv(I - rho * W) # (S, S)

# %%
Y = np.zeros((N,S), dtype=np.float)
BETA = np.zeros((N,Kr), dtype=np.float)
for n in range(N):
    b = BETA_mean + np.random.randn(Kr) * BETA_std
    MU = Ainv @ (Xr[n] @ b + Xf[n] @ ALPHA + np.random.normal(size=(S,))) # (S, )
    pr = 1 / (1 + np.exp(-MU)) # (S, )
    Y[n,:] = np.random.binomial(1, pr) # (S, )
    BETA[n,:] = b

# %%
alpha_inits = np.zeros((Kf,))
zeta_inits = np.zeros((Kr,))
Sigma_inits = 0.1 * np.eye(Kr)
A = 1.04
nu = 2
alpha_prior_mean = np.zeros((Kf,), dtype=np.float)
alpha_prior_var = sp.identity(Kf, format='csc') * 1e-1
rho_a = 1.01
niter = 1000
nretain = 500
griddy_n = 100
thinning = 10

# %%
def spatial_logit(
    X, Y, W,
    Kr, Kf,
    alpha_inits = np.ones((Kf,)),
    zeta_inits = np.ones((Kr,)),
    Sigma_inits = 10 * np.eye(Kr),
    A = 1.04,
    nu = 2,
    alpha_prior_mean = np.zeros((Kf,), dtype=np.float),
    alpha_prior_var = sp.identity(Kf, format='csc') * 10**8,
    rho_a = 1.01,
    niter = 1000, nretain = 500, griddy_n = 100, thinning = 10
    ):

    # dimensions
    N, S, K = X.shape
    Xr, Xf = X[:,:,:Kr], X[:,:,Kr:]

    # number of discard in bayesian sampling
    ndiscard = niter - nretain

    # pdf of a Beta distribution: https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
    beta_prob = lambda rho, a: 1/beta(a,a) * ((1 + rho)**(a-1) * (1 - rho)**(a-1)) / (2**(2*a - 1))

    # save the posterior draws here
    post_beta = np.zeros((N, Kr, nretain), dtype=np.float)
    post_zeta = np.zeros((Kr, nretain), dtype=np.float)
    post_a = np.zeros((Kr, nretain), dtype=np.float)
    post_sigma = np.zeros((Kr, Kr, nretain), dtype=np.float)
    post_alpha = np.zeros((Kf, nretain), dtype=np.float)
    post_rho = np.zeros(nretain, dtype=np.float)
    post_y = np.zeros((N, S, nretain), dtype=np.float)
    post_omega = np.zeros((N, S, nretain), dtype=np.float)

    # pre-calculate some terms for faster draws
    alpha_prior_var_inv = splinalg.inv(alpha_prior_var)
    kappa = Y - 1/2 # (N,S)
    I = sp.identity(S, format='csc')
    AiXr = np.einsum('ij,njk->nik', Ainv.toarray(), Xr)
    AiXf = np.einsum('ij,njk->nik', Ainv.toarray(), Xf)

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
    Xf_flat = Xf.reshape(-1,Kf) #(N*S, Kf)
    Xr_flat = Xr.reshape(-1,Kr) #(N*S, Kf)
    kappa_flat = kappa.reshape(-1,) #(N*S,)
    # curr.alpha = np.linalg.inv(Xf_flat.T @ Xf_flat) @ Xf_flat.T @ kappa_flat
    # curr.beta = np.linalg.inv(Xr_flat.T @ Xr_flat) @ Xr_flat.T @ kappa_flat
    invAsq = (A**(-2)) ** np.ones((Kr,))
    curr.A = I - curr.rho * W
    curr.Ainv = splinalg.inv(curr.A).toarray()
    curr.AiX = np.einsum('ij,njk->nik', curr.Ainv, X)
    curr.AiXr, curr.AiXf = curr.AiX[:,:,:Kr], curr.AiX[:,:,Kr:]
    curr.a = np.random.gamma((nu+Kr)/2, 1/invAsq)
    curr.Sigma = invwishart.rvs(nu + Kr - 1, 2 * nu * np.diag(curr.a))
    curr.zeta = zeta_inits + np.linalg.cholesky(curr.Sigma) @ np.random.randn(Kr,) / np.sqrt(N)
    curr.mu = X @ np.concatenate([curr.zeta, alpha_inits], axis=0) # (N, S, Kr+Kf) * (Kr+Kf,) -> (N,S)
    curr.alpha = alpha_inits
    curr.beta = np.tile(curr.zeta, (N,1))

    ### Gibbs sampling
    print("Gibbs sampling...")
    for iter in tqdm(range(niter)):
        # draw omega
        curr.om = random_polyagamma(z=curr.mu, size=(N,S))

        # draw alpha and beta
        z = kappa / curr.om #(N,S)
        AiXrom = curr.AiXr * np.sqrt(curr.om)[:,:,np.newaxis] # (Ainv Xr)*sqrt(omega) (N, S, K)
        AiXfom = curr.AiXf * np.sqrt(curr.om)[:,:,np.newaxis] # (Ainv Xr)*sqrt(omega) (N, S, K)
        Zom = z * np.sqrt(curr.om) #sqrt(omega)*z (N,S)
        AiXf2 = np.zeros((Kf,Kf))
        AiXr2 = np.zeros((N,Kr,Kr))
        AiXfz = np.zeros((Kf,))
        AiXrz = np.zeros((N,Kr))
        for n in range(N):
            AiXf2 += AiXfom[n].T @ AiXfom[n] #(K,S)*(S,K)->(K,K)
            AiXfz += AiXfom[n].T @ (Zom[n] - AiXrom[n] @ curr.beta[n]) #(K,S)*(S,)->(K,)
            AiXr2[n] = AiXrom[n].T @ AiXrom[n] #(K,S)*(S,K)->(K,K)
            AiXrz[n] = AiXrom[n].T @ (Zom[n] - AiXfom[n] @ curr.alpha) #(K,S)*(S,)->(K,)
        post_alpha_var = np.linalg.inv(alpha_prior_var_inv + AiXf2)
        post_alpha_mean = post_alpha_var @ (alpha_prior_var_inv @ alpha_prior_mean + AiXfz)
        post_alpha_mean = np.array(post_alpha_mean).reshape(-1,)
        curr.alpha = np.random.multivariate_normal(mean=post_alpha_mean, cov=post_alpha_var)
        curr.invSigma = np.linalg.inv(curr.Sigma)
        curr.beta = np.zeros((N,Kr), dtype=np.float)
        for n in range(N):
            post_beta_var = np.linalg.inv(curr.invSigma + AiXr2[n])
            post_beta_mean = post_beta_var @ (curr.invSigma @ curr.zeta + AiXrz[n])
            curr.beta[n] = post_beta_mean + np.linalg.cholesky(post_beta_var) @ np.random.randn(Kr,)

        # draw rho using griddy Gibbs
        curr.params = np.concatenate([curr.beta, np.tile(curr.alpha, (N,1))], axis=1)
        mus = np.einsum('rnsk,nk->rns', YAiXs, curr.params) # \sum_n \sum_i y_ni * u_ni
        mus = np.sum(mus, axis=(1,2)) # (griddy_n,)
        summu = np.einsum('nskr,nk->rns', AiXs, curr.params) # (griddy_n, N, S)
        summu = np.log(1 + np.exp(summu)).sum(axis=(1,2)) # (griddy_n,)
        ll = mus - summu + np.log(beta_prob(rrhos, rho_a)) # log-odds (or log-likelihood) of joint prob: y*log(p(y=1)) + log(p(rho))
        den = ll - np.max(ll) # normalization
        x = np.exp(den) # p(y|rho,beta) * p(rho): joint probability
        # denominator: int_rho p(y,rho,beta) drho
        # approximate integral by piecewise linear, thus trapezoid: (rho_{i+1} - rho_i) * (p_i + p_{i+1}) / 2
        isum = np.sum(
            (rrhos[1:] - rrhos[:-1]) * (x[1:] + x[:-1]) / 2
        )
        z = np.abs(x/isum) # bayes theorem -> posterior probability of rho p(rho|y,beta)
        den = np.cumsum(z)
        rnd = np.random.uniform() * np.sum(z)
        idx = np.max(np.append([0], np.where(den <= rnd)))
        curr.rho = rrhos[idx]
        curr.A = I - curr.rho * W
        curr.AiX = AiXs[:,:,:,idx]
        curr.Ainv = splinalg.inv(curr.A).toarray()
        curr.AiXr, curr.AiXf = curr.AiX[:,:,:Kr], curr.AiX[:,:,Kr:]
        curr.mu = np.einsum('nsk,nk->ns', curr.AiX, curr.params)

        # draw other params
        curr.a = np.random.gamma((nu+Kr)/2, 1/invAsq + nu * np.diag(curr.invSigma))
        betaS = curr.beta - curr.zeta
        curr.Sigma = invwishart.rvs(nu + Kr - 1, 2 * nu * np.diag(curr.a) + betaS.T @ betaS)
        curr.zeta = np.mean(curr.beta, axis=0) + np.linalg.cholesky(curr.Sigma) @ np.random.randn(Kr,) / np.sqrt(N)

        if iter > ndiscard:
            s = iter - ndiscard
            post_alpha[:,s] = curr.alpha
            post_zeta[:,s] = curr.zeta
            post_beta[:,:,s] = curr.beta
            post_sigma[:,:,s] = curr.Sigma
            post_a[:,s] = curr.a
            post_rho[s] = curr.rho
            mu = np.exp(curr.mu)
            post_y[:,:,s] = mu / (1 + mu)
            post_omega[:,:,s] = curr.om

    return post_beta, post_zeta, post_a, post_sigma, post_alpha, post_rho, post_y, post_omega

# %%
res = spatial_logit(X, Y, W, Kr=2, Kf=4, niter=2000, nretain=1000)
post_beta, post_zeta, post_a, post_sigma, post_alpha, post_rho, post_y, post_omega = res

### calculate posterior mean of beta and sigma
alpha_mean_hat = np.mean(post_alpha, axis=1)
zeta_mean_hat = np.mean(post_zeta, axis=1)
y_mean_hat = np.mean(post_y, axis=2)
rho_post_mean = np.mean(post_rho)
# error = np.mean(Y - y_mean_hat)
ic(alpha_mean_hat)
ic(ALPHA)
ic(zeta_mean_hat)
ic(BETA_mean)
ic(rho_post_mean)
ic(rho)

# %%
sigma_mean_hat = np.mean(post_sigma, axis=2)
sigma_mean_hat
