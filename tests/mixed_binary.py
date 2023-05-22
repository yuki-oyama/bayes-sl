import numpy as np
from polyagamma import random_polyagamma
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
Kf = 2       # number of fixed parameters
#vars: intercept (random), dist (random), dist*male, dist*age, f1, f2

# %%
## define variables
dist_km = np.random.lognormal(size=(S,))
green  = np.random.binomial(1,0.6,size=(S,))
construction = np.random.binomial(1,0.6,size=(S,))
traffic = np.random.uniform(size=(S,))
X = np.vstack([dist_km, green, construction, traffic]).T
X = np.tile(X, (N,1,1))
Xr, Xf = X[:,:,:Kr], X[:,:,Kr:]

# %%
## define true parameters
BETA_mean = np.array([-2.0, 1.2])
BETA_std = np.array([1.1, 0.5])
ALPHA = np.array([-0.8, -0.4])

# %%
BETA = BETA_mean + (np.random.randn(N,Kr) * BETA_std)
params = np.concatenate([BETA, np.tile(ALPHA, (N,1))], axis=1)
MU = np.einsum('nsk,nk->ns', X, params) + np.random.randn(N,S)
P = 1 / (1 + np.exp(-MU))
Y = np.random.binomial(1, P)

# %%
alpha_inits = np.zeros((Kf,))
zeta_inits = np.zeros((Kr,))
Sigma_inits = 0.1 * np.eye(Kr)
A = 1.04
nu = 2
alpha_prior_mean = np.zeros((Kf,), dtype=np.float)
alpha_prior_var = sp.identity(Kf, format='csc') * 1e-1
niter = 1000
nretain = 500
thinning = 10

# %%
def spatial_logit(
    X, Y,
    Kr, Kf,
    alpha_inits = np.ones((Kf,)),
    zeta_inits = np.ones((Kr,)),
    Sigma_inits = 10 * np.eye(Kr),
    A = 1.04,
    nu = 2,
    alpha_prior_mean = np.zeros((Kf,), dtype=np.float),
    alpha_prior_var = sp.identity(Kf, format='csc') * 10**8,
    niter = 1000, nretain = 500, griddy_n = 100, thinning = 10
    ):

    # dimensions
    N, S, K = X.shape
    Xr, Xf = X[:,:,:Kr], X[:,:,Kr:]

    # number of discard in bayesian sampling
    ndiscard = niter - nretain

    # save the posterior draws here
    post_beta = np.zeros((N, Kr, nretain), dtype=np.float)
    post_zeta = np.zeros((Kr, nretain), dtype=np.float)
    post_a = np.zeros((Kr, nretain), dtype=np.float)
    post_sigma = np.zeros((Kr, Kr, nretain), dtype=np.float)
    post_alpha = np.zeros((Kf, nretain), dtype=np.float)
    post_y = np.zeros((N, S, nretain), dtype=np.float)
    post_omega = np.zeros((N, S, nretain), dtype=np.float)

    # pre-calculate some terms for faster draws
    alpha_prior_var_inv = splinalg.inv(alpha_prior_var)
    kappa = Y - 1/2 # (N,S)

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
        Xrom = Xr * np.sqrt(curr.om)[:,:,np.newaxis] # (Ainv Xr)*sqrt(omega) (N, S, K)
        Xfom = Xf * np.sqrt(curr.om)[:,:,np.newaxis] # (Ainv Xf)*sqrt(omega) (N, S, K)
        Zom = z * np.sqrt(curr.om) #sqrt(omega)*z (N,S)
        Xf2 = np.zeros((Kf,Kf))
        Xr2 = np.zeros((N,Kr,Kr))
        Xfz = np.zeros((Kf,))
        Xrz = np.zeros((N,Kr))
        for n in range(N):
            Xf2 += Xfom[n].T @ Xfom[n] #(K,S)*(S,K)->(K,K)
            Xfz += Xfom[n].T @ (Zom[n] - Xrom[n] @ curr.beta[n]) #(K,S)*(S,)->(K,)
            Xr2[n] = Xrom[n].T @ Xrom[n] #(K,S)*(S,K)->(K,K)
            Xrz[n] = Xrom[n].T @ (Zom[n] - Xfom[n] @ curr.alpha) #(K,S)*(S,)->(K,)
        post_alpha_var = np.linalg.inv(alpha_prior_var_inv + Xf2)
        post_alpha_mean = post_alpha_var @ (alpha_prior_var_inv @ alpha_prior_mean + Xfz)
        post_alpha_mean = np.array(post_alpha_mean).reshape(-1,)
        curr.alpha = np.random.multivariate_normal(mean=post_alpha_mean, cov=post_alpha_var)
        curr.invSigma = np.linalg.inv(curr.Sigma)
        curr.beta = np.zeros((N,Kr), dtype=np.float)
        for n in range(N):
            post_beta_var = np.linalg.inv(curr.invSigma + Xr2[n])
            post_beta_mean = post_beta_var @ (curr.invSigma @ curr.zeta + Xrz[n])
            curr.beta[n] = post_beta_mean + np.linalg.cholesky(post_beta_var) @ np.random.randn(Kr,)

        # draw rho using griddy Gibbs
        curr.params = np.concatenate([curr.beta, np.tile(curr.alpha, (N,1))], axis=1)
        curr.mu = np.einsum('nsk,nk->ns', X, curr.params)

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
            mu = np.exp(curr.mu)
            post_y[:,:,s] = mu / (1 + mu)
            post_omega[:,:,s] = curr.om

    return post_beta, post_zeta, post_a, post_sigma, post_alpha, post_y, post_omega

# %%
res = spatial_logit(X, Y, Kr=2, Kf=2, niter=4000, nretain=2000)
post_beta, post_zeta, post_a, post_sigma, post_alpha, post_y, post_omega = res

### calculate posterior mean of beta and sigma
alpha_mean_hat = np.mean(post_alpha[:,1:], axis=1)
zeta_mean_hat = np.mean(post_zeta[:,1:], axis=1)
sigma_mean_hat = np.mean(post_sigma[:,:,1:], axis=2)
y_mean_hat = np.mean(post_y[:,:,1:], axis=2)
# error = np.mean(Y - y_mean_hat)
ic(alpha_mean_hat)
ic(ALPHA)
ic(zeta_mean_hat)
ic(sigma_mean_hat)
ic(BETA_mean)
