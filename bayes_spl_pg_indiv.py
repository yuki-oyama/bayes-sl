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
## generate synthetic data
N = 50      # number of individuals
S = 200     # number of spatial units
K = 3       # number of parameters

# %%
## define variables
intercept = np.random.uniform(size=(S,N,1))
X_n       = np.ones((N,K-1))
X_n[:,0]  = np.random.binomial(1,0.6,size=(N,))
X_n[:,1]  = np.random.binomial(1,0.2,size=(N,))
X_s       = np.random.normal(size=(S,))
X_s -= np.mean(X_s)
X_s /= np.std(X_s)
X_s = np.vstack([X_s, X_s]).T  # for interactions
X = X_s[:,np.newaxis,:] * X_n[np.newaxis,:,:] # (S,N,K-1)
X = np.concatenate([intercept, X], axis=2) # (S,N,K)

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
rho = 0.2
BETA = np.array([0.1, -2.0, -1.2])
AI = splinalg.inv(I - rho * W) # (S, S)

# %%
MU = AI @ (X @ BETA + np.random.normal(size=(S,N))) # (S, N)
pr = 1 / (1 + np.exp(-MU)) # (S, N)
Y = np.random.binomial(1, pr) # (S, N)

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
    S, N, K = X.shape

    # number of discard in bayesian sampling
    ndiscard = niter - nretain

    # pdf of a Beta distribution: https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
    beta_prob = lambda rho, a: 1/beta(a,a) * ((1 + rho)**(a-1) * (1 - rho)**(a-1)) / (2**(2*a - 1))

    # save the posterior draws here
    postb = np.zeros((K, nretain), dtype=np.float)
    postr = np.zeros(nretain, dtype=np.float)
    posty = np.zeros((S, N, nretain), dtype=np.float)
    postom = np.zeros((S, N, nretain), dtype=np.float)

    # pre-calculate some terms for faster draws
    beta_prior_var_inv = splinalg.inv(beta_prior_var)
    kappa = Y - 1/2 # (S,N)
    I = sp.identity(S, format='csc')
    AiX = np.einsum('ij,jkl->ikl', AI.toarray(), X)

    # set-up for griddy gibbs
    Ais = np.zeros((S, S, griddy_n), dtype=np.float)
    AiXs = np.zeros((S, N, K, griddy_n), dtype=np.float)
    YAiXs = np.zeros((griddy_n, S, N, K), dtype=np.float)
    rrhos = np.linspace(-1, 1, griddy_n + 2)
    rrhos = rrhos[1:-1]

    ### For fast calculation in griddy Gibbs
    print("Pre-calculate griddy Gibbs...")
    for ii in tqdm(range(griddy_n)):
        tempA = I - rrhos[ii] * W
        Ai = splinalg.inv(tempA)
        Ais[:,:,ii] = Ai.toarray()
        AiXs[:,:,:,ii] = np.einsum('ij,jkl->ikl', Ai.toarray(), X)
        YAiXs[ii,:,:,:] = Y[:,:,np.newaxis] * AiXs[:,:,:,ii]

    # starting values (won't matter after sufficient draws)
    curr = lambda: None
    curr.rho = 0
    # starting from OLS estimates
    Xflat = X.reshape(-1,K)
    kappa_flat = kappa.reshape(-1,)
    curr.beta = np.linalg.inv(Xflat.T @ Xflat) @ Xflat.T @ kappa_flat
    curr.A = I - curr.rho * W
    curr.AiX = np.einsum('ij,jkl->ikl', splinalg.inv(curr.A).toarray(), X)
    curr.mu = curr.AiX @ curr.beta
    curr.xb = X @ curr.beta

    ### Gibbs sampling
    print("Gibbs sampling...")
    for iter in tqdm(range(niter)):
        # draw omega
        curr.om = random_polyagamma(z=curr.mu, size=(S,N))
        z = kappa / curr.om
        curr.Az = curr.A @ z

        # draw beta
        tx = curr.AiX * np.sqrt(curr.om)[:,:,np.newaxis]
        tz = z * np.sqrt(curr.om)
        tx = tx.reshape(-1,K)
        tz = tz.reshape(-1,)
        V = np.linalg.inv(beta_prior_var_inv + tx.T @ tx)
        b = V @ (beta_prior_var_inv @ beta_prior_mean + tx.T @ tz)
        b = np.array(b).reshape(-1,)
        curr.beta = np.random.multivariate_normal(mean=b, cov=V)
        curr.xb = X @ curr.beta
        curr.mu = curr.AiX @ curr.beta

        # draw rho using griddy Gibbs
        mus = YAiXs @ curr.beta
        mus = np.sum(mus, axis=(1,2))
        summu = AiXs.transpose(3,0,1,2) @ curr.beta
        summu = np.log(1 + np.exp(summu)).sum(axis=(1,2))

        ll = mus - summu + np.log(beta_prob(rrhos, rho_a))
        den = ll - np.max(ll) # normalization
        x = np.exp(den) # p(y=1)^y * p(rho)
        isum = np.sum(
            (rrhos[1:] - rrhos[:-1]) * (x[1:] + x[:-1]) / 2
        ) # approximate integral by piecewise linear
        z = np.abs(x/isum) # not necessary
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

### calculate posterior mean of beta and rho
beta_mean_hat = np.mean(postb, axis=1)
prob_mean_hat = np.mean(posty, axis=2)
rho_post_mean = np.mean(postr)
ic(beta_mean_hat)
ic(rho_post_mean)

# accuracy in terms (take MAP; just for a naive index)
y_mean_hat = (prob_mean_hat > 0.5) * 1
error = np.sum(np.abs(Y - y_mean_hat))/np.sum(np.ones_like(Y))
ic(error)
