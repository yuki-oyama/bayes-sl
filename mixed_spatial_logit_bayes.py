import numpy as np
from polyagamma import random_polyagamma
from scipy.stats import invwishart
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import linalg as splinalg
from scipy.special import beta
from tqdm import tqdm
from icecream import ic

# Samplers
def next_omega(paramAll, AiX, nInd, nSpc):
    mu = np.einsum('nsk,nk->ns', AiX, paramAll)
    omega = random_polyagamma(z=mu, size=(nInd, nSpc))
    return omega

def next_paramFixRnd(
        paramFix, paramRnd,
        xFix, nFix,
        xRnd, nRnd,
        zeta, invSigma,
        omega, kappa,
        priMuFix, priInvVarFix,
        AiX,
        nInd, nSpc):

    # preliminary
    z = kappa / omega # kappa = Y - 1/2, size N x S
    AiX_fix, AiX_rnd = AiX[:,:,:nFix], AiX[:,:,nFix:]
    AiXO_fix = AiX_fix * np.sqrt(omega)[:,:,np.newaxis]
    AiXO_rnd = AiX_rnd * np.sqrt(omega)[:,:,np.newaxis]
    zO = z * np.sqrt(omega)

    # draw alpha and beta
    varFix = np.zeros((nFix,nFix), dtype=np.float64)
    varRnd = np.zeros((nInd,nRnd,nRnd), dtype=np.float64)
    muFix = np.zeros((nFix,), dtype=np.float64)
    muRnd = np.zeros((nInd,nRnd))
    for n in range(nInd):
        varFix += AiXO_fix[n].T @ AiXO_fix[n] # K x S @ S x K -> K x K
        muFix += AiXO_fix[n].T @ (zO[n] - AiXO_rnd[n] @ paramRnd[n]) # K x S @ S -> K
        varRnd[n] = AiXO_rnd[n].T @ AiXO_rnd[n] # K x S @ S x K -> K x K
        muRnd[n] = AiXO_rnd[n].T @ (zO[n] - AiXO_fix[n] @ paramFix) # K x S @ S -> K

    # paramFix
    postVarFix = np.linalg.inv(priInvVarFix + varFix)
    postMuFix = postVarFix @ (priInvVarFix @ priMuFix + muFix)
    postMuFix = np.array(postMuFix).reshape(-1,)
    paramFix = np.random.multivariate_normal(mean=postMuFix, cov=postVarFix)

    # paramRnd
    paramRnd = np.zeros((nInd,nRnd), dtype=np.float64)
    for n in range(nInd):
        postVarRnd = np.linalg.inv(invSigma + varRnd[n])
        postMuRnd = postVarRnd @ (invSigma @ zeta + muRnd[n])
        paramRnd[n] = postMuRnd + np.linalg.cholesky(postVarRnd) @ np.random.randn(nRnd,)

    paramAll = np.concatenate([np.tile(paramFix, (nInd,1)), paramRnd], axis=1)
    return paramFix, paramRnd, paramAll

def next_iwDianA(invSigma, nu, invAsq, nRnd):
    iwDiagA = np.random.gamma((nu + nRnd) / 2, 1 / invAsq + nu * np.diag(invSigma))
    return iwDiagA

def next_Sigma(paramRnd, zeta, nu, iwDiagA, nRnd):
    betaS = paramRnd - zeta
    Sigma = invwishart.rvs(nu + nRnd - 1, 2 * nu * np.diag(iwDiagA) + betaS.T @ betaS)
    invSigma = np.linalg.inv(Sigma)
    return Sigma, invSigma

def next_zeta(paramRnd, Sigma, nRnd, nInd):
    zeta = paramRnd.mean(axis=0) + np.linalg.cholesky(Sigma) @ np.random.randn(nRnd,) / np.sqrt(nInd)
    return zeta

def next_rho(paramAll,
            rhos, rho_a,
            yAiXs, AiXs,
            I, spW,
            nFix, nRnd, nGrid,
            beta_prob = lambda rho, a: 1/beta(a,a) * ((1 + rho)**(a-1) * (1 - rho)**(a-1)) / (2**(2*a - 1))
            ):
    # draw rho by griddy Gibbs sampler
    yMu = np.einsum('rnsk,nk->rns', yAiXs, paramAll) # y_ni * u_ni
    Mu = np.einsum('rnsk,nk->rns', AiXs, paramAll) # u_ni
    LL = yMu.sum(axis=(1,2)) - np.log(1 + np.exp(Mu)).sum(axis=(1,2)) # (nGrid,) taking sum_n sum_i y_ni * u_ni - log(1 + exp(u_ni))
    LL += np.log(beta_prob(rhos, rho_a)) # prior prob of rho
    p = np.exp(LL - np.max(LL)) # normalized joint prob: p(y|rho,beta) * p(rho)
    # approximate integral by piecewise linear, thus trapezoid: (rho_{i+1} - rho_i) * (p_i + p_{i+1}) / 2
    Z = np.sum(
        (rhos[1:] - rhos[:-1]) * (p[1:] + p[:-1]) / 2
    )
    postP = np.abs(p/Z) # bayes theorem -> posterior probability of rho p(rho|y,beta)
    den = np.cumsum(postP)
    rnd = np.random.uniform() * np.sum(postP)
    idx = np.max(np.append([0], np.where(den <= rnd)))
    rho = rhos[idx]
    A = I - rho * spW
    AiX = AiXs[idx]
    # invA = invAs[idx]
    return rho, AiX

def estimate(seed,
            y, spW,
            nIter, nIterBurn, nGrid,
            paramFix_inits, zeta_inits, Sigma_inits,
            invAsq, nu, rho_a,
            xFix, nFix,
            xRnd, nRnd,
            nInd, nSpc,
            priMuFix, priInvVarFix
            ):

    nRetain = nIter - nIterBurn
    post_paramFix = np.zeros((nRetain, nFix), dtype=np.float64)
    post_paramRnd = np.zeros((nRetain, nInd, nRnd), dtype=np.float64)
    post_zeta = np.zeros((nRetain, nRnd), dtype=np.float64)
    post_iwDiagA = np.zeros((nRetain, nRnd), dtype=np.float64)
    post_Sigma = np.zeros((nRetain, nRnd, nRnd), dtype=np.float64)
    post_rho = np.zeros(nRetain, dtype=np.float64)
    post_y = np.zeros((nRetain, nInd, nSpc), dtype=np.float64)
    post_omega = np.zeros((nRetain, nInd, nSpc), dtype=np.float64)

    ###
    # Initialization
    ###
    I = sp.identity(nSpc, format='csc')
    X = np.concatenate([xFix, xRnd], axis=2)
    rho = 0.
    A = I - rho * spW
    invA = splinalg.inv(A).toarray()
    AiX = np.einsum('ij,njk->nik', invA, X)

    kappa = y - 1/2

    # parameters
    paramFix = paramFix_inits
    zeta = zeta_inits
    Sigma = Sigma_inits
    invSigma = np.linalg.inv(Sigma)
    paramRnd = np.tile(zeta, (nInd, 1))
    paramAll = np.concatenate([np.tile(paramFix, (nInd,1)), paramRnd], axis=1)

    ###
    # Pre-computation for griddy Gibbs
    ###
    print("Pre-calculate griddy Gibbs...")
    invAs = np.zeros((nGrid, nSpc, nSpc), dtype=np.float64)
    AiXs = np.zeros((nGrid, nInd, nSpc, nFix+nRnd), dtype=np.float64)
    yAiXs = np.zeros((nGrid, nInd, nSpc, nFix+nRnd), dtype=np.float64)
    rhos = np.linspace(-1, 1, nGrid + 2)
    rhos = rhos[1:-1]
    for i in tqdm(range(nGrid)):
        A_rho = I - rhos[i] * spW
        invAs[i] = splinalg.inv(A_rho).toarray()
        AiXs[i] = np.einsum('ij,njk->nik', invAs[i], X)
        yAiXs[i] = y[:,:,np.newaxis] * AiXs[i]

    ###
    # Estimation
    ###
    for iter in tqdm(range(nIter)):
        iwDiagA = next_iwDianA(invSigma, nu, invAsq, nRnd)
        Sigma, invSigma = next_Sigma(paramRnd, zeta, nu, iwDiagA, nRnd)
        zeta = next_zeta(paramRnd, Sigma, nRnd, nInd)
        omega = next_omega(paramAll, AiX, nInd, nSpc)
        paramFix, paramRnd, paramAll = next_paramFixRnd(
                paramFix, paramRnd,
                xFix, nFix, xRnd, nRnd,
                zeta, invSigma,
                omega, kappa,
                priMuFix, priInvVarFix,
                AiX, nInd, nSpc)
        rho, AiX = next_rho(paramAll, rhos, rho_a,
                    yAiXs, AiXs, I, spW,
                    nFix, nRnd, nGrid)
        mu = np.einsum('nsk,nk->ns', AiX, paramAll)

        if iter > nIterBurn:
            s = iter - nIterBurn - 1
            post_paramFix[s] = paramFix
            post_paramRnd[s] = paramRnd
            post_zeta[s] = zeta
            post_Sigma[s] = Sigma
            post_iwDiagA[s] = iwDiagA
            post_rho[s] = rho
            post_y[s] = np.exp(mu) / (1 + np.exp(mu))
            post_omega[s] = omega

    return post_paramFix, post_paramRnd, post_zeta, post_iwDiagA, post_Sigma, post_rho, post_y, post_omega

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
    invA = splinalg.inv(I - rho * spW)

    # observation
    paramRnd = paramRnd_mean + (np.diag(paramRnd_std) @ np.random.randn(nRnd,nInd)).T
    paramAll = np.concatenate([np.tile(paramFix, (nInd,1)), paramRnd], axis=1)
    mu = np.einsum('nsk,nk->ns', x, paramAll) + np.random.randn(nInd, nSpc)
    mu = np.einsum('ij,nj->ni', invA.toarray(), mu)
    prob = 1 / (1 + np.exp(-mu))
    y = np.random.binomial(1, prob)

    return x, xFix, xRnd, y, spW


###
# Run
###
# %%
if __name__ == '__main__':
    # generate synthetic data
    nInd, nSpc = 50, 200
    nFix, nRnd = 4, 2
    x, xFix, xRnd, y, spW = generate_data(nInd, nSpc)

    seed = 111
    nIter = 10000
    nIterBurn = 5000
    nGrid = 100
    rho_a = 1.01

    paramFix_inits = np.zeros(nFix, dtype=np.float64)
    zeta_inits = np.zeros(nRnd, dtype=np.float64)
    Sigma_inits = 0.1 * np.eye(nRnd)

    A = 1.04
    invAsq = A**(-2)
    nu = 2

    priMuFix = np.zeros((nFix,), dtype=np.float)
    priVarFix = sp.identity(nFix, format='csc') * 1e-1
    priInvVarFix = splinalg.inv(priVarFix)

    # %%
    post_params = estimate(seed, y, spW,
            nIter, nIterBurn, nGrid,
            paramFix_inits, zeta_inits, Sigma_inits,
            invAsq, nu, rho_a,
            xFix, nFix,
            xRnd, nRnd,
            nInd, nSpc,
            priMuFix, priInvVarFix)

    post_paramFix, post_paramRnd, post_zeta, post_iwDiagA, post_Sigma, post_rho, post_y, post_omega = post_params

    # %%
    ### calculate posterior mean of beta and sigma
    alpha_mean_hat = np.mean(post_paramFix, axis=0)
    zeta_mean_hat = np.mean(post_zeta, axis=0)
    sigma_mean_hat = np.mean(post_Sigma, axis=0)
    y_mean_hat = np.mean(post_y, axis=0)
    rho_post_mean = np.mean(post_rho)
    # error = np.mean(Y - y_mean_hat)
    ic(alpha_mean_hat)
    ic(zeta_mean_hat)
    ic(rho_post_mean)
    ic(sigma_mean_hat)
