import numpy as np
import pandas as pd
from polyagamma import random_polyagamma
from scipy.stats import invwishart
import scipy.sparse as sp
from scipy.special import beta
from tqdm import tqdm
from math import floor

class spLogit(object):

    def __init__(self,
            seed = 123,
            nInd = None,
            nSpc = None,
            nFix = None,
            nRnd = None,
            x = None,
            y = None,
            W = None,
            xFixName = None,
            xRndName = None,
            A = 1.04,
            nu = 2.,
            rho_a = 1.01,
            paramFix_inits = None,
            zeta_inits = None,
            Sigma_inits = None,
            rho_init = 0.,
            priMuFix = None,
            priVarFix = None,
            spatialLag = True,
            eval_effect = False,
            ):
        # model
        self.spatialLag = spatialLag
        # effect size
        self.eval_effect = eval_effect
        # numbers
        self.nInd = nInd
        self.nSpc = nSpc
        self.nFix = nFix
        self.nRnd = nRnd
        # variables
        self.x = x
        self.xFix = x[:,:,:nFix] if x is not None else None
        self.xRnd = x[:,:,nFix:] if x is not None else None
        self.xFixName = xFixName
        self.xRndName = xRndName
        self.y = y
        self.kappa = y - 1/2 if y is not None else None
        # spatial matrix
        self.W = W
        self.I = sp.identity(nSpc, format='csc') if nSpc is not None else None
        # hyperparameters
        self.invAsq = A**(-2)
        self.nu = nu
        self.rho_a = rho_a
        # initial parameters
        self.paramFix_inits = paramFix_inits
        self.zeta_inits = zeta_inits
        self.Sigma_inits = Sigma_inits
        self.rho_init = rho_init
        self.priMuFix = priMuFix
        self.priVarFix = priVarFix
        # function
        self.beta_prob = lambda rho, a: 1/beta(a,a) * ((1 + rho)**(a-1) * (1 - rho)**(a-1)) / (2**(2*a - 1))

    def load_data_from_spData(self, spData):
        # numbers
        self.nInd = spData['nInd']
        self.nSpc = spData['nSpc']
        self.nFix = spData['nFix']
        self.nRnd = spData['nRnd']
        # variables
        self.x = spData['x']
        self.xFix = self.x[:,:,:self.nFix]
        self.xRnd = self.x[:,:,self.nFix:]
        self.xFixName = spData['xFixName']
        self.xRndName = spData['xRndName']
        self.y = spData['y']
        self.kappa = self.y - 1/2
        # spatial matrix
        self.W = spData['W']
        self.I = sp.identity(self.nSpc, format='csc')
    
    def load_init_params(self, param_df):
        self.rho_init = param_df.loc['rho']['mean']
        if self.nFix > 0:
            fix_keys = [nm + 'Fix' for nm in self.xFixName]
            self.paramFix_inits = param_df.loc[fix_keys]['mean'].values
        if self.nRnd > 0:
            rnd_keys = [nm + 'Fix' for nm in self.xRndName]
            self.zeta_inits = param_df.loc[rnd_keys]['mean'].values
            self.Sigma_inits = np.diag(param_df.loc[rnd_keys]['std. dev.'].values)
            # rndMean_keys = [nm + 'RndMean' for nm in self.xRndName]
            # rndStd_keys = [nm + 'RndStd' for nm in self.xRndName]
            # self.zeta_inits = param_df.loc[rndMean_keys]['mean'].values
            # self.Sigma_inits = np.diag(param_df.loc[rndStd_keys]['mean'].values)

    def update_omega(self):
        mu = np.einsum('nsk,nk->ns', self.AiX, self.paramAll) #+ self.AiE
        self.omega = random_polyagamma(z=mu, size=(self.nInd, self.nSpc))

    def update_paramFixRnd(self):
        # preliminary
        z = self.kappa / self.omega # kappa = Y - 1/2, size N x S
        AiX_fix, AiX_rnd = self.AiX[:,:,:self.nFix], self.AiX[:,:,self.nFix:]
        AiXO_fix = AiX_fix * np.sqrt(self.omega)[:,:,np.newaxis]
        AiXO_rnd = AiX_rnd * np.sqrt(self.omega)[:,:,np.newaxis]
        zO = z * np.sqrt(self.omega)
        # draw alpha and beta
        varFix = np.zeros((self.nFix, self.nFix), dtype=np.float64)
        varRnd = np.zeros((self.nInd, self.nRnd, self.nRnd), dtype=np.float64)
        muFix = np.zeros((self.nFix,), dtype=np.float64)
        muRnd = np.zeros((self.nInd, self.nRnd))
        for n in range(self.nInd):
            varFix += AiXO_fix[n].T @ AiXO_fix[n] # K x S @ S x K -> K x K
            varRnd[n] = AiXO_rnd[n].T @ AiXO_rnd[n] # K x S @ S x K -> K x K
            if self.nRnd > 0:
                muFix += AiXO_fix[n].T @ (zO[n] - AiXO_rnd[n] @ self.paramRnd[n]) # K x S @ S -> K # - self.Ai @ self.epsilon[n]
            else:
                muFix += AiXO_fix[n].T @ (zO[n]) # K x S @ S -> K # - self.Ai @ self.epsilon[n]
            if self.nFix > 0:
                muRnd[n] = AiXO_rnd[n].T @ (zO[n] - AiXO_fix[n] @ self.paramFix) # K x S @ S -> K # - self.Ai @ self.epsilon[n]
            else:
                muRnd[n] = AiXO_rnd[n].T @ (zO[n]) # K x S @ S -> K # - self.Ai @ self.epsilon[n]

        # paramFix
        if self.nFix > 0:
            postVarFix = np.linalg.inv(self.priInvVarFix + varFix)
            postMuFix = postVarFix @ (self.priInvVarFix @ self.priMuFix + muFix)
            postMuFix = np.array(postMuFix).reshape(-1,)
            self.paramFix = np.random.multivariate_normal(mean=postMuFix, cov=postVarFix)

        # paramRnd
        if self.nRnd > 0:
            paramRnd = np.zeros((self.nInd,self.nRnd), dtype=np.float64)
            for n in range(self.nInd):
                postVarRnd = np.linalg.inv(self.invSigma + varRnd[n])
                postMuRnd = postVarRnd @ (self.invSigma @ self.zeta + muRnd[n])
                paramRnd[n] = postMuRnd + np.linalg.cholesky(postVarRnd) @ np.random.randn(self.nRnd,)
            self.paramRnd = paramRnd
        self.paramAll = np.concatenate([np.tile(self.paramFix, (self.nInd,1)), self.paramRnd], axis=1)

    def update_iwDiagA(self):
        self.iwDiagA = np.random.gamma((self.nu + self.nRnd) / 2,
                                            1 / (self.invAsq + self.nu * np.diag(self.invSigma)))

    def update_Sigma(self):
        betaS = self.paramRnd - self.zeta
        self.Sigma = invwishart.rvs(self.nu + self.nInd + self.nRnd - 1,
                                        2 * self.nu * np.diag(self.iwDiagA) + betaS.T @ betaS)
        self.invSigma = np.linalg.inv(self.Sigma)

    def update_zeta(self):
        self.zeta = self.paramRnd.mean(axis=0) +\
                        np.linalg.cholesky(self.Sigma) @ np.random.randn(self.nRnd,) / np.sqrt(self.nInd)

    def update_rho(self):
        # draw rho by griddy Gibbs sampler
        yMu = np.einsum('rnsk,nk->rns', self.yAiXs, self.paramAll) # y_ni * u_ni
        Mu = np.einsum('rnsk,nk->rns', self.AiXs, self.paramAll) # u_ni
        deno = Mu.copy()
        deno[Mu < 500] = np.log(1 + np.exp(Mu[Mu < 500]))
        LL = yMu.sum(axis=(1,2)) - deno.sum(axis=(1,2)) # (nGrid,) taking sum_n sum_i y_ni * u_ni - log(1 + exp(u_ni))
        LL += np.log(self.beta_prob(self.rhos, self.rho_a)) # prior prob of rho
        p = np.exp(LL - np.max(LL)) # normalized joint prob: p(y|rho,beta) * p(rho)
        # approximate integral by piecewise linear, thus trapezoid: (rho_{i+1} - rho_i) * (p_i + p_{i+1}) / 2
        Z = np.sum(
            (self.rhos[1:] - self.rhos[:-1]) * (p[1:] + p[:-1]) / 2
        )
        postP = np.abs(p/Z) # bayes theorem -> posterior probability of rho p(rho|y,beta)
        den = np.cumsum(postP)
        rnd = np.random.uniform() * np.sum(postP)
        idx = np.max(np.append([0], np.where(den <= rnd)))
        self.idx = idx
        self.rho = self.rhos[idx]
        self.AiX = self.AiXs[idx]
        # self.Ai = self.Ais[idx]
        # self.AiE = np.einsum('ij,nj->ni', self.Ai, self.epsilon)
    
    def update_epsilon(self):
        self.epsilon = np.random.randn(self.nInd, self.nSpc)

    def estimate(self, nIter, nIterBurn, nGrid, iterThin=1):
        nRetain = (nIter - nIterBurn) // iterThin
        post_paramFix = np.zeros((nRetain, self.nFix), dtype=np.float64)
        post_paramRnd = np.zeros((nRetain, self.nInd, self.nRnd), dtype=np.float64)
        post_zeta = np.zeros((nRetain, self.nRnd), dtype=np.float64)
        post_iwDiagA = np.zeros((nRetain, self.nRnd), dtype=np.float64)
        post_Sigma = np.zeros((nRetain, self.nRnd, self.nRnd), dtype=np.float64)
        post_rho = np.zeros(nRetain, dtype=np.float64)
        post_y = np.zeros((nRetain, self.nInd, self.nSpc), dtype=np.float64)
        post_omega = np.zeros((nRetain, self.nInd, self.nSpc), dtype=np.float64)
        post_idx = np.zeros(nRetain, dtype=np.int64)

        # init_params
        if self.paramFix_inits is None:
            self.paramFix_inits = np.zeros(self.nFix, dtype=np.float64)
        if self.zeta_inits is None:
            self.zeta_inits = np.zeros(self.nRnd, dtype=np.float64)
        if self.Sigma_inits is None:
            self.Sigma_inits = 0.1 * np.eye(self.nRnd)
        if self.priMuFix is None:
            self.priMuFix = np.zeros((self.nFix,), dtype=np.float64)
        if self.priVarFix is None:
            self.priVarFix = sp.identity(self.nFix, format='csc') * 1e-1
        if self.nFix > 0:
            self.priInvVarFix = sp.linalg.inv(self.priVarFix)

        # pre-computation for griddy gibbs
        print("Pre-computation for Griddy Gibbs")
        if self.spatialLag:
            self.Ais = np.zeros((nGrid, self.nSpc, self.nSpc), dtype=np.float64)
            self.AiXs = np.zeros((nGrid, self.nInd, self.nSpc, self.nFix+self.nRnd), dtype=np.float64)
            self.yAiXs = np.zeros((nGrid, self.nInd, self.nSpc, self.nFix+self.nRnd), dtype=np.float64)
            self.rhos = np.linspace(-1, 1, nGrid + 2)
            self.rhos = self.rhos[1:-1]
            for i in tqdm(range(nGrid)):
                A = self.I - self.rhos[i] * self.W
                invA = sp.linalg.inv(A).toarray()
                self.Ais[i] = invA
                self.AiXs[i] = np.einsum('ij,njk->nik', invA, self.x)
                self.yAiXs[i] = self.y[:,:,np.newaxis] * self.AiXs[i]

        # initialization
        self.paramFix = self.paramFix_inits
        self.zeta = self.zeta_inits
        self.Sigma = self.Sigma_inits
        self.invSigma = np.linalg.inv(self.Sigma)
        self.paramRnd = np.tile(self.zeta, (self.nInd, 1))
        self.paramAll = np.concatenate([np.tile(self.paramFix, (self.nInd,1)), self.paramRnd], axis=1)
        if self.rho_init is not None:
            self.rho = self.rho_init
            A = self.I - self.rho * self.W
            invA = sp.linalg.inv(A).toarray()
            # self.Ai = invA
            self.AiX = np.einsum('ij,njk->nik', invA, self.x)
        else:
            self.rho = self.rhos[nGrid//2] if self.spatialLag else 0.
            # self.Ai = self.Ais[nGrid//2] if self.spatialLag else 0.
            self.AiX = self.AiXs[nGrid//2] if self.spatialLag else self.x
        # self.epsilon = np.random.randn(self.nInd, self.nSpc)
        # self.AiE = np.einsum('ij,nj->ni', self.Ai, self.epsilon)
        self.idx = nGrid//2
        
        # estimation
        print("Estimation")
        for i in tqdm(range(nIter)):
            # self.update_epsilon()
            if self.nRnd > 0:
                self.update_iwDiagA()
                self.update_Sigma()
                self.update_zeta()
            self.update_omega()
            self.update_paramFixRnd()
            if self.spatialLag:
                self.update_rho()
            if i >= nIterBurn:
                if (i - nIterBurn) % iterThin == 0:
                    s = floor((i - nIterBurn) / iterThin)
                    # mu = np.einsum('ij,nj->ni', self.Ai, self.epsilon)
                    mu = np.zeros((self.nInd, self.nSpc), dtype=np.float64) #+ self.AiE
                    if self.nFix > 0:
                        post_paramFix[s] = self.paramFix
                        mu += self.AiX[:,:,:self.nFix] @ self.paramFix
                    if self.nRnd > 0:
                        post_paramRnd[s] = self.paramRnd
                        post_zeta[s] = self.zeta
                        post_Sigma[s] = self.Sigma
                        post_iwDiagA[s] = self.iwDiagA
                        mu += np.einsum('nsk,nk->ns', self.AiX[:,:,self.nFix:], self.paramRnd)
                        # mu += self.AiX[:,:,self.nFix:] @ self.paramRnd.mean(axis=0)
                    post_rho[s] = self.rho
                    post_omega[s] = self.omega
                    post_y[s] = 1 / (1 + np.exp(-mu))
                    post_idx[s] = self.idx

        postParams = {
            'rho': post_rho,
            'paramFix': post_paramFix,
            'paramRnd': post_paramRnd,
            'zeta': post_zeta,
            'Sigma': post_Sigma,
            'y': post_y,
        }
        postRes = self.analyze_posterior(post_rho, post_paramFix, post_paramRnd, post_zeta, post_Sigma)
        if self.eval_effect:
            elasRes, meRes = self.compute_effect_size(post_paramFix, post_paramRnd, post_y, post_idx)
        else:
            elasRes, meRes = None, None
        modelFits = self.evaluate_modelfit(post_y)
        simFits = self.simloglike(postParams)
        modelFits.update(**simFits)
        return postRes, modelFits, postParams, elasRes, meRes

    def analyze_posterior(self,
            post_rho, post_paramFix, post_paramRnd,
            post_zeta, post_Sigma):
        postRes = {}
        postRes['rho'] = self.get_postStats(post_rho)
        if self.nFix > 0:
            # postRes['paramFix'] = self.get_postStats(post_paramFix)
            for k, paramName in enumerate(self.xFixName):
                postRes[paramName + 'Fix'] = self.get_postStats(post_paramFix[:,k])
        if self.nRnd > 0:
            # postRes['zeta'] = self.get_postStats(post_zeta)
            # postRes['Sigma'] = self.get_postStats(post_Sigma)
            dim = 1 # over individuals
            postMean = np.mean(post_paramRnd, axis = dim)
            postStd = np.std(post_paramRnd, axis = dim)
            for k, paramName in enumerate(self.xRndName):
                postRes[paramName + 'RndMean'] = self.get_postStats(postMean[:,k])
                postRes[paramName + 'RndStd'] = self.get_postStats(postStd[:,k])
                postRes[paramName + 'Zeta'] = self.get_postStats(post_zeta[:,k])
                postRes[paramName + 'SigmaDiag'] = self.get_postStats(post_Sigma[:,k,k])
                for l, paramName2 in enumerate(self.xRndName):
                    if l > k:
                        postRes[paramName + '_' + paramName2 + '_Sigma'] = self.get_postStats(post_Sigma[:,k,l])
            # paramRnd
            # dim = (0,1)
            # postMean = np.mean(post_paramRnd, axis = dim)
            # postQl, postQr = np.quantile(post_paramRnd, [0.025, 0.975], axis = dim)
            # postStd = np.std(post_paramRnd, axis = dim)
            # paramRndFlat = post_paramRnd.reshape(-1, self.nRnd)
            # postCorr = np.corrcoef(paramRndFlat[:,0], paramRndFlat[:,1])
            # postRes['paramRnd'] = {
            #     'mean': postMean,
            #     'std. dev.': postStd,
            #     '2.5%': postQl,
            #     '97.5%': postQr,
            #     'corr.': postCorr
            # }
        return postRes

    def get_postStats(self, postParam):
        dim = 0 if postParam.ndim > 1 else None
        postMean = np.mean(postParam, axis = dim)
        postQl, postQr = np.quantile(postParam, [0.025, 0.975], axis = dim)
        postStd = np.std(postParam, axis = dim)
        return {
            'mean': postMean,
            'std. dev.': postStd,
            '2.5%': postQl,
            '97.5%': postQr
        }
    
    def compute_effect_size_prev(self, post_paramFix, post_paramRnd, post_y, post_rho):
        elasRes = {}
        meRes = {}
        # fixed parameters
        if self.nFix > 0:
            for k in range(self.nFix):
                paramName = self.xFixName[k]
                alphaR = post_paramFix[:,k] # R x 1
                d_elasK, d_meK = [], []
                id_elasK, id_meK = [], []
                for n in range(self.nInd):
                    for i in range(self.nSpc):
                        yR = post_y[:,n,i] # R x 1
                        xf = self.xFix[n,i,k] # Kf x 1
                        wxf = self.W[i].dot(self.xFix[n,:,k]) # (1 x S) x (S x 1)
                        d_elas = yR * alphaR * xf
                        id_elas = yR * alphaR * (post_rho * wxf)
                        d_elasK.append(self.get_postStats(d_elas))
                        id_elasK.append(self.get_postStats(id_elas))
                        d_me = yR * (1 - yR) * alphaR
                        id_me = yR * (1 - yR) * alphaR * (post_rho * self.W[i].sum())
                        d_meK.append(self.get_postStats(d_me))
                        id_meK.append(self.get_postStats(id_me))
                d_eDf = pd.DataFrame(d_elasK)
                d_mDf = pd.DataFrame(d_meK)
                id_eDf = pd.DataFrame(id_elasK)
                id_mDf = pd.DataFrame(id_meK)
                elasRes[paramName + 'Fix' + 'Direct'] = {
                    'mean': d_eDf['mean'].mean(),
                    'std. dev.': d_eDf['std. dev.'].mean(),
                    '2.5%': d_eDf['2.5%'].mean(),
                    '97.5%': d_eDf['97.5%'].mean()
                }
                meRes[paramName + 'Fix' + 'Direct'] = {
                    'mean': d_mDf['mean'].mean(),
                    'std. dev.': d_mDf['std. dev.'].mean(),
                    '2.5%': d_mDf['2.5%'].mean(),
                    '97.5%': d_mDf['97.5%'].mean()
                }
                elasRes[paramName + 'Fix' + 'InDirect'] = {
                    'mean': id_eDf['mean'].mean(),
                    'std. dev.': id_eDf['std. dev.'].mean(),
                    '2.5%': id_eDf['2.5%'].mean(),
                    '97.5%': id_eDf['97.5%'].mean()
                }
                meRes[paramName + 'Fix' + 'InDirect'] = {
                    'mean': id_mDf['mean'].mean(),
                    'std. dev.': id_mDf['std. dev.'].mean(),
                    '2.5%': id_mDf['2.5%'].mean(),
                    '97.5%': id_mDf['97.5%'].mean()
                }
        if self.nRnd > 0:
            for k in range(self.nRnd):
                paramName = self.xRndName[k]
                d_elasK, d_meK = [], []
                id_elasK, id_meK = [], []
                for n in range(self.nInd):
                    betaR = post_paramRnd[:,n,k] # R x 1
                    for i in range(self.nSpc):
                        yR = post_y[:,n,i] # R x 1
                        xf = self.xRnd[n,i,k] # Kf x 1
                        wxf = self.W[i].dot(self.xFix[n,:,k]) # (1 x S) x (S x 1)
                        d_elas = yR * betaR * xf
                        id_elas = yR * betaR * (post_rho * wxf)
                        d_elasK.append(self.get_postStats(d_elas))
                        id_elasK.append(self.get_postStats(id_elas))
                        d_me = yR * (1 - yR) * betaR
                        id_me = yR * (1 - yR) * betaR * (post_rho * self.W[i].sum())
                        d_meK.append(self.get_postStats(d_me))
                        id_meK.append(self.get_postStats(id_me))
                d_eDf = pd.DataFrame(d_elasK)
                d_mDf = pd.DataFrame(d_meK)
                id_eDf = pd.DataFrame(id_elasK)
                id_mDf = pd.DataFrame(id_meK)
                elasRes[paramName + 'Rnd' + 'Direct'] = {
                    'mean': d_eDf['mean'].mean(),
                    'std. dev.': d_eDf['std. dev.'].mean(),
                    '2.5%': d_eDf['2.5%'].mean(),
                    '97.5%': d_eDf['97.5%'].mean()
                }
                meRes[paramName + 'Rnd' + 'Direct'] = {
                    'mean': d_mDf['mean'].mean(),
                    'std. dev.': d_mDf['std. dev.'].mean(),
                    '2.5%': d_mDf['2.5%'].mean(),
                    '97.5%': d_mDf['97.5%'].mean()
                }
                elasRes[paramName + 'Rnd' + 'InDirect'] = {
                    'mean': id_eDf['mean'].mean(),
                    'std. dev.': id_eDf['std. dev.'].mean(),
                    '2.5%': id_eDf['2.5%'].mean(),
                    '97.5%': id_eDf['97.5%'].mean()
                }
                meRes[paramName + 'Rnd' + 'InDirect'] = {
                    'mean': id_mDf['mean'].mean(),
                    'std. dev.': id_mDf['std. dev.'].mean(),
                    '2.5%': id_mDf['2.5%'].mean(),
                    '97.5%': id_mDf['97.5%'].mean()
                }
        return elasRes, meRes

    def compute_effect_size(self, post_paramFix, post_paramRnd, post_y, post_idx):
        elasRes = {}
        meRes = {}
        nRetain = post_idx.shape[0]
        post_Ai = self.Ais[post_idx]
        I = self.I.toarray()
        # fixed parameters
        if self.nFix > 0:
            for k in range(self.nFix):
                paramName = self.xFixName[k]
                if paramName not in ["InRiver", "InAvenue"]:
                    continue
                print(f"Evaluating fixed effects for {paramName}...")
                alphaR = post_paramFix[:,k] # R x 1
                d_elasK, d_meK = [], []
                id_elasK, id_meK = [], []
                for n in tqdm(range(self.nInd)):
                    yR = post_y[:,n,:] # R x S
                    xf = self.xFix[n,:,k] # S x 1 (j)
                    elas_nk = (1 - yR[:,:,np.newaxis]) * xf[np.newaxis,np.newaxis,:] * \
                                post_Ai * alphaR[:,np.newaxis,np.newaxis] # R x S x S
                    me_nk = yR[:,:,np.newaxis] * (1 - yR[:,:,np.newaxis]) * post_Ai * alphaR[:,np.newaxis,np.newaxis] # R x S x S
                    d_elas_nk = (elas_nk * I[np.newaxis,:,:]).sum(axis=(1,2)) / self.nSpc # R x 1
                    id_elas_nk = elas_nk.sum(axis=(1,2)) / self.nSpc - d_elas_nk # R x 1
                    d_me_nk = (me_nk * I[np.newaxis,:,:]).sum(axis=(1,2)) / self.nSpc # R x 1
                    id_me_nk = me_nk.sum(axis=(1,2)) / self.nSpc - d_me_nk # R x 1
                    d_elasK.append(self.get_postStats(d_elas_nk))
                    id_elasK.append(self.get_postStats(id_elas_nk))
                    d_meK.append(self.get_postStats(d_me_nk))
                    id_meK.append(self.get_postStats(id_me_nk))
                d_eDf = pd.DataFrame(d_elasK)
                d_mDf = pd.DataFrame(d_meK)
                id_eDf = pd.DataFrame(id_elasK)
                id_mDf = pd.DataFrame(id_meK)
                elasRes[paramName + 'Fix' + 'Direct'] = {
                    'mean': d_eDf['mean'].mean(),
                    'std. dev.': d_eDf['std. dev.'].mean(),
                    '2.5%': d_eDf['2.5%'].mean(),
                    '97.5%': d_eDf['97.5%'].mean()
                }
                meRes[paramName + 'Fix' + 'Direct'] = {
                    'mean': d_mDf['mean'].mean(),
                    'std. dev.': d_mDf['std. dev.'].mean(),
                    '2.5%': d_mDf['2.5%'].mean(),
                    '97.5%': d_mDf['97.5%'].mean()
                }
                elasRes[paramName + 'Fix' + 'InDirect'] = {
                    'mean': id_eDf['mean'].mean(),
                    'std. dev.': id_eDf['std. dev.'].mean(),
                    '2.5%': id_eDf['2.5%'].mean(),
                    '97.5%': id_eDf['97.5%'].mean()
                }
                meRes[paramName + 'Fix' + 'InDirect'] = {
                    'mean': id_mDf['mean'].mean(),
                    'std. dev.': id_mDf['std. dev.'].mean(),
                    '2.5%': id_mDf['2.5%'].mean(),
                    '97.5%': id_mDf['97.5%'].mean()
                }
        if self.nRnd > 0:
            for k in range(self.nRnd):
                paramName = self.xRndName[k]
                if paramName not in ["Dist", "Tree", "Bldg", "Road", "Sky"]:
                    continue
                print(f"Evaluating random effects for {paramName}...")
                d_elasK, d_meK = [], []
                id_elasK, id_meK = [], []
                for n in tqdm(range(self.nInd)):
                    yR = post_y[:,n,:] # R x S
                    xf = self.xRnd[n,:,k] # S x 1 (j)
                    betaR = post_paramRnd[:,n,k] # R x 1
                    elas_nk = (1 - yR[:,:,np.newaxis]) * xf[np.newaxis,np.newaxis,:] * post_Ai * betaR[:,np.newaxis,np.newaxis] # R x S x S
                    me_nk = yR[:,:,np.newaxis] * (1 - yR[:,:,np.newaxis]) * post_Ai * betaR[:,np.newaxis,np.newaxis] # R x S x S
                    d_elas_nk = (elas_nk * I[np.newaxis,:,:]).sum(axis=(1,2)) / self.nSpc # R x 1
                    id_elas_nk = elas_nk.sum(axis=(1,2)) / self.nSpc - d_elas_nk # R x 1
                    d_me_nk = (me_nk * I[np.newaxis,:,:]).sum(axis=(1,2)) / self.nSpc # R x 1
                    id_me_nk = me_nk.sum(axis=(1,2)) / self.nSpc - d_me_nk # R x 1
                    d_elasK.append(self.get_postStats(d_elas_nk))
                    id_elasK.append(self.get_postStats(id_elas_nk))
                    d_meK.append(self.get_postStats(d_me_nk))
                    id_meK.append(self.get_postStats(id_me_nk))
                d_eDf = pd.DataFrame(d_elasK)
                d_mDf = pd.DataFrame(d_meK)
                id_eDf = pd.DataFrame(id_elasK)
                id_mDf = pd.DataFrame(id_meK)
                elasRes[paramName + 'Rnd' + 'Direct'] = {
                    'mean': d_eDf['mean'].mean(),
                    'std. dev.': d_eDf['std. dev.'].mean(),
                    '2.5%': d_eDf['2.5%'].mean(),
                    '97.5%': d_eDf['97.5%'].mean()
                }
                meRes[paramName + 'Rnd' + 'Direct'] = {
                    'mean': d_mDf['mean'].mean(),
                    'std. dev.': d_mDf['std. dev.'].mean(),
                    '2.5%': d_mDf['2.5%'].mean(),
                    '97.5%': d_mDf['97.5%'].mean()
                }
                elasRes[paramName + 'Rnd' + 'InDirect'] = {
                    'mean': id_eDf['mean'].mean(),
                    'std. dev.': id_eDf['std. dev.'].mean(),
                    '2.5%': id_eDf['2.5%'].mean(),
                    '97.5%': id_eDf['97.5%'].mean()
                }
                meRes[paramName + 'Rnd' + 'InDirect'] = {
                    'mean': id_mDf['mean'].mean(),
                    'std. dev.': id_mDf['std. dev.'].mean(),
                    '2.5%': id_mDf['2.5%'].mean(),
                    '97.5%': id_mDf['97.5%'].mean()
                }
        return elasRes, meRes

    def evaluate_modelfit(self, post_y):
        nRetain = post_y.shape[0]
        # log pointwise predictive density
        LPPD = 0.
        RSME = 0.
        FPR = 0.
        for n in range(self.nInd):
            for s in range(self.nSpc):
                y_pred = post_y[:,n,s]
                if self.y[n,s] == 1:
                    # LPPD += np.log(y_pred).sum() / nRetain
                    LPPD += np.log(y_pred).mean()
                else:
                    # LPPD += np.log(1-y_pred).sum() / nRetain
                    LPPD += np.log(1-y_pred).mean()
                RSME += np.sqrt((y_pred - self.y[n,s])**2).mean()
                FPR += np.abs((y_pred > 0.5) - self.y[n,s]).mean()
        yMean = np.mean(post_y, axis = 0)
        # root mean square error
        # RSME = np.sqrt(((yMean - self.y)**2).sum() / (self.nInd * self.nSpc))
        RSME /= (self.nInd * self.nSpc)
        # first preference recovery
        # FPR = 1 - np.abs((yMean > 0.5) - self.y).sum() / (self.nInd * self.nSpc)
        FPR = 1 - FPR / (self.nInd * self.nSpc)
        return {'LPPD': LPPD, 'RSME': RSME, 'FPR': FPR}

    def simloglike(self, postParams):
        simDraws = 10000
        if self.nFix > 0 and self.nRnd == 0:
            simDraws_star = 1
        else:
            simDraws_star = simDraws
        
        pSim = np.zeros((simDraws_star, self.nInd, self.nSpc))

        postMean_rho = postParams['rho'].mean(axis=0)
        A = self.I - postMean_rho * self.W
        invA = sp.linalg.inv(A).toarray()
        AiX = np.einsum('ij,njk->nik', invA, self.x)
        
        paramFix = 0
        paramRnd = 0
        if self.nFix > 0: paramFix = postParams['paramFix'].mean(axis=0)
        if self.nRnd > 0: 
            postMean_zeta = postParams['zeta'].mean(axis=0) 
            postMean_chSigma = np.linalg.cholesky(postParams['Sigma'].mean(axis=0))

        for i in np.arange(simDraws_star):
            mu = np.zeros((self.nInd, self.nSpc), dtype=np.float64)
            # if simDraws_star > 1 and postMean_rho > 0:
            #     mu += np.einsum('ij,nj->ni', invA, np.random.randn(self.nInd, self.nSpc)) # invA @ epsilon
            if self.nFix > 0:
                mu += AiX[:,:,:self.nFix] @ paramFix
            if self.nRnd > 0:
                paramRnd = postMean_zeta + (postMean_chSigma @ np.random.randn(self.nRnd, self.nInd)).T
                mu += np.einsum('nsk,nk->ns', AiX[:,:,self.nFix:], paramRnd)
            mu = np.clip(mu, -100, 100)
            pSim[i, :, :] = 1 / (1 + np.exp(-mu))
        
        logLik = 0.
        pSimMean = pSim.mean(axis=0)
        # print(pSimMean[0,:])
        for n in range(self.nInd):
            for s in range(self.nSpc):
                if self.y[n,s] == 1:
                    logLik += np.log(pSimMean[n,s])
                else:
                    logLik += np.log(1-pSimMean[n,s])
        
        # root mean square error
        RSME = np.sqrt(((pSimMean - self.y)**2).sum() / (self.nInd * self.nSpc))
        # first preference recovery
        FPR = 1 - np.abs((pSimMean > 0.5) - self.y).sum() / (self.nInd * self.nSpc)
        
        print(' ')
        print('Log-likelihood (simulated at posterior means): ' + str(logLik)) 
        return {'SimLogLik': logLik, 'SimRSME': RSME, 'SimFPR': FPR}
