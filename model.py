import numpy as np
from polyagamma import random_polyagamma
from scipy.stats import invwishart
import scipy.sparse as sp
from scipy.special import beta
from tqdm import tqdm

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
            ):
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

    def update_omega(self):
        mu = np.einsum('nsk,nk->ns', self.AiX, self.paramAll)
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
                muFix += AiXO_fix[n].T @ (zO[n] - AiXO_rnd[n] @ self.paramRnd[n]) # K x S @ S -> K
            else:
                muFix += AiXO_fix[n].T @ zO[n] # K x S @ S -> K
            if self.nFix > 0:
                muRnd[n] = AiXO_rnd[n].T @ (zO[n] - AiXO_fix[n] @ self.paramFix) # K x S @ S -> K
            else:
                muRnd[n] = AiXO_rnd[n].T @ zO[n] # K x S @ S -> K

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
                                            1 / self.invAsq + self.nu * np.diag(self.invSigma))

    def update_Sigma(self):
        betaS = self.paramRnd - self.zeta
        self.Sigma = invwishart.rvs(self.nu + self.nRnd - 1,
                                        2 * self.nu * np.diag(self.iwDiagA) + betaS.T @ betaS)
        self.invSigma = np.linalg.inv(self.Sigma)

    def update_zeta(self):
        self.zeta = self.paramRnd.mean(axis=0) +\
                        np.linalg.cholesky(self.Sigma) @ np.random.randn(self.nRnd,) / np.sqrt(self.nInd)

    def update_rho(self):
        # draw rho by griddy Gibbs sampler
        yMu = np.einsum('rnsk,nk->rns', self.yAiXs, self.paramAll) # y_ni * u_ni
        Mu = np.einsum('rnsk,nk->rns', self.AiXs, self.paramAll) # u_ni
        LL = yMu.sum(axis=(1,2)) - np.log(1 + np.exp(Mu)).sum(axis=(1,2)) # (nGrid,) taking sum_n sum_i y_ni * u_ni - log(1 + exp(u_ni))
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
        self.rho = self.rhos[idx]
        self.AiX = self.AiXs[idx]

    def estimate(self, nIter, nIterBurn, nGrid):
        nRetain = nIter - nIterBurn
        post_paramFix = np.zeros((nRetain, self.nFix), dtype=np.float64)
        post_paramRnd = np.zeros((nRetain, self.nInd, self.nRnd), dtype=np.float64)
        post_zeta = np.zeros((nRetain, self.nRnd), dtype=np.float64)
        post_iwDiagA = np.zeros((nRetain, self.nRnd), dtype=np.float64)
        post_Sigma = np.zeros((nRetain, self.nRnd, self.nRnd), dtype=np.float64)
        post_rho = np.zeros(nRetain, dtype=np.float64)
        post_y = np.zeros((nRetain, self.nInd, self.nSpc), dtype=np.float64)
        post_omega = np.zeros((nRetain, self.nInd, self.nSpc), dtype=np.float64)

        # init_params
        if self.paramFix_inits is None:
            self.paramFix_inits = np.zeros(self.nFix, dtype=np.float64)
        if self.zeta_inits is None:
            self.zeta_inits = np.zeros(self.nRnd, dtype=np.float64)
        if self.Sigma_inits is None:
            self.Sigma_inits = 0.1 * np.eye(self.nRnd)
        if self.priMuFix is None:
            self.priMuFix = np.zeros((self.nFix,), dtype=np.float)
        if self.priVarFix is None:
            self.priVarFix = sp.identity(self.nFix, format='csc') * 1e-1
        if self.nFix > 0:
            self.priInvVarFix = sp.linalg.inv(self.priVarFix)

        # pre-computation for griddy gibbs
        self.AiXs = np.zeros((nGrid, self.nInd, self.nSpc, self.nFix+self.nRnd), dtype=np.float64)
        self.yAiXs = np.zeros((nGrid, self.nInd, self.nSpc, self.nFix+self.nRnd), dtype=np.float64)
        self.rhos = np.linspace(-1, 1, nGrid + 2)
        self.rhos = self.rhos[1:-1]
        for i in tqdm(range(nGrid)):
            A = self.I - self.rhos[i] * self.W
            invA = sp.linalg.inv(A).toarray()
            self.AiXs[i] = np.einsum('ij,njk->nik', invA, self.x)
            self.yAiXs[i] = self.y[:,:,np.newaxis] * self.AiXs[i]

        # initialization
        self.paramFix = self.paramFix_inits
        self.zeta = self.zeta_inits
        self.Sigma = self.Sigma_inits
        self.invSigma = np.linalg.inv(self.Sigma)
        self.paramRnd = np.tile(self.zeta, (self.nInd, 1))
        self.paramAll = np.concatenate([np.tile(self.paramFix, (self.nInd,1)), self.paramRnd], axis=1)
        self.rho = self.rhos[nGrid//2]
        self.AiX = self.AiXs[nGrid//2]

        # estimation
        for iter in tqdm(range(nIter)):
            if self.nRnd > 0:
                self.update_iwDiagA()
                self.update_Sigma()
                self.update_zeta()
            self.update_omega()
            self.update_paramFixRnd()
            self.update_rho()
            if iter >= nIterBurn:
                s = iter - nIterBurn
                mu = np.einsum('nsk,nk->ns', self.AiX, self.paramAll)
                if self.nFix > 0:
                    post_paramFix[s] = self.paramFix
                if self.nRnd > 0:
                    post_paramRnd[s] = self.paramRnd
                    post_zeta[s] = self.zeta
                    post_Sigma[s] = self.Sigma
                    post_iwDiagA[s] = self.iwDiagA
                post_rho[s] = self.rho
                post_omega[s] = self.omega
                post_y[s] = np.exp(mu) / (1 + np.exp(mu))

        postParams = {
            'rho': post_rho,
            'paramFix': post_paramFix,
            'paramRnd': post_paramRnd,
            'zeta': post_zeta,
            'Sigma': post_Sigma,
            'y': post_y,
        }
        postRes = self.analyze_posterior(post_rho, post_paramFix, post_paramRnd, post_zeta, post_Sigma)
        modelFits = self.evaluate_modelfit(post_y)
        return postRes, modelFits, postParams

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
                # postRes[paramName + 'Zeta'] = self.get_postStats(post_zeta[:,k])
                postRes[paramName + 'RndMean'] = self.get_postStats(postMean[:,k])
                postRes[paramName + 'RndStd'] = self.get_postStats(postStd[:,k])
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

    def evaluate_modelfit(self, post_y):
        nRetain = post_y.shape[0]
        # log pointwise predictive density
        LPPD = 0.
        for n in range(self.nInd):
            for s in range(self.nSpc):
                y_pred = post_y[:,n,s]
                if self.y[n,s] == 1:
                    LPPD += np.log(y_pred).sum() / nRetain
                else:
                    LPPD += np.log(1-y_pred).sum() / nRetain
        yMean = np.mean(post_y, axis = 0)
        # root mean square error
        RSME = np.sqrt(((yMean - self.y)**2).sum() / (self.nInd * self.nSpc))
        # first preference recovery
        FPR = 1 - np.abs((yMean > 0.5) - self.y).sum() / (self.nInd * self.nSpc)
        return {'LPPD': LPPD, 'RSME': RSME, 'FPR': FPR}
