from . import beta_priors as bp
import numpy as np
from scipy import stats
import os
import pickle
import pystan

from rpy2 import robjects
from rpy2.robjects.packages import importr
utils = importr('utils')
# utils.install_packages('tmvnsim')
tmvnsim = importr('tmvnsim')
#utils.install_packages('tmg')
tmg = importr('tmg')

class PosteriorSampler:

    def __init__(self,
                 beta_prior=None,
                 pst_mean=None,
                 pst_cov=None,
                 pst_prec=None):
        self.beta_prior = beta_prior
        self.pst_mean = pst_mean
        if (pst_cov is None) + (pst_prec is None) != 1:
            raise ValueError('Eaxcatly one of pst_cov/pst_prec must be set')
        if pst_prec is None:
            self.pst_cov = pst_cov
        if pst_cov is None:
            self.pst_prec = pst_prec
            self.pst_cov = np.linalg.inv(pst_prec)
            # ^----- can we use something else to invert posdef matrix?
        if hasattr(self, 'simplex_sampler') is False:
            self.simplex_sampler=None
        elif hasattr(self, 'positive_sampler') is False:
            self.positive_sampler=None
        if hasattr(self, 'id_constrained_sampler') is False:
            self.id_constrained_sampler=None
        if hasattr(self, 'diag_constrained_sampler') is False:
            self.diag_constrained_sampler=None
        if hasattr(self, 'unconstrained_sampler') is False:
            self.unconstrained_sampler=None
        if hasattr(self, 'general_sampler') is False:
            self.general_sampler=None
        self.select_prior_appropriate_sampler()

    def select_prior_appropriate_sampler(self):
        if isinstance(self.beta_prior, bp.SimplexBetaPrior):
            sampler = self.simplex_sampler
        elif self.beta_prior.__class__ is bp.PositiveBetaPrior:
            sampler = self.positive_sampler
        elif isinstance(self.beta_prior, bp.IdentityConstrainedBetaPrior):
            sampler = self.id_constrained_sampler
        elif isinstance(self.beta_prior, bp.DiagConstrainedBetaPrior):
            sampler = self.diag_constrained_sampler
        elif isinstance(self.beta_prior, bp.UnconstrainedBetaPrior):
            sampler = self.unconstrained_sampler
        elif isinstance(self.beta_prior, bp.BetaPrior):
            sampler = self.general_sampler
        else:
            raise ValueError("beta_prior is wrong")
        self.sampler = sampler

class CholeskyTransformUnitNormal(PosteriorSampler):

    def __init__(self,
                 beta_prior=None,
                 pst_mean=None,
                 pst_cov=None,
                 pst_prec=None):
        super().__init__(beta_prior=beta_prior,
                         pst_mean=pst_mean,
                         pst_cov=pst_cov,
                         pst_prec=pst_prec)
        self.T = np.linalg.cholesky(self.pst_prec)
        # to-do:
        # next line is inverse of triangular matrix
        # can replace np.linalg.inv with linalg.lapack.clapack.dtrtri
        # once I figure out f2py ...
        # http://scipy.github.io/old-wiki/pages/Cookbook/F2Py.html
        self.Tinv = np.linalg.inv(self.T)
        self.L = self.Tinv.T

    def unconstrained_sampler(self, n_smp):
        z = np.random.normal(0, 1, size=(self.pst_mean.size, n_smp))
        beta = self.pst_mean + np.dot(self.L, z).T
        mnv = stats.multivariate_normal(mean=self.pst_mean, cov=self.pst_cov)
        return beta, mnv.logpdf(beta), None


class StanHMC(PosteriorSampler):

    def __init__(self,
                 beta_prior=None,
                 pst_mean=None,
                 pst_cov=None,
                 pst_prec=None):
        super().__init__(beta_prior=beta_prior,
                         pst_mean=pst_mean,
                         pst_cov=pst_cov,
                         pst_prec=pst_prec)

    def unconstrained_sampler(self,
                              iter=1000,
                              chains=5,
                              control={}):
        stanmodfile = 'stanmods/hmc_unconstrained.pkl'
        if os.path.isfile(stanmodfile):
            model = pickle.load(open(stanmodfile, 'rb'))
        else:
            code = """
            data {
                int<lower=1> p;
                vector[p] mu;
                cov_matrix[p] Sigma;
            }
            parameters {
                vector[p] beta;
            }
            model {
                beta ~ multi_normal(mu, Sigma);
            }
            """
            model = pystan.StanModel(model_code=code)
            with open(stanmodfile, 'wb') as f:
                pickle.dump(model, f)
        data = {'p':len(self.pst_mean),
                'mu':self.pst_mean,
                'Sigma':self.pst_cov}
        fit = model.sampling(data=data, iter=iter, chains=chains)
        return fit['beta'], fit['lp__'], fit

    def positive_sampler(self,
                         iter=1000,
                         chains=5,
                         control={}):
        stanmodfile = 'stanmods/hmc_positive.pkl'
        if os.path.isfile(stanmodfile):
            model = pickle.load(open(stanmodfile, 'rb'))
        else:
            code = """
            data {
                int<lower=1> p;
                vector[p] mu;
                cov_matrix[p] Sigma;
            }
            parameters {
                vector<lower=0>[p] beta;
            }
            model {
                beta ~ multi_normal(mu, Sigma);
            }
            """
            model = pystan.StanModel(model_code=code)
            with open(stanmodfile, 'wb') as f:
                pickle.dump(model, f)
        data = {'p':len(self.pst_mean),
                'mu':self.pst_mean,
                'Sigma':self.pst_cov}
        fit = model.sampling(data=data,
                             iter=iter,
                             chains=chains,
                             control=control)
        return fit['beta'], fit['lp__'], fit


class StanTMVN(PosteriorSampler):

    def __init__(self,
                 beta_prior=None,
                 pst_mean=None,
                 pst_cov=None,
                 pst_prec=None):
        super().__init__(beta_prior=beta_prior,
                         pst_mean=pst_mean,
                         pst_cov=pst_cov,
                         pst_prec=pst_prec)
        self.L = np.linalg.cholesky(self.pst_cov)

    def positive_sampler(self,
                         iter=1000,
                         chains=5,
                         control={}):
        stanmodfile = 'stanmods/tMVN.pkl'
        if os.path.isfile(stanmodfile):
            model = pickle.load(open(stanmodfile, 'rb'))
        else:
            stancodefile = 'stanmods/tMVN.txt'
            code = open(stancodefile, "r").read()
            model = pystan.StanModel(model_code=code)
            with open(stanmodfile, 'wb') as f:
                pickle.dump(model, f)
        data = {'K':self.beta_prior.p,
                'b':np.zeros(self.beta_prior.p),
                's':np.zeros(self.beta_prior.p)+1,
                'mu':self.pst_mean,
                'L':self.L}
        fit = model.sampling(data=data,
                             iter=iter,
                             chains=chains,
                             control=control)
        return fit['y'], fit['lp__'], fit


class StanGHK(PosteriorSampler):

    def __init__(self,
                 beta_prior=None,
                 pst_mean=None,
                 pst_cov=None,
                 pst_prec=None):
        super().__init__(beta_prior=beta_prior,
                         pst_mean=pst_mean,
                         pst_cov=pst_cov,
                         pst_prec=pst_prec)
        self.L = np.linalg.cholesky(self.pst_cov)

    def positive_sampler(self,
                         iter=1000,
                         chains=5,
                         control={}):
        stanmodfile = 'stanmods/gibbs_positive.pkl'
        if os.path.isfile(stanmodfile):
            model = pickle.load(open(stanmodfile, 'rb'))
        else:
            stancodefile = 'stanmods/gibbs_positive.txt'
            code = open(stancodefile, "r").read()
            model = pystan.StanModel(model_code=code)
            with open(stanmodfile, 'wb') as f:
                pickle.dump(model, f)
        data = {'p':len(self.pst_mean),
                'mu':self.pst_mean,
                'L':self.L}
        fit = model.sampling(data=data,
                             iter=iter,
                             chains=chains,
                             control=control)
        return fit['beta'], fit['lp__'], fit


class ExactHMC(PosteriorSampler):

    def __init__(self,
                 beta_prior=None,
                 pst_mean=None,
                 pst_cov=None,
                 pst_prec=None):
        super().__init__(beta_prior=beta_prior,
                         pst_mean=pst_mean,
                         pst_cov=pst_cov,
                         pst_prec=pst_prec)

    def wrap_tmg(self,
                 n=1000,
                 precision=None,
                 covariance=None,
                 mean=None,
                 initial=None,
                 f=None,
                 g=None,
                 q=None,
                 burn_in=100):
        if (precision is None) + (covariance is None) != 1:
            raise ValueError('Exactly one of precision or covariance must be set')
        if precision is None:
            precision = np.linalg.inv(covariance)
        d = precision.shape[0]
        M = robjects.FloatVector(precision.ravel())
        M = robjects.r['matrix'](M, nrow=d)
        if mean is None:
            mean = np.zeros(d)
        r = np.dot(precision, mean)
        r = robjects.FloatVector(r)
        if initial is None:
            initial = np.zeros(d) + 1e-5
        initial = robjects.FloatVector(initial)
        if f is None:
            f = robjects.NULL
        else:
            f = robjects.FloatVector(f.ravel())
            f = robjects.r['matrix'](f, nrow=d)
        if g is None:
            g = robjects.NULL
        else:
            g = robjects.FloatVector(g)
        if q is None:
            q = robjects.NULL
        else:
            q = robjects.ListVector(q)
        kw_args = {'f':f, 'g':g, 'q':q, 'burn.in':burn_in}
        result = tmg.rtmg(n, M, r, initial, **kw_args)
        return np.array(result)

    def positive_sampler(self, nsamp=1000, init=1e-5, burn_in=100):
        k = len(self.pst_mean)
        smp = self.wrap_tmg(n=nsamp,
                            precision=self.pst_prec,
                            mean=self.pst_mean,
                            initial=np.ones(k)*init,
                            f=np.identity(k),
                            g=np.zeros(k),
                            burn_in=burn_in)
        return smp



# end
