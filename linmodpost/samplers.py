from . import beta_priors as bp
import numpy as np
from scipy import stats
import os
import pickle
import pystan

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

class Emcee(PosteriorSampler):
    pass


class StanHMC(PosteriorSampler):

    def bp_model(self, x):
        return x**2

    def __init__(self,
                 beta_prior=None,
                 model_grid=None,
                 data=None):
        self.spx_bp_model = None
        self.pos_bp_model = 3
        self.idc_bp_model = None
        self.dgc_bp_model = None
        super().__init__(beta_prior=beta_prior,
                         model_grid=model_grid,
                         data=data)


class StanGibbs(PosteriorSampler):
    pass


class ExactHMC_R(PosteriorSampler):
    pass


class ExactHMC_Python(PosteriorSampler):
    pass
