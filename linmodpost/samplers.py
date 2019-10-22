from . import beta_priors as bp
import numpy as np
from scipy import stats

class Samples:
    """Class for samples. Contains methods to calculate:
    1) MAP, i.e. the the maximum probability sample
    2) average 1D marginalised samples, either:
        i) median
        ii) mode estimtaed by fitting beta distribution
        iii) mode estimtaed as the "halfsample mode"
    3) Bayesian Credible Intervals (BCIs) - i.e. intervals that contain/exclude
    a specified fraction of samples, either:
        i) even-tailed interval
        ii) highest-density interval
    4)


    Parameters
    ----------
    x : array of (n_smp, smp_shape)
        sample values
    logprob : array (n_smp,)
        log proabability values

    """

    def __init__(self, x=None, logprob=None):
        self.x = x
        self.n_smp = x.shape[0]
        self.smp_shape = x.shape[1::]
        self.smp_dim = len(self.smp_shape)
        self.sx = np.sort(x, axis=0)
        self.logprob = logprob

    def check_p_in_p_out(self, p_in, p_out):
        """Check p_in/p_out parameters for BCIs i.e. the fraction of samples
        to include/exclude in the intervals.

        """
        if (p_in is None) + (p_out is None) != 1:
            raise ValueError('Exactly one of p_in or p_out must be set')
        if p_in is None:
            p_out = np.atleast_1d(p_out)
            p_in = 1. - p_out
        if p_out is None:
            p_in = np.atleast_1d(p_in)
            p_out = 1. - p_in
        return p_in, p_out

    def even_tailed_bcis(self,
                         p_in=None,
                         p_out=None):
        p_in, p_out = self.check_p_in_p_out(p_in, p_out)
        p_lim_even_tailed = np.concatenate((p_out/2., 1.-p_out/2.))
        bcis = np.percentile(self.x, 100.*p_lim_even_tailed, axis=0)
        bcis = np.reshape(bcis, (2, len(p_in))+self.smp_shape)
        bcis = np.swapaxes(bcis, 0, 1)
        return bcis

    def highest_density_single_bci(self, p_in):
        """
        Adatped from PYMC3
        """
        n_smp_in = int(np.floor(p_in * self.n_smp))
        n_smp_out = self.n_smp - n_smp_in
        possible_widths = self.sx[n_smp_in:] - self.sx[:n_smp_out]
        if len(possible_widths) == 0:
            raise ValueError('Too few elements for interval calculation')
        # argmin for minimum widths of first axis, i.e. sample axis
        min_width_idx = np.argmin(possible_widths, 0)
        min_width_idx = np.ravel(min_width_idx)
        # construct tuple for indices of remaining axes of self.sx
        idx_x_shape = [np.arange(i) for i in self.smp_shape]
        idx_x_shape = np.meshgrid(*idx_x_shape, indexing='ij')
        idx_x_shape = [np.ravel(idx) for idx in idx_x_shape]
        idx_x_shape = tuple(idx_x_shape)
        # get indices/values of bci edges
        idx_min = (min_width_idx,) + idx_x_shape
        hdi_min = self.sx[idx_min]
        idx_max = (min_width_idx+n_smp_in,) + idx_x_shape
        hdi_max = self.sx[idx_max]
        bci = np.array([hdi_min, hdi_max])
        return np.reshape(bci, (2,)+self.smp_shape)

    def highest_density_bcis(self,
                             p_in=None,
                             p_out=None):
        p_in, p_out = self.check_p_in_p_out(p_in, p_out)
        bcis = np.zeros((len(p_in), 2) + self.smp_shape)
        idx = tuple([slice(0,end) for end in self.smp_shape])
        for i, p_in0 in enumerate(p_in):
            bcis[(i,)+idx] = self.highest_density_single_bci(p_in0)
        return bcis

    def calc_min_interval(self, alpha):
        """Internal method to determine the minimum interval of
        a given width
        Assumes that x is sorted numpy array.
        STOLEN FROM PYMC3 !!!
        """
        x = np.sort(self.x)
        n = len(x)
        cred_mass = 1.0 - alpha
        interval_idx_inc = int(np.floor(cred_mass * n))
        n_intervals = n - interval_idx_inc
        interval_width = x[interval_idx_inc:] - x[:n_intervals]
        if len(interval_width) == 0:
            raise ValueError('Too few elements for interval calculation')
        min_idx = np.argmin(interval_width)
        hdi_min = x[min_idx]
        hdi_max = x[min_idx + interval_idx_inc]
        raise ValueError()
        return hdi_min, hdi_max

    def get_hdr_interval(x, alpha=0.05):
        '''
        high density region interval
        i.e. smallest interval containing 100*(1-alpha)% of samples x
        STOLEN FROM PYMC3 !!!
        '''
        self.sort_idx = np.sort(x, axis=0)
        return np.array(calc_min_interval(sx, alpha))

    def get_bcis(xarr,
                 bci_levels=[0.5, 0.95]):
        '''
        Get 1d bayesian credible intervals defined by higheest density regions
        looping over a 2d array of samples
        '''
        _, nx = xarr.shape
        alpha_levels = 1. - np.array(bci_levels)
        nalph = len(alpha_levels)
        bcis = np.zeros((nx, nalph, 2))
        for i in range(nx):
            for j, alpha in enumerate(alpha_levels):
                bcis[i, j, :] = get_hdr_interval(xarr[:,i], alpha)
        return bcis


class PosteriorSampler:

    def __init__(self,
                 beta_prior=None,
                 pst_mean=None,
                 pst_cov=None,
                 pst_prec=None):
        self.beta_prior = beta_prior
        self.pst_mean = pst_mean
        self.pst_cov = pst_cov
        self.pst_prec = pst_prec
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
        if self.pst_cov is None:
            self.pst_cov = np.linalg.inv(self.pst_prec)

    def unconstrained_sampler(self, n_smp):
        z = np.random.normal(0, 1, size=(self.pst_mean.size, n_smp))
        beta = self.pst_mean + np.dot(self.L, z).T
        return beta

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
