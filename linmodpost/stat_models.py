import numpy as np
from scipy import stats
from scipy.linalg import orth as scipy_linalg_orth

class BayesianLinearModel:

    def __init__(self,
                 y=None,
                 X=None,
                 sig=None,
                 cov=None,
                 sig_0=None,
                 mu_0=0.):
        # set values
        self.y = y
        self.X = X
        n, p = X.shape
        self.n = n
        self.p = p
        self.sig_0 = sig_0
        self.mu_0 = mu_0
        # check sig/cov parmeters set correctly
        c1 = sig is None
        c2 = cov is None
        if (c1 and c2) or (not c1 and not c2):
            errmsg = 'One and only one of sig and cov must be set'
            raise ValueError(errmsg)
        if sig is not None:
            self.has_sig = True
            self.sig = sig
        else:
            self.has_sig = False
            self.cov = cov
            self.cov_inv = np.linalg.inv(cov)
        self.get_poterior_cov()
        self.get_b()
        self.get_poterior_mean()

    def get_poterior_cov(self):
        if self.has_sig:
            c1 = self.sig**-2.*np.dot(self.X.T, self.X)
        else:
            c1 = np.einsum('ij,jk,kl',
                           self.X.T,
                           self.cov_inv,
                           self.X,
                           optimize=True)
        c2 = self.sig_0**-2 * np.identity(self.p)
        cov_n_inv = c1 + c2
        self.cov_n_inv = cov_n_inv

    def get_b(self):
        if self.has_sig:
            b1 = self.sig**-2. * np.dot(self.X.T, self.y)
        else:
            b1 = np.einsum('ij,jk,k',
                           self.X.T,
                           self.cov_inv,
                           self.y,
                           optimize=True)
        b2 = self.sig_0**-2 * self.mu_0*np.ones(self.p)
        b = b1 + b2
        self.b = b

    def get_poterior_mean(self):
        """Get posterior mean mu_n by solving the OLS problem A mu_n = b, where:
        A = cov_n_inv
        b = X.T cov_inv y + sig_0^-2 mu_0
        """
        mu_n, _, _, _ = np.linalg.lstsq(self.cov_n_inv, self.b, rcond=None)
        self.mu_n = mu_n




# end
