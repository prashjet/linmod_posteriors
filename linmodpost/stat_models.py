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

class Projector:

    # X = (n,p) data matrix
    # A = (m,n) "projection" matrix, full rank
    # y ~ Norm(X beta, Cov) <==> Ay ~ Norm(AX beta, ACovA^T)
    # but what is relation between the functions...
    #   f(beta) = Norm(y ; X beta, Cov)
    #   g(beta) = Norm(Ay ; AX beta, ACovA^T) ...?
    # guess f(beta) ~ Norm(const*g(beta), sig^2)
    # where sig^2 = variance of X lost under A

    def __init__(self, A=None, X=None, plot_proj_lost_var=True):
        # check shapes
        if A.shape[1]!=X.shape[0]:
            raise ValueError('A and X are incompatible shapes')
        # check A is full rank
        if np.linalg.matrix_rank(A)!=np.min(A.shape):
            raise ValueError('A is not full rank')
        # give A orthonormal columns
        A = scipy_linalg_orth(A.T).T
        self.A = A
        self.X = X
        n, p = X.shape
        self.n = n
        self.p = p
        S = np.dot(self.X, self.X.T)
        self.total_var = np.trace(S)
        ASAT = np.dot(A, np.dot(S, A.T))
        self.captured_var = np.trace(ASAT)
        self.lost_var = self.total_var - self.captured_var
        self.sig = np.sqrt(self.total_var - self.captured_var)

    def plot_beta_simtest(self, log_g, log_f, plot_proj_lost_var=True):
        def minmax(x):
            return np.array([np.min(x), np.max(x)])
        # get Delta log Ls
        Del = log_g - log_f
        mean_Del = np.mean(Del)
        Del0 = Del - np.mean(Del)
        # create figure
        fig = plt.figure(figsize=(10, 6))
        ax0 = fig.add_subplot(2, 2, 1)
        ax1 = fig.add_subplot(2, 2, 3)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_xlabel('log $L(y)$')
        ax0.set_ylabel('log $L(Ay)$')
        ax1.set_ylabel('$\Delta$ log L')
        ax2.set_ylabel('pdf')
        ax2.set_xlabel('$\Delta$ log L')
        ax = [ax0, ax1, ax2]
        ax0.set_xticks([])
        # first axis
        ax[0].plot(minmax(log_f), minmax(log_f), ':k', label='1:1')
        ax[0].plot(log_f, log_g, '.', ms=10, alpha=0.5)
        ax[0].plot(minmax(log_f),
                   minmax(log_f)+mean_Del,
                   '-k',
                   label='1:1 - $<\Delta \log L>$')
        ax[0].legend()
        # second axis
        ax[1].plot(log_f, Del0, '.', ms=10, alpha=0.5)
        ax[1].axhline([0], ls='-', color='k')
        # third axis
        ax[2].hist(Del0, bins=100, density=True, alpha=0.3)
        ax[2].axvline([0], color='k')
        xarr = np.linspace(np.min(Del0), np.max(Del0), 100)
        sig_real = np.std(Del0)
        nrm = stats.norm(0, sig_real)
        xarr = np.linspace(np.min(Del0), np.max(Del0), 100)
        ax[2].plot(xarr, nrm.pdf(xarr), color='C0')
        ax[2].text(0.9, 0.8,
                   "Real $\sigma$ = {0:0.2e}".format(sig_real),
                   transform=ax[2].transAxes,
                   ha='right')
        if plot_proj_lost_var:
            nrm = stats.norm(0, self.sig)
            ax[2].plot(xarr, nrm.pdf(xarr), color='C1')
            ax[2].text(0.9, 0.9,
                       "Projection $\sigma =$ {0:0.2e}".format(self.sig),
                       transform=ax[2].transAxes,
                       ha='right')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        return

    def beta_sim_test(self,
                      y,
                      cov,
                      beta_prior=None,
                      force_positive_beta=False,
                      n_beta_smp=1000,
                      plot=True,
                      plot_proj_lost_var=True):
        if beta_prior is None:
            beta_prior = stats.multivariate_normal(np.zeros(self.p),
                                                   np.identity(self.p))
        beta_smp = beta_prior.rvs(n_beta_smp).T
        if force_positive_beta:
            beta_smp = np.abs(beta_smp)
        # get log f(beta)
        mvn = stats.multivariate_normal(y, cov)
        Xbeta = np.dot(self.X, beta_smp)
        log_f = mvn.logpdf(Xbeta.T)
        # get log g(beta)
        Ay = np.dot(self.A, y)
        AcovAT = np.dot(np.dot(self.A, cov), self.A.T)
        mvn = stats.multivariate_normal(Ay, AcovAT, allow_singular=True)
        A_X_beta = np.dot(self.A, Xbeta)
        log_g = mvn.logpdf(A_X_beta.T)
        if plot:
            self.plot_beta_simtest(log_g,
                                   log_f,
                                   plot_proj_lost_var=plot_proj_lost_var)
        return

################################
# matrix factorisors
################################

class MatrixFactorisor:

    def __init__(self, X=None, center='both', q=None):
        self.orthogonal_loadings = None
        n, p = X.shape
        self.n = n
        self.p = p
        self.min_n_p = np.min([n, p])
        if center=='rows':
            self.mu_row = np.mean(X, 0)
            self.X = X - self.mu_row
        elif center=='cols':
            self.mu_col = np.mean(X, 1)
            self.X = (X.T - self.mu_col).T
        elif center=='both':
            self.mu_row = np.mean(X, 0)
            X = X - self.mu_row
            self.mu_col = np.mean(X, 1)
            X = (X.T - self.mu_col).T
            self.X = X
        elif center==False:
            self.X = X
        else:
            raise ValueError('unknown option for center')

    def factorise(self, q):
        # placeholder
        self.q = q
        self.Z = None
        self.W = None
        pass

    def get_nullspace_of_loading_matrix(self):
        # placeholder
        self.U = None
        pass

    def analyse_residuals(self):
        self.E = self.X - np.dot(self.Z, self.W.T)
        self.covE = np.cov(self.E)

    def get_U(self):
        # U = (pseudo) inverse of W^T
        if self.orthogonal_loadings:
            self.U = self.W
        else:
            self.U = np.linalg.pinv(self.W.T)

    def get_V(self):
        # placeholder
        # V = (p, p-q) matrix
        # p-q columns of V + q columns of U form a basis for R^p
        pass

    def check_added_likelihood_variance(self, data_cov):
        self.WTW = np.dot(self.W.T, self.W)
        self.Cinv = 1./self.sig2 * self.WTW
        self.min_diag_Cinv = np.min(np.diag(self.Cinv))
        self.C = self.sig2 * np.linalg.inv(self.WTW)
        tmp = np.ones(self.q)
        self.added_var = np.einsum('i,ij,j', tmp, self.C, tmp)
        self.max_frac_added_var = np.max(self.added_var/np.diag(data_cov))

class SVD(MatrixFactorisor):

    def __init__(self, X=None, center=None):
        super().__init__(X=X, center=center)
        self.orthogonal_loadings = True
        u, s, vt = np.linalg.svd(self.X, full_matrices=True)
        self.svd_u = u
        self.svd_s = s
        self.svd_vt = vt
        self.total_var = np.sum(self.svd_s**2)
        self.lost_var = self.total_var - np.cumsum(self.svd_s**2)

    def plot_lost_variance(self, qlim=None, ax=None):
        qarr = np.arange(self.min_n_p)+1
        if ax==None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(qarr, self.lost_var, '.-')
        ax.set_xlabel('q')
        ax.set_ylabel('Lost Var. [$\sigma^2$]')
        # add right hand axis labels as % of total
        ax2 = ax.twinx()
        ax2.set_ylabel('Lost Var. [% of total]')
        ax2.plot(qarr,
                 100.*self.lost_var/self.total_var,
                 alpha=0)
        # format
        for ax0 in [ax, ax2]:
            ax0.set_xscale('log')
            ax0.set_yscale('log')
        if qlim is not None:
            qlim = np.array(qlim)
            ax.set_xlim(qlim)
            ax.set_ylim(self.lost_var[[qlim[1]-1, qlim[0]-1]])
        plt.gcf().tight_layout()
        return plt.gcf(), ax

    def factorise(self, q):
        self.q = q
        self.Z = np.dot(self.svd_u[:, 0:q],
                        np.diag(self.svd_s[0:q]))
        self.W = self.svd_vt[0:q, :].T

    def get_V(self):
        self.V = self.svd_vt[self.q:, :].T
