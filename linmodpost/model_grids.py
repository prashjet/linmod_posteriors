import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class modelGrid:

    def __init__(self,
                 n,
                 lmd_min,
                 lmd_max,
                 npars,
                 par_dims,
                 par_lims,
                 par_mask_function,
                 normalise_models):
        self.n = n
        self.lmd_min = lmd_min
        self.lmd_max = lmd_max
        self.lmd = np.linspace(lmd_min, lmd_max, n)
        self.delta_lmd = (lmd_max-lmd_min)/n
        self.npars = npars
        self.par_dims = par_dims
        self.par_lims = par_lims
        self.par_mask_function = par_mask_function
        self.check_par_input()
        self.get_par_grid()
        p = self.pars.shape[-1]
        self.p = p
        self.min_n_p = np.min([n, p])
        self.X = self.get_X(self.lmd[:, np.newaxis], self.pars)

    def add_constant(self, c):
        self.X += c

    def normalise(self):
        self.X = self.X/self.delta_lmd/np.median(np.sum(self.X, 0))

    def check_par_input(self):
        check = (len(self.par_dims) == self.npars)
        if check is False:
            raise ValueError('par_dims is wrong length')
        check = (np.array(self.par_lims).shape == (self.npars, 2))
        if check is False:
            raise ValueError('par_lims is wrong shape')
        if np.product(self.par_dims) > 1e5:
            raise ValueError('total size of par grid > 1e5')
        if self.par_mask_function is not None:
            # evaluate par_mask_function on zero-array to check input shape
            tmp = np.zeros((self.npars, 3))
            self.par_mask_function(tmp)
        pass

    def par_description(self):
        print(self.par_string)

    def get_par_grid(self):
        # get param grid
        par_edges = []
        par_widths = []
        par_cents = []
        par_idx_arrays = []
        ziparr = zip(self.par_dims, self.par_lims)
        for i_p, (n_p, (lo, hi)) in enumerate(ziparr):
            edges = np.linspace(lo, hi, n_p+1)
            par_edges += [edges]
            par_widths += [edges[1:] - edges[0:-1]]
            par_cents += [(edges[0:-1] + edges[1:])/2.]
            par_idx_arrays += [np.arange(n_p)]
        self.par_edges = par_edges
        self.par_widths = par_widths
        self.par_cents = par_cents
        par_grid_nd = np.meshgrid(*self.par_cents, indexing='ij')
        par_grid_nd = np.array(par_grid_nd)
        pars = par_grid_nd.reshape(self.npars, -1)
        par_idx = np.meshgrid(*par_idx_arrays, indexing='ij')
        par_idx = np.array(par_idx)
        par_idx = par_idx.reshape(self.npars, -1)
        # mask out entries
        if self.par_mask_function is not None:
            par_mask_nd = self.par_mask_function(par_grid_nd)
            par_mask_1d = self.par_mask_function(pars)
        else:
            par_mask_nd = np.zeros(self.par_dims, dtype=bool)
            par_mask_1d = np.zeros(np.product(self.par_dims),
                                   dtype=bool)
        self.par_mask_nd = par_mask_nd
        unmasked = par_mask_1d==False
        pars = pars[:, unmasked]
        par_idx = par_idx[:, unmasked]
        # store final arrays
        self.par_idx = par_idx
        self.pars = pars

    def get_X(self, lmd, pars):
        """Placeholder for get_X"""
        pass

    def plot_models(self,
                    ax=None,
                    kw_plot={'color':'k',
                             'alpha':0.1}):
        if ax==None:
            ax = plt.gca()
        ax.plot(self.lmd, self.X, **kw_plot)
        ax.set_xlabel('$\lambda$')
        plt.show()

    def reshape_beta(self, beta):
        shape = beta.shape
        if shape[-1]!=self.p:
            raise ValueError('beta must have shape (..., {0})'.format(self.p))
        dims = tuple(shape[0:-1]) + tuple(self.par_dims)
        beta_reshape = np.zeros(dims)
        idx = tuple([slice(0,None) for i in range(len(beta.shape)-1)])
        idx += tuple([self.par_idx[i] for i in range(self.npars)])
        beta_reshape[idx] = beta
        return beta_reshape

class CenteredNormal(modelGrid):

    def __init__(self,
                 n=300,
                 lmd_min=-3,
                 lmd_max=3.,
                 par_lims=((-1,0),),
                 par_dims=(30,),
                 par_mask_function=None,
                 normalise_models=True):
        npars = 1
        s = ("1) log10(sigma) of normal pdf")
        self.par_string = s
        pltsym = ['$\\log\\sigma$']
        self.par_pltsym = pltsym
        super().__init__(n,
                         lmd_min,
                         lmd_max,
                         npars,
                         par_dims,
                         par_lims,
                         par_mask_function,
                         normalise_models)

    def get_X(self, lmd, pars):
        logsig = pars[0, :]
        return stats.norm.pdf(lmd, loc=0., scale=10.**logsig)


class Normal(modelGrid):

    def __init__(self,
                 n=100,
                 lmd_min=-3,
                 lmd_max=3.,
                 par_lims=((-0.5,0.), (-2,2)),
                 par_dims=(30, 10),
                 par_mask_function=None,
                 normalise_models=True):
        npars = 2
        s = ("1) log10(sigma) of normal pdf\n"
             "2) mean of normal pdf")
        self.par_string = s
        pltsym = ['$\\log\\sigma$', '$\\mu$']
        self.par_pltsym = pltsym
        super().__init__(n,
                         lmd_min,
                         lmd_max,
                         npars,
                         par_dims,
                         par_lims,
                         par_mask_function,
                         normalise_models)

    def get_X(self, lmd, pars):
        logsig = pars[0, :]
        loc = pars[1, :]
        return stats.norm.pdf(lmd, loc=loc, scale=10.**logsig)
