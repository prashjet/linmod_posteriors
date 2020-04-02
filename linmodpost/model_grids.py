import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from read_miles import read_miles

class modelGrid:

    def __init__(self,
                 n,
                 lmd_min,
                 lmd_max,
                 npars,
                 par_dims,
                 par_lims,
                 par_mask_function,
                 normalise_models,
                 X=None):
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
        if X==None:
            self.X = self.get_X(self.lmd[:, np.newaxis], self.pars)
        if normalise_models:
            self.normalise()
        self.override_ticks = False

    def add_constant(self, c):
        self.X += c

    def normalise(self):
        self.X = self.X/self.delta_lmd/np.sum(self.X, 0)

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

class LineSpectrum(modelGrid):

    def __init__(self,
                 n=100,
                 lmd_min=-3,
                 lmd_max=3.,
                 par_lims=((-1,1),),
                 par_dims=(30,),
                 par_mask_function=None,
                 normalise_models=True,
                 n_lines=5,
                 sig_lines=0.1,
                 mu_lines=None,
                 s0=None,
                 s1=None,
                 s2=None):
        npars = 1
        s = ("1) z, controls relative line strengths\n")
        self.par_string = s
        pltsym = ['$z$']
        self.par_pltsym = pltsym
        self.n_lines = n_lines
        self.sig_lines = sig_lines
        if mu_lines is None:
            self.mu_lines = np.random.uniform(lmd_min, lmd_max, n_lines)
        else:
            self.mu_lines = np.array(mu_lines)
        if s0 is None:
            self.s0 = np.random.uniform(-1, 1, n_lines)
        else:
            self.s0 = np.array(s0)
        if s1 is None:
            self.s1 = np.random.uniform(-1, 1, n_lines)
        else:
            self.s1 = np.array(s1)
        if s2 is None:
            self.s2 = np.random.uniform(-1, 1, n_lines)
        else:
            self.s2 = np.array(s1)
        super().__init__(n,
                         lmd_min,
                         lmd_max,
                         npars,
                         par_dims,
                         par_lims,
                         par_mask_function,
                         normalise_models)

    def get_linestrengths(self, z):
        return self.s0 + self.s1*z + self.s2*z**2.

    def get_X(self, lmd, pars):
        z = pars[0, :]
        nrm = stats.norm(loc=self.mu_lines, scale=self.sig_lines)
        lmd = np.broadcast_to(lmd, (self.n_lines,)+lmd.shape )
        lines = nrm.pdf(lmd.T)
        s = self.get_linestrengths(z[:, np.newaxis, np.newaxis])
        lines = lines * s
        X = np.sum(lines, -1)
        return X.T

class ContinuumLineSpectrum(modelGrid):

    def __init__(self,
                 n=100,
                 lmd_min=-3,
                 lmd_max=3.,
                 par_lims=((-1,1), (-1,1)),
                 par_dims=(30, 20),
                 par_mask_function=None,
                 normalise_models=True,
                 n_lines=5,
                 sig_lines=0.1):
        npars = 2
        s = ("1) z, controls relative line strengths\n",
             "2) c_slope, linear slope of the continuum")
        self.par_string = s
        pltsym = ['$z$', '$c_\mathrm{slope}$']
        self.par_pltsym = pltsym
        self.n_lines = n_lines
        self.sig_lines = sig_lines
        self.mu_lines = np.random.uniform(lmd_min, lmd_max, n_lines)
        self.s0 = np.random.uniform(-1, 1, n_lines)
        self.s1 = np.random.uniform(-1, 1, n_lines)
        self.s2 = np.random.uniform(-1, 1, n_lines)
        super().__init__(n,
                         lmd_min,
                         lmd_max,
                         npars,
                         par_dims,
                         par_lims,
                         par_mask_function,
                         normalise_models)

    def get_linestrengths(self, z):
        return self.s0 + self.s1*z + self.s1*z**2.

    def get_X(self, lmd, pars):
        z = pars[0, :]
        c_slope = pars[1, :]
        self.c_slope = c_slope
        nrm = stats.norm(loc=self.mu_lines, scale=self.sig_lines)
        tmp = np.broadcast_to(lmd, (self.n_lines,)+lmd.shape)
        lines = nrm.pdf(tmp.T)
        strengths = self.get_linestrengths(z[:, np.newaxis, np.newaxis])
        lines = lines * strengths
        tmp = np.broadcast_to(lmd, (len(c_slope),)+lmd.shape)
        continuum = (c_slope*(tmp**2.).T).T
        continuum = np.squeeze(continuum)
        X = continuum + np.sum(lines, -1)
        return X.T

class CenteredNormalVarSigmaFlux(modelGrid):

    def __init__(self,
                 n=300,
                 lmd_min=-3,
                 lmd_max=3.,
                 par_lims=((0,1),(-5,0)),
                 par_dims=(30,50),
                 par_mask_function=None,
                 normalise_models=True):
        npars = 2
        s = ("1) $\\alpha$ = line flux",
             "2) $\\beta$ =  log($\\sigma^2$)")
        self.par_string = s
        pltsym = ['$\\alpha$', '$\\beta$']
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
        alpha = pars[0, :]
        beta = pars[1, :]
        dlmd = self.delta_lmd
        N = stats.norm(loc=0., scale=np.exp(beta)**0.5)
        line = N.cdf(lmd+dlmd/2.) - N.cdf(lmd-dlmd/2.)
        return alpha * line

class MilesSSP(modelGrid):

    def __init__(self,
                 miles_mod_directory='MILES_BASTI_CH_baseFe',
                 n=None,
                 lmd_min=None,
                 lmd_max=None,
                 age_lim=None,
                 z_lim=None,
                 thin_age=1,
                 thin_z=1,
                 normalise_models=True):

        ssps = read_miles.MilesSSPs(mod_dir=miles_mod_directory,
                                    age_lim=age_lim,
                                    z_lim=z_lim,
                                    thin_age=thin_age,
                                    thin_z=thin_z)
        if n is None:
            n = ssps.X.shape[0]
        else:
            ssps.interpolate_wavelength(n_pixels=n,
                                        lmd_min=lmd_min,
                                        lmd_max=lmd_max)
        self.ssps = ssps
        npars = 2
        s = ("1) metallicity of stellar population",
             "2) age of stellar population")
        self.par_string = s
        pltsym = ['Z [M/H]', 'Age [Gyr]']
        self.par_pltsym = pltsym
        self.n = n
        self.lmd_min = ssps.lmd[0]
        self.lmd_max = ssps.lmd[-1]
        self.lmd = ssps.lmd
        self.delta_lmd = (self.lmd_max-self.lmd_min)/(n-1)
        self.npars = 2
        self.par_dims = (ssps.nz, ssps.nt)
        self.par_lims = ((0,1),(0,1))
        self.par_mask_function = None
        self.check_par_input()
        self.get_par_grid()
        p = self.pars.shape[-1]
        self.p = p
        self.min_n_p = np.min([n, p])
        self.X = ssps.X
        # remaining pars from modgrid
        if normalise_models:
            self.normalise()
        self.override_ticks = False

    def set_tick_parameters(self,
                            n_z_ticks=5,
                            n_t_ticks=5,
                            n_round=2):
        # set parameters for adding custom tickmarks on plots
        self.override_ticks = True
        # z ticks
        fill_value = (np.min(self.ssps.z_unq), np.max(self.ssps.z_unq))
        self.y_ticks = np.linspace(0, 1, n_z_ticks)
        interpolator = interp1d(self.par_cents[0],
                                self.ssps.z_unq,
                                bounds_error=False,
                                fill_value=fill_value)
        y_tick_labs = interpolator(self.y_ticks)
        y_tick_labs = np.round(y_tick_labs, n_round)
        self.y_tick_labs = y_tick_labs
        # t ticks
        fill_value = (np.min(self.ssps.t_unq), np.max(self.ssps.t_unq))
        self.x_ticks = np.linspace(0, 1, n_t_ticks)
        interpolator = interp1d(self.par_cents[1],
                                self.ssps.t_unq,
                                bounds_error=False,
                                fill_value=fill_value)
        x_tick_labs = interpolator(self.x_ticks)
        x_tick_labs = np.round(x_tick_labs, n_round)
        self.x_tick_labs = x_tick_labs



# end
