import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class Plotter:

    def __init__(self,
                 modgrid=None,
                 data=None,
                 beta_smp=None,
                 df=None,
                 percentiles=np.array([5, 50]),
                 col_data='C0',
                 col_true='k',
                 col_post='C1'):
        # set values
        self.modgrid = modgrid
        if self.modgrid is not None:
            if self.modgrid.npars==2:
                plim = self.modgrid.par_lims
                plim_rng = np.array(plim)[:,1] - np.array(plim)[:,0]
                plim_ratio = 1.*plim_rng[1]/plim_rng[0]
                pdim_ratio = 1.*self.modgrid.par_dims[1]/self.modgrid.par_dims[0]
                aspect = plim_ratio / pdim_ratio
            elif self.modgrid.npars==1:
                plim = self.modgrid.par_lims
                plim_rng = plim[0][1] - plim[0][0]
                plim_ratio = 1./plim_rng
                pdim_ratio = 1./self.modgrid.par_dims[0]
                aspect = pdim_ratio/plim_ratio
            else:
                pass
        self.imshow_aspect = aspect
        self.data = data
        self.beta_smp = beta_smp
        self.df = df
        self.n_pc = len(percentiles)
        self.pc = percentiles
        self.pc_even_tailed = np.concatenate((percentiles/2.,
                                              100-percentiles/2.))

    def plot_modgrid(self, mod_cmap=plt.cm.viridis):
        # get colors
        models = self.modgrid.X.T
        v = [np.array([self.modgrid.lmd, model]).T for model in models]
        cols = np.linspace(0, 1, self.modgrid.p)
        lc = LineCollection(np.array(v),
                            colors=mod_cmap(cols))
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
        # plot models
        ax[0].add_collection(lc)
        if self.modgrid.npars==2:
            mod_key_arr = self.modgrid.reshape_beta(cols)
            plim = self.modgrid.par_lims
            extent = np.concatenate(plim[::-1])
        elif self.modgrid.npars==1:
            mod_key_arr = np.array([cols])
            plim = self.modgrid.par_lims
            extent = np.concatenate((plim[0], [0,1]))
        else:
            raise ValueError('plot_modgrid assumes 1D or 2D model grid')
        ax[1].imshow(mod_key_arr,
                     cmap=mod_cmap,
                     extent=extent,
                     aspect=self.imshow_aspect)
        # format ax0
        minx, maxx = np.min(self.modgrid.lmd), np.max(self.modgrid.lmd)
        miny, maxy = np.min(self.modgrid.X), np.max(self.modgrid.X)
        ax[0].set_xlim(minx, maxx)
        ax[0].set_ylim(miny, maxy)
        ax[0].set_ylabel('Models')
        ax[0].set_xlabel('$\lambda$')
        # format ax1
        if self.modgrid.npars==2:
            ax[1].set_xlabel(self.modgrid.par_pltsym[1])
            ax[1].set_ylabel(self.modgrid.par_pltsym[0])
        else:
            ax[1].set_xlabel(self.modgrid.par_pltsym[0])
            ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()

    def plot_data(self, plot_true_ybar=True):
        fig, ax = plt.subplots(1, 1, figsize=(7,3.5))
        # plot data
        ax.plot(self.data.lmd, self.data.y, '.', ms=1, label='$y$')
        if plot_true_ybar:
            if hasattr(self.data, 'ybar'):
                ax.plot(self.data.lmd,
                        self.data.ybar,
                        '-k',
                        label='true $\\bar{y}$')
        # format
        ax.legend()
        ax.set_ylabel('Data')
        ax.set_xlabel('$\lambda$')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)

    def plot_data_model(self, plot_true_ybar=True):
        fig, ax = plt.subplots(2, 1, figsize=(7,7), sharex=True)
        # plot data
        ax[0].plot(self.data.lmd, self.data.y, '.', ms=1, label='data')
        # plot P(\bar{y} | y)
        y_rec_smp = np.dot(self.modgrid.X, self.beta_smp.T)
        y_rec_lims = np.percentile(y_rec_smp, self.pc_even_tailed, axis=1)
        ax[0].plot(self.data.lmd, y_rec_lims.T, color='C1')
        for i in range(self.n_pc):
            ax[0].fill_between(self.data.lmd,
                               y_rec_lims[0,:],
                               y_rec_lims[3,:],
                               color='C1',
                               alpha=0.3)
            ax[0].fill_between(self.data.lmd,
                               y_rec_lims[1,:],
                               y_rec_lims[2,:],
                               color='C1',
                               alpha=0.3)
        ax[0].plot([], [], lw=5, alpha=0.5, color='C1', label='$model$')
        # plot truth
        if plot_true_ybar:
            if hasattr(self.data, 'ybar'):
                ax[0].plot(self.data.lmd,
                           self.data.ybar,
                           '-k',
                           label='true')
        # plot residuals relative to median recovered ybar
        y_rec_med = np.percentile(y_rec_smp, 50., axis=1)
        ax[1].plot(self.data.lmd, self.data.y - y_rec_med, '.', ms=3)
        ax[1].fill_between(self.data.lmd,
                           y_rec_lims[0,:] - y_rec_med,
                           y_rec_lims[3,:] - y_rec_med,
                           color='C1',
                           alpha=0.3)
        ax[1].fill_between(self.data.lmd,
                           y_rec_lims[1,:] - y_rec_med,
                           y_rec_lims[2,:] - y_rec_med,
                           color='C1',
                           alpha=0.3)
        ax[1].plot(self.data.lmd, (y_rec_lims  - y_rec_med).T, color='C1')
        ax[1].plot(self.data.lmd, self.data.ybar - y_rec_med, '-k')
        # format
        ax[0].legend()
        ax[0].set_ylabel('Data')
        ax[1].set_ylabel('Residuals')
        ax[1].set_xlabel('$\lambda$')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)

    def plot_df(self, plot_true_df=True, cmap_df=plt.cm.viridis):

        if self.modgrid.npars==1:
            if plot_true_df:
                fig, ax = plt.subplots(2, 1, figsize=(7,7), sharex=True)
            else:
                fig, ax0 = plt.subplots(1, 1, figsize=(7,3.5), sharex=True)
                ax = [ax0]
            beta_lims = np.percentile(self.beta_smp,
                                      self.pc_even_tailed,
                                      axis=0)
            ax[0].fill_between(self.modgrid.par_cents[0],
                               beta_lims[1,:],
                               beta_lims[2,:],
                               color='C1',
                               alpha=0.3)
            ax[0].fill_between(self.modgrid.par_cents[0],
                               beta_lims[0,:],
                               beta_lims[3,:],
                               color='C1',
                               alpha=0.3)
            ax[0].plot([], [], lw=5, alpha=0.5, color='C1', label='$model$')
            if plot_true_df:
                ax[0].plot(self.modgrid.par_cents[0],
                           self.df.F,
                           '-k',
                           label='true')
                beta_med = np.percentile(self.beta_smp, 50, axis=0)
                ax[1].plot(self.modgrid.par_cents[0],
                           self.df.F - beta_med,
                           '-k')
                ax[1].fill_between(self.modgrid.par_cents[0],
                                   beta_lims[1,:] - beta_med,
                                   beta_lims[2,:] - beta_med,
                                   color='C1',
                                   alpha=0.3)
                ax[1].fill_between(self.modgrid.par_cents[0],
                                   beta_lims[0,:] - beta_med,
                                   beta_lims[3,:] - beta_med,
                                   color='C1',
                                   alpha=0.3)
            ax[0].legend()
            ax[0].set_ylabel('Distribution Function')
            if plot_true_df:
                ax[1].set_ylabel('Residuals')
                ax[1].set_xlabel(self.modgrid.par_pltsym[0])
            else:
                ax[0].set_xlabel(self.modgrid.par_pltsym[0])
            fig.tight_layout()
            fig.subplots_adjust(hspace=0)

        if self.modgrid.npars==2:

            fig = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)

            ax[0]




            pass
