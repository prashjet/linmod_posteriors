import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . samples import Samples

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side
    from a prescribed midpoint value) e.g.
    im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-10, vmax=10))
    taken from e.g. https://github.com/matplotlib/matplotlib/issues/14892
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

class Plotter:

    def __init__(self,
                 modgrid=None,
                 data=None,
                 beta_smp=None,
                 df=None,
                 credible_interval_type='even_tailed',
                 p_in_credible_intervals=np.array([0.5, 0.95]),
                 imshow_aspect='square_pixels',
                 kw_mod_plot={},
                 kw_data_plot={},
                 kw_true_plot={},
                 kw_post1d_plot={},
                 kw_post2d_plot={}):
        # set input values
        self.modgrid = modgrid
        self.data = data
        self.beta_smp = beta_smp
        self.df = df
        # set bayesian credible interval parameters
        if credible_interval_type=='even_tailed':
            self.get_bcis = 'even_tailed_bcis'
        elif credible_interval_type=='highest_density':
            self.get_bcis = 'highest_density_bcis'
        else:
            raise ValueError('Unknown credible_interval_type')
        self.set_credible_interval_levels(p_in_credible_intervals)
        # set/update plot formatting keywords
        self.kw_mod_plot = {'cmap':plt.cm.viridis}
        self.kw_mod_plot.update(kw_mod_plot)
        self.kw_data_plot = {'color':'C0', 'marker':'.', 'ls':'None', 'ms':3}
        self.kw_data_plot.update(kw_data_plot)
        self.kw_true_plot = {'color':'k', 'ls':'-'}
        self.kw_true_plot.update(kw_true_plot)
        self.kw_post1d_plot = {'color':'C1', 'alpha':0.3}
        self.kw_post1d_plot.update(kw_post1d_plot)
        self.kw_post2d_plot={'cmap':plt.cm.cividis}
        self.kw_post2d_plot.update(kw_post2d_plot)
        # set aspect ratio for calls to imshow
        self.kw_resid_plot = self.kw_post2d_plot.copy()
        self.kw_resid_plot.update({'cmap':plt.cm.bwr,
                                   'norm':MidpointNormalize(midpoint=0)})
        self.configure_imshow(imshow_aspect=imshow_aspect)

    def set_credible_interval_levels(self, p_in_credible_intervals):
        self.p_in = np.array(p_in_credible_intervals)
        self.n_bcis = len(p_in_credible_intervals)

    def override_ticks(self, ax):
        if self.modgrid.override_ticks:
            ax.set_xticks(self.modgrid.x_ticks)
            ax.set_xticklabels(self.modgrid.x_tick_labs)
            ax.set_yticks(self.modgrid.y_ticks)
            ax.set_yticklabels(self.modgrid.y_tick_labs)

    def configure_imshow(self,
                         imshow_aspect='square_pixels',
                         interpolation='none'):
        par_dims = self.modgrid.par_dims
        if imshow_aspect=='square_pixels':
            # aspect ratio for square pixels in imshow calls
            if self.modgrid is not None:
                if self.modgrid.npars==2:
                    plim = self.modgrid.par_lims
                    plim_rng = np.array(plim)[:,1] - np.array(plim)[:,0]
                    plim_ratio = 1.*plim_rng[1]/plim_rng[0]
                    pdim_ratio = 1.*par_dims[1]/par_dims[0]
                    aspect = plim_ratio/pdim_ratio
                elif self.modgrid.npars==1:
                    plim = self.modgrid.par_lims
                    plim_rng = plim[0][1] - plim[0][0]
                    plim_ratio = 1./plim_rng
                    pdim_ratio = 1./par_dims[0]
                    aspect = pdim_ratio/plim_ratio
                else:
                    pass
        elif imshow_aspect=='square_axes':
            # aspect ratio for square pixels in imshow calls
            if self.modgrid is not None:
                if self.modgrid.npars==2:
                    plim = self.modgrid.par_lims
                    plim_rng = np.array(plim)[:,1] - np.array(plim)[:,0]
                    plim_ratio = 1.*plim_rng[1]/plim_rng[0]
                    aspect = plim_ratio
                elif self.modgrid.npars==1:
                    plim = self.modgrid.par_lims
                    plim_rng = plim[0][1] - plim[0][0]
                    plim_ratio = 1./plim_rng
                    aspect = plim_ratio
                else:
                    pass
        elif imshow_aspect=='auto':
            # auto: fill the axes
            aspect = 'auto'
        elif isinstance(imshow_aspect, (int, float)):
            aspect = float(imshow_aspect)
        else:
            raise ValueError('unknown imshow_aspect')
        self.imshow_aspect = aspect
        self.kw_post2d_plot.update({'aspect':aspect})
        self.kw_mod_plot.update({'aspect':aspect})
        self.kw_resid_plot.update({'aspect':aspect})
        # set imshow extent
        if self.modgrid.npars==1:
            plim = self.modgrid.par_lims
            extent = np.concatenate((plim[0], [0,1]))
        elif self.modgrid.npars==2:
            plim = self.modgrid.par_lims
            extent = np.concatenate(plim[::-1])
        else:
            pass
        self.kw_post2d_plot.update({'extent':extent})
        self.kw_mod_plot.update({'extent':extent})
        self.kw_resid_plot.update({'extent':extent})
        # set interpolation
        self.kw_post2d_plot.update({'interpolation':interpolation})
        self.kw_mod_plot.update({'interpolation':interpolation})
        self.kw_resid_plot.update({'interpolation':interpolation})
        # set origin
        self.kw_post2d_plot.update({'origin':'lower'})
        self.kw_mod_plot.update({'origin':'lower'})
        self.kw_resid_plot.update({'origin':'lower'})

    def plot_modgrid(self, y_offset=0):
        # get colors
        models = self.modgrid.X.T
        v = [np.array([self.modgrid.lmd, models[i]+i*y_offset]).T for i in range(self.modgrid.p)]
        cols = np.linspace(0, 1, self.modgrid.p)
        if self.modgrid.npars==1:
            mod_key_arr = np.array([cols])
        elif self.modgrid.npars==2:
            mod_key_arr = self.modgrid.reshape_beta(cols)
        cmap = self.kw_mod_plot.pop('cmap')
        lc = LineCollection(np.array(v),
                            colors=cmap(cols))
        self.kw_mod_plot.update({'cmap':cmap})
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
        # plot models
        ax[0].add_collection(lc)
        ax[1].imshow(mod_key_arr,
                     **self.kw_mod_plot)
        # format ax0
        minx, maxx = np.min(self.modgrid.lmd), np.max(self.modgrid.lmd)
        miny, maxy = np.min(self.modgrid.X), np.max(self.modgrid.X)
        ax[0].set_xlim(minx, maxx)
        ax[0].set_ylim(miny, maxy)
        ax[0].set_ylabel('Flux')
        ax[0].set_xlabel('$\lambda$ [angstrom]')
        # format ax1
        if self.modgrid.npars==2:
            ax[1].set_xlabel(self.modgrid.par_pltsym[1])
            ax[1].set_ylabel(self.modgrid.par_pltsym[0])
        else:
            ax[1].set_xlabel(self.modgrid.par_pltsym[0])
            ax[1].get_yaxis().set_visible(False)
        self.override_ticks(ax[1])
        fig.tight_layout()
        return

    def plot_data(self, plot_true_ybar=True):
        fig, ax = plt.subplots(1, 1, figsize=(7,3.5))
        # plot data
        ax.plot(self.data.lmd, self.data.y, label='$y$', **self.kw_data_plot)
        if plot_true_ybar:
            if hasattr(self.data, 'ybar'):
                ax.plot(self.data.lmd,
                        self.data.ybar,
                        label='true $\\bar{y}$',
                        **self.kw_true_plot)
        # format
        ax.legend()
        ax.set_ylabel('Data')
        ax.set_xlabel('$\lambda$ [angstrom]')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        return

    def plot_dummy_model1d(self,
                           ax=None,
                           label='model',
                           lw=6):
        ax.plot([], [], lw=5, label=label, **self.kw_post1d_plot)

    def plot_data_model(self, plot_true=True):
        fig, ax = plt.subplots(2, 1, figsize=(7,7), sharex=True)
        # plot data
        ax[0].plot(self.data.lmd,
                   self.data.y,
                   label='data',
                   **self.kw_data_plot)
        # plot P(\bar{y} | y)
        y_rec_smp = np.dot(self.modgrid.X, self.beta_smp.x.T).T
        y_rec_smp = Samples(y_rec_smp)
        y_rec_bcis = y_rec_smp.__getattribute__(self.get_bcis)(self.p_in)
        for i in range(self.n_bcis):
            ax[0].fill_between(self.data.lmd,
                               y_rec_bcis[i, 0, :],
                               y_rec_bcis[i, 1, :],
                               **self.kw_post1d_plot)
        self.plot_dummy_model1d(ax=ax[0], label='model')
        # plot truth
        if plot_true:
            if hasattr(self.data, 'ybar'):
                ax[0].plot(self.data.lmd,
                           self.data.ybar,
                           label='true',
                           **self.kw_true_plot)
        # plot residuals relative to median recovered ybar
        y_rec_med = y_rec_smp.get_median()
        ax[1].plot(self.data.lmd,
                   self.data.y - y_rec_med,
                   **self.kw_data_plot)
        for i in range(self.n_bcis):
            ax[1].fill_between(self.data.lmd,
                               y_rec_bcis[i, 0, :] - y_rec_med,
                               y_rec_bcis[i, 1, :] - y_rec_med,
                               **self.kw_post1d_plot)
        if plot_true:
            if hasattr(self.data, 'ybar'):
                ax[1].plot(self.data.lmd,
                           self.data.ybar - y_rec_med,
                           **self.kw_true_plot)
        # format
        ax[0].legend()
        ax[0].set_ylabel('Data')
        ax[1].set_ylabel('Residuals')
        ax[1].set_xlabel('$\lambda$ [angstrom]')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        return fig

    def plot_df_1d(self,
                   plot_true=True):
        if self.modgrid.npars!=1:
            raise ValueError('Model not 1D')
        if plot_true:
            # plot truth, posterior + residuals below
            fig, ax = plt.subplots(2, 1, figsize=(7,7), sharex=True)
        else:
            # plot posterior
            fig, ax0 = plt.subplots(1, 1, figsize=(7,3.5), sharex=True)
            ax = [ax0]
        beta_bcis = self.beta_smp.__getattribute__(self.get_bcis)(self.p_in)
        for i in range(self.n_bcis):
            ax[0].fill_between(self.modgrid.par_cents[0],
                               beta_bcis[i, 0, :],
                               beta_bcis[i, 1, :],
                               **self.kw_post1d_plot)
        self.plot_dummy_model1d(ax=ax[0], label='model')
        if plot_true:
            ax[0].plot(self.modgrid.par_cents[0],
                       self.df.F,
                       label='true',
                       **self.kw_true_plot)
            beta_med = self.beta_smp.get_median()
            ax[1].plot(self.modgrid.par_cents[0],
                       self.df.F - beta_med,
                       '-k')
            for i in range(self.n_bcis):
                ax[1].fill_between(self.modgrid.par_cents[0],
                                   beta_bcis[i, 0, :] - beta_med,
                                   beta_bcis[i, 1, :] - beta_med,
                                   **self.kw_post1d_plot)
        ax[0].legend()
        ax[0].set_ylabel('Distribution Function')
        if plot_true:
            ax[1].set_ylabel('Residuals')
            ax[1].set_xlabel(self.modgrid.par_pltsym[0])
        else:
            ax[0].set_xlabel(self.modgrid.par_pltsym[0])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        return fig

    def plot_df_2d(self,
                   plot_true=True,
                   plot_posterior=True,
                   posterior_view='median',
                   df_clim='joint_rng',
                   resid_type='significance',
                   resid_clim='auto',
                   resid_nsmp_min=30):
        """Plot 2D distribution functions as images

        Parameters
        ----------
        plot_true : Boolean
            whether to plot the true distribution function
        plot_posterior : Boolean
            whether to plot the posterior of the distribution function
        posterior_view : string
            how to plot the posterior, choices:
            1) median
            2) mode
            3) MAP
        df_clim : string, float, or array shape (2,). Options:
            when plotting true and posterior of the DF, scale colorbar to:
            1) 'true_rng' i.e. min/max of the true distribution
            2) 'posterior_rng' i.e. min/max of the posterior_view
            3) 'joint_rng' i.e. joint min/max of the true/posterior_view
            4) float vlim, then clim=(-vlim, vlim)
            5) array (vmin, vmax)
        resid_type : string or float. Options:
            1) 'diff' raw differences between posterior view and truth
            2) 'RMSE_normed' - differences normalised by RMSE i.e. root mean
			squared error (relative to central panel)
            3) 'MAD_normed' - differences normalised by MAD i.e. median absolute
			deviation (relative to central panel)
            4) 'significance' - in gaussian sigma
        resid_clim : string, float or array shape (2,). Options:
            1) 'auto', i.e. symmetric, maximum = max(abs(residual))
            2) float vlim, then clim=(-vlim, vlim)
            3) array (vmin, vmax)
        resid_nsmp_min : int
            if resid_type='significance', this is minimum number of samples to
            consider when calculting significances (see Samples class)
        """
        if self.modgrid.npars!=2:
            raise ValueError('Model not 2D')
        # make figure
        if (plot_true and plot_posterior):
            # plot truth, posterior, residuals
            fig, ax = plt.subplots(1, 3,
                                   figsize=(7.5, 3.5),
                                   sharey=True)
        elif (plot_true or plot_posterior):
            # plot either truth or posterior
            fig, ax0 = plt.subplots(1, 1, figsize=(3, 3.5), sharex=True)
            ax = [ax0]
        else:
            raise ValueError('One of plot_true or plot_posterior must be True')
        # get posterior image
        if plot_posterior:
            if posterior_view=='median':
                post_view = self.beta_smp.get_median()
                title = 'median model'
            elif posterior_view=='mode':
                post_view = self.beta_smp.get_halfsample_mode()
                title = 'modal model'
            elif posterior_view=='MAP':
                post_view = self.beta_smp.get_MAP()
                title = 'MAP model'
            else:
                raise ValueError('Unknown posterior_view')
        # get df colorlims
        if df_clim=='true_rng':
            vmin, vmax = np.min(self.df.F), np.max(self.df.F)
        elif df_clim=='posterior_rng':
            vmin, vmax = np.min(post_view), np.max(post_view)
        elif df_clim=='joint_rng':
            vmin = np.min(np.concatenate((self.df.F.ravel(), post_view)))
            vmax = np.max(np.concatenate((self.df.F.ravel(), post_view)))
        elif isinstance(df_clim, (float,int)):
            vmin = -1.*df_clim
            vmax = 1.*df_clim
        elif isinstance(df_clim, (np.ndarray,list,tuple)):
            vmin = df_clim[0]
            vmax = df_clim[1]
        else:
            raise ValueError('Unknown df_clim')
        if (plot_true and plot_posterior):
            if resid_type=='diff':
                resid_img = post_view - self.df.F.ravel()
                cbar_lab = 'mod - true'
            elif resid_type=='RMSE_normed':
                resid_img = post_view - self.df.F.ravel()
                squared_error = (post_view - self.beta_smp.x)**2.
                mean_squared_error = np.mean(squared_error, axis=0)
                rmse = np.sqrt(mean_squared_error)
                resid_img /= rmse
                cbar_lab = '(mod - true)/RMSE'
            elif resid_type=='MAD_normed':
                resid_img = post_view - self.df.F.ravel()
                absolute_deviation = np.abs(post_view - self.beta_smp.x)
                mad = np.median(absolute_deviation, axis=0)
                resid_img /= mad
                cbar_lab = '(mod - true)/MAD'
            elif resid_type=='significance':
                resid_img = self.beta_smp.gaussian_significances_1d(
                            truth=self.df.F.ravel(),
                            center=post_view,
                            resid_nsmp_min=resid_nsmp_min)
                cbar_lab = 'significance [$\\sigma$]'
            else:
                raise ValueError('Unknown resid_type')
        if plot_posterior and plot_true:
            # get df colorlims
            if resid_clim=='auto':
                vlim = np.max(np.abs(resid_img))
                self.kw_resid_plot['norm'].vmin = -vlim
                self.kw_resid_plot['norm'].vmax = vlim
            elif isinstance(resid_clim, (float,int)):
                self.kw_resid_plot['norm'].vmin = -1.0001*resid_clim
                self.kw_resid_plot['norm'].vmax = 1.0001*resid_clim
            elif isinstance(resid_clim, (np.ndarray,list,tuple)):
                self.kw_resid_plot['norm'].vmin = 1.0001*resid_clim[0]
                self.kw_resid_plot['norm'].vmax = 1.0001*resid_clim[1]
            else:
                raise ValueError('Unknown resid_clim')
        # plot images
        ax_cnt = 0
        if plot_true:
            # plot truth
            imdf = ax[ax_cnt].imshow(self.df.F,
                                     **self.kw_post2d_plot,
									 norm=colors.LogNorm(vmin=vmin, vmax=vmax))
            imdf = ax[ax_cnt].imshow(self.df.F,
                                     **self.kw_post2d_plot,
									 norm=colors.LogNorm(vmin=vmin, vmax=vmax))
            ax[ax_cnt].set_title('true')
            ax_cnt += 1
        if plot_posterior:
            post_view = self.modgrid.reshape_beta(post_view)
            post_view[post_view<=0] = np.min(post_view[post_view>0.])
            imdf = ax[ax_cnt].imshow(post_view,
                                     **self.kw_post2d_plot,
									 norm=colors.LogNorm(vmin=vmin, vmax=vmax))
            ax[ax_cnt].set_title(title)
            ax_cnt += 1
        if (plot_true and plot_posterior):
            # plot residuals
            resid_img = self.modgrid.reshape_beta(resid_img)
            imres = ax[ax_cnt].imshow(resid_img,
                                      **self.kw_resid_plot)
            ax[ax_cnt].set_title('residuals')
        # plot format
        for ax0 in ax:
            ax0.set_xlabel(self.modgrid.par_pltsym[1])
            self.override_ticks(ax0)
        ax[0].set_ylabel(self.modgrid.par_pltsym[0])
        fig.subplots_adjust(wspace=0)
        fig.tight_layout()
        # add colorbars
        y0 = ax[0].get_position().y0 - 0.25
        height = 0.05
        if (plot_true and plot_posterior):
            x0 = ax[0].get_position().x0
            tmp = ax[1].get_position()
            right = tmp.x0 + tmp.width
            width = right - x0
            cax_df = fig.add_axes([x0, y0, width, height])
            tmp = ax[2].get_position()
            x0 = tmp.x0
            width = tmp.width
            cax_resid = fig.add_axes([x0, y0, width, height])
        else:
            tmp = ax[0].get_position()
            x0 = tmp.x0
            width = tmp.width
            cax_df = fig.add_axes([x0, y0, width, height])
        cbdf = fig.colorbar(imdf, cax=cax_df, orientation='horizontal')
        cbdf.set_label('weights')
        if (plot_true and plot_posterior):
            cbres = fig.colorbar(imres,
                                 cax=cax_resid,
                                 orientation='horizontal',
                                 extend='both')
            cbres.set_label(cbar_lab)
        return








# end
