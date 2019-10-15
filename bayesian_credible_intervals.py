import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import linmodpost as lmp

# Example of two different bayesian credible intervals (bcis)
# i.e. an interval containing a specified probability mass of a distribution
# 1) equal-tailed BCI
#    has equal probability either side of the interval
# 2) highest-density BCI
#    the smallest (contiguous) interval containing the specified probability

# sample values from four different beta distributions
n_smp = 10000
a = np.array([2., 2., 10., 1])
b = np.array([2., 5., 1., 1.5])
samples = stats.beta.rvs(a, b, size=(n_smp,) + a.shape)
samples = lmp.samplers.Samples(samples)

# calculate equal-tailed and highest-density BCIs containing probabilities p_in
p_in = [0.68, 0.95, 0.997]
et_bci = samples.even_tailed_bcis(p_in=p_in)
hd_bci = samples.highest_density_bcis(p_in=p_in)

# plot results
kw_hist = {'bins':50, 'histtype':'step', 'density':True}
kw_text = {'size':20, 'va':'center', 'ha':'center'}

fig, ax = plt.subplots(4, 2, figsize=(6, 8), sharex='col')
ax = ax.T
for i in range(4):
    ax[0,i].hist(samples.x[:,i], **kw_hist)
    ax[1,i].hist(samples.x[:,i], **kw_hist)
    for j in range(3):
        ax[0,i].axvspan(et_bci[j, 0, i], et_bci[j, 1, i], alpha=0.2)
        ax[1,i].axvspan(hd_bci[j, 0, i], hd_bci[j, 1, i], alpha=0.2)
    for j in range(2):
        ax[j,i].get_yaxis().set_visible(False)
        ax[j,i].get_xaxis().set_visible(False)
fig.text(0.25, 0.95, 'Equal Tailed', size=20, va='center', ha='center')
fig.text(0.75, 0.95, 'Highest Density', size=20, va='center', ha='center')
fig.tight_layout()
fig.subplots_adjust(top=0.9)
plt.savefig('../plots/bayesian_credible_intervals.png')
plt.close()
