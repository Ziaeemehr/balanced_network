import numpy as np
import pylab as pl
pl.switch_backend('agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from run import *
from sys import exit
import os


if not os.path.exists("../data/fig"):
    os.makedirs("../data/fig")

os.chdir('../data/')

nc = len(node_coupling)
ns = len(std_noise)


for t in range(len(tau_syn_ex)):
    for d in range(len(delay)):
        vol_syn, st_vol_syn = np.zeros(ns), np.zeros(ns)
        spike_syn, st_spike_syn = np.zeros(ns), np.zeros(ns)
        mean, st_mean = np.zeros(ns), np.zeros(ns)
        var, st_var = np.zeros(ns), np.zeros(ns)
        stdv, st_stdv = np.zeros(ns), np.zeros(ns)
        skw, st_skw = np.zeros(ns), np.zeros(ns)
        kurt, st_kurt = np.zeros(ns), np.zeros(ns)

        for j in range(ns):

            ifname = str('npz/par-%.6f-%.6f-%.6f-%.6f' % 
            (node_coupling[0], std_noise[j], tau_syn_ex[t], delay[d]))
            C = np.load(ifname+'.npz')
            if num_sim > 1:
                moments = np.mean(C['moments_isi'], axis=0)
                st_moments = np.std(C['moments_isi'], axis=0)

                spike_syn[j],st_spike_syn[j] = np.mean(C['spike_syn']), np.std(C['spike_syn'])
                vol_syn[j]  ,st_vol_syn      = np.mean(C['vol_syn'])  , np.std(C['vol_syn'])
                mean[j] ,st_mean[j] = moments[0], st_moments[0]
                var[j]  ,st_var[j]  = moments[1], st_moments[1]
                stdv[j] ,st_stdv[j] = moments[2], st_moments[2]
                skw[j]  ,st_skw[j]  = moments[3], st_moments[3]
                kurt[j] ,st_kurt[j] = moments[4], st_moments[4]
            else:
                print 'something was wrong'
                exit(0)


colors = ['k', 'maroon', "crimson", "y", 'royalblue', 'magenta']

fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(12,8))
pl.autoscale(enable=True, axis='y', tight=True)
x = std_noise
ax[0, 0].errorbar(x, vol_syn, yerr=st_vol_syn, fmt='-o', c=colors[0], elinewidth=1, ecolor='g')
ax[1,0].errorbar(x, spike_syn, yerr=st_spike_syn, fmt='-o', c=colors[1], elinewidth=1, ecolor='g')
ax[0,1].errorbar(x, mean, yerr=st_mean, fmt='-o', c=colors[2], elinewidth=1, ecolor='g')
ax[0,2].errorbar(x, skw, yerr=st_skw, fmt='-o', c=colors[3], elinewidth=1, ecolor='g')
ax[1,1].errorbar(x, kurt, yerr=st_kurt, fmt='-o', c=colors[4], elinewidth=1, ecolor='g')
# ax[1,1].errorbar(x, var, yerr=st_var, fmt='--o')
ax[1,2].errorbar(x, stdv, yerr=st_stdv, fmt='-o', c=colors[5], elinewidth=1, ecolor='g')
ax[0,0].set_ylabel('voltage synchrony')
ax[1,0].set_ylabel('spike synchrony')
ax[0,1].set_ylabel('mean')
ax[0,2].set_ylabel('skewness')
ax[1,1].set_ylabel('kurtosis')
ax[1,2].set_ylabel("std")
ax[1,2].set_ylim(0,100)
for i in range(2):
    for j in range(3):
        ax[i, j].set_xlabel(r'$\sigma$', fontsize=20)
        ax[i,j].set_xticks(np.arange(x[6],x[-1],1))
pl.tight_layout()
fig.savefig('fig/f.pdf')
# pl.show()


