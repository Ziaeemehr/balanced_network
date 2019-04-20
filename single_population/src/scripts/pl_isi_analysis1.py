import numpy as np
import pylab as pl
pl.switch_backend('agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.filters import gaussian_filter1d
from run import *
from sys import exit
import os

if not os.path.exists("../data/fig"):
    os.makedirs("../data/fig")
os.chdir('../data/')

#---------------------------------------#
def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))
#---------------------------------------#
def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))
#---------------------------------------#
nc = len(node_coupling)
ns = len(std_noise)

fig2, ax = pl.subplots(1, figsize=(8, 5))
from cycler import cycler
NUM_COLORS = len(range(0, ns, 3))
cm = pl.get_cmap('gist_rainbow')  # , gist_yarg
ax.set_prop_cycle(cycler('color', [cm(1.*ii/NUM_COLORS)
                    for ii in range(NUM_COLORS)]))


fwhm = 20
sigma = fwhm2sigma(fwhm)

for t in range(len(tau_syn_ex)):
    for d in range(len(delay)):
        for j in range(0,ns,3):
            ifname = str('npz/par-%.6f-%.6f-%.6f-%.6f' %
                         (node_coupling[0], std_noise[j], tau_syn_ex[t], delay[d]))
            C = np.load(ifname+'.npz')
            isi = C["t_isi"]
            hist, bins = np.histogram(isi, bins=150, normed=False)
            center = (bins[:-1] + bins[1:]) / 2
            filtered = gaussian_filter1d(hist, sigma, mode='reflect')
            # ax.plot(center, hist, label=str('%.2f'%std_noise[j]))
            ax.plot(center, filtered, lw=2.5, label=str('%.2f'%std_noise[j]))
ax.set_xlim(0,250)
ax.legend(loc='best', frameon=False, fontsize=12, ncol=2)
ax.set_xlabel("isi (ms)")
ax.set_ylabel("smoothed histogram")
pl.savefig("isi.pdf")
pl.show()



