import numpy as np
import pylab as pl
pl.switch_backend('agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from run import *


if not os.path.exists("../data/fig"):
    os.makedirs("../data/fig")

def plot_phase_space(R, Y, X, name="f"):
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    x_step = X[1] - X[0]
    y_step = Y[1] - Y[0]

    f, ax = pl.subplots(1, figsize=(10, 10))
    im = ax.imshow(R, interpolation='nearest', 
        cmap='afmhot', vmin=0, vmax=1, aspect='auto')
    ax.invert_yaxis()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)
    step = 10
    ax.set_xticks(np.arange(0, len(X), step))
    ax.set_xticklabels(str("%.1f" % i)for i in X[0::step])
    ax.set_yticks(np.arange(0, len(Y), step))
    ax.set_yticklabels(str("%.1f" % i)for i in Y[::step])
    ax.set_xlabel(r"$g$", fontsize=16)
    ax.set_title(name, fontsize=16)
    ax.set_ylabel(r"$\sigma$", fontsize=16)

    pl.savefig('../data/fig/'+name+".png")
    pl.close()

os.chdir('../data/')

nc = len(node_coupling)
ns = len(std_noise)
vol_syn = np.zeros((nc, ns))
burst_syn = np.zeros((nc,ns))

for t in range(len(tau_syn_ex)):
    for d in range(len(delay)):
        for i in range(nc):
            for j in range(ns):
                ifname = str('npz/par-%.6f-%.6f-%.6f-%.6f' % 
                (node_coupling[i], std_noise[j], tau_syn_ex[t], delay[d]))
                C = np.load(ifname+'.npz')
                burst_syn[i,j] = C['burst']
                vol_syn[i,j] = C['vol']
        ofname = str('tau-%.2f-delay-%.2f'%(tau_syn_ex[t], delay[d]))
        np.savez(ofname, 
        vol=vol_syn, 
        burst=burst_syn, 
        g=node_coupling, 
        sigma=std_noise,
        delay=delay[d],
        tau=tau_syn_ex[t])

        plot_phase_space(burst_syn.T, std_noise, node_coupling,'burst-'+ofname)
        plot_phase_space(vol_syn.T, std_noise, node_coupling, 'voltage-'+ofname)
