import os
import lib
import sys
import numpy as np
import pl as ll
import pylab as pl
from single_neuron import params, I_e, std_noise

os.chdir('../data/')


def plot_dispersion_indices(I, sigma):

    c = np.load("npz/data.npz")
    I = c['I']
    sigma = c['std_noise']
    stdv = c['stdv']
    fano = c['fano']
    n_spikes = c['n_spikes']
    rate = n_spikes/float(params['t_sim'] - params['t_trans'])*1000.0

    ll.plot_phase_space(stdv.T, I, sigma, name='stdv',
                        xtickstep=xtickstep, ytickstep=ytickstep,
                        xlabel=r"Input current ($pA$)",
                        ylabel='noise amplitude',
                        title="std")
    ll.plot_phase_space(fano.T, I, sigma, name='fano',
                        xtickstep=xtickstep, ytickstep=ytickstep,
                        xlabel=r"Input current ($pA$)",
                        ylabel='noise amplitude',
                        title='Fano factor')

    ll.plot_phase_space(rate.T, I, sigma, name='n_spikes',
                        xtickstep=xtickstep, ytickstep=ytickstep,
                        xlabel=r"Input current ($pA$)",
                        ylabel='noise amplitude',
                        title="spike rate")


if __name__ == "__main__":
    xtickstep = 25
    ytickstep = 25
    plot_dispersion_indices(params['I_e'], params['std_noise'])

    ll.plot_bunch_rasters_activity(I_e[::2], std_noise[::2], xlim=(14000, 20000))
