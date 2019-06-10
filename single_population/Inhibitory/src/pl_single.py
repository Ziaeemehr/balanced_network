import os
import lib
import sys
import numpy as np
import pl as ll
import pylab as pl
from single_neuron import params

os.chdir('../data/')


def plot_dispersion_indices(I, sigma):

    c = np.load("npz/data.npz")
    I = c['I']
    sigma = c['std_noise']
    stdv = c['stdv']
    fano = c['fano']
    n_spikes = c['n_spikes']


    ll.plot_phase_space(stdv.T, I, sigma, name='stdv',
                        xtickstep=xtickstep, ytickstep=ytickstep,
                        xlabel=r"Input current ($pA$)",
                        ylabel='noise amplitude')
    ll.plot_phase_space(fano.T, I, sigma, name='fano',
                        xtickstep=xtickstep, ytickstep=ytickstep,
                        xlabel=r"Input current ($pA$)",
                        ylabel='noise amplitude')

    ll.plot_phase_space(n_spikes.T, I, sigma, name='n_spikes',
                        xtickstep=xtickstep, ytickstep=ytickstep,
                        xlabel=r"Input current ($pA$)",
                        ylabel='noise amplitude')

if __name__ == "__main__":
    xtickstep = 10
    ytickstep = 10
    plot_dispersion_indices(params['I_e'], params['std_noise'])
