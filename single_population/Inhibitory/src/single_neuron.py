import sys
import lib
import nest
from time import time
import numpy as np
import pylab as pl
import matplotlib.gridspec as gridspec
sys.argv.append('--quiet')

t_start = time()


params = {
    'I_e': 370.0,  # np.arange(370, 375, 0.5),
    't_sim': 20000.0,
    't_trans': 1000.0,
    'std_noise': 20.0,  # np.arange(20.0, 30.0, 1.0),
    'tau_syn_ex': 2.0,
    'mean_noise': 0.0,
    'noise_weight': 1.0,
    'neuron_model': 'iaf_psc_alpha',
}

I_e = np.arange(360, 385, 1.0, dtype=float)
std_noise = np.arange(10.0, 50.0, 1.0, dtype=float)


if __name__ == "__main__":

    t_start = time()
    
    dt = 0.1
    n_I = len(I_e)
    n_s = len(std_noise)
    fano_factor = np.zeros((n_I, n_s))
    n_spikes = np.zeros((n_I, n_s))
    stdv = np.zeros((n_I, n_s))

    for i in range(n_I):
        for s in range(n_s):

            print "I = %10.3f  sigma = %10.3f " % (
                I_e[i], std_noise[s]
            )
            params['I_e'] = I_e[i]
            params['std_noise'] = std_noise[s]
            sol = lib.single_iaf_neuron(dt)
            sol.set_params(**params)
            sol.run()
            n_spikes[i, s] = len(sol.ts)
            if len(sol.ts) > 10:
                t_isi = lib.isi(sol.ts, sol.gids)
                fano_factor[i, s] = lib.fano_factor(t_isi)
                stdv[i, s] = np.std(t_isi)
    np.savez("../data/npz/data",
             I=I_e,
             stdv=stdv,
             fano=fano_factor,
             std_noise=std_noise,
             n_spikes=n_spikes)

    lib.display_time(time()- t_start)
