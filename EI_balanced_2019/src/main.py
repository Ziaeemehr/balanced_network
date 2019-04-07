# Abolfazl Ziaeemehr
# Institute for Advanced Studies in
# Basic Sciences (IASBS)
# tel: +98 3315 2148
# github.com/ziaeemehr

import sys
sys.argv.append('--quiet')
import numpy as np
import pylab as pl
import networkx as nx
from sys import exit, argv
from lib import *
from time import time 

#---------------------------------------------------------#
order  = 2500
params = {
    "t_sim"      : 1000.0,
    "t_trans"    : 50.0,
    "NE"         : 4 * order,
    "NI"         : 1 * order,
    "I_e"        : 0.0,
    "dt"         : 0.1,
    "delay"      : 1.0,
    "j_exc_exc"  : 0.33,       # EE connection strength
    "j_exc_inh"  : 1.5,        # EI connection strength
    "j_inh_exc"  : -6.2,       # IE connection strength
    "j_inh_inh"  : -12.0,      # II connection strength
    "epsilonEE"  : 0.15,       # EE connection probability
    "epsilonIE"  : 0.2,        # IE connection probability
    "epsilonEI"  : 0.2,        # EI connection probability
    "epsilonII"  : 0.2,        # II connection probability
    "tau_syn_ex" : 2.0,
    "tau_syn_in" : 2.0,
    "poiss_to_exc_w" : 5.0,  # weight
    "poiss_to_inh_w" : 2.0,  # weigth
    "poiss_rate_exc" : 8000.0,
    "poiss_rate_inh" : 6300.0,
    "vol_step" : 1,
    "nthreads" : 4,
    "num_sim"  : 1,
    "N_rec_exc" : 4 * order,
    "N_rec_inh" : 1 * order,
    "neuron_model" : "iaf_psc_alpha"
}


start = time()

sol = iaf_neuron()
sol.set_params( **params)

sol.run(params['t_sim'], params['t_trans'])
sol.visualize([0, params['t_sim']+ params['t_trans']])


display_time(time()-start)

# a1, a2 = sol.measure_synchrony(calculate_vol_syn=True)
# vol_syn[ens], spike_syn[ens] = a1, a2
# t_isi.append(sol.t_isi)
# moments.append(sol.moments)


# np.savez('../data/npz/'+ofname,
#          vol_syn=vol_syn, spike_syn=spike_syn, g=node_coupling,
#          sigma=std_noise, delay=delay, tau_syn_ex=tau_syn_ex,
#          t_isi=t_isi, moments_isi=moments)

# if PLOT:
#     sol.visualize([0.0, t_sim+t_trans-10.0])

# print '%15.5f %15.5f %15.5f %15.5f' % (
#     node_coupling, std_noise, vol_syn, spike_syn)

