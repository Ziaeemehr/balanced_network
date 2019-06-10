# Abolfazl Ziaeemehr
# Institute for Advanced Studies in
# Basic Sciences (IASBS)
# tel: +98 3315 2148
# github.com/ziaeemehr

from time import time
from run import N, I_e, dt, t_sim, t_trans
from run import num_sim, nthreads
from lib import iaf_neuron
from sys import exit, argv
import networkx as nx
import pylab as pl
import numpy as np
import sys
sys.argv.append('--quiet')

# ---------------------------------------------------------#

node_coupling = float(argv[1])
noise_weight = float(argv[2])
mean_noise = float(argv[3])
std_noise = float(argv[4])
tau_syn_ex = float(argv[5])
delay = float(argv[6])

vol_step = 1
adj = np.loadtxt('dat/C.dat', dtype=int)

params = {
    'N': N,
    'adj': adj,
    'I_e': I_e,
    'delay': delay,
    't_sim': t_sim,
    't_trans': t_trans,
    'num_sim': num_sim,
    'nthreads': nthreads,
    'vol_step': vol_step,
    'std_noise': std_noise,
    'tau_syn_ex': tau_syn_ex,
    'mean_noise': mean_noise,
    'noise_weight': noise_weight,
    'weight_coupling': node_coupling,
    'neuron_model': 'iaf_psc_alpha',
}

# ---------------------------------------------------------#

ofname = str('par-%.6f-%.6f-%.6f-%.6f' % (
    node_coupling, std_noise, tau_syn_ex, delay))

for ens in range(num_sim):
    sol = iaf_neuron(dt, nthreads)
    sol.set_params(**params)
    sol.run()


# print '%15.5f %15.5f %15.5f %15.5f' % (
#     node_coupling, std_noise, vol_syn, spike_syn)
