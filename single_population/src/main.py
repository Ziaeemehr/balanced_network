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

#---------------------------------------------------------#

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
    'node_coupling': node_coupling,
}

#---------------------------------------------------------#

ofname = str('par-%.6f-%.6f-%.6f-%.6f' % (
    node_coupling, std_noise, tau_syn_ex, delay))

vol_syn = np.zeros(num_sim)
spike_syn = np.zeros(num_sim)
t_isi = []
moments = []
for ens in range(num_sim):
    sol = iaf_neuron(dt, nthreads)
    # adj, N, dt, tau_syn_ex, delay, nthreads)
    sol.set_params(**params)
    # I_e, node_coupling, noise_weight,
    #             mean_noise, std_noise, num_sim, vol_step)
    sol.run()
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
