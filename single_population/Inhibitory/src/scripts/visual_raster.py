import nest as nt
import nest
import numpy as np
import pylab as pl
import networkx as nx
from elephant.statistics import isi
from sys import exit
from iaf_alpha import *
from time import time 

seed = 1256
np.random.seed(seed)

N = 200
p = 0.5
G = nx.erdos_renyi_graph(N, p, seed=seed, directed=False)
adj = np.asarray(nx.to_numpy_matrix(G), dtype=int)

I_e = 350.0
dt = 0.1

node_coupling = np.arange(1, 10, 3, dtype=float)
std_noise = np.arange(1, 10, 3, dtype=float)

node_coupling = [3.0]
std_noise = [10.0]

noise_weight = 5.0
mean_noise = 0.0
t_trans = 50.0
t_sim = 2000.0
num_sim = 1
nthreads = 1
vol_step = 1

#---------------------------------------------------------#

if __name__ == "__main__":
    
    nc, ns = len(node_coupling), len(std_noise)
    for i in range(nc):
        for j in range(ns):

            sol = iaf_neuron(adj,N, dt, nthreads)
            sol.set_params( I_e, node_coupling[i], noise_weight, 
                            mean_noise, std_noise[j], num_sim)
            sol.run(t_sim, t_trans)
            vol_syn, burst_syn = sol.analysis(vol_step, True)
            print '%15.5f %15.5f %15.5f %15.5f' % (
                node_coupling[i], std_noise[j], vol_syn, burst_syn)
            sol.visualize([t_trans+t_sim-100.0, t_sim+t_trans])
            sol.visualize([0.0, t_sim+t_trans])
    pl.show()
