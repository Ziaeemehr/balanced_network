import os
import lib
import numpy as np
import pylab as pl
import networkx as nx
from sys import exit
from time import time
try:
    from threading import Thread
    from joblib import Parallel, delayed
except:
    print "Error importing joblib"
    exit(0)

# ---------------------------------------------------------#


def run_command(arg):
    k, w, mu, sig, tau, d, step = arg
    command = "python main.py %g %g %g %g %g %g %d" % (
        k, w, mu, sig, tau, d, step)
    os.system(command)
# ---------------------------------------------------------#


def batch_run():
    arg = []
    nc = len(weight_coupling)
    ns = len(std_noise)

    for i in range(nc):
        for j in range(ns):
            for t in range(len(tau_syn_in)):
                for d in range(len(delay)):
                    arg.append([
                        weight_coupling[i],
                        noise_weight,
                        mean_noise,
                        std_noise[j],
                        tau_syn_in[t],
                        delay[d],
                        vol_step
                    ])

    Parallel(n_jobs=8)(
        map(delayed(run_command), arg))
# ---------------------------------------------------------#


N = 100
p = 1.0
t_trans = 1000.0
t_sim = 2000.0
dt = 0.1
I_e = 370.0
weight_coupling = -1.0 * np.arange(0.0, 1.1, .1)
std_noise = np.arange(5, 25, 5)
tau_syn_in = np.arange(0.5, 10, 0.5)
delay = [0.5]
noise_weight = 1.0
mean_noise = 0.0
num_sim = 1
nthreads = 1
vol_step = 5

PLOT = False

# ---------------------------------------------------------#


if __name__ == "__main__":
    seed = 1256
    start = time()

    adj = lib.make_er_graph(N, p, seed)
    np.savetxt("dat/C.txt", adj, fmt="%d")

    batch_run()
    lib.display_time(time()-start)
