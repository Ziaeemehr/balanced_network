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
    k, w, mu, sig, tau, d = arg
    command = "python main.py %g %g %g %g %g %g" % (
        k, w, mu, sig, tau, d)
    os.system(command)
# ---------------------------------------------------------#


def batch_run():
    arg = []
    nc = len(weight_coupling)
    ns = len(std_noise)

    for i in range(nc):
        for j in range(ns):
            for t in range(len(tau_syn_ex)):
                for d in range(len(delay)):
                    arg.append([
                        weight_coupling[i],
                        noise_weight,
                        mean_noise,
                        std_noise[j],
                        tau_syn_ex[t],
                        delay[d],
                    ])

    Parallel(n_jobs=8)(
        map(delayed(run_command), arg))
# ---------------------------------------------------------#


N = 100
p = 1.0
t_trans = 2000.0
t_sim = 2000.0
dt = 0.1
I_e = 370.0
weight_coupling = np.arange(0.2, 4.0, 0.1)
std_noise = np.arange(7, 60, 4)
delay = [1.0]
noise_weight = 1.0
mean_noise = 0.0
num_sim = 1
nthreads = 1
vol_step = 1
tau_syn_ex = [2.0]

PLOT = False

# ---------------------------------------------------------#


if __name__ == "__main__":
    seed = 1256
    start = time()
    lib.make_er_graph(N, p, seed)
    batch_run()
    lib.display_time(time()-start)
