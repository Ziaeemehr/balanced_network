import os
import lib
import numpy as np 
import pylab as pl 
from main import params, g, eta
from sys import exit
path = "../data/"
os.chdir(path)


hist = True
n = 50
esel = range(n)
isel = range(params['NE'], params['NE']+n)
nthreads = params['nthreads']
hist_binwidth = params['hist_binwidth']
xlim=[params['t_trans'], params['t_sim']]

for i in g:
    for j in eta:
        subname = str("%.3f-%.3f"%(i, j))

        efname = str("E-%s.npz"%subname)
        ifname = str("I-%s.npz"%subname) 

        print efname
        # exit(0)
        e = np.load("npz/"+efname)

        lib.raster_plot_from_data(e['t'], e['gid'], hist, hist_binwidth, xlim, esel)
        pl.savefig("fig/E-"+subname+".png")

# pl.figure()
# lib.raster_plot_from_file(ifname, hist, hist_binwidth, xlim, isel)
# pl.savefig("../data/fig/I-"+subname+".png")

# pl.close()
