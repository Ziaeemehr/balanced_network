import lib
import numpy as np 
import pylab as pl 
from main import params

path = "../data/text/"


hist = True
n = 50
esel = range(n)
isel = range(params['NE'], params['NE']+n)
g = params['g']
eta = params['eta']
nthreads = params['nthreads']
hist_binwidth = params['hist_binwidth']
xlim=[params['t_trans'], params['t_sim']]

subname = str("%.3f-%.3f"%(g, eta))

efname =  [str("%sE-%s-%d.gdf"%(path,subname, i)) 
    for i in range(nthreads)]
ifname =  [str("%sI-%s-%d.gdf"%(path,subname, i)) 
    for i in range(nthreads)]

lib.raster_plot_from_file(efname, hist, hist_binwidth, xlim, esel)
pl.savefig("../data/fig/E-"+subname+".png")

pl.figure()
lib.raster_plot_from_file(ifname, hist, hist_binwidth, xlim, isel)
pl.savefig("../data/fig/I-"+subname+".png")

# pl.close()
