import os
import lib
import numpy as np
import pylab as pl
from sys import exit
from main import params, g, eta
from scipy.signal import welch
from time import time

path = "../data/"
os.chdir(path)
start = time()

PLOT_HIST   = False
RHYTHM      = True
PLOT_RHYTHM = True
PLOT_R      = False
PLOT_FREQ   = False

# parameter for raster plots
n = 50
esel = range(n)
isel = range(params['NE'], params['NE']+n)
# ----------------------------------------------#
hist_binwidth = params['hist_binwidth']
xlim = [params['t_sim']-200, params['t_sim']]
fwhm = params['fwhm']
fs = 1000.0/params['dt']
labels = ['E', 'I']

eRate = []
iRate = []
len_g = len(g)
len_eta = len(eta)
freq = np.zeros((len_g, len_eta))

for i in range(len_g):
    erate = []
    irate = []
    for j in range(len_eta):
        subname = str("%.3f-%.3f" % (g[i], eta[j]))

        efname = str("E-%s.npz" % subname)
        ifname = str("I-%s.npz" % subname)
        E = np.load("npz/"+efname)
        I = np.load("npz/"+ifname)

        pop = [E, I]

        erate.append(E['rate'])
        irate.append(I['rate'])

        if PLOT_HIST:
            for p in range(2):
                ts = pop[p]['t']
                gids = pop[p]['gid']
                if len(ts):
                    lib.raster_plot_from_data( ts, gids, hist, hist_binwidth, xlim, esel)
                    pl.savefig("fig/"+labels[p]+subname+".png")
                    pl.close()
                else:
                    print "no plot for %s at g= %g, eta= %g" % (labels[p], g[i], eta[j])

        if RHYTHM:
            if PLOT_RHYTHM:
                fig0, ax = pl.subplots(4, figsize=(10, 10))
                pl.subplots_adjust(hspace=.4)
            
            for p in range(2):
                ts = pop[p]['t']
                gids = pop[p]['gid']
                
                if len(ts):
                    t_bins, heights = lib.calculate_histogram(ts, gids, hist_binwidth)
                    t_bins, heights = lib.calculate_histogram_(ts, gids, hist_binwidth)

                    filtered = lib.filter_gaussian(heights, fwhm)
                    # print len(filtered)
                    f, P = welch(filtered, fs=fs)

                    freq[i,j]= f[np.argmax(P)]

                    if PLOT_RHYTHM:
                        lib.plot_rhythms_from_file(
                            t_bins, heights, filtered, ax[p], xlim, 
                            str("%s, rate=%.1f" % (labels[p], pop[p]['rate'])))
                
                        lib.plot_power_spectrum(f, P, ax[p+2])
                
            if PLOT_RHYTHM:
                pl.savefig('fig/r-'+subname+".png")
                pl.close()
            

    eRate.append(erate)
    iRate.append(irate)

iRate = np.asarray(iRate)
eRate = np.asarray(eRate)
np.savez("npz/workspace", 
    freq=freq,
    iRate=iRate, 
    eRate=eRate)

if PLOT_FREQ:
    lib.imshow_plot(freq, fname="fig/freq.png")

if PLOT_R:
    lib.plot_R(eRate, eta, g, "Erate.png",
               xlabel=r"$\eta$", xtickstep=5,
               ylabel="g", ytickstep=5, title="Exc-rate")

    lib.plot_R(iRate, eta, g, "Irate.png",
               xlabel=r"$\eta$", xtickstep=5,
               ylabel="g", ytickstep=5, title="Inh-rate")

lib.display_time(time()-start)