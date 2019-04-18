# brunel_delta_nest.py

# Abolfazl Ziaeemehr
# Institute for Advanced Studies in
# Basic Sciences (IASBS)
# tel: +98 3315 2148
# github.com/ziaeemehr
# a.ziaeemehr@gmail.com


import os
import lib
import numpy as np
import pylab as pl 
from time import time 


order  = 250
params = {
    "nthreads"   : 8,
    "dt"         : 0.1,
    "t_sim"      : 10000.0,
    "t_trans"    : 5000.0,
    "NE"         : 4 * order,
    "NI"         : 1 * order,
    "delay"      : 1.0,
    "epsilon"    : 0.1,
    "eta"        : 4.0,
    "g"          : 6.0,
    "tau_m"      : 20.0,
    "V_th"       : 20.0,
    "V_m"        : 0.0,
    "V_reset"    : 10.0,
    "E_L"        : 0.0,
    "t_ref"      : 2.0,
    "C_m"        : 1.0,
    "J"          : 0.1,
    "I_e"        : 0.0,
    "fwhm"       : 20,
    "hist_binwidth" : 0.1,
    "neuron_model" : "iaf_psc_delta",
}

params["N_rec_E"] = params['NE']
params["N_rec_I"] = params['NI']

# g   = np.arange( 3, 6, .5 )
# eta = np.arange( 1, 6, 0.5 )

g   = [3.5]#np.arange( 3, 4, .1 )
eta = [1.8]#np.arange( 1, 2, 0.1 )


if __name__ == "__main__":
    
    start = time()
    for i in g:
        for j in eta:
            print "g = %.2f, eta = %.2f" % (i, j)
            params['g'] = i
            params['eta'] = j
            sol = lib.Brunel(params['dt'], params['nthreads'])
            sol.set_params(**params)
            sol.run(params['t_sim'])
            # sol.visualize( 50,
            #     xlim=[params['t_trans'], params['t_sim']], 
            #     rhythm=True, hist=True, 
            #     fwhm=params['fwhm'])
    print '*'*50
    lib.display_time(time()-start)

