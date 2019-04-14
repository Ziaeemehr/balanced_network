# brunel_delta_nest.py

# Abolfazl Ziaeemehr
# Institute for Advanced Studies in
# Basic Sciences (IASBS)
# tel: +98 3315 2148
# github.com/ziaeemehr
# a.ziaeemehr@gmail.com


import nest 
import nest.raster_plot
import pylab as pl 
from time import time 
import numpy 
import os
from lib import Brunel


order  = 250
params = {
    "nthreads"   : 4,
    "dt"         : 0.1,
    "t_sim"      : 200.0,
    "t_trans"    : 50.0,
    "NE"         : 4 * order,
    "NI"         : 1 * order,
    "delay"      : 1.0,
    "epsilon"    : 0.1,
    "eta"        : 2.0,
    "g"          : 5.0,
    "tau_m"      : 20.0,
    "V_th"       : 20.0,
    "V_m"        : 0.0,
    "V_reset"    : 0.0,
    "E_L"        : 0.0,
    "t_ref"      : 2.0,
    "C_m"        : 1.0,
    "J"          : 0.1,
    "I_e"        : 0.0,
    "neuron_model" : "iaf_psc_delta",
}

params["N_rec_E"] = 50 #params['NE']
params["N_rec_I"] = 50 #params['NI']
    
sol = Brunel(params['dt'], params['nthreads'])
sol.set_params(**params)
sol.run(params['t_sim'])
sol.visualize([0, params['t_sim']])
