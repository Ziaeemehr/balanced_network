import sys
import lib
import nest
from time import time
import numpy as np
import pylab as pl
import matplotlib.gridspec as gridspec
sys.argv.append('--quiet')

t_start = time()

I_e = 370.0
t_sim = 1000.0
t_trans = 100.0
tau_syn_in = 2.0
mean_noise = 0.0
std_noise = 20.0
noise_weight = 1.0
neuron_model = 'iaf_psc_alpha'
dt = 0.1

TO_NPZ = False

nest.ResetKernel()
nest.set_verbosity('M_WARNING')

nest.SetKernelStatus(
    {'resolution': dt,
        "data_path": '../data',
        "overwrite_files": True,
        "print_time": False,
     })


# building -----------------------------------------------#
neuron = nest.Create(neuron_model)
nest.SetStatus(
    neuron, {'I_e': I_e,
             "tau_syn_in": tau_syn_in})

spikedetector = nest.Create("spike_detector",
                            params={"withgid": True,
                                    "withtime": True})
multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime": True,
                            "record_from": ["V_m"]})

noise = nest.Create('noise_generator')
nest.SetStatus(noise, [{'mean': mean_noise,
                        'std': std_noise}])

# connecting ----------------------------------------------#
nest.Connect(neuron, spikedetector)
syn_dict = {'weight': noise_weight}
nest.Connect(noise, neuron, syn_spec=syn_dict)
nest.Connect(multimeter, neuron)


# running ------------------------------------------------#
# nest.Simulate(t_trans)
# nest.SetStatus(spikedetector, {'n_events': 0})
nest.Simulate(t_sim)

ts, gids = lib.get_spike_times(spikedetector)

if TO_NPZ:
    subname = str('%.3f-%.3f' % (weight_coupling, std_noise))
    np.savez("../data/npz/"+"spk-"+subname, ts=ts, gids=gids)

if len(ts) > 10:
    t_isi = lib.isi(ts, gids)
    # print t_isi
    print "std of isi : ", np.std(t_isi)
    print "fano factor of isi : ", lib.fano_factor(t_isi)

lib.display_time(time()-t_start)


# Extracting and plotting data from devices
dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
Vts = dmm["events"]["times"]


fig, ax = pl.subplots(2, figsize=(5, 4), sharex=True)

fig = pl.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(4, 1, hspace=0.0)
ax = []
ax.append(fig.add_subplot(gs1[:3]))
ax.append(fig.add_subplot(gs1[3]))

ax[0].plot(Vms, lw=1, c='k')
ax[1].plot(ts, gids, 'ko')
ax[0].set_xlim(0, t_sim)
ax[1].set_xlim(0, t_sim)
pl.savefig("../data/fig/single", dpi=600)
# pl.show()
