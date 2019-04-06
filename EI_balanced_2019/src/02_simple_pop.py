import numpy as np 
import pylab as pl
import nest 
from sys import exit 
from time import time

np.random.seed(12587)

start = time()
nest.set_verbosity("M_WARNING")
dt = 0.01 
N = 10
t_sim = 100.0
I_e = 0.0
tauMem = 20.0
epsilonEE = 1.0
p_rate = 2000.0 #N*50.0
poiss_to_exc_pop = 0.1     # weight
j_exc_exc = 0.5            # EE connection strength
std_exc_exc = 0.0           # std connection strength
delay = 1.0
V_th  = 20.0
C_m   = 1.0
t_ref = 2.0
E_L   = 0.0
V_reset = 0.0
neuron_model = "iaf_psc_alpha"

nthreads = 4
nest.ResetKernel()
nest.SetKernelStatus({'resolution': dt,
                      "local_num_threads": nthreads})

# Create and seed RNGs--------------------------------
msd = 1000  # master seed
n_vp = nest.GetKernelStatus('total_num_virtual_procs')
msdrange1 = range(msd, msd + n_vp)
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)
nest.SetKernelStatus({'grng_seed': msd + n_vp,
                      'rng_seeds': msdrange2})

neuron_params = {"C_m": C_m,
                 "tau_m": tauMem,
                 "t_ref": t_ref,
                 "E_L": E_L,
                 "V_reset": V_reset,
                 "V_m": 0.0,
                 "V_th": V_th}

#################### BUILDING ####################
pop = nest.Create(neuron_model, N, params=neuron_params)

node_info = nest.GetStatus(pop)
local_nodes = [(ni['global_id'], ni['vp'])
                for ni in node_info if ni['local']]
for gid, vp in local_nodes:
    nest.SetStatus(
        [gid], {'V_m': pyrngs[vp].uniform(V_reset, V_th)})

# V_reset = nest.GetDefaults('iaf_psc_exp')['V_reset']
# # V_th = nest.GetDefaults('iaf_psc_exp')['V_th']
# Vms = V_reset+(V_th-V_reset)*np.random.rand(len(pop))
# nest.SetStatus(pop, "V_m", Vms)

# print nest.GetStatus(pop, "V_m")
multimeter = nest.Create("multimeter", N)
nest.SetStatus(multimeter, {"withtime": True,
                            "record_from": ["V_m"]})

spikedetector = nest.Create("spike_detector",
                            params={"withgid": True,
                                    "withtime": True})

noise = nest.Create("poisson_generator", N, params={"rate": p_rate})
# nest.SetStatus(noise, {"rate":p_rate})

#################### CONNECTING ####################

syn_dict = {"model": "static_synapse",
            "weight": {"distribution": "normal", 
            "mu": j_exc_exc, 
            "sigma": std_exc_exc},
            "delay": delay}

conn_dict_EE = {"rule": "pairwise_bernoulli",
                        "p": epsilonEE,
                        "autapses": False,
                        "multapses": False}

nest.CopyModel("static_synapse", "excitatory",
               {"weight": poiss_to_exc_pop,
               "delay": delay})

nest.Connect(pop, pop, conn_dict_EE,syn_spec=syn_dict)
# conn = nest.GetConnections(pop)
# for i in nest.GetStatus(conn, ['source', 'target']):
#     print i
nest.Connect(noise, pop, "one_to_one", syn_spec='excitatory')
nest.Connect(multimeter, pop, "one_to_one")


nest.Connect(pop, spikedetector)
nest.Simulate(t_sim)

print "Done in %g seconds." % (time()-start)
################### PLOTTING ###################
fig, ax = pl.subplots(2, figsize=(12,8), sharex=True)
for i in range(0,N):
    dmm = nest.GetStatus(multimeter)[i]
    Vms = dmm["events"]["V_m"]
    tsv = dmm["events"]["times"]
    ax[0].plot(tsv,Vms, label=i)
t = np.arange(0, t_sim)
ax[0].plot(t, [V_th]*len(t), lw=2, ls="--", c="gray")


dSD = nest.GetStatus(spikedetector, keys='events')[0]
evs = dSD['senders']
ts = dSD["times"]
ax[1].plot(ts,evs,'.',c='k')

# ax[0].legend(frameon=False)

ax[1].set_xlabel("time(ms)", fontsize=20)
ax[0].set_ylabel("V(mV)", fontsize=16)
ax[1].set_ylabel("neuron index", fontsize=16)
ax[0].set_title("%s, I=%.3f pA" % (neuron_model, I_e), fontsize=20)
ax[0].margins(y=0.1)
ax[1].margins(y=0.01)
pl.savefig("../data/fig/02.pdf")
# pl.show()
