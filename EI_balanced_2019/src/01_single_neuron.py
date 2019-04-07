import numpy as np 
import pylab as pl
import nest 

nest.ResetKernel()
nest.set_verbosity("M_WARNING")

# parameters -------------------------------------------- #
dt = 0.1 
t_sim = 1500.0
p_rate = 1700.0
delay = 1.0
J_ex = 0.1             # amplitude of excitatory postsynaptic potential
tau_m = 20.0           # time constant of membrane potential in ms
V_th = 20.0            # Spike threshold
C_m = 1.0
t_ref = 2.0            # Duration of refractory period
E_L = 0.0              # Resting membrane potential
V_reset = 0.0
V_m = 0.0

neuron_model = "iaf_psc_alpha"
nest.SetKernelStatus({'resolution': dt})

neuron_params = {"C_m": C_m,
                 "tau_m": tau_m,
                 "t_ref": t_ref,
                 "E_L": E_L,
                 "V_reset": V_reset,
                 "V_m": V_m,
                 "V_th": V_th}

neuron = nest.Create(neuron_model, params=neuron_params)

# DEFAULT = nest.GetDefaults("iaf_psc_alpha")
# for i in DEFAULT:
#     print i, "\t", DEFAULT[i]
multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime": True,
                            "record_from": ["V_m"]})
spikedetector = nest.Create("spike_detector",
                            params={"withgid": True,
                                    "withtime": True})

nest.CopyModel("static_synapse", "excitatory",
               {"weight": J_ex, "delay": delay})

noise = nest.Create("poisson_generator", params={"rate":p_rate})
nest.Connect(noise, neuron, syn_spec="excitatory")

nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikedetector)
nest.Simulate(t_sim)


################### PLOTTING ###################
fig, ax = pl.subplots(2, figsize=(12, 6), sharex=True)
dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
ax[0].plot(ts,Vms, lw=2, c='k')
ax[0].plot(ts, [V_th]*len(ts), lw=2, ls="--", c="gray")
dSD = nest.GetStatus(spikedetector, keys='events')[0]
evs = dSD['senders']
ts = dSD["times"]
ax[1].plot(ts,evs,'o', c='r')
ax[1].set_xlabel("time(ms)", fontsize=20)
ax[0].set_ylabel("V(mV)", fontsize=16)
ax[1].set_ylabel("neuron index", fontsize=16)
ax[1].set_yticks([1])
ax[0].set_title("iaf-psc-alpha", fontsize=20)
ax[0].margins(y=0.1)
for i in range(2):
    ax[i].tick_params(labelsize=18)
pl.savefig("../data/fig/01.pdf")
# pl.show()
