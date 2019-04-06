
# Here are the parameters for the iaf_cond_alpha neuron model.

​poiss_rate_ex = 8000.0            
poiss_rate_in  = p_rate_ex - 1700.0

j_exc_exc = 0.33      # EE connection strength
j_exc_inh = 1.5        # EI connection strength
j_inh_exc = -6.2       # IE connection strength
j_inh_inh = -12.0       # II connection strength

epsilonEE   = 0.15      # EE connection probability
epsilonIE   = 0.2       # IE connection probability
epsilonEI   = 0.2       # EI connection probability
epsilonII   = 0.2       # II connection probability

neuron_params= {'V_th' :-54.,
                'V_reset'   :-70.,
                'tau_syn_ex': 1.0,
                'tau_syn_in': 1.}

poiss_to_exc_pop = 0.25   #weight
poiss_to_inh_pop  = 0.4   #weight

# These are parameters I am using in my own simulations. Hope you find them helpful in your work too. Please let me know if something is missing.

# For another example, you can use Brunel (2000) paper as:
g = 5.0  # ratio inhibitory weight/excitatory weight
eta = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability

tauSyn = 0.5  # synaptic time constant in ms
tauMem = 20.0  # time constant of membrane potential in ms
CMem = 250.0  # capacitance of membrane in in pF
theta = 20.0  # membrane threshold potential in mV
neuron_params = {"C_m": CMem,
                 "tau_m": tauMem,
                 "tau_syn_ex": tauSyn,
                 "tau_syn_in": tauSyn,
                 "t_ref": 2.0,
                 "E_L": 0.0,
                 "V_reset": 0.0,
                 "V_m": 0.0,
                 "V_th": theta}
J = 0.1        # postsynaptic amplitude in mV
J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
J_ex = J / J_unit  # amplitude of excitatory postsynaptic current
J_in = -g * J_ex    # amplitude of inhibitory postsynaptic current

nu_th = (theta * CMem) / (J_ex * CE * exp(1) * tauMem * tauSyn)
nu_ex = eta * nu_th
p_rate = 1000.0 * nu_ex * CE

# To simulate a network of iaf_psc_alpha neurons.