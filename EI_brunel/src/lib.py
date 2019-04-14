import nest 
import nest.raster_plot
import pylab as pl 
from time import time 
import numpy as np
import os

class Brunel(object):
    '''
    Implementation of the sparsely connected random network,
    described by Brunel (2000) J. Comp. Neurosci.
    Parameters are chosen for the asynchronous irregular
    state (AI).
    '''

    data_path = "../data/text/"
    built = False       # True, if build() was called
    connected = False   # True, if connect() was called

    def __init__(self, dt, nthreads):
        self.name = self.__class__.__name__
        nest.ResetKernel()
        # nest.set_verbosity('M_QUIET')

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        nest.SetKernelStatus({
                      "resolution": dt, 
                      "print_time": True,
                      "overwrite_files": True, 
                      "data_path": self.data_path,
                      "local_num_threads": nthreads})
        
        np.random.seed(125472)

        # Create and seed RNGs
        msd = 1000      # master seed
        n_vp = nest.GetKernelStatus('total_num_virtual_procs')
        msdrange1 = range(msd, msd + n_vp)
        self.pyrngs = [np.random.RandomState(s) for s in msdrange1]
        msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)
        nest.SetKernelStatus({'grng_seed': msd + n_vp,
                              'rng_seeds': msdrange2})        

    def set_params(self, **par):

        self.nthreads           = par['nthreads']
        self.t_sim              = par['t_sim']
        self.t_trans            = par['t_trans']
        self.NE                 = par["NE"]
        self.NI                 = par["NI"]
        self.delay              = par["delay"]
        self.epsilon            = par['epsilon']
        self.eta                = par['eta']
        self.g                  = par['g']
        self.N_rec_E            = par['N_rec_E']
        self.N_rec_I            = par['N_rec_I']
        self.tau_m              = par['tau_m']
        self.V_th               = par['V_th']
        self.V_m                = par['V_m']
        self.V_reset            = par['V_reset']
        self.E_L                = par['E_L']
        self.t_ref              = par['t_ref']
        self.C_m                = par['C_m']
        self.J                  = par['J']
        self.I_e                = par["I_e"]
        self.neuron_model       = par['neuron_model']

        
        self.neuron_params = {
            "C_m"       : self.C_m,
            "tau_m"     : self.tau_m,
            "t_ref"     : self.t_ref,
            "E_L"       : self.E_L,
            "V_th"      : self.V_th,
            "V_m"       : self.V_m,
            "I_e"       : self.I_e,
            "V_reset"   : self.V_reset,
        }

        self.N_neurons = self.NE + self.NI  # number of neurons in total
        self.J_ex = self.J                  # amplitude of excitatory postsynaptic potential
        self.J_in = -self.g * self.J_ex     # amplitude of inhibitory postsynaptic potential


    def calibrate(self):
        '''
        Compute all parameter dependent variables of the model.
        '''
        self.CE = int(self.epsilon * self.NE)  # number of excitatory synapses per neuron
        self.CI = int(self.epsilon * self.NI)  # number of inhibitory synapses per neuron
        C_tot = int(self.CI + self.CE)  # total number of synapses per neuron
        nu_th = self.V_th / (self.J * self.CE * self.tau_m)
        nu_ex = self.eta * nu_th
        p_rate = 1000.0 * nu_ex * self.CE


        nest.SetDefaults("iaf_psc_delta", self.neuron_params)
        nest.SetDefaults("poisson_generator", {"rate": p_rate})


    def build(self):
        '''
        Create all nodes, used in the model.
        '''
        if self.built:
            return
        self.calibrate()

        self.nodes_ex = nest.Create("iaf_psc_delta", self.NE)


        self.nodes_in = nest.Create("iaf_psc_delta", self.NI)
        self.noise = nest.Create("poisson_generator")
        self.espikes = nest.Create("spike_detector")
        self.ispikes = nest.Create("spike_detector")

        nest.SetStatus(self.espikes, [{"label": "brunel-py-ex",
                                "withtime": True,
                                "withgid": True,
                                "to_file": True}])

        nest.SetStatus(self.ispikes, [{"label": "brunel-py-in",
                                "withtime": True,
                                "withgid": True,
                                "to_file": True}])
        node_info = nest.GetStatus(self.nodes_ex+self.nodes_in)
        local_nodes = [(ni['global_id'], ni['vp'])
                    for ni in node_info if ni['local']]
        for gid, vp in local_nodes:
            nest.SetStatus([gid], {'V_m': self.pyrngs[vp].uniform(-self.V_th, self.V_th)})
    
        self.built = True
    
    def connect(self):
        '''
        Connect all nodes in the model.
        '''

        if self.connected:
            return 
        if not self.built:
            self.build()

        nest.CopyModel("static_synapse", "excitatory")

        nest.CopyModel("static_synapse", "inhibitory",
                    {"weight": self.J_in, "delay": self.delay})

        nest.CopyModel("static_synapse", "excitatory_input",
                    {"weight": self.J_ex, "delay": self.delay})

        nest.Connect(self.noise, self.nodes_ex, syn_spec="excitatory_input")
        nest.Connect(self.noise, self.nodes_in, syn_spec="excitatory_input")

        nest.Connect(self.nodes_ex[:self.N_rec_E], self.espikes)
        nest.Connect(self.nodes_in[:self.N_rec_I], self.ispikes)

        print "Excitatory connections"

        conn_params_ex = {'rule': 'fixed_indegree', 'indegree': self.CE}

        nest.Connect(self.nodes_ex, self.nodes_ex + self.nodes_in, conn_params_ex,
                    {'model': 'excitatory',
                    'delay': self.delay,
                    'weight': {'distribution': 'uniform',
                                'low': 0.5*self.J_ex,
                                'high': 1.5*self.J_ex}})

        print "Inhibitory connections"
        conn_params_in = {'rule': 'fixed_indegree', 'indegree': self.CI}
        nest.Connect(self.nodes_in, self.nodes_ex + self.nodes_in, conn_params_in, "inhibitory")

        self.connected = True

    def run(self, simtime=300):
        '''
        Simulate the model for simtime milliseconds and print the firing 
        rates of the network during this period.
        '''
        if not self.connected:
            self.connect()
        nest.Simulate(simtime)

        events_ex = nest.GetStatus(self.espikes, "n_events")[0]
        events_in = nest.GetStatus(self.ispikes, "n_events")[0]

        rate_ex = events_ex / simtime * 1000.0 / self.N_rec_E
        rate_in = events_in / simtime * 1000.0 / self.N_rec_I

        num_synapses = (nest.GetDefaults("excitatory")["num_connections"] +
                        nest.GetDefaults("inhibitory")["num_connections"])

        print("Brunel network simulation (Python)")
        print("Number of neurons : {0}".format(self.N_neurons))
        print("Number of synapses: {0}".format(num_synapses))
        print("       Exitatory  : {0}".format(int(self.CE * self.N_neurons) + self.N_neurons))
        print("       Inhibitory : {0}".format(int(self.CI * self.N_neurons)))
        print("Excitatory rate   : %.2f Hz" % rate_ex)
        print("Inhibitory rate   : %.2f Hz" % rate_in)
        


    def visualize(self, tlimits):
        '''
        plot rasterplots
        '''
        
        # NE = self.NE
        # NI = self.NI

        # pl.figure()
        # nest.raster_plot.from_device(
        #     self.espikes, 
        #     hist=True, 
        #     title="Excitatory",
        #     grayscale=False)
        # pl.savefig("../data/fig/E.pdf")
        # pl.close()

        # pl.figure()
        # nest.raster_plot.from_device(
        #     self.spikes_inh,
        #     hist=True,
        #     title="Inhibitory",
        #     grayscale=True)
        # pl.savefig("../data/fig/I.pdf")

        # fig, ax = pl.subplots(1, figsize=(15,10))
        # self.my_raster_plot(self.espikes, ax, 'k' )
        # self.my_raster_plot(self.ispikes, ax, 'r')
        # pl.show()

        # nest.raster_plot.from_device(self.espikes, hist=True)
        # pl.show()
        self.raster_plot_from_device(self.espikes, hist=True)
        pl.savefig("../data/fig/E.pdf")
        # pl.show()

        pl.figure()
        self.raster_plot_from_device(self.ispikes, hist=True)
        pl.savefig("../data/fig/I.pdf")





    
    # -------------------------------------------------------------------#
    def raster_plot_from_device(self, 
        detec, hist=False, hist_binwidth=5.0):

        # ---------------------------------------------------------------#
        def _histogram(a, bins=10, bin_range=None, normed=False):
            from numpy import asarray, iterable, linspace, sort, concatenate

            a = asarray(a).ravel()

            if bin_range is not None:
                mn, mx = bin_range
                if mn > mx:
                    raise ValueError("max must be larger than min in range parameter")

            if not iterable(bins):
                if bin_range is None:
                    bin_range = (a.min(), a.max())
                mn, mx = [mi + 0.0 for mi in bin_range]
                if mn == mx:
                    mn -= 0.5
                    mx += 0.5
                bins = linspace(mn, mx, bins, endpoint=False)
            else:
                if (bins[1:] - bins[:-1] < 0).any():
                    raise ValueError("bins must increase monotonically")

            # best block size probably depends on processor cache size
            block = 65536
            n = sort(a[:block]).searchsorted(bins)
            for i in range(block, a.size, block):
                n += sort(a[i:i + block]).searchsorted(bins)
            n = concatenate([n, [len(a)]])
            n = n[1:] - n[:-1]

            if normed:
                db = bins[1] - bins[0]
                return 1.0 / (a.size * db) * n, bins
            else:
                return n, bins

        # ---------------------------------------------------------------#
        if nest.GetStatus(detec, "to_memory")[0]:
            if not nest.GetStatus(detec)[0]["model"] == "spike_detector":
                raise nest.NESTError("Please provide a spike_detector.")
            
            ev = nest.GetStatus(detec, "events")[0]
            ts = ev["times"]
            gids = ev["senders"]
        
        else:
            raise nest.NESTError("No data to plot. Make sure that to_memory is set.")
        

        if not len(ts):
            raise nest.NESTError("No events recorded!")

        if hist:
            ax1 = pl.axes([0.1, 0.3, 0.85, 0.6])
            plotid = pl.plot(ts, gids, '.')
            pl.ylabel("Time (ms)")
            pl.xticks([])
            xlim = pl.xlim()

            pl.axes([0.1, 0.1, 0.85, 0.17])
            t_bins = np.arange(
                np.amin(ts), 
                np.amax(ts), 
                float(hist_binwidth))
            n, bins = _histogram(ts, bins=t_bins)
            num_neurons = len(np.unique(gids))
            heights = 1000 * n / (hist_binwidth * num_neurons)
            pl.bar(t_bins, heights, width=hist_binwidth, color='royalblue', edgecolor='black')
            pl.yticks([int(x) for x in np.linspace(0.0, int(max(heights) * 1.1) + 5, 4)])
            pl.ylabel("Rate (Hz)")
            pl.xlabel("Time (ms)")
            pl.xlim(xlim)
            pl.axes(ax1)
        else:
            plotid = pl.plot(ts, gids, '.')
            pl.xlabel("Time (ms)")
            pl.ylabel("Neuron ID")
    # -------------------------------------------------------------------#
    def my_raster_plot(self, spike_detector, ax, color):
            dSD = nest.GetStatus(spike_detector, keys='events')[0]
            evs = dSD['senders']
            tsd = dSD["times"]
            ax.plot(tsd, evs, '.', c=color, markersize=3)
            ax.set_xlabel("Time (ms)", fontsize=18)
            ax.set_ylabel("Neuron ID", fontsize=18)
            ax.tick_params(labelsize=18)
            return (evs, tsd)