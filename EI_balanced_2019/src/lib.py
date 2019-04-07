# Abolfazl Ziaeemehr
# Institute for Advanced Studies in
# Basic Sciences (IASBS)
# tel: +98 3315 2148 
# github.com/ziaeemehr

import nest
import sys
import numpy as np 
import pylab as pl 
from sys import exit 
import networkx as nx
# pl.switch_backend('agg')
from scipy.stats import kurtosis, skew
import matplotlib.gridspec as gridspec
import nest.raster_plot

seed = 1256
np.random.seed(seed)

#-------------------------------------------------------------------#
class iaf_neuron(object):
    '''
    a balanced network of leaky integrate and fire neurons
    '''

    built = False       # True, if build()   was called
    connected = False   # True, if connect() was called    

    tau_m = 20.0
    t_ref = 2.0
    E_L = 0.0
    V_reset = 0.0
    V_m = 0.0
    V_th = 20.0
    C_m = 1.0
    

    def __init__(self):
    
        self.name = self.__class__.__name__
        nest.ResetKernel()
        nest.set_verbosity('M_QUIET') #M_QUIET  M_WARNING
    
    #---------------------------------------------------------------#
    def set_params(self, **par):

        self.NE                 = par["NE"]
        self.NI                 = par["NI"]
        self.delay              = par["delay"]
        self.I_e                = par["I_e"]
        self.dt                 = par['dt']
        self.t_sim              = par["t_sim"]
        self.t_trans            = par["t_trans"]
        self.j_exc_exc          = par["j_exc_exc"]
        self.j_exc_inh          = par['j_exc_inh']
        self.j_inh_exc          = par['j_inh_exc']
        self.j_inh_inh          = par['j_inh_inh']
        self.epsilonEE          = par['epsilonEE']
        self.epsilonIE          = par['epsilonIE']
        self.epsilonEI          = par['epsilonEI']
        self.epsilonII          = par['epsilonII']
        self.tau_syn_ex         = par["tau_syn_ex"]
        self.tau_syn_in         = par["tau_syn_in"]
        self.poiss_to_exc_w     = par['poiss_to_exc_w']
        self.poiss_to_inh_w     = par['poiss_to_inh_w']
        self.poiss_rate_exc     = par['poiss_rate_exc']
        self.poiss_rate_inh     = par['poiss_rate_inh']
        self.N_rec_exc           = par["N_rec_exc"]
        self.N_rec_inh           = par["N_rec_inh"]
        self.num_sim            = par["num_sim"]
        self.nthreads           = par['nthreads']
        self.vol_step           = par['vol_step']
        self.neuron_model       = par['neuron_model']
        
        neuron_params = {
            "C_m"       : self.C_m,
            "tau_m"     : self.tau_m,
            "t_ref"     : self.t_ref,
            "E_L"       : self.E_L,
            "V_th"      : self.V_th,
            "V_m"       : self.V_m,
            "I_e"       : self.I_e,
            "V_reset"   : self.V_reset,
        }

        nest.SetKernelStatus({'resolution': self.dt,
                              "local_num_threads": self.nthreads,
                              "overwrite_files":True,
                              "data_path": "../data/text/"})

        # Create and seed RNGs
        msd = 1000  # master seed
        n_vp = nest.GetKernelStatus('total_num_virtual_procs')
        msdrange1 = range(msd, msd + n_vp)
        self.pyrngs = [np.random.RandomState(s) for s in msdrange1]
        msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)
        nest.SetKernelStatus({'grng_seed': msd + n_vp,
                              'rng_seeds': msdrange2})

        # nest.SetDefaults("poisson_generator", {"rate": p_rate})
        nest.SetDefaults(self.neuron_model, neuron_params)

    #---------------------------------------------------------------#
    def build(self):
        '''
        create nodes and devices used in the model
        '''
        if self.built:
            return 

        print "building ..."
        
        self.nodes_exc = nest.Create(self.neuron_model, self.NE)
        self.nodes_inh = nest.Create(self.neuron_model, self.NI)        

        self.poiss_gen_exc = nest.Create("poisson_generator", 
                                        self.NE, params={'rate': self.poiss_to_exc_w})
        self.poiss_gen_inh = nest.Create("poisson_generator",
                                        self.NI, params={'rate': self.poiss_to_inh_w})
        self.shared_poiss_gen_exc = nest.Create("poisson_generator", 1,
                                        params={'rate': self.poiss_to_exc_w})

        # print nest.GetStatus(self.nodes_exc, "V_th") 

        # def initial_v(nodes, n):
        #     Vm = self.V_reset+(self.V_th-self.V_reset)*np.random.rand(n)
        #     nest.SetStatus(nodes, "V_m", Vm)

        # if self.num_sim == 1:
        #     np.random.seed(1256)
        #     initial_v(self.nodes_exc, self.NE)
        #     initial_v(self.nodes_inh, self.NI)

        # else:
        #     initial_v(self.nodes_exc, self.NE)
        #     initial_v(self.nodes_inh, self.NI)

        node_info = nest.GetStatus(self.nodes_exc+self.nodes_inh)
        local_nodes = [(ni['global_id'], ni['vp'])
                       for ni in node_info if ni['local']]
        for gid, vp in local_nodes:
            nest.SetStatus(
                [gid], {'V_m': self.pyrngs[vp].uniform(self.V_reset, self.V_th)})
        
            
        # print nest.GetStatus(self.nodes_exc, "V_m")

        # self.multimeters = nest.Create("multimeter", self.N)
        # nest.SetStatus(self.multimeters, 
        #                 {"withtime": True, 
        #                 "interval": (self.vol_step *self.dt),
        #                 "record_from": ["V_m"]})


        self.spikes_exc = nest.Create("spike_detector")
        self.spikes_inh = nest.Create("spike_detector")

        nest.SetStatus(self.spikes_exc, [{"label": "ex",
                                       "withtime": True,
                                       "withgid": True,
                                       "to_file": True}])  # "to_file": True

        nest.SetStatus(self.spikes_inh, [{"label": "in",
                                       "withtime": True,
                                       "withgid": True,
                                       "to_file": True}])

        self.built = True 
    
    #---------------------------------------------------------------#
    def connect(self):
        '''
        make connecttions between nodes and devices.
        '''

        if self.connected:
            return 
        if not self.built:
            self.build()
        
        print "connecting ..."

        # nest.CopyModel("static_synapse", "poiss_to_exc_w",
        #                {"weight": self.poiss_to_exc_w, "delay": self.delay})
        # nest.CopyModel("static_synapse", "inhibitory",
        #                {"weight": self.J_in, "delay": self.delay})

        conn_dict_EE = {"rule": "pairwise_bernoulli",
                        "p": self.epsilonEE,
                        "autapses": False,
                        "multapses": False}
        conn_dict_EI = {"rule": "pairwise_bernoulli",
                        "p": self.epsilonEI,
                        "autapses": False,
                        "multapses": False}
        conn_dict_IE = {"rule": "pairwise_bernoulli",
                        "p": self.epsilonIE,
                        "autapses": False,
                        "multapses": False}
        conn_dict_II = {"rule": "pairwise_bernoulli",
                        "p": self.epsilonII,
                        "autapses": False,
                        "multapses": False}

        nest.Connect(
            self.nodes_exc, 
            self.nodes_exc, 
            conn_dict_EE, 
            syn_spec={"weight":self.j_exc_exc, "delay":self.delay})


        nest.Connect(
            self.nodes_exc, 
            self.nodes_inh, 
            conn_dict_EI, 
            syn_spec={"weight":self.j_exc_inh, "delay":self.delay})

        nest.Connect(
            self.nodes_inh, 
            self.nodes_exc,
            conn_dict_IE, 
            syn_spec={"weight": self.j_inh_exc, "delay": self.delay})

        nest.Connect(
            self.nodes_inh,
            self.nodes_inh,
            conn_dict_II, 
            syn_spec={"weight": self.j_inh_inh, "delay": self.delay})

        nest.Connect(
            self.poiss_gen_exc, 
            self.nodes_exc, 
            "one_to_one", 
            syn_spec={"weight": self.poiss_to_exc_w})
        nest.Connect(
            self.poiss_gen_inh, 
            self.nodes_inh, 
            "one_to_one", 
            syn_spec={"weight": self.poiss_to_inh_w})
        
        nest.Connect(
            self.shared_poiss_gen_exc,
            self.nodes_exc,
            syn_spec={'weight':self.poiss_to_exc_w})

        # nest.Connect(self.multimeters, self.neurons, 'one_to_one')

        nest.Connect(self.nodes_exc[:self.N_rec_exc], self.spikes_exc)
        nest.Connect(self.nodes_inh[:self.N_rec_inh], self.spikes_inh)

        self.connected = True

    #---------------------------------------------------------------#
    def run(self, t_sim=1000.0, t_trans = 500.0):
        
        if not self.built:
            self.build()
        if not self.connected:
            self.connect()
        
        self.t_trans = t_trans
        self.t_sim = t_sim

        nest.Simulate(t_trans+t_sim)
        # nest.SetStatus(self.spikedetectors, {'n_events': 0})
        # nest.Simulate(simtime)

    #---------------------------------------------------------------#
    def visualize(self, tlimits):
        '''
        plot the results
        '''
        NE = self.NE
        NI = self.NI
        t_sim = self.t_sim
        t_trans = self.t_trans
        t_total = t_sim+t_trans

        pl.figure()
        nest.raster_plot.from_device(
            self.spikes_exc, 
            hist=True, 
            title="Excitatory",
            grayscale=True)
        pl.savefig("../data/fig/E.pdf")
        # pl.close()
        pl.figure()
        nest.raster_plot.from_device(
            self.spikes_inh,
            hist=True,
            title="Inhibitory",
            grayscale=True)
        pl.savefig("../data/fig/I.pdf")

        events_ex = nest.GetStatus(self.spikes_exc, "n_events")[0]
        events_in = nest.GetStatus(self.spikes_inh, "n_events")[0]
        
        rate_ex = events_ex / t_total * 1000.0 / self.N_rec_exc
        rate_in = events_in / t_total * 1000.0 / self.N_rec_inh
        print("Excitatory rate   : %.2f Hz" % rate_ex)
        print("Inhibitory rate   : %.2f Hz" % rate_in)
        # pl.show()

        '''
        tmin = tlimits[0]
        tmax = tlimits[1]

        fig = pl.figure(figsize=(12, 7))
        gs1 = gridspec.GridSpec(4, 1, hspace=.5)
        axs = []
        axs.append(fig.add_subplot(gs1[0:3, 0]))
        axs.append(fig.add_subplot(gs1[3, 0]))
        
        def raster_plot(spike_detector, ax, color):
            dSD = nest.GetStatus(spike_detector, keys='events')[0]
            evs = dSD['senders']
            tsd = dSD["times"]
            ax.plot(tsd, evs, color, markersize=3)
            return (evs, tsd)

        events_in, tsdi = raster_plot(self.spikes_inh, axs[0], 'r.')
        events_ex, tsde = raster_plot(self.spikes_exc, axs[0], 'k.')


        # self.t_isi = []
        # for i in range(N):
        #     indices = [j for j, x in enumerate(evs) if x == i]
        #     spike_i = tsd[indices]
        #     t1 = calculate_isi(spike_i)
        #     self.t_isi.extend(t1)


    
        # Hist = 0
        # bin_width=5.0 #ms
        # nbin = int((self.t_trans+self.t_sim)/bin_width)
        # def histogram(EVS, TSD, N, n_bin):
        #     Hist = 0
        #     for i in range(N):
        #             indices = [j for j, x in enumerate(EVS) if x == i]
        #             spike_i = TSD[indices]
        #             hist, bins = np.histogram(spike_i, bins=n_bin)
        #             Hist = Hist + hist
        #     return Hist , bins   

        # Hist_e, bins = histogram(events_ex, tsde, NE, nbin)
        # # Hist_i, _ = histogram(events_in, tsdi, NI, nbin)
        # Hist = Hist_e# + Hist_i
        # Hist = Hist/float(NE)

        # for i in range(NE):
        #     indices = [j for j, x in enumerate(events_ex) if x == i]
        #     spike_i = tsde[indices]
        #     hist, bins = np.histogram(spike_i, bins=nbin)  # bins=20
        #     Hist += hist
        # width = (bins[1] - bins[0])
        # center = (bins[:-1] + bins[1:]) / 2.0
        
        # for i in range(NI):
        #     indices = [j for j, x in enumerate(events_in) if x == i]
        #     spike_i = tsdi[indices]
        #     hist, bins = np.histogram(spike_i, bins=nbin)  # bins=20
        #     Hist += hist
        
        # axs[1].bar(center, Hist/float(NE+NI), align='center', width=width)
        # axs[1].set_xlim(0, t_trans+t_sim)

        # print center


        axs[0].set_ylabel('Neuron ID, Ex, In', fontsize=16)
        axs[0].set_xlabel("time(ms)", fontsize=13)
        axs[0].set_xlim(tmin, tmax)
        axs[0].set_ylim(-1, NE+NI+1)
        fig.savefig('../data/fig/F.pdf')
        pl.close()

        pl.figure()
        nest.raster_plot.from_device(self.spikes_exc)
        # nest.raster_plot.from_device(self.spikes_exc)
        pl.show()

    #     # axs[2].set_yticks([0,N/2, N])
    #     fname = str('%.6f-%.6f' % (self.node_coupling, self.std_noise))
        # if N > 10:
        #     for i in range(0, N, 20):
        #         dmm = nest.GetStatus(self.multimeters)[i]

        #         Vms = dmm["events"]["V_m"]
        #         tsv = dmm["events"]["times"]
        #         axs[0].plot(tsv, Vms, lw=0.5)
        '''
#---------------------------------------------------------------#
def display_time(time):
    hour = int(time/3600)
    minute = (int(time % 3600))/60
    second = time-(3600.*hour+60.*minute)
    print "Done in %d hours %d minutes %.6f seconds" \
        % (hour, minute, second)
#---------------------------------------------------------------#
def calculate_isi(x):
    isi = np.diff(x)
    return isi

    # def measure_synchrony(self, calculate_vol_syn=False):
    #     '''
    #     calculate the coherency from spike trains and voltages
    #     '''
    #     N = self.N
    #     spike_syn = vol_syn = 0.0 
        
    #     dSD = nest.GetStatus(self.spikedetectors, keys='events')[0]
    #     evs = dSD['senders']
    #     tsd = dSD["times"]

    #     # claculate interspike interval for each neuron
    #     self.t_isi = []
    #     for i in range(N):
    #         indices = [j for j, x in enumerate(evs) if x == i]
    #         spike_i = tsd[indices]
    #         t1 = calculate_isi(spike_i)
    #         self.t_isi.extend(t1)
        
    #     # calculate the moments of isi
    #     self.moments = [np.mean(self.t_isi), 
    #                     np.var(self.t_isi),
    #                     np.std(self.t_isi),
    #                     skew(self.t_isi),
    #                     kurtosis(self.t_isi)]
        
    #     # put a thereshold (10 spikes) for number of spikes in spiketrain to calculate the synchrony
    #     threshold_n_spikes = 10
    #     if len(tsd) < threshold_n_spikes:
    #         spike_syn = 0.0     
    #     else:
    #         spike_syn = calculate_spike_synchrony(tsd, self.N)

    #     #-------------------------------------------------------------#

    #     if calculate_vol_syn:
    #         if len(tsd) < threshold_n_spikes:
    #             vol_syn = 0.0
    #         else:
    #             voltages = []
    #             index_cut = int(self.t_trans/self.dt)
    #             for i in range(self.N):
    #                 dmm = nest.GetStatus(self.multimeters)[i]
    #                 vms = dmm["events"]["V_m"][index_cut:]
    #                 voltages.append(vms)
    #             voltages = np.asarray(voltages)

    #             n = self.N 
    #             m = len(voltages[0])
                
    #             vg = np.sum(voltages, axis=0)
    #             vg /=(n+0.0)
                
    #             vg_mean = np.mean(vg)
                
    #             # vg_std = np.std(vg)
    #             O = np.sum((vg - vg_mean)*(vg-vg_mean))
    #             O /= (m+0.0)
                
    #             denom = 0.0            
    #             for i in range(n):
    #                 v = voltages[i,:]
    #                 vm = np.mean(v)
    #                 sigma = np.sum((v-vm)*(v-vm))
    #                 sigma /= (m+0)
    #                 denom += sigma
    #             denom /=(n+0.0)

    #             vol_syn = O/(denom+0.0)
            
    #     return vol_syn, spike_syn


    # #---------------------------------------------------------------#
# def calculate_isi(x):
#     isi = np.diff(x)
#     return isi
#---------------------------------------------------------------#
# def calculate_spike_synchrony(spiketrains, N):
        
#     t = calculate_isi(np.sort(spiketrains))
#     t2 = t*t
#     t_m = np.mean(t)
#     t2_m = np.mean(t2)
#     t_m2 = t_m*t_m
#     burs = ((np.sqrt(t2_m - t_m2) /
#             (t_m+0.0))-1.0)/(np.sqrt(N)+0.0)

#     return burs
#---------------------------------------------------------------#
# randomise initial potentials
# node_info = nest.GetStatus(self.neurons)
# local_nodes = [(ni['global_id'], ni['vp'])
#             for ni in node_info if ni['local']]
# for gid, vp in local_nodes:
#     # print gid, vp
#     nest.SetStatus([gid], {'V_m': self.pyrngs[vp].uniform(-70.0, -56.0)})
