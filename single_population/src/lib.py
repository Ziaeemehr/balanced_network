# Abolfazl Ziaeemehr
# Institute for Advanced Studies in
# Basic Sciences (IASBS)
# tel: +98 3315 2148
# github.com/ziaeemehr

import sys
import nest
import pylab as pl
import numpy as np
from sys import exit
import networkx as nx
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis, skew
sys.argv.append('--quiet')
pl.switch_backend('agg')

seed = 1256
np.random.seed(seed)
#---------------------------------------------------------------#


class iaf_neuron(object):
    """
    iaf_psc_alpha - Leaky integrate-and-fire neuron model
    """

    built = False       # True, if build()   was called
    connected = False   # True, if connect() was called
    data_path = "../data"

    def __init__(self, dt, nthreads):
    # adj, N, dt, tau_syn_ex, delay,  nthreads):

        self.name = self.__class__.__name__

        nest.ResetKernel()
        nest.set_verbosity('M_WARNING')

        self.dt = dt
        # self.delay = delay
        # self.adj = adj
        # self.N = N
        # self.tau_syn_ex = tau_syn_ex

        nest.SetKernelStatus(
            {'resolution': self.dt,
             'local_num_threads': nthreads,
             "data_path": self.data_path,
             "overwrite_files": True,
             "print_time": False,
             })

        msd = 1000      # master seed
        n_vp = nest.GetKernelStatus('total_num_virtual_procs')
        msdrange1 = range(msd, msd + n_vp)
        self.pyrngs = [np.random.RandomState(s) for s in msdrange1]
        msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)
        nest.SetKernelStatus({'grng_seed': msd + n_vp,
                              'rng_seeds': msdrange2})

    #---------------------------------------------------------------#
    def set_params(self, **par):

        self.node_coupling = par['node_coupling']
        self.noise_weight = par['noise_weight']
        self.mean_noise = par['mean_noise']
        self.tau_syn_ex = par['tau_syn_ex']
        self.std_noise = par['std_noise']
        self.vol_step = par['vol_step']
        self.num_sim = par['num_sim']
        self.t_trans = par['t_trans']
        self.delay = par['delay']
        self.t_sim = par['t_sim']
        self.I_e = par['I_e']
        self.adj = par['adj']
        self.N = par['N']


    #---------------------------------------------------------------#

    def build(self):
        '''
        create nodes and devices used in the model
        '''
        if self.built:
            return

        self.neurons = nest.Create('iaf_psc_alpha', self.N)
        nest.SetStatus(
            self.neurons, {'I_e': self.I_e, "tau_syn_ex": self.tau_syn_ex})

        # self.dc = nest.Create("poisson_generator")

        # randomise initial potentials
        # node_info = nest.GetStatus(self.neurons)
        # local_nodes = [(ni['global_id'], ni['vp'])
        #             for ni in node_info if ni['local']]
        # for gid, vp in local_nodes:
        #     # print gid, vp
        #     nest.SetStatus([gid], {'V_m': self.pyrngs[vp].uniform(-70.0, -56.0)})

        if self.num_sim == 1:
            np.random.seed(1256)
            for gid in self.neurons:
                nest.SetStatus([gid], {'V_m': np.random.uniform(-70.0, -56)})
        else:
            for gid in self.neurons:
                nest.SetStatus([gid], {'V_m': np.random.uniform(-70, -56.0)})

        # self.multimeters = nest.Create("multimeter", self.N)
        # nest.SetStatus(self.multimeters,
        #                {"withtime": True,
        #                 "interval": (self.vol_step * self.dt),
        #                 "record_from": ["V_m"]})

        self.spikedetectors = nest.Create("spike_detector",
                                          params={"withgid": True,
                                                  "withtime": True})

        self.noise = nest.Create('noise_generator')
        nest.SetStatus(
            self.noise, [{'mean': self.mean_noise, 'std': self.std_noise}])

        self.built = True

    #---------------------------------------------------------------#
    def connect(self):
        '''
        Connect all nodes in the model.
        '''

        if self.connected:
            return
        if not self.built:
            self.build()

        # make graph from the adj matrix
        G = nx.from_numpy_matrix(self.adj, create_using=nx.DiGraph())

        # connect using the edges in the graph
        for edge in G.edges():
            from_idx = edge[0]
            to_idx = edge[1]
            nest.Connect([self.neurons[from_idx]],
                         [self.neurons[to_idx]],
                         'one_to_one',
                         syn_spec={'weight': self.node_coupling, 'delay': self.delay})

        # nest.Connect(self.multimeters, self.neurons, 'one_to_one')
        nest.Connect(self.neurons, self.spikedetectors)

        syn_dict = {'weight': self.noise_weight}
        nest.Connect(self.noise, self.neurons, syn_spec=syn_dict)

        self.connected = True

    #---------------------------------------------------------------#
    def run(self):
        if not self.connected:
            self.connect()

        # nest.Simulate(self.t_trans)
        # nest.SetStatus(self.spikedetectors, {'n_events': 0})
        nest.Simulate(self.t_sim)

    #---------------------------------------------------------------#
    def measure_synchrony(self, calculate_vol_syn=False):
        '''
        calculate the coherency from spike trains and voltages
        '''
        N = self.N
        spike_syn = vol_syn = 0.0

        dSD = nest.GetStatus(self.spikedetectors, keys='events')[0]
        evs = dSD['senders']
        tsd = dSD["times"]

        # claculate interspike interval for each neuron
        self.t_isi = []
        for i in range(N):
            indices = [j for j, x in enumerate(evs) if x == i]
            spike_i = tsd[indices]
            t1 = calculate_isi(spike_i)
            self.t_isi.extend(t1)

        # calculate the moments of isi
        self.moments = [np.mean(self.t_isi),
                        np.var(self.t_isi),
                        np.std(self.t_isi),
                        skew(self.t_isi),
                        kurtosis(self.t_isi)]

        # put a thereshold (10 spikes) for number of spikes in spiketrain to calculate the synchrony
        threshold_n_spikes = 10
        if len(tsd) < threshold_n_spikes:
            spike_syn = 0.0
        else:
            spike_syn = calculate_spike_synchrony(tsd, self.N)

        #-------------------------------------------------------------#

        if calculate_vol_syn:
            if len(tsd) < threshold_n_spikes:
                vol_syn = 0.0
            else:
                voltages = []
                index_cut = int(self.t_trans/self.dt)
                for i in range(self.N):
                    dmm = nest.GetStatus(self.multimeters)[i]
                    vms = dmm["events"]["V_m"][index_cut:]
                    voltages.append(vms)
                voltages = np.asarray(voltages)

                n = self.N
                m = len(voltages[0])

                vg = np.sum(voltages, axis=0)
                vg /= (n+0.0)

                vg_mean = np.mean(vg)

                # vg_std = np.std(vg)
                O = np.sum((vg - vg_mean)*(vg-vg_mean))
                O /= (m+0.0)

                denom = 0.0
                for i in range(n):
                    v = voltages[i, :]
                    vm = np.mean(v)
                    sigma = np.sum((v-vm)*(v-vm))
                    sigma /= (m+0)
                    denom += sigma
                denom /= (n+0.0)

                vol_syn = O/(denom+0.0)

        return vol_syn, spike_syn

    #---------------------------------------------------------------#

    def visualize(self, tlimits):
        '''
        plot the results
        '''
        N = self.N
        tmin = tlimits[0]
        tmax = tlimits[1]

        fig = pl.figure(figsize=(8, 8))
        gs1 = gridspec.GridSpec(6, 2, hspace=1.0)
        axs = []
        axs.append(fig.add_subplot(gs1[0:3, 0]))
        axs.append(fig.add_subplot(gs1[0:3, 1]))
        axs.append(fig.add_subplot(gs1[3:10, :]))

        # if N > 10:
        #     for i in range(0, N, 20):
        #         dmm = nest.GetStatus(self.multimeters)[i]

        #         Vms = dmm["events"]["V_m"]
        #         tsv = dmm["events"]["times"]
        #         axs[0].plot(tsv, Vms, lw=0.5)

        dSD = nest.GetStatus(self.spikedetectors, keys='events')[0]
        evs = dSD['senders'].tolist()
        tsd = dSD["times"]

        axs[0].hist(self.t_isi, bins=150)
        axs[0].set_ylim(0, 200)
        axs[0].set_xlim(0, 150)

        moments = ['mean', 'var', 'std', 'skew', 'kurt']
        ypos = np.arange(1, len(moments)+1)
        axs[1].set_xticks(ypos)
        axs[1].bar(ypos, self.moments, align='center', alpha=0.5)
        axs[1].set_xticklabels(moments)
        axs[1].set_ylim(0, 100)

        axs[2].plot(tsd, evs, 'k.', markersize=3)

        axs[2].set_ylabel('Neuron ID')
        axs[2].set_xlabel("time(ms)")
        # for i in range(1,3):
        #     axs[i].set_xlim(tmin, tmax)
        # axs[2].set_yticks([0,N/2, N])
        fname = str('%.6f-%.6f' % (self.node_coupling, self.std_noise))
        fig.savefig('../data/fig/f-'+fname+'.png')
        pl.close()

#---------------------------------------------------------------#


def display_time(time):
    hour = int(time/3600)
    minute = (int(time % 3600))/60
    second = time-(3600.*hour+60.*minute)
    print "Done in %d hours %d minutes %.6f seconds" \
        % (hour, minute, second)


def calculate_isi(x):
    isi = np.diff(x)
    return isi
#---------------------------------------------------------------#


def calculate_spike_synchrony(spiketrains, N):

    t = calculate_isi(np.sort(spiketrains))
    t2 = t*t
    t_m = np.mean(t)
    t2_m = np.mean(t2)
    t_m2 = t_m*t_m
    burs = ((np.sqrt(t2_m - t_m2) /
             (t_m+0.0))-1.0)/(np.sqrt(N)+0.0)

    return burs
#---------------------------------------------------------------#


def make_er_graph(N, p, seed=1256):
    import os
    if not os.path.exists('dat'):
        os.makedirs('dat')
    G = nx.erdos_renyi_graph(N, p, seed=seed, directed=False)
    adj = np.asarray(nx.to_numpy_matrix(G), dtype=int)
    np.savetxt('dat/C.dat', adj, fmt="%d")
