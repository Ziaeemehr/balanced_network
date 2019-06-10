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
import pyspike as spk
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis, skew
from scipy.signal import butter, hilbert
from scipy.signal import welch, filtfilt
from scipy.ndimage.filters import gaussian_filter1d


sys.argv.append('--quiet')
# pl.switch_backend('agg')

seed = 1256
np.random.seed(seed)
# ---------------------------------------------------------------#


class single_iaf_neuron(object):
    built = False
    connected = False
    data_path = '../data'

    def __init__(self, dt):
        self.name = self.__class__.__name__
        nest.ResetKernel()
        nest.set_verbosity('M_WARNING')

        self.dt = dt

        nest.SetKernelStatus(
            {'resolution': self.dt,
             "data_path": self.data_path,
             "overwrite_files": True,
             "print_time": False,
             })

    def set_params(self, **par):

        self.neuron_model = par['neuron_model']
        self.noise_weight = par['noise_weight']
        self.mean_noise = par['mean_noise']
        self.tau_syn_ex = par['tau_syn_ex']
        self.std_noise = par['std_noise']
        self.t_trans = par['t_trans']
        self.t_sim = par['t_sim']
        self.I_e = par['I_e']

    def build(self):
        '''
        create nodes and devices used in the model
        '''
        if self.built:
            return

        self.neuron = nest.Create(self.neuron_model)
        nest.SetStatus(self.neuron, {'I_e': self.I_e,
                                     "tau_syn_ex": self.tau_syn_ex})

        self.spikedetector = nest.Create("spike_detector",
                                         params={"withgid": True,
                                                 "withtime": True})

        self.multimeter = nest.Create("multimeter")

        nest.SetStatus(self.multimeter, {"withtime": True,
                                    "record_from": ["V_m"]})

        self.noise = nest.Create('noise_generator')
        nest.SetStatus(self.noise, [{'mean': self.mean_noise,
                                     'std': self.std_noise}])

        self.built = True

    def connect(self):
        '''
        Connect nodes and devices.
        '''

        if not self.built:
            self.build()

        if self.connected:
            return

        nest.Connect(self.neuron, self.spikedetector)

        syn_dict = {'weight': self.noise_weight}
        nest.Connect(self.noise, self.neuron, syn_spec=syn_dict)
        nest.Connect(self.multimeter, self.neuron)

        self.connected = True

    def run(self, to_npz=False):

        if not self.built:
            self.build()
        if not self.connected:
            self.connect()

        nest.Simulate(self.t_trans)
        nest.SetStatus(self.spikedetector, {'n_events': 0})
        nest.Simulate(self.t_sim)

        self.ts, self.gids = get_spike_times(self.spikedetector)

        if to_npz:
            path = "../data/npz/"
            subname = str('%.3f-%.3f' % (
                self.weight_coupling, self.std_noise))
            np.savez(path+"spk-"+subname, ts=self.ts, gids=self.gids)

# ---------------------------------------------------------------#


class iaf_neuron(object):
    """
    iaf_psc_alpha - Leaky integrate-and-fire neuron model
    """

    built = False       # True, if build()   was called
    connected = False   # True, if connect() was called
    data_path = "../data"

    def __init__(self, dt, nthreads):

        self.name = self.__class__.__name__

        nest.ResetKernel()
        nest.set_verbosity('M_WARNING')

        self.dt = dt

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

    # ---------------------------------------------------------------#
    def set_params(self, **par):

        self.weight_coupling = par['weight_coupling']
        self.neuron_model = par['neuron_model']
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

    # ---------------------------------------------------------------#

    def build(self):
        '''
        create nodes and devices used in the model
        '''
        if self.built:
            return

        self.neurons = nest.Create(self.neuron_model, self.N)
        nest.SetStatus(
            self.neurons, {'I_e': self.I_e,
                           "tau_syn_ex": self.tau_syn_ex})

        # self.dc = nest.Create("poisson_generator")

        # randomise initial potentials
        node_info = nest.GetStatus(self.neurons)
        local_nodes = [(ni['global_id'], ni['vp'])
                       for ni in node_info if ni['local']]
        for gid, vp in local_nodes:
            nest.SetStatus(
                [gid], {'V_m': self.pyrngs[vp].uniform(-70.0, -56.0)})

        # if self.num_sim == 1:
        #     np.random.seed(1256)
        #     for gid in self.neurons:
        #         nest.SetStatus([gid], {'V_m': np.random.uniform(-70.0, -56)})
        # else:
        #     for gid in self.neurons:
        #         nest.SetStatus([gid], {'V_m': np.random.uniform(-70, -56.0)})

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
            self.noise, [{'mean': self.mean_noise,
                          'std': self.std_noise}])

        self.built = True

    # ---------------------------------------------------------------#
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
                         syn_spec={'weight': self.weight_coupling,
                                   'delay': self.delay})

        # nest.Connect(self.multimeters, self.neurons, 'one_to_one')
        nest.Connect(self.neurons, self.spikedetectors)

        syn_dict = {'weight': self.noise_weight}
        nest.Connect(self.noise, self.neurons, syn_spec=syn_dict)

        self.connected = True

    # ---------------------------------------------------------------#

    def run(self):
        if not self.connected:
            self.connect()

        nest.Simulate(self.t_trans)
        nest.SetStatus(self.spikedetectors, {'n_events': 0})
        nest.Simulate(self.t_sim)

        ts, gids = get_spike_times(self.spikedetectors)

        path = "../data/npz/"
        subname = str('%.3f-%.3f' % (
            self.weight_coupling, self.std_noise))

        np.savez(path+"spk-"+subname, ts=ts, gids=gids)
    # ---------------------------------------------------------------#


def display_time(time):
    ''' print wall time '''

    hour = int(time/3600)
    minute = (int(time % 3600))/60
    second = time-(3600.*hour+60.*minute)
    print "Done in %d hours %d minutes %.6f seconds" \
        % (hour, minute, second)
# ---------------------------------------------------------------#


def isi(tsd, gids):
    '''
    calculate interspike interval of given population
    '''
    neurons = np.unique(gids)
    t_pop = []
    for i in neurons:
        indices = np.where(gids == i)
        spikes = np.sort(tsd[indices])
        isi_i = np.diff(spikes)
        t_pop.extend(isi_i)

    return np.asarray(t_pop)
# ---------------------------------------------------------------#


def calculate_spike_synchrony(ts, gids):

    from numpy import sqrt, nanmean, sort, diff, unique
    t = ts.ravel()
    N = len(unique(gids))
    tau = diff(sort(t))
    tau2 = tau*tau
    tau_m = nanmean(tau)
    tau2_m = nanmean(tau2)
    tau_m2 = tau_m * tau_m
    sync = (((sqrt(tau2_m - tau_m2)) / float(tau_m)) - 1.0) / float(sqrt(N))

    return sync
# ---------------------------------------------------------------#


def spike_trains_list_of_list(ts, gids):

    neurons = np.unique(gids)

    t_pop = []
    for i in neurons:
        indices = np.where(gids == i)
        spikes = ts[indices]
        t_pop.append(spikes)

    return t_pop
# ---------------------------------------------------------------#


def calculate_spike_synchrony_pyspike(ts, gids, interval=None):

    if interval is None:
        tmin = np.min(ts)
        tmax = np.max(ts)
        interval = [tmin, tmax]

    s = spike_trains_list_of_list(ts, gids)
    spike_trains = []
    for i in range(len(s)):
        spike_trains.append(spk.SpikeTrain(s[i], interval, is_sorted=False))

    spk_sync = spk.spike_sync(spike_trains, interval=interval, max_tau=0.1)

    return spk_sync
# ---------------------------------------------------------------#


def make_er_graph(N, p, seed=1256):
    import os
    if not os.path.exists('dat'):
        os.makedirs('dat')
    G = nx.erdos_renyi_graph(N, p, seed=seed, directed=False)
    adj = np.asarray(nx.to_numpy_matrix(G), dtype=int)
    np.savetxt('dat/C.dat', adj, fmt="%d")
# ---------------------------------------------------------------#


def get_spike_times(detec):

    if nest.GetStatus(detec, "to_memory")[0]:
        if not nest.GetStatus(detec)[0]["model"] == "spike_detector":
            raise nest.NESTError("Please provide a spike_detector.")

        ev = nest.GetStatus(detec, "events")[0]
        ts = ev["times"]
        gids = ev["senders"]

    else:
        raise nest.NESTError(
            "No data to plot. Make sure that to_memory is set.")
    return ts, gids
# -----------------------------------------------------------------------#


def raster_plot_from_data(
        ts, gids, hist_binwidth=5.0,
        xlim=None, sel=None):
    """
    Plot raster from data
    """

    if not len(ts):
        raise nest.NESTError("No events recorded!")

    if sel is not None:
        val = extract_events(ts, gids, sel=sel)
    else:
        val = np.zeros((len(ts), 2))
        val[:, 0] = gids
        val[:, 1] = ts

    if val.shape[0] > 1:
        make_plot(ts, gids, val, hist_binwidth, sel, xlim)

# -----------------------------------------------------------------------#


def make_plot(ts, gids, val, hist_binwidth, sel, xlim):
    """
    Generic plotting routine that constructs a raster plot along with
    an optional histogram (common part in all routines above)
    """

    import matplotlib.gridspec as gridspec
    fig = pl.figure(figsize=(8, 6))
    gs1 = gridspec.GridSpec(4, 1, hspace=0.1)
    axs = []

    xlabel = "Time (ms)"
    ylabel = "Neuron ID"
    if xlim is None:
        xlim = pl.xlim()

    axs.append(fig.add_subplot(gs1[0:3]))
    axs.append(fig.add_subplot(gs1[3]))

    axs[0].plot(val[:, 1], val[:, 0], '.', markersize=5)
    axs[0].set_ylabel(ylabel)
    axs[0].set_xticks([])
    axs[0].set_xlim(xlim)

    t_bins = np.arange(
        np.amin(ts),
        np.amax(ts),
        float(hist_binwidth))
    n, bins = histogram(ts, bins=t_bins)
    num_neurons = len(np.unique(gids))
    heights = 1000 * n / (hist_binwidth * num_neurons)
    axs[1].bar(t_bins, heights, width=hist_binwidth,
               color='royalblue', edgecolor='black')
    axs[1].set_yticks([int(x) for x in np.linspace(
        0.0, int(max(heights) * 1.1) + 5, 4)])
    axs[1].set_ylabel("Rate (Hz)")
    axs[1].set_xlabel(xlabel)
    axs[1].set_xlim(xlim)
    # else:
    #     axs.append(fig.add_subplot(gs1[:]))

    #     axs[0].plot(val[:,1], val[:,0], '.')
    #     axs[0].set_xlabel(xlabel)
    #     axs[0].set_ylabel(ylabel)
# -------------------------------------------------------------------#


def histogram(a, bins=10, bin_range=None, normed=False):
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


def filter_gaussian(signal, fwhm):

    def fwhm2sigma(fwhm):
        return fwhm / np.sqrt(8 * np.log(2))

    sigma = fwhm2sigma(fwhm)
    filtered = gaussian_filter1d(signal, sigma, mode='reflect')

    return filtered
#--------------------------------------------------------------#


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
#--------------------------------------------------------------#


def filter_bandpass(ar_signal, sampling_rate, lowcut, highcut, order=5):
    b, a = butter_bandpass(lowcut, highcut, sampling_rate, order)

    return filtfilt(b, a, ar_signal)
#--------------------------------------------------------------#


def plot_rhythms_from_file(
        t_bins, heights, filtered,
        ax=None, xlim=None,  title=None):

    ax.plot(t_bins, heights, color='royalblue', alpha=0.5, label="activity")
    ax.plot(t_bins, filtered, lw=1, c='k', label='filtered')

    ax.set_yticks([int(x) for x in np.linspace(
        0.0, int(max(heights) * 1.1) + 5, 4)])

    if xlim:
        ax.set_xlim(xlim)
    # ax.set_title(title, fontsize=13)
    if title:
        ax.text(0.1, 0.9, title,
                ha='center',
                va='center',
                transform=ax.transAxes)

    ax.set_ylabel("Rate (Hz)")
    ax.set_xlabel("Time (ms)")
    ax.legend(loc='upper right')
    # pl.legend(frameon=False)
# ---------------------------------------------------------------#


def extract_events(ts, gids, time=None, sel=None):
    """
    Extracts all events within a given time interval or are from a
    given set of neurons.
    - data is a matrix such that

    - time is a list with at most two entries such that
      time=[t_max] extracts all events with t< t_max
      time=[t_min, t_max] extracts all events with t_min <= t < t_max
    - sel is a list of gids such that
      sel=[gid1, ... , gidn] extracts all events from these gids.
      All others are discarded.
    Both time and sel may be used at the same time such that all
    events are extracted for which both conditions are true.
    """

    val = []

    if time:
        t_max = time[-1]
        if len(time) > 1:
            t_min = time[0]
        else:
            t_min = 0

    for t, gid in zip(ts, gids):

        if time and (t < t_min or t >= t_max):
            continue
        if not sel or gid in sel:
            val.append([gid, t])

    # indices = np.in1d(gids, sel).nonzero()[0]
    # val = np.zeros((len(indices), 2))
    # val[:, 0] = gids[indices]
    # val[:, 1] = ts[indices]

    return np.array(val)

# ------------------------------------------------------------------------#


def calculate_histogram(ts, gids, hist_binwidth):

    t_bins = np.arange(
        np.amin(ts),
        np.amax(ts),
        float(hist_binwidth))

    n, bins = histogram(ts, bins=t_bins)
    num_neurons = len(np.unique(gids))
    heights = 1000 * n / (hist_binwidth * num_neurons)

    return t_bins, heights
# --------------------------------------------------------------#


def plot_power_spectrum(f, P, ax, label=None, xlim=None):

    ax.semilogx(f, P, lw=2, label=label)

    # if label:
    #     ax.text(0.05, 0.9, label,
    #             ha='center',
    #             va='center',
    #             transform=ax.transAxes)
    if xlim:
        ax.set_xlim(xlim)
    if label:
        ax.legend(loc='upper right')
    ax.set_xlabel("frequency")
    ax.set_ylabel("Power")
# --------------------------------------------------------------#


def fano_factor(dist):
    """
    Compute the Fano factor sigma^2_n / mean_n of a given distribution 
    """
    fn = np.var(dist) / np.mean(dist)
    return fn


#     def visualize(self, tlimits):
#         '''
#         plot the results
#         '''
#         N = self.N
#         tmin = tlimits[0]
#         tmax = tlimits[1]

#         fig = pl.figure(figsize=(8, 8))
#         gs1 = gridspec.GridSpec(6, 2, hspace=1.0)
#         axs = []
#         axs.append(fig.add_subplot(gs1[0:3, 0]))
#         axs.append(fig.add_subplot(gs1[0:3, 1]))
#         axs.append(fig.add_subplot(gs1[3:10, :]))

#         # if N > 10:
#         #     for i in range(0, N, 20):
#         #         dmm = nest.GetStatus(self.multimeters)[i]

#         #         Vms = dmm["events"]["V_m"]
#         #         tsv = dmm["events"]["times"]
#         #         axs[0].plot(tsv, Vms, lw=0.5)

#         dSD = nest.GetStatus(self.spikedetectors, keys='events')[0]
#         evs = dSD['senders'].tolist()
#         tsd = dSD["times"]

#         axs[0].hist(self.t_isi, bins=150)
#         axs[0].set_ylim(0, 200)
#         axs[0].set_xlim(0, 150)

#         moments = ['mean', 'var', 'std', 'skew', 'kurt']
#         ypos = np.arange(1, len(moments)+1)
#         axs[1].set_xticks(ypos)
#         axs[1].bar(ypos, self.moments, align='center', alpha=0.5)
#         axs[1].set_xticklabels(moments)
#         axs[1].set_ylim(0, 100)

#         axs[2].plot(tsd, evs, 'k.', markersize=3)

#         axs[2].set_ylabel('Neuron ID')
#         axs[2].set_xlabel("time(ms)")
#         # for i in range(1,3):
#         #     axs[i].set_xlim(tmin, tmax)
#         # axs[2].set_yticks([0,N/2, N])
#         fname = str('%.6f-%.6f' % (self.weight_coupling, self.std_noise))
#         fig.savefig('../data/fig/f-'+fname+'.png')
#         pl.close()
