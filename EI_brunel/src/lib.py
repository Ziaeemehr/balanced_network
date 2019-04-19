from scipy.signal import welch, filtfilt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import butter, hilbert
import os
import nest
import numpy as np
import pylab as pl
from time import time
import nest.raster_plot
# pl.switch_backend('agg')


class Brunel(object):
    '''
    Implementation of the sparsely connected random network,
    described by Brunel (2000) J. Comp. Neurosci.
    Parameters are chosen for the asynchronous irregular
    state (AI).
    '''

    data_path = "../data/text"
    built = False       # True, if build() was called
    connected = False   # True, if connect() was called

    def __init__(self, dt, nthreads):
        self.name = self.__class__.__name__
        nest.ResetKernel()
        nest.set_verbosity('M_QUIET')

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        nest.SetKernelStatus({
            "resolution": dt,
            "print_time": False,
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

        self.nthreads = par['nthreads']
        self.t_sim = par['t_sim']
        self.t_trans = par['t_trans']
        self.NE = par["NE"]
        self.NI = par["NI"]
        self.delay = par["delay"]
        self.epsilon = par['epsilon']
        self.eta = par['eta']
        self.g = par['g']
        self.N_rec_E = par['N_rec_E']
        self.N_rec_I = par['N_rec_I']
        self.tau_m = par['tau_m']
        self.V_th = par['V_th']
        self.V_m = par['V_m']
        self.V_reset = par['V_reset']
        self.E_L = par['E_L']
        self.t_ref = par['t_ref']
        self.C_m = par['C_m']
        self.J = par['J']
        self.I_e = par["I_e"]
        self.neuron_model = par['neuron_model']
        self.hist_binwidth = par['hist_binwidth']

        self.neuron_params = {
            "C_m": self.C_m,
            "tau_m": self.tau_m,
            "t_ref": self.t_ref,
            "E_L": self.E_L,
            "V_th": self.V_th,
            "V_m": self.V_m,
            "I_e": self.I_e,
            "V_reset": self.V_reset,
        }

        self.N_neurons = self.NE + self.NI  # number of neurons in total
        self.J_ex = self.J                  # amplitude of excitatory postsynaptic potential
        # amplitude of inhibitory postsynaptic potential
        self.J_in = -self.g * self.J_ex

    def calibrate(self):
        '''
        Compute all parameter dependent variables of the model.
        '''
        self.CE = int(
            self.epsilon * self.NE)  # number of excitatory synapses per neuron
        # number of inhibitory synapses per neuron
        self.CI = int(self.epsilon * self.NI)
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

        nest.SetStatus(self.espikes, [
            {"label": str("E-%.3f-%.3f" % (self.g, self.eta)),
             "withtime": True,
             "withgid": True,
             "to_file": False,
             "use_gid_in_filename": False}])

        nest.SetStatus(self.ispikes, [
            {"label": str("I-%.3f-%.3f" % (self.g, self.eta)),
             "withtime": True,
             "withgid": True,
             "to_file": False,
             "use_gid_in_filename": False}])
        node_info = nest.GetStatus(self.nodes_ex+self.nodes_in)
        local_nodes = [(ni['global_id'], ni['vp'])
                       for ni in node_info if ni['local']]
        for gid, vp in local_nodes:
            nest.SetStatus(
                [gid], {'V_m': self.pyrngs[vp].uniform(-self.V_th, self.V_th)})

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

        conn_params_ex = {'rule': 'fixed_indegree', 'indegree': self.CE}

        nest.Connect(self.nodes_ex, self.nodes_ex + self.nodes_in, conn_params_ex,
                     {'model': 'excitatory',
                      'delay': self.delay,
                      'weight': {'distribution': 'uniform',
                                 'low': 0.5*self.J_ex,
                                 'high': 1.5*self.J_ex}})

        conn_params_in = {'rule': 'fixed_indegree', 'indegree': self.CI}
        nest.Connect(self.nodes_in, self.nodes_ex +
                     self.nodes_in, conn_params_in, "inhibitory")

        self.connected = True

    def run(self):
        '''
        Simulate the model for simtime milliseconds and print the firing 
        rates of the network during this period.
        '''

        if not self.connected:
            self.connect()

        nest.Simulate(self.t_trans)

        nest.SetStatus(self.espikes, {'n_events': 0})
        nest.SetStatus(self.ispikes, {'n_events': 0})

        nest.Simulate(self.t_sim)

        events_ex = nest.GetStatus(self.espikes, "n_events")[0]
        events_in = nest.GetStatus(self.ispikes, "n_events")[0]

        rate_ex = events_ex / self.t_sim * 1000.0 / self.N_rec_E
        rate_in = events_in / self.t_sim * 1000.0 / self.N_rec_I

        num_synapses = (nest.GetDefaults("excitatory")["num_connections"] +
                        nest.GetDefaults("inhibitory")["num_connections"])

        # print "*" * 50
        # print("Brunel network simulation (Python)")
        # print("Number of neurons : {0}".format(self.N_neurons))
        # print("Number of synapses: {0}".format(num_synapses))
        # print("       Exitatory  : {0}".format(
        #     int(self.CE * self.N_neurons) + self.N_neurons))
        # print("       Inhibitory : {0}".format(int(self.CI * self.N_neurons)))
        # print("Excitatory rate   : %.2f Hz" % rate_ex)
        # print("Inhibitory rate   : %.2f Hz" % rate_in)

        path = "../data/npz/"
        subname = str("%.3f-%.3f" % (self.g, self.eta))
        e_t, e_gid = get_spike_times(self.espikes)
        i_t, i_gid = get_spike_times(self.ispikes)

        np.savez(path+"E-"+subname,
                 t=e_t,
                 gid=e_gid,
                 rate=rate_ex)
        np.savez(path+"I-"+subname,
                 t=i_t,
                 gid=i_gid,
                 rate=rate_in)

    # ---------------------------------------------------------------#

    def my_raster_plot(self, spike_detector, ax, color):
        dSD = nest.GetStatus(spike_detector, keys='events')[0]
        evs = dSD['senders']
        tsd = dSD["times"]
        ax.plot(tsd, evs, '.', c=color, markersize=3)
        ax.set_xlabel("Time (ms)", fontsize=18)
        ax.set_ylabel("Neuron ID", fontsize=18)
        ax.tick_params(labelsize=18)
        return (evs, tsd)
    # -------------------------------------------------------------------#

    def visualize(self, n=50, hist=False, rhythm=False, fwhm=4, xlim=None):
        '''
        plot rasterplots
        '''
        print "plotting ..."

        subname = str("%.3f-%.3f" % (self.g, self.eta))
        path = "../data/fig/"

        if hist:
            raster_plot_from_device(self.espikes, hist=hist,
                                    hist_binwidth=self.hist_binwidth,
                                    xlim=xlim, sel=range(n))
            pl.savefig(path+'E-'+subname+'.png')
            pl.close()

            raster_plot_from_device(self.ispikes, hist=hist,
                                    hist_binwidth=self.hist_binwidth,
                                    xlim=xlim, sel=range(self.NE, self.NE+n))
            pl.savefig(path+'I-'+subname+'.png')
            pl.close()

        if rhythm:
            fig0, ax0 = pl.subplots(1, figsize=(10, 5))
            plot_rhythms_from_device(self.espikes, ax0,
                                     fwhm=fwhm, hist_binwidth=self.hist_binwidth)
            pl.savefig(path+'r-'+subname+'.png')
            pl.close()

    # -------------------------------------------------------------------#


def raster_plot_from_device(
        detec, hist=False, hist_binwidth=5.0,
        xlim=None, sel=None):

    ts, gids = get_spike_times(detec)

    val = extract_events(ts, gids, sel=sel)
    # t  = val[:,1]
    # gs = val[:,0]
    if not len(ts):
        raise nest.NESTError("No events recorded!")

    make_plot(ts, gids, val, hist, hist_binwidth, sel, xlim)
# -----------------------------------------------------------------------#


def raster_plot_from_file(
        fname, hist=False, hist_binwidth=5.0,
        xlim=None, sel=None):
    """
    Plot raster from file
    """
    if nest.is_iterable(fname):
        data = None
        for f in fname:
            if data is None:
                data = np.loadtxt(f)
            else:
                data = np.concatenate((data, np.loadtxt(f)))
    else:
        data = np.loadtxt(fname)

    ts = data[:, 1]
    gids = data[:, 0]
    if not len(ts):
        raise nest.NESTError("No events recorded!")

    val = extract_events(ts, gids, sel=sel)
    make_plot(ts, gids, val, hist, hist_binwidth, sel, xlim)
# -----------------------------------------------------------------------#


def raster_plot_from_data(
        ts, gids, hist=False, hist_binwidth=5.0,
        xlim=None, sel=None):
    """
    Plot raster from data
    """

    if not len(ts):
        raise nest.NESTError("No events recorded!")

    val = extract_events(ts, gids, sel=sel)
    make_plot(ts, gids, val, hist, hist_binwidth, sel, xlim)
# -----------------------------------------------------------------------#


def make_plot(ts, gids, val, hist, hist_binwidth, sel, xlim):
    """
    Generic plotting routine that constructs a raster plot along with
    an optional histogram (common part in all routines above)
    """

    import matplotlib.gridspec as gridspec
    fig = pl.figure(figsize=(12, 9))
    gs1 = gridspec.GridSpec(4, 1, hspace=0.1)
    axs = []

    xlabel = "Time (ms)"
    ylabel = "Neuron ID"
    if xlim == None:
        xlim = pl.xlim()

    if hist:
        axs.append(fig.add_subplot(gs1[0:3]))
        axs.append(fig.add_subplot(gs1[3]))

        axs[0].plot(val[:, 1], val[:, 0], '.')
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


def plot_rhythms_from_device(detec, ax, fwhm, hist_binwidth=5.0):

    from scipy.ndimage.filters import gaussian_filter1d

    def fwhm2sigma(fwhm):
        return fwhm / np.sqrt(8 * np.log(2))

    ts, gids = get_spike_times(detec)

    if not len(ts):
        raise nest.NESTError("No events recorded!")

    t_bins = np.arange(
        np.amin(ts),
        np.amax(ts),
        float(hist_binwidth))
    n, bins = histogram(ts, bins=t_bins)
    num_neurons = len(np.unique(gids))
    heights = 1000 * n / (hist_binwidth * num_neurons)
    ax.bar(t_bins, heights, width=hist_binwidth, color='royalblue',
           edgecolor='black', alpha=0.1)
    ax.set_yticks([int(x) for x in np.linspace(
        0.0, int(max(heights) * 1.1) + 5, 4)])
    ax.set_ylabel("Rate (Hz)")

    sigma = fwhm2sigma(fwhm)
    filtered = gaussian_filter1d(heights, sigma, mode='reflect')
    ax.plot(t_bins, filtered, marker='o', markersize=1, c='k', label='conv')
    pl.legend(frameon=False)
    pl.xlabel("Time (ms)")
# ---------------------------------------------------------------#


def calculate_histogram(ts, gids, hist_binwidth):

    t_bins = np.arange(
        np.amin(ts),
        np.amax(ts),
        float(hist_binwidth))

    n, bins = histogram(ts, bins=t_bins)
    num_neurons = len(np.unique(gids))
    heights = 1000 * n / (hist_binwidth * num_neurons)

    return t_bins, heights
# ---------------------------------------------------------------#


def filter_gaussian(signal, fwhm):

    def fwhm2sigma(fwhm):
        return fwhm / np.sqrt(8 * np.log(2))

    sigma = fwhm2sigma(fwhm)
    filtered = gaussian_filter1d(signal, sigma, mode='reflect')

    return filtered
# ---------------------------------------------------------------#


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

    return np.array(val)
#------------------------------------------------------------------------#


def plot_R(R, X, Y, name="R", xtickstep=1, ytickstep=1,
           xlabel=None, ylabel=None, title=None):
    ''' 
    plot R in 2D plane of X and Y axises 
    '''

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    r, c = R.shape
    assert((r > 1) & (c > 1))

    x_step = X[1] - X[0]
    y_step = Y[1] - Y[0]

    f, ax = pl.subplots(1, figsize=(6, 6))
    im = ax.imshow(R, interpolation='nearest', cmap='afmhot')
    ax.invert_yaxis()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)

    ax.set_xticks(np.arange(0, len(X), xtickstep))
    ax.set_xticklabels(str("%.1f" % i)for i in X[::xtickstep])
    ax.set_yticks(np.arange(0, len(Y), ytickstep))
    ax.set_yticklabels(str("%.1f" % i)for i in Y[::ytickstep])

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    if title:
        ax.set_title(title, fontsize=16)

    pl.savefig("fig/"+name+".png")
    pl.close()
#--------------------------------------------------------------#


def display_time(time):
    ''' 
    show real time elapsed
    '''

    hour = int(time/3600)
    minute = (int(time % 3600))/60
    second = time-(3600.*hour+60.*minute)
    print "Done in %d hours %d minutes %09.6f seconds" \
        % (hour, minute, second)
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


def plot_power_spectrum(f, P, ax, title=None, xlim=None):

    ax.semilogx(f, P, lw=1)

    if title:
        ax.text(0.05, 0.9, title,
                ha='center',
                va='center',
                transform=ax.transAxes)
    if xlim:
        ax.set_xlim(xlim)
    # ax.legend(loc='best')
#--------------------------------------------------------------#


def imshow_plot(data, interpolation="nearest", cmap='afmhot', fname="fig"):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = pl.figure(100, figsize=(6, 6))
    ax = pl.subplot(111)
    im = ax.imshow(data, interpolation=interpolation,
                   cmap=cmap)  # , cmap=pl.cm.ocean
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)
    pl.savefig(fname)
    pl.close()
