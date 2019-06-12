import os
import lib
import numpy as np
import pylab as pl
from sys import exit
from time import time
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch, filtfilt
from scipy.signal import butter, hilbert
from run import weight_coupling, tau_syn_in
from run import N, t_trans, t_sim, dt, I_e, std_noise

path = "../data/"
os.chdir(path)


# -----------------------------------------------------------------------#


def plot_frequeicy(xtickstep=5, ytickstep=5, vmax=None, vmin=None):
    try:
        c = np.load("npz/Freq.npz")
    except:
        raise Exception("file not found!")
    freq = c['freq']
    g = c['g']
    sigma = c['sigma']

    plot_phase_space(freq, sigma, g, "freq",
                     xtickstep=xtickstep,
                     ytickstep=ytickstep,
                     xlabel=r"$\sigma$",
                     ylabel='g',
                     vmax=vmax,
                     vmin=vmin)
    # lib.imshow_plot(ifreq, fname="fig/ifreq.png")
    # lib.imshow_plot(efreq, fname="fig/efreq.png")

# -----------------------------------------------------------------------#


def plot_raster(g, sigma):

    len_g = len(g)
    len_sigma = len(sigma)
    for i in range(len_g):
        for j in range(len_sigma):
            subname = str("%.3f-%.3f" % (g[i], sigma[j]))

            fname = str("spk-%s.npz" % subname)
            pop = np.load("npz/"+fname)

            ts = pop['ts']
            gids = pop['gids']

            if len(ts):
                lib.raster_plot_from_data(
                    ts, gids, hist_binwidth, xlim, sel)
                pl.savefig("fig/spk-"+subname+".png")
                pl.close()
            else:
                print "no plot at g= %g, sigma= %g" % (g[i], sigma[j])

# -----------------------------------------------------------------------#


def plot_raster_fast(fname, ax):

    number = 10
    cmap = pl.get_cmap('afmhot')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]

    pop = np.load("npz/"+fname)
    ts = pop['ts']
    gids = pop['gids']

    neurons = np.unique(gids)

    if len(ts) > 3:
        tmp = lib.calculate_spike_synchrony(ts, gids)
        R = 0 if ((tmp > 1) or (tmp < 0)) else tmp
        R = int(round(R, 1)*10)

        for ii in neurons:
            indices = np.where(gids == ii)
            spikes = ts[indices]
            ax.plot(spikes, [ii]*len(spikes), '.',
                    c=colors[R], markersize=8)
    else:
        print "empty plot for %s" % fname
# -----------------------------------------------------------------------#


def plot_rhythm(g, sigma, PLOT=False):
    """
    calculate the frequency and plot the histogram of
    activity and filtered activety
    """

    len_g = len(g)
    len_sigma = len(sigma)
    freq = np.zeros((len_g, len_sigma))

    for i in range(len_g):
        for j in range(len_sigma):
            subname = str("%.3f-%.3f" % (g[i], sigma[j]))

            fname = str("spk-%s.npz" % subname)
            pop = np.load("npz/"+fname)

            if PLOT:
                fig0, ax = pl.subplots(2, figsize=(8, 5))
                pl.subplots_adjust(hspace=.4)

            ts = pop['ts']
            gids = pop['gids']
            if len(ts):

                t_bins, heights = lib.calculate_histogram(
                    ts, gids, hist_binwidth)

                filtered = lib.filter_gaussian(heights, fwhm)
                # filtered = lib.filter_bandpass(heights, fs, 2, 1000, 5)

                f, P = welch(filtered, fs=fs, nperseg=nperseg)
                freq[i, j] = f[np.argmax(P)]

                if PLOT:
                    lib.plot_rhythms_from_file(
                        t_bins, heights, filtered, ax[0], xlim)

                    lib.plot_power_spectrum(f, P, ax[1])

            if PLOT:
                pl.savefig('fig/r-'+subname+".png")
                pl.close()

    # np.savez("npz/Freq",
    #          freq=freq, g=g, sigma=sigma)
# -----------------------------------------------------------------------#


def plot_activity(fname, fwhm, ax):

    number = 10
    cmap = pl.get_cmap('afmhot')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]

    # fname = str("spk-%s.npz" % subname)
    pop = np.load("npz/"+fname)

    ts = pop['ts']
    gids = pop['gids']
    if len(ts) > 3:
        tmp = lib.calculate_spike_synchrony(ts, gids)
        R = 0 if ((tmp > 1) or (tmp < 0)) else tmp
        R = int(round(R, 1)*10)

        t_bins, heights = lib.calculate_histogram(
            ts, gids, hist_binwidth)
        filtered = lib.filter_gaussian(heights, fwhm)

        # ax1.tick_params(axis='y', labelcolor='royalblue')
        ax.plot(t_bins, filtered, lw=2, c=colors[R])
    else:
        print "empty plot for %s" % fname
# -----------------------------------------------------------------------#


def plot_isi(hist_binwidth=5):

    nbin = int(params['t_sim']/hist_binwidth)
    for i in range(len_g):
        for j in range(len_sigma):

            fig0, ax = pl.subplots(1, figsize=(4, 4))
            subname = str("%.3f-%.3f" % (g[i], sigma[j]))

            fname = str("E-%s.npz" % subname)
            pop = np.load("npz/"+fname)

            ts = pop[p]['t']
            gids = pop[p]['gid']

            if len(ts):
                t_isi = lib.isi(ts, gids)
                ax.hist(t_isi, bins=nbin, density=True)

            pl.savefig('fig/isi-'+subname+".png")
            pl.close()
# -----------------------------------------------------------------------#


def plot_moments(g, eta, xlabel=None, xlim=None, xticks=None):
    from scipy.stats import kurtosis, skew
    from collections import Iterable
    assert (isinstance(g, Iterable)), "g should be itetable"

    n = len(g)
    fig, ax = pl.subplots(nrows=5, ncols=2, figsize=(8, 5), sharex=True)
    pl.subplots_adjust(hspace=0)
    immt = np.zeros((n, 5))
    emmt = np.zeros((n, 5))

    for i in range(n):

        subname = str("%.3f-%.3f" % (g[i], eta))

        efname = str("E-%s.npz" % subname)
        ifname = str("I-%s.npz" % subname)
        E = np.load("npz/"+efname)
        I = np.load("npz/"+ifname)

        pop = [E, I]

        for p in range(2):
            ts = pop[p]['t']
            gids = pop[p]['gid']

            if len(ts):
                t_isi = lib.isi(ts, gids)
                if p == 0:
                    emmt[i, :] = [np.mean(t_isi), np.var(t_isi), np.std(t_isi),
                                  skew(t_isi), kurtosis(t_isi)]
                if p == 1:
                    immt[i, :] = [np.mean(t_isi), np.var(t_isi), np.std(t_isi),
                                  skew(t_isi), kurtosis(t_isi)]

    mmt = [emmt, immt]
    label1 = ['mean', 'var', 'std', 'skew', 'kurtosis']
    for p in range(2):
        m = mmt[p]
        for i in range(5):
            ax[i][p].plot(g, m[:, i], label=label1[i])
            ax[i][p].legend()
    for p in range(2):
        ax[0][p].set_title(labels[p])
        if xlabel:
            ax[-1][p].set_xlabel(xlabel)

    pl.savefig("fig/moments.png")
# -----------------------------------------------------------------------#


def plot_phase_space(R, X, Y, name="R", xtickstep=1, ytickstep=1,
                     xlabel=None, ylabel=None, title=None,
                     vmax=None, vmin=None):
    '''
    plot R in 2D plane of X and Y axises
    '''
    print len(X), len(Y), R.shape

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    r, c = R.shape
    assert((r > 1) & (c > 1))

    x_step = X[1] - X[0]
    y_step = Y[1] - Y[0]

    f, ax = pl.subplots(1, figsize=(6, 6))
    im = ax.imshow(R, interpolation='nearest',
                   cmap='afmhot', vmax=vmax, vmin=vmin)
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

# -----------------------------------------------------------------------#


def plot_spike_synchrony(g, sigma, tau,
                         xtickstep=1, ytickstep=1,
                         xlabel=None, ylabel=None):

    len_g, len_sigma, len_tau = len(g), len(sigma), len(tau)

    for j in range(len_sigma):
        R = np.zeros((len_tau, len_g))
        for k in range(len_tau):
            for i in range(len_g):

                subname = str("%.3f-%.3f-%.3f" % (g[i], sigma[j], tau[k]))
                fname = str("spk-%s.npz" % subname)
                pop = np.load("npz/"+fname)

                ts = pop['ts']
                gids = pop['gids']

                if len(ts):
                    tmp = lib.calculate_spike_synchrony(ts, gids)
                    R[k, i] = 0 if ((tmp > 1) or (tmp < 0)) else tmp
                    if tmp > 1:
                        print "%.2f %.2f %.2f %10d, %15.4f" % (
                            g[i], sigma[j], tau[k], len(ts), tmp)
        plot_phase_space(R.T, tau, g, "R-"+str("%.3f" % sigma[j]),
                         xtickstep=xtickstep,
                         ytickstep=ytickstep,
                         ylabel=ylabel,
                         xlabel=xlabel)

        np.savez("npz/R-"+str("%.3f" % sigma[j])+".npz",
                 R=R,
                 g=g,
                 tau=tau,
                 sigma=sigma)
# -----------------------------------------------------------------------#


def plot_bunch_rasters_activity(g, sigma, xlim=None):

    len_g = len(g)
    len_sigma = len(sigma)

    for i in range(len_g):
        for j in range(len_sigma):

            fig = pl.figure(figsize=(6, 5))
            gs1 = gridspec.GridSpec(4, 1, hspace=0.0)
            axs = []
            axs.append(fig.add_subplot(gs1[:3]))
            axs.append(fig.add_subplot(gs1[3]))

            subname = str("%.3f-%.3f" % (g[i], sigma[j]))
            fname = str("spk-%s.npz" % subname)
            plot_raster_fast(fname, axs[0])

            plot_activity(fname, fwhm, axs[1])
            axs[0].set_xticks([])
            axs[0].axis('off')
            if xlim:
                axs[0].set_xlim(xlim)
                axs[1].set_xlim(xlim)

            # axs[1].set_ylim(0, 300)
            # axs[1].axis('off')
            pl.savefig("fig/spk-"+subname+".png")
            pl.close()
# -----------------------------------------------------------------------#


def plot_map_raster_activity(g, sigma):
    g = g[::-1]
    len_g = len(g)
    len_sigma = len(sigma)
    R = 0

    print len_g, len_sigma

    fig = pl.figure(figsize=(40, 30))
    outer = gridspec.GridSpec(len_g, len_sigma, wspace=0.05, hspace=0.02)

    subnames = []
    glabels = []
    for i in range(len_g):
        for j in range(len_sigma):
            subname = str("%.3f-%.3f" % (g[i], sigma[j]))
            subnames.append(subname)
            glabels.append(str('%.1f' % g[i]))
    glabels = glabels
    slabels = sigma

    for i in range(len_g*len_sigma):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                 subplot_spec=outer[i],
                                                 wspace=0.0,
                                                 hspace=0.0)

        fname = str("spk-%s.npz" % subnames[i])

        for j in range(2):
            ax = pl.Subplot(fig, inner[j])
            # ax.set_xlabel(slabels[i % (len_sigma)], fontsize=20)
            if j == 0:
                plot_raster_fast(fname, ax)
                # if (i % len_sigma) == 0:
                #     ax.set_ylabel(glabels[i], fontsize=20)
            else:
                plot_activity(fname, fwhm, ax)
                ax.set_ylim(0, 300)

            ax.set_xlim(t_sim+t_trans-1000, t_sim+t_trans)
            # ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

            if (i+1) > ((len_g-1)*len_sigma) and j == 1:
                ax.set_xlabel(slabels[i % (len_sigma)], fontsize=20)
            if ((i % len_sigma) == 0) and j == 0:
                ax.set_ylabel(glabels[i], fontsize=20)

    # show only the outside spines
    # all_axes = fig.get_axes()
    # print len(all_axes)
    # for ax in all_axes:
    #     for sp in ax.spines.values():
    #         sp.set_visible(False)
    #     if ax.is_first_row():
    #         ax.spines['top'].set_visible(True)
    #     if ax.is_last_row():
    #         ax.spines['bottom'].set_visible(True)
    #     if ax.is_first_col():
    #         ax.spines['left'].set_visible(True)
    #     if ax.is_last_col():
    #         ax.spines['right'].set_visible(True)

    fig.text(0.5, 0.07, r'$\sigma$', ha='center', fontsize=40)
    fig.text(0.07, 0.5, r'$g$', va='center',
             fontsize=40)
    pl.savefig("fig/spk.png")
    # pl.savefig("fig/spk.pdf")
# -----------------------------------------------------------------------#


def plot_voltage(fname, ax):

    c = np.load('npz/'+fname)
    t = c['t']
    v = c['v']
    # for i in range(N):
    #     ax.plot(t, v[i, :], lw=1)
    ax.plot(t, v, lw=2, c='k')
# -----------------------------------------------------------------------#


def plot_I_syn(fname, ax):

    c = np.load('npz/'+fname)
    t = c['t']
    I = c['I_syn_in']
    ax.plot(t, I, lw=2, c='k')
# -----------------------------------------------------------------------#


def plot_raster_voltage_Isyn(g, sigma, tau, xlim=None):
    len_g = len(g)
    len_sigma = len(sigma)
    len_tau = len(tau)
    for j in range(len_sigma):
        for i in range(len_g):
            for k in range(len_tau):

                fig = pl.figure(figsize=(6, 6))
                gs1 = gridspec.GridSpec(5, 1, hspace=0.0)
                axs = []
                axs.append(fig.add_subplot(gs1[:3]))
                axs.append(fig.add_subplot(gs1[3]))
                axs.append(fig.add_subplot(gs1[4]))

                subname = str("%.3f-%.3f-%.3f" % (g[i], sigma[j], tau[k]))
                spkname = str("spk-%s.npz" % subname)
                vname = str("v-%s.npz" % subname)
                plot_raster_fast(spkname, axs[0])

                plot_voltage(vname, axs[1])
                plot_I_syn(vname, axs[2])

                axs[0].set_xticks([])
                axs[0].axis('off')
                # axs[1].set_ylim(0, 300)
                for ii in range(3):
                    if xlim:
                        axs[ii].set_xlim(xlim)
                    else:
                        axs[ii].set_xlim(t_trans, t_trans + t_sim)
                axs[2].set_xlabel("Time (ms)")
                axs[1].set_ylabel(r"$V_{global}$")
                axs[2].set_ylabel(r"$I_{syn}$")
                # axs[1].axis('off')
                pl.tight_layout()
                pl.savefig("fig/spk_v-"+subname+".png")
                pl.close()
# -----------------------------------------------------------------------#


def plot_map_raster_voltage(g, sigma, tau, xlim=None, title=None):

    g = g[::-1]
    len_g = len(g)
    len_tau = len(tau)
    len_sigma = len(sigma)
    R = 0

    print len_g, len_tau, len_sigma

    fig = pl.figure(figsize=(40, 30))
    # pl.title(title, fontsize=40)
    outer = gridspec.GridSpec(len_g, len_tau, wspace=0.05, hspace=0.02)

    subnames = []
    glabels = []
    for i in range(len_g):
        for k in range(len_tau):
            subname = str("%.3f-%.3f-%.3f" % (g[i], sigma[0], tau[k]))
            subnames.append(subname)
            glabels.append(str('%.1f' % g[i]))

    tlabels = tau
    for i in range(len_g*len_tau):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[i],
            wspace=0.0,
            hspace=0.0)

        fname = str("spk-%s.npz" % subnames[i])
        vname = str("v-%s.npz" % subnames[i])

        for j in range(2):
            ax = pl.Subplot(fig, inner[j])
            if j == 0:
                plot_raster_fast(fname, ax)
            else:
                plot_voltage(vname, ax)
                # plot_I_syn(vname, ax)
                # ax.set_ylim(-8, 1)
                ax.set_ylim(-58, -55)

            if xlim:
                ax.set_xlim(xlim)
            # ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

            if (i+1) > ((len_g-1)*len_tau) and j == 1:
                ax.set_xlabel(tlabels[i % (len_tau)], fontsize=20)
            if ((i % len_tau) == 0) and j == 0:
                ax.set_ylabel(glabels[i], fontsize=20)

    fig.text(0.5, 0.07, r'$\tau_{syn}$', ha='center', fontsize=45)
    fig.text(0.07, 0.5, r'$g_{syn}$', va='center', fontsize=45)
    fig.text(0.5, 0.9, title, ha='center', fontsize=45)
    pl.savefig("fig/spk-"+str('%.3f' % sigma[0])+".png")


# -----------------------------------------------------------------------#
n = 100
len_g = len(weight_coupling)
len_sigma = len(std_noise)
sel = None  # range(n)
# ----------------------------------------------#
hist_binwidth = 0.1
xlim = [t_sim-200, t_sim]
fwhm = 30
fs = 1000.0/dt
nperseg = 2**13

if __name__ == "__main__":

    start = time()
    # g = weight_coupling[::2]
    # sigma = std_noise

    # plot_raster_voltage_Isyn(weight_coupling[::1],
    #                          std_noise,
    #                          tau_syn_in[::1],
    #                          xlim=(t_trans + t_sim - 500.0,
    #                                t_trans + t_sim))
    for s in std_noise[1:]:
        plot_map_raster_voltage(
            weight_coupling[1::1],
            [s],
            tau_syn_in[::2],
            xlim=(t_trans + t_sim - 500.0,
                  t_trans + t_sim),
            # title=str(r"$\sigma=$%.2f, raster, $I_{Syn}$" % s),
            title=str(r"$\sigma=$%.2f, raster, $V_{glob}$" % s))
    # plot_bunch_rasters_activity(g, sigma)
    # plot_map_raster_activity(g, sigma)

    # plot_spike_synchrony(weight_coupling,
    #                      std_noise,
    #                      tau_syn_in,
    #                      xtickstep=2,
    #                      ytickstep=2,
    #                      xlabel='tau',
    #                      ylabel='g')
    #  ylabel=r"$g_{syn}$",
    #  xlabel=r"$\tau$")
    # plot_rhythm(g, sigma, True)
    # plot_firing_rate(g, eta, 1, 1)
    # plot_frequeicy(xtickstep=10, ytickstep=10,
    # vmax=60, vmin=0)
    # plot_isi()
    # plot_moments(g, eta[0], xlabel="g")

    lib.display_time(time()-start)
