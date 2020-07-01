#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shower."""

import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import os
import yaml

from yaml import Loader
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa


def extract_scatterings(dir_summary, tag, epoch=0):
    """Extract scattering coefficients."""
    # Load arguments
    file_args = os.path.join(os.path.join(dir_summary, tag), 'args.yaml')
    args = yaml.load(open(file_args).read(), Loader=Loader)
    j = args['layers']['j']
    q = args['layers']['q']

    file_path = os.path.join(dir_summary, tag, 'scatterings.h5')
    with h5py.File(file_path, 'r') as hf:
        time = hf['time'].value
        scat = hf['epoch_{:05d}'.format(epoch)].value
    return time, scat, j, q


def show_entropy(dir_summary, tag, epoch=0, dir_output=None,
                 save='entropy.png'):
    """Show seismic data in scattering latent space."""
    # Manage output directory
    if dir_output is None:
        dir_output = os.path.join(dir_summary, tag)
    else:
        dir_output = os.path.join(dir_output, tag)
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

    # Load latent variables
    file_path = os.path.join(dir_summary, tag, 'scatterings.h5')
    with h5py.File(file_path, 'r') as hf:
        time = hf['time'].value
        scat = hf['epoch_{:05d}'.format(epoch)].value

    # Cummulatives
    scat = np.exp(scat)
    energy = np.sum(scat, axis=1)
    scat /= scat.sum(axis=0, keepdims=True)
    entropy = - np.sum(scat * np.log(scat), axis=1)

    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(time, entropy)
    ax[0].plot(time, energy)
    ax[0].set_xlabel('Time')
    xticks = md.AutoDateLocator()
    ax[0].xaxis.set_major_locator(xticks)
    ax[0].xaxis.set_major_formatter(md.AutoDateFormatter(xticks))
    ax[0].set_ylabel('Entropy')
    ax[0].set_yscale('log')
    ax[0].grid()
    ax[0].set_title('Scattering entropy')

    ax[1].plot(entropy, energy, '.', ms=3)
    ax[1].set_xlabel('Energy')
    ax[1].set_ylabel('Entropy')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].grid()
    ax[1].set_title('Phase diagram')

    if save is not None:
        fig.savefig(os.path.join(dir_output, save), dpi=150)
    return fig, ax


def show_detections(dir_summary, tag, epoch=0, dir_output=None,
                    save='detections_{:05d}.png', norm=None):
    """Show seismic data in scattering latent space."""
    # Manage output directory
    if dir_output is None:
        dir_output = os.path.join(dir_summary, tag)
    else:
        dir_output = os.path.join(dir_output, tag)
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

    # Load latent variables
    file_path = os.path.join(dir_summary, tag, 'clusters.h5')
    with h5py.File(file_path, 'r') as hf:
        time = hf['time'].value
        hot = hf['epoch_{:05d}'.format(epoch)]['hot'].value

    # Cummulatives
    hot_unique = np.unique(hot)
    print(hot_unique)
    n_samples = len(hot)
    n_cat = len(hot_unique)
    for i, h in enumerate(hot_unique):
        hot[hot == h] = i
    one_hot = np.zeros((n_samples, n_cat))
    one_hot[np.arange(n_samples), hot] = 1
    cumulatives = np.cumsum(one_hot, axis=0)
    if norm is not None:
        cumulatives /= cumulatives.max(axis=0, keepdims=True)

    # Create figure
    fig, ax = plt.subplots(1, figsize=(10, 4))
    for cumulative in cumulatives.T:
        ax.plot(time, cumulative)
    ax.set_xlabel('Time')
    xticks = md.AutoDateLocator()
    ax.xaxis.set_major_locator(xticks)
    ax.xaxis.set_major_formatter(md.AutoDateFormatter(xticks))
    ax.set_ylabel('Number of events')
    ax.grid()
    ax.set_title('Cumulative within-clusters detections')

    if save is not None:
        fig.savefig(os.path.join(dir_output, save.format(epoch)), dpi=150)
    return fig, ax


def show_latent(dir_summary, tag, epoch=0, dir_output=None, save='latent.png'):
    """Show seismic data in scattering latent space."""
    # Manage output directory
    if dir_output is None:
        dir_output = os.path.join(dir_summary, tag)
    else:
        dir_output = os.path.join(dir_output, tag)
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

    # Load latent variables
    file_path = os.path.join(dir_summary, tag, 'clusters.h5')
    with h5py.File(file_path, 'r') as hf:
        latent = hf['epoch_{:05d}'.format(epoch)]['features'].value
        means = hf['epoch_{:05d}'.format(epoch)]['means'].value
        covariances = hf['epoch_{:05d}'.format(epoch)]['covariance'].value

    # Create figure
    fig, ax = plt.subplots(1)
    ax.plot(*latent[:, :2].T, 'k.', ms=3)
    # for i, (mean, covariance) in enumerate(zip(means, covariances)):
    #     add_covariance(ax, mean, covariance, index=i)

    ax.set_xlabel('First latent variable')
    ax.set_ylabel('Second latent variable')
    ax.grid()
    ax.set_title('Scattering latent space')

    if save is not None:
        fig.savefig(os.path.join(dir_output, save), dpi=150)
    return fig, ax


def show_clusters(dir_summary, tag, epoch=0, dir_output=None,
                  save='clusters.png'):
    """Show seismic data in scattering latent space."""
    # Manage output directory
    if dir_output is None:
        dir_output = os.path.join(dir_summary, tag)
    else:
        dir_output = os.path.join(dir_output, tag)
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

    # Load latent variables
    file_path = os.path.join(dir_summary, tag, 'clusters.h5')
    with h5py.File(file_path, 'r') as hf:
        latent = hf['epoch_{:05d}'.format(epoch)]['features'].value
        hot = hf['epoch_{:05d}'.format(epoch)]['hot'].value

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for h in np.unique(hot):
        ax.plot(*latent[hot == h].T, '.', ms=3, alpha=.4)

    # for i, (mean, covariance) in enumerate(zip(means, covariances)):
    #     add_covariance(ax, mean, covariance, index=i)

    ax.set_xlabel('First latent variable')
    ax.set_ylabel('Second latent variable')
    ax.grid()
    ax.set_title('Scattering latent space')
    # plt.show()

    if save is not None:
        fig.savefig(os.path.join(dir_output, save), dpi=150)
    return fig, ax


def show_graph(dir_summary, tag, dir_output=None, save='graph.png'):
    """Representation of the graph saved from the main execution.

    Plots the different time and frequency properties of the graph at each
    layer.

    Arguments
    ---------
    dir_summary: str
        Root path to the summary directory.
    tag: str
        Run tag. Should include the subdirectory path if exist.

    Keyword arguments
    -----------------
    dir_output: str or None
        Root path where to store the image. By default, ``dir_sumary``.
    save: bool
        If ``True`` (default), save the figure in the output path, under name
        ``graph.png``.

    Returns
    -------
    fig: :class:`~matplotlib.pyplot.Figure()`
        The figure instance.
    ax: :class:`~matplotlib.pyplot.Axes()`
        The axe instance.
    """
    # Load graph
    file_graph = os.path.join(os.path.join(dir_summary, tag), 'arch.yaml')

    # Manage output directory
    if dir_output is None:
        dir_output = os.path.join(dir_summary, tag)
    else:
        dir_output = os.path.join(dir_output, tag)
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

    # Load graph
    graph = yaml.load(open(file_graph).read(), Loader=Loader)
    depth = len(graph)
    keys = list(graph.keys())

    # Create figure
    fig, ax = plt.subplots(2, sharex=True)

    # Exctract info
    patch_shape = [graph[key]['patch_shape'] for key in keys]
    sampling_rate = [graph[key]['sampling_rate'] for key in keys]
    largest_period = [graph[key]['largest_period'] for key in keys[1:-1]]

    # Plot unmber of samples
    ax[0].plot(patch_shape, 'o')
    ax[0].grid()
    ax[0].set_yscale('log', basey=2)
    ax[0].set_title('Graph time scales')
    ax[0].set_ylabel('Patch shape')

    for i, (p_max, f_max) in enumerate(
            zip(largest_period, sampling_rate[:-2])):
        h1 = ax[1].plot(2 * [i + 1], [1 / p_max, f_max / 2], 'k-_')
    for i, f_max in enumerate(sampling_rate[1:]):
        h2 = ax[1].plot(i + 1, f_max, 'x', c='C1')
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Frequency (Hz)')
    ax[1].grid()

    # xaxis ticks
    ticks_layer = ['L{}'.format(i) for i in range(1, depth - 1)]
    ax[1].set_xticks(range(depth))
    ax[1].set_xticklabels(['INPUT'] + ticks_layer + ['SC'])
    ax[1].set_xlabel('Layer index')
    ax[1].legend([h1[0], h2[0]], ['Frequency range', 'Sampling rate'])

    if save is not None:
        fig.savefig(os.path.join(dir_output, save), dpi=150)

    return fig, ax


def add_covariance(ax, mean, covariance, index=0):
    """Draw covariance on given axe."""
    # Get ellipse axes
    v, w = np.linalg.eigh(covariance)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle *= 180 / np.pi

    # Draw surface ellipse
    ell = Ellipse(mean, *v, 180 + angle, alpha=.4, lw=0, zorder=2,
                  facecolor='C{}'.format(index))
    ax.add_artist(ell)

    # Draw edge ellipse
    ell = Ellipse(mean, *v, 180 + angle, lw=.4, facecolor='None', zorder=3,
                  edgecolor='C{}'.format(index))
    ax.add_artist(ell)

    pass
