#!/usr/bin/env python3

# Dependencies:
# sudo apt-get install python3-matplotlib python3-scipy

# Usage:
# python dendro.py LISTS...

# Plot code derived from Jed.

#CLUSTER_ALGO = 'single' # nearest point algorithm
#CLUSTER_ALGO = 'complete' # farthest point algorithm
CLUSTER_ALGO = 'centroid'
#FONT = 'TeX Gyre Termes'
#FONT = 'TeX Gyre Heros'
#FONT = 'AR PL UMing CN'
FONT = 'Latin Modern Math' # inkscape can edit with this font losslessly
#FONT = 'Arial'
MATH_FONT = 'Latin Modern Math'

TICKS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

import csv
import math
import os
import sys

import numpy as np

import matplotlib

matplotlib.use('pgf')
pgf_options = {
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'xelatex',
    'pgf.preamble': [
        r'\usepackage{unicode-math}',
    ] + ([
        r'\setmainfont{%s}' % FONT,
        r'\setsansfont{%s}' % FONT,
        r'\setmonofont{%s}' % FONT,
    ] if FONT else []) + ([
        r'\setmathfont{%s}' % MATH_FONT,
    ] if MATH_FONT else [])
}
matplotlib.rcParams.update(pgf_options)

from matplotlib.backends.backend_pgf import RendererPgf
RendererPgf.draw_image_old = RendererPgf.draw_image
def draw_image(self, gc, x, y, im, *args, **kwargs):
    if len(im.shape) == 3 and im.shape[2] == 4 and np.all(im[:, :, 3] == 255):
        # Remove unused transparency channel.  This generates the pdf without
        # unnecessary smasks, which make it friendlier to edit with tools.
        im = im[:, :, :3]
    return self.draw_image_old(gc, x, y, im, *args, **kwargs)
RendererPgf.draw_image = draw_image

from matplotlib import pyplot as plt

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from strings import Strings
import metrics

METRIC_DICT = {name: getattr(metrics, name)
               for name in dir(metrics)
               if not name.startswith('_')
               and callable(getattr(metrics, name))}

def read_strings(path, name=None):
    '''Read newline-delimited utf-8 strings from file as Strings instance.'''
    try:
        with open(path, 'rb') as f:
            strings = []
            for i, line in enumerate(f):
                try:
                    line = line.decode('utf-8')
                except UnicodeDecodeError:
                    sys.stderr.write('%s: error decoding line %d\n' %
                                     (path, i + 1))
                else:
                    stripped = line.strip()
                    if stripped:
                        strings.append(stripped)
    except EnvironmentError as e:
        sys.stderr.write('Error reading `%s\': %s\n' % (path, e))
    else:
        if name is None:
            name = os.path.basename(path).rsplit('.', 1)[0]
        return Strings(strings, name)
    return None

def create_distance_matrix(lists, dist):
    n = len(lists)
    M = np.empty((n, n))
    for i, xs in enumerate(lists):
        for j, ys in enumerate(lists):
            M[i][j] = dist(xs, ys)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            D[i][j] = D[j][i] = 1.0 - max(M[i][j], M[j][i])
    return D, M

def plot(plot_name, names_lists, D, M, fig_size=None):
    dpi = 100
    matrix_frac = 0.5
    if fig_size is None:
        fig_size = D.shape[0] / dpi / matrix_frac
        fig_size *= math.ceil(12 / fig_size)

    P = np.empty_like(D)
    P[:, :] = 2.0
    n = 0
    for names_list in names_lists:
        m = len(names_list)
        P[n:n + m,n:n + m] = 0.0
        n += m
    D = D + P

    names = [name for names_list in names_lists for name in names_list]
    ticklabel_size = max(int(72 * matrix_frac * fig_size // n) - 1, 1)

    # Compute dendrograms.
    C = squareform(D, force='tovector')
    Y = sch.linkage(C, method=CLUSTER_ALGO, optimal_ordering=True)

    # Create figure.
    fig = plt.figure(figsize=(fig_size, fig_size))

    # Plot first dendrogram.
    ax1 = fig.add_axes([0.02, 0.25, 0.2, matrix_frac])
    Z1 = sch.dendrogram(Y, orientation='left', color_threshold=1.2, ax=ax1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.invert_yaxis()

    # Plot second dendrogram.
    ax2 = fig.add_axes([0.23, 0.76, matrix_frac, 0.2])
    Z2 = sch.dendrogram(Y, color_threshold=1.2, ax=ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.23, 0.25, matrix_frac, matrix_frac])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    H = M[idx1,:]
    H = H[:,idx2]
    im = axmatrix.matshow(H, aspect='auto', vmin=0.0, vmax=1.0, cmap=plt.cm.YlGnBu)
    idx1_names = [names[i] for i in idx1]
    idx2_names = [names[i] for i in idx2]

    # Print matrix
    with open('%s.csv' % plot_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([''] + idx1_names)
        for name, row in zip(idx2_names, H):
            writer.writerow([name] + list(row))

    # Set up Y axis ticks.
    axmatrix.set_yticks(range(n))
    axmatrix.yaxis.set_label_position('right')
    axmatrix.yaxis.tick_right()

    while True:
        axmatrix.set_yticklabels(idx2_names, fontsize=ticklabel_size)
        if ticklabel_size == 1:
            break
        _, bbox = axmatrix.yaxis.get_ticklabel_extents(fig.canvas.get_renderer())
        bbox = bbox.inverse_transformed(fig.transFigure)
        ticklabel_width = bbox.width
        max_ticklabel_width = 0.92 - (0.23 + matrix_frac) - 2 * max(bbox.x0 - matrix_frac - 0.23, 0)
        if ticklabel_width < max_ticklabel_width:
            break
        ticklabel_size = max_ticklabel_width * ticklabel_size // ticklabel_width
        ticklabel_size = min(ticklabel_size, ticklabel_size - 1)
        ticklabel_size = max(ticklabel_size, 1)

    # Set up X axis ticks.
    axmatrix.set_xticks(range(n))
    axmatrix.set_xticklabels(idx1_names, fontsize=ticklabel_size, rotation=90)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    # Plot colorbar.
    colorbar_left = 0.92 if any(idx2_names) else 0.23 + matrix_frac + 0.02
    axcolor = fig.add_axes([colorbar_left, 0.25, 0.02, matrix_frac])
    axcolor.tick_params(labelsize=max(fig_size * 1.5, 1))
    cb = plt.colorbar(im, cax=axcolor, ticks=TICKS)
    cb.patch.set_visible(False)

    # Save figure.
    fig.savefig('dendro-%s.pdf' % plot_name, dpi=dpi, transparent=True)

    # Create heatmap-only figure.
    fig = plt.figure(figsize=(fig_size, fig_size))

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.23, 0.25, matrix_frac, matrix_frac])
    im = axmatrix.matshow(H, aspect='auto', vmin=0.0, vmax=1.0, cmap=plt.cm.YlGnBu)

    # Set up Y axis ticks.
    axmatrix.set_yticks(range(n))
    axmatrix.set_yticklabels(idx2_names, fontsize=ticklabel_size)
    axmatrix.yaxis.set_label_position('right')
    axmatrix.yaxis.tick_right()

    # Set up X axis ticks.
    axmatrix.set_xticks(range(n))
    axmatrix.set_xticklabels(idx1_names, fontsize=ticklabel_size, rotation=90)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    # Plot colorbar.
    colorbar_left = 0.92 if any(idx2_names) else 0.23 + matrix_frac + 0.02
    axcolor = fig.add_axes([colorbar_left, 0.25, 0.02, matrix_frac])
    axcolor.tick_params(labelsize=max(fig_size * 1.5, 1))
    cb = plt.colorbar(im, cax=axcolor, ticks=TICKS)
    cb.patch.set_visible(False)

    # Save heatmap-only figure.
    fig.savefig('heatmap-%s.pdf' % plot_name, dpi=dpi, transparent=True)

def create_distance_matrix_and_plot(plot_name, lists_list, metric_d, metric_m=None, *args, **kwargs):
    all_lists = [l for lists in lists_list for l in lists]
    D, M = create_distance_matrix(all_lists, metric_d)
    if metric_m is not None and metric_d != metric_m:
        _, M = create_distance_matrix(all_lists, metric_m)
    names = [[xs.name for xs in lists] for lists in lists_list]
    plot(plot_name, names, D, M, *args, **kwargs)
    return D, M

def main(show_labels=True, *args, **kwargs):
    if len(sys.argv) <= 2:
        sys.stderr.write('Usage: %s LISTS...\n' % sys.argv[0])
        sys.exit(1)
    lists = []
    name = None if show_labels else ''
    for path in sys.argv[1:]:
        strings = read_strings(path, name)
        if strings is not None:
            lists.append(strings)
    if not lists:
        sys.stderr.write('No lists specified\n')
        sys.exit(1)
    for metric_name, metric in sorted(METRIC_DICT.items()):
        create_distance_matrix_and_plot(metric_name, [lists], metric, None, *args, **kwargs)

if __name__ == '__main__':
    main()
