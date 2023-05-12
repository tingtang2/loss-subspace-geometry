import argparse
import os

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

parser = argparse.ArgumentParser(description='Plane visualization')
parser.add_argument('--dir',
                    type=str,
                    default='../src/',
                    metavar='DIR',
                    help='training directory')
parser.add_argument('--file',
                    type=str,
                    default='plane',
                    help='training file')
parser.add_argument('--scale',
                    type=str,
                    default='log')

parser.add_argument('--log_alpha',
                    type=float,
                    default=-5.0)

parser.add_argument('--curve',
                    type=bool,
                    default=False)

args = parser.parse_args()

savedir_root = '/home/tristan/loss-subspace-geometry-project/loss-subspace-geometry-save/images'

file = np.load(os.path.join(args.dir, args.file+'.npz'))
print(file['grid'].shape)
# matplotlib.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{sansmath}')
# matplotlib.rc('font', **{
#     'family': 'sans-serif',
#     'sans-serif': ['DejaVu Sans']
# })

matplotlib.rc('xtick.major', pad=12)
matplotlib.rc('ytick.major', pad=12)
matplotlib.rc('grid', linewidth=0.8)

sns.set_style('whitegrid')


class LogNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        log_v = np.ma.log(value - self.vmin)
        log_v = np.ma.maximum(log_v, self.log_alpha)
        return 0.9 * (log_v - self.log_alpha) / (
            np.log(self.vmax - self.vmin) - self.log_alpha)


def plane(grid, values, scale, vmax=None, log_alpha=-5, N=7, cmap='jet_r'):
    cmap = plt.get_cmap(cmap)
    if vmax is None:
        clipped = values.copy()
    else:
        clipped = np.minimum(values, vmax)
    if scale == "log":
        log_gamma = (np.log(clipped.max() - clipped.min()) - log_alpha) / N
        levels = clipped.min() + np.exp(log_alpha + log_gamma * np.arange(N + 1))
        levels[0] = clipped.min()
        levels[-1] = clipped.max()
        norm = LogNormalize(clipped.min() - 1e-8,
                    clipped.max() + 1e-8,
                    log_alpha=log_alpha)
    elif scale == "std":
        gamma = (clipped.max() - clipped.min())/N
        levels = clipped.min() + gamma*np.arange(N+1)
        norm = None
    else:
        raise Exception("invalid scale argument")

    levels = np.concatenate((levels, [1e10]))

    contour = plt.contour(grid[:, 0, 0],
                          grid[0, :, 1],
                          values,
                          cmap=cmap,
                          norm=norm,
                          linewidths=2.5,
                          zorder=1,
                          levels=levels)
    contourf = plt.contourf(grid[:, 0, 0],
                            grid[0, :, 1],
                            values,
                            cmap=cmap,
                            norm=norm,
                            levels=levels,
                            zorder=0,
                            alpha=0.55)
    colorbar = plt.colorbar(format='%.2g')
    label_numbers = colorbar.get_ticks()
    labels = colorbar.ax.get_yticklabels()
    for i, label in enumerate(labels):
        label.set_text(f'{label_numbers[i]:.2f}')

    labels[-1].set_text(r'$>\,$' + labels[-2].get_text())
    colorbar.ax.set_yticklabels(labels)
    return contour, contourf, colorbar


plt.figure(figsize=(12.4, 7))

log_alpha = args.log_alpha

contour, contourf, colorbar = plane(file['grid'],
                                    file['tr_loss'],
                                    vmax=None,
                                    log_alpha=log_alpha,
                                    N=7,
                                    scale=args.scale)

bend_coordinates = np.array(file['bend_coordinates'])
print(bend_coordinates)

plt.scatter(bend_coordinates[:2,0],
            bend_coordinates[:2,1],
            marker='o',
            c='k',
            s=120,
            zorder=2)
plt.scatter(bend_coordinates[2, 0],
            bend_coordinates[2, 1],
            marker='D',
            c='k',
            s=120,
            zorder=2)
if args.curve:
    curve_coordinates = np.array(file['curve_coordinates'])
    plt.plot(curve_coordinates[:, 0],
            curve_coordinates[:, 1],
            linewidth=4,
            c='k',
            label='$w(t)$',
            zorder=4)


plt.margins(0.0)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
colorbar.ax.tick_params(labelsize=18)
plt.savefig(os.path.join(savedir_root, args.file+'_logalpha_'+str(args.log_alpha)+'_train.pdf'),
            format='pdf',
            bbox_inches='tight')

plt.figure(figsize=(12.4, 7))


contour, contourf, colorbar = plane(file['grid'],
                                    file['te_err'],
                                    vmax=None,
                                    log_alpha=log_alpha,
                                    N=7,
                                    scale=args.scale)


plt.scatter(bend_coordinates[:2,0],
            bend_coordinates[:2,1],
            marker='o',
            c='k',
            s=120,
            zorder=2)
plt.scatter(bend_coordinates[2, 0],
            bend_coordinates[2, 1],
            marker='D',
            c='k',
            s=120,
            zorder=2)
if args.curve:
    plt.plot(curve_coordinates[:, 0],
            curve_coordinates[:, 1],
            linewidth=4,
            c='k',
            label='$w(t)$',
            zorder=4)
    
plt.margins(0.0)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
colorbar.ax.tick_params(labelsize=18)
plt.savefig(os.path.join(savedir_root, args.file+'_logalpha_'+str(args.log_alpha)+'_test.pdf'),
            format='pdf',
            bbox_inches='tight')
# plt.show()