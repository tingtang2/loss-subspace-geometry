import argparse
import sys
import os

import numpy as np
import skdim
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import orth
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.distributions.exponential import Exponential
from tqdm import tqdm

from models.mlp import SubspaceNN
from other_scripts.curvature import estimate


def estimate_tangent_plane(X: np.ndarray, num_neighbors: int) -> int:
    knn = NearestNeighbors(n_neighbors=num_neighbors + 1)

    knn.fit(X)

    dim_estimates = []

    for point in range(X.shape[0]):
        k_neighbors = knn.kneighbors(X[point].reshape(1, -1),
                                     return_distance=False)

        basis = orth(X[k_neighbors[0, 1:]].T)
        dim_estimates.append(basis.shape[1])

    return sum(dim_estimates) / len(dim_estimates)

def calc_ds(data):
    diff = np.diff(data, axis = 0)
    ds = np.expand_dims(np.linalg.norm(diff, axis=1), axis=1)
    return ds

def calc_T(data):
    diff = np.diff(data, axis = 0)
    T = diff/np.linalg.norm(diff, axis=1)[:, None]
    return T

def calc_dT(data):
    T = calc_T(data)
    dT = np.diff(T, axis=0)
    return dT

def calc_curvature(data, alpha):
    dT = calc_dT(data)
    ds = calc_ds(data)[1:]
    curvature = np.linalg.norm(dT/ds, axis=1)
    return curvature

def plot_curvature(alpha, curvature, savedir, filename):
    plt.figure(figsize=(12.4, 7))
    plt.plot(alpha[1:-1], curvature)
    plt.xlabel('alpha')
    plt.ylabel('curvature')
    plt.savefig(os.path.join(savedir, filename+'.pdf'),
            format='pdf',
            bbox_inches='tight')

def estimate_curvature(X: np.ndarray) -> int:

    return 0

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Analyze geometry of neural loss subspaces')

    parser.add_argument(
        '--dir',
        default=
        # '/gpfs/commons/home/tchen/loss_sub_space_geometry_project/loss-subspace-geometry-save/models/subspace_vanilla_mlp_0.pt',
        '/home/tristan/loss-subspace-geometry-project/loss-subspace-geometry-save/',
        help='path to sampled weight file')
    parser.add_argument(
        '--file')
    parser.add_argument(
        '--estimation-type',
        default='knn',
        help='kind of estimation (dimensionality/curvature) to perform')
    parser.add_argument(
       '--savedir',
       default= '/home/tristan/loss-subspace-geometry-project/loss-subspace-geometry-save/images'
    )

    args = parser.parse_args()
    

    file = np.load(os.path.join(args.dir, args.file+'.npz'))
    sample_weights = file['arr_0']
    if 'alpha' in file:
       alpha = file['alpha']
    else:
       alpha = np.linspace(0, 1, 1000)

    k_s = [3, 5, 8, 13]
    # for k in tqdm(k_s):
    #     print(
    #         f'num: neighbors: {k}, dim estimate: {skdim.id.KNN(k=k).fit(sample_weights).dimension_}'
    #     )
    curvature = calc_curvature(sample_weights, alpha)
    mean_curvature = np.mean(curvature)
    plot_curvature(alpha, curvature, args.savedir, args.file)

    print(mean_curvature)
    return 0


if __name__ == '__main__':
    sys.exit(main())
