import argparse
import sys

import numpy as np
import skdim
import torch
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


def get_weights(model: nn.Module,
                t: float,
                t_2: float = 0,
                type: str = 'line'):
    weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # add attribute for weight dimensionality and subspace dimensionality
            if type == 'line':
                setattr(module, f'alpha', t)
            elif type == 'simplex':
                setattr(module, f't1', t)
                setattr(module, f't2', t_2)

            weights.extend([module.get_weight(), module.bias.data])
    return np.concatenate([w.detach().cpu().numpy().ravel() for w in weights])


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Analyze geometry of neural loss subspaces')

    parser.add_argument(
        '--model-path',
        default=
        '/gpfs/commons/home/tchen/loss_sub_space_geometry_project/loss-subspace-geometry-save/models/subspace_vanilla_mlp_0.pt',
        # '/home/tristan/loss-subspace-geometry-project/loss-subspace-geometry-save/',
        help='path to saved model files')
    parser.add_argument('--num-samples',
                        default=200,
                        type=int,
                        help='number of samples from model subspace')
    parser.add_argument(
        '--estimation-type',
        default='knn',
        help='kind of estimation (dimensionality/curvature) to perform')

    args = parser.parse_args()
    configs = args.__dict__

    # load model
    data_dim = 784
    hidden_size = 512
    out_dim = 10
    dropout_prob = 0.3
    seed = 11202022
    device = torch.device('cpu')

    if 'simplex' in configs['model_path']:
        num_weights = 3
    else:
        num_weights = 2

    model = SubspaceNN(input_dim=data_dim,
                       hidden_dim=hidden_size,
                       out_dim=out_dim,
                       dropout_prob=dropout_prob,
                       seed=seed,
                       num_weights=num_weights).to(device)
    checkpoint = torch.load(configs['model_path'])
    model.load_state_dict(checkpoint)

    sample_weights = []
    if 'simplex' in configs['model_path']:
        # TODO: Look at random vs. tiling for simplex sampling (random looks bad)
        # dist = Exponential(rate=1)
        # Z = dist.sample(sample_shape=(configs['num_samples'], 3))
        # Z = Z / Z.sum()
        # for idx in range(Z.shape[0]):
        #     sample_weights.append(
        #         get_weights(model=model,
        #                     t=Z[idx][1],
        #                     t_2=Z[idx][2],
        #                     type='simplex'))
        ts = np.linspace(0.0, 1.0, configs['num_samples'])
        t_2s = np.linspace(0.0, 1.0, configs['num_samples'])

        # tile simplex subspace
        xx, yy = np.meshgrid(ts, t_2s)
        for i in range(ts.shape[0]):
            for j in range(t_2s.shape[0]):
                t = xx[i, j]
                t_2 = yy[i, j]
                sample_weights.append(
                    get_weights(model=model, t=t, t_2=t_2, type='simplex'))

    else:
        # sample from line
        sample_alphas = np.linspace(start=0,
                                    stop=1,
                                    num=configs['num_samples'])

        for alpha in sample_alphas:
            sample_weights.append(get_weights(model=model, t=alpha))

    k_s = [3, 5, 8, 13]
    print(f'num samples of weights: {len(sample_weights)}')
    if configs['estimation_type'] == 'knn':
        for k in tqdm(k_s):
            print(
                f'num: neighbors: {k}, dim estimate: {estimate_tangent_plane(X=np.vstack(sample_weights),num_neighbors=k)}'
            )
    elif configs['estimation_type'] == 'mle':
        for k in tqdm(k_s):
            print(
                f'num: neighbors: {k}, dim estimate: {skdim.id.MLE(K=k, neighborhood_based=False).fit_predict(np.vstack(sample_weights))}'
            )
    elif configs['estimation_type'] == 'skdim_knn':
        for k in tqdm(k_s):
            print(
                f'num: neighbors: {k}, dim estimate: {skdim.id.KNN(k=k).fit(np.vstack(sample_weights)).dimension_}'
            )
    elif configs['estimation_type'] == 'triangle_empirical_curvature':
        estimate(dataset=np.vstack(sample_weights))

    return 0


if __name__ == '__main__':
    sys.exit(main())
