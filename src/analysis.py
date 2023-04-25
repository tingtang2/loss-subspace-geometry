import argparse
import sys

import numpy as np
import torch
from models.mlp import SubspaceNN
from scipy.linalg import orth
from sklearn.neighbors import NearestNeighbors
from torch import nn


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


def get_weights(model: nn.Module, alpha: float):
    weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # add attribute for weight dimensionality and subspace dimensionality
            setattr(module, f'alpha', alpha)
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

    args = parser.parse_args()
    configs = args.__dict__

    # load model
    data_dim = 784
    hidden_size = 512
    out_dim = 10
    dropout_prob = 0.3
    seed = 11202022
    device = torch.device('cuda')

    model = SubspaceNN(input_dim=data_dim,
                       hidden_dim=hidden_size,
                       out_dim=out_dim,
                       dropout_prob=dropout_prob,
                       seed=seed).to(device)
    checkpoint = torch.load(configs['model_path'])
    model.load_state_dict(checkpoint)

    # sample from line
    sample_alphas = np.linspace(start=0, stop=1, num=100)
    sample_weights = []

    for alpha in sample_alphas:
        sample_weights.append(get_weights(model=model, alpha=alpha))

    k_s = [3, 5, 8, 13]

    for k in k_s:
        print(
            estimate_tangent_plane(X=np.vstack(sample_weights),
                                   num_neighbors=k))

    return 0


if __name__ == '__main__':
    sys.exit(main())
