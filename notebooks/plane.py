import os
import sys

import numpy as np
import tabulate
import torch
from torch import nn
from tqdm import tqdm

# adding Folder_2/subfolder to the system path
sys.path.insert(
    0,
    '/gpfs/commons/home/tchen/loss_sub_space_geometry_project/loss-subspace-geometry/src'
)

import argparse

import torchvision
import torchvision.transforms as transforms
from models.mlp import NN, SubspaceNN

parser = argparse.ArgumentParser(
    description='Computes values for plane visualization')
parser.add_argument(
    '--subspace-shape',
    # default='line',
    default='simplex',
    help='shape of subspace you want to visualize')
parser.add_argument(
    '--model-path',
    default=
    # '/gpfs/commons/home/tchen/loss_sub_space_geometry_project/loss-subspace-geometry-save/models/subspace_vanilla_mlp_0.pt',
    '/gpfs/commons/home/tchen/loss_sub_space_geometry_project/loss-subspace-geometry-save/models/simplex_subspace_vanilla_mlp_0.pt',
    help='path to model weights file')
args = parser.parse_args()
configs = args.__dict__

# configs
data_dim = 784
hidden_size = 512
out_dim = 10
dropout_prob = 0.3
seed = 11202022
device = torch.device('cuda')

if configs['subspace_shape'] == 'line':
    num_weights = 2
elif configs['subspace_shape'] == 'simplex':
    num_weights = 3

curve_model = SubspaceNN(input_dim=data_dim,
                         hidden_dim=hidden_size,
                         out_dim=out_dim,
                         dropout_prob=dropout_prob,
                         seed=seed,
                         num_weights=num_weights).to(device)
checkpoint = torch.load(configs['model_path'])
curve_model.load_state_dict(checkpoint)

# more configs
curve_points = 61
grid_points = 21
margin_left = 0.2
margin_right = 0.2
margin_bottom = 0.2
margin_top = 0.2

curve_parameters = list(curve_model.parameters())
w = []

if configs['subspace_shape'] == 'line':
    w.append(
        np.concatenate([
            p.data.cpu().numpy().ravel() for p in [
                curve_parameters[0], curve_parameters[1], curve_parameters[3],
                curve_parameters[4]
            ]
        ]))

    isolated_model = NN(input_dim=data_dim,
                        hidden_dim=hidden_size,
                        out_dim=out_dim,
                        dropout_prob=dropout_prob).to(device)
    isolated_checkpoint = torch.load(
        '/gpfs/commons/home/tchen/loss_sub_space_geometry_project/loss-subspace-geometry-save/models/vanilla_mlp_0.pt'
    )
    isolated_model.load_state_dict(isolated_checkpoint)

    w.append(
        np.concatenate([
            p.data.cpu().numpy().ravel()
            for p in list(isolated_model.parameters())
        ]))

    w.append(
        np.concatenate([
            p.data.cpu().numpy().ravel() for p in [
                curve_parameters[2], curve_parameters[1], curve_parameters[5],
                curve_parameters[4]
            ]
        ]))
elif configs['subspace_shape'] == 'simplex':
    w.append(
        np.concatenate([
            p.data.cpu().numpy().ravel() for p in [
                curve_parameters[0], curve_parameters[1], curve_parameters[4],
                curve_parameters[5]
            ]
        ]))
    w.append(
        np.concatenate([
            p.data.cpu().numpy().ravel() for p in [
                curve_parameters[2], curve_parameters[1], curve_parameters[6],
                curve_parameters[5]
            ]
        ]))
    w.append(
        np.concatenate([
            p.data.cpu().numpy().ravel() for p in [
                curve_parameters[3], curve_parameters[1], curve_parameters[7],
                curve_parameters[5]
            ]
        ]))


# set up for grid for plane plotting
def get_xy(point, origin, vector_x, vector_y):
    return np.array(
        [np.dot(point - origin, vector_x),
         np.dot(point - origin, vector_y)])


print('Weight space dimensionality: %d' % w[0].shape[0])

u = w[2] - w[0]
dx = np.linalg.norm(u)
u /= dx

v = w[1] - w[0]
v -= np.dot(u, v) * u
dy = np.linalg.norm(v)
v /= dy

bend_coordinates = np.stack(get_xy(p, w[0], u, v) for p in w)


def get_weights(model: nn.Module, t: int, t_2: int = 0, type: str = 'line'):
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


if configs['subspace_shape'] == 'line':
    ts = np.linspace(0.0, 1.0, curve_points)
    curve_coordinates = []
    for t in np.linspace(0.0, 1.0, curve_points):
        weights = get_weights(model=curve_model, t=t)
        curve_coordinates.append(get_xy(weights, w[0], u, v))

    isolated_model_weights = w[2]
    curve_coordinates.append(get_xy(isolated_model_weights, w[0], u, v))
    curve_coordinates = np.stack(curve_coordinates)

elif configs['subspace_shape'] == 'simplex':
    ts = np.linspace(0.0, 1.0, curve_points)
    t_2s = np.linspace(0.0, 1.0, curve_points)

    # tile simplex subspace
    xx, yy = np.meshgrid(ts, t_2s)
    curve_coordinates = []
    for i in range(ts.shape[0]):
        for j in range(t_2s.shape[0]):
            t = xx[i, j]
            t_2 = yy[i, j]
            weights = get_weights(model=curve_model,
                                  t=t,
                                  t_2=t_2,
                                  type='simplex')
            curve_coordinates.append(get_xy(weights, w[0], u, v))

    curve_coordinates = np.stack(curve_coordinates)

G = grid_points
alphas = np.linspace(0.0 - margin_left, 1.0 + margin_right, G)
betas = np.linspace(0.0 - margin_bottom, 1.0 + margin_top, G)

tr_loss = np.zeros((G, G))
tr_nll = np.zeros((G, G))
tr_acc = np.zeros((G, G))
tr_err = np.zeros((G, G))

te_loss = np.zeros((G, G))
te_nll = np.zeros((G, G))
te_acc = np.zeros((G, G))
te_err = np.zeros((G, G))

grid = np.zeros((G, G, 2))

# even more configs for evaluating on FashionMNIST

data_dir = '/gpfs/commons/home/tchen/loss_sub_space_geometry_project/data/'
batch_size = 128

transform = transforms.Compose([transforms.ToTensor()])
FashionMNIST_data_train = torchvision.datasets.FashionMNIST(
    data_dir, train=True, transform=transform, download=False)

train_set, val_set = torch.utils.data.random_split(FashionMNIST_data_train,
                                                   [50000, 10000])
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_set,
                                           batch_size=len(val_set),
                                           shuffle=False)

criterion = nn.CrossEntropyLoss(reduction='sum')


def eval(model: nn.Module, loader):
    running_loss = 0.0
    num_right = 0

    model.eval()

    for i, (x, y) in enumerate(loader):
        reshaped_x = x.reshape(x.size(0), 784)
        y_hat = model(reshaped_x.to(device))
        num_right += torch.sum(
            y.to(device) == torch.argmax(y_hat, dim=-1)).detach().cpu().item()

        running_loss += criterion(y_hat, y.to(device)).item()

    return {
        'nll': running_loss / len(loader.dataset),
        'loss': running_loss / len(loader.dataset),
        'accuracy': num_right * 100.0 / len(loader.dataset),
    }


base_model = NN(input_dim=data_dim,
                hidden_dim=hidden_size,
                out_dim=out_dim,
                dropout_prob=dropout_prob).to(device)

columns = [
    'X', 'Y', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll',
    'Test error (%)'
]

for i, alpha in enumerate(tqdm(alphas)):
    for j, beta in enumerate(betas):
        p = w[0] + alpha * dx * u + beta * dy * v

        offset = 0
        for parameter in base_model.parameters():
            size = np.prod(parameter.size())
            value = p[offset:offset + size].reshape(parameter.size())
            parameter.data.copy_(torch.from_numpy(value))
            offset += size

        tr_res = eval(model=base_model, loader=train_loader)
        te_res = eval(model=base_model, loader=valid_loader)

        tr_loss_v, tr_nll_v, tr_acc_v = tr_res['loss'], tr_res['nll'], tr_res[
            'accuracy']
        te_loss_v, te_nll_v, te_acc_v = te_res['loss'], te_res['nll'], te_res[
            'accuracy']

        c = get_xy(p, w[0], u, v)
        grid[i, j] = [alpha * dx, beta * dy]

        tr_loss[i, j] = tr_loss_v
        tr_nll[i, j] = tr_nll_v
        tr_acc[i, j] = tr_acc_v
        tr_err[i, j] = 100.0 - tr_acc[i, j]

        te_loss[i, j] = te_loss_v
        te_nll[i, j] = te_nll_v
        te_acc[i, j] = te_acc_v
        te_err[i, j] = 100.0 - te_acc[i, j]

        values = [
            grid[i, j, 0], grid[i, j, 1], tr_loss[i, j], tr_nll[i, j],
            tr_err[i, j], te_nll[i, j], te_err[i, j]
        ]
        table = tabulate.tabulate([values],
                                  columns,
                                  tablefmt='simple',
                                  floatfmt='10.4f')
        if j == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

np.savez(os.path.join(
    '/gpfs/commons/home/tchen/loss_sub_space_geometry_project/loss-subspace-geometry-save/',
    'simplex_plane.npz'),
         ts=ts,
         bend_coordinates=bend_coordinates,
         curve_coordinates=curve_coordinates,
         alphas=alphas,
         betas=betas,
         grid=grid,
         tr_loss=tr_loss,
         tr_acc=tr_acc,
         tr_nll=tr_nll,
         tr_err=tr_err,
         te_loss=te_loss,
         te_acc=te_acc,
         te_nll=te_nll,
         te_err=te_err)
