import logging

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import trange

from models.mlp import NN
from trainers.base_trainer import BaseTrainer


class MLPTrainer(BaseTrainer):

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()

            reshaped_x = x.reshape(x.size(0), 784)

            y_hat = self.model(reshaped_x.to(self.device))
            loss = self.criterion(y_hat, y.to(self.device))

            loss.backward()
            running_loss += loss.item()

            self.optimizer.step()

        return running_loss / len(loader.dataset)

    def eval(self, loader: DataLoader):
        num_right = 0
        running_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                reshaped_x = x.reshape(x.size(0), 784)
                y_hat = self.model(reshaped_x.to(self.device))
                num_right += torch.sum(
                    y.to(self.device) == torch.argmax(
                        y_hat, dim=-1)).detach().cpu().item()

                running_loss += self.criterion(y_hat, y.to(self.device)).item()

        return num_right / len(loader.dataset), (running_loss /
                                                 len(loader.dataset))

    def run_experiment(self):
        self.create_dataloaders()

        self.model = NN(input_dim=self.data_dim,
                        hidden_dim=self.hidden_size,
                        out_dim=self.out_dim,
                        dropout_prob=self.dropout_prob).to(self.device)

        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate)

        training_loss = []
        val_loss = []
        training_accuracy = []
        val_accuracy = []

        best_val_loss = 1e+5
        early_stopping_counter = 0

        for i in trange(1, self.epochs + 1):
            training_loss.append(self.train_epoch(self.train_loader))
            training_accuracy.append(self.eval(self.train_loader)[0])

            acc, loss = self.eval(self.valid_loader)
            val_accuracy.append(acc)
            val_loss.append(loss)

            logging.info(
                f'epoch: {i} training loss: {training_loss[-1]:.3f} val loss:{val_loss[-1]:.3f} training accuracy: {training_accuracy[-1]:.3f} val acc: {val_accuracy[-1]:.3f}, patience: {early_stopping_counter}'
            )

            if loss < best_val_loss:
                self.save_model(self.name)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter == self.early_stopping_threshold:
                break


class FashionMNISTMLPTrainer(MLPTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.name = 'vanilla_mlp'
        self.early_stopping_threshold = 10

        self.data_dim = 784
        self.out_dim = 10

    def create_dataloaders(self):
        transform = transforms.Compose([transforms.ToTensor()])
        FashionMNIST_data_train = torchvision.datasets.FashionMNIST(
            self.data_dir, train=True, transform=transform, download=False)

        train_set, val_set = torch.utils.data.random_split(
            FashionMNIST_data_train, [50000, 10000])
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(
            val_set, batch_size=len(val_set), shuffle=False)
