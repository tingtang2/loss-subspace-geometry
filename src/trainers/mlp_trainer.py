import torch
import torchvision
import torchvision.transforms as transforms

from trainers.base_trainer import BaseTrainer

from models.mlp import MLP


class MLPTrainer(BaseTrainer):

    def __init__(self, dropout_prob, **kwargs) -> None:
        super().__init__(dropout_prob=dropout_prob, **kwargs)

        self.model = MLP(dropout=dropout_prob).to(self.device)

        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate)

        self.name = 'vanilla_mlp'

        self.early_stopping_threshold = 10


class FashionMNISTMLPTrainer(MLPTrainer):

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
