import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import read_dir, setup_seed

femnist_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(femnist_dir, 'train')
test_path = os.path.join(femnist_dir, 'test')
train_clients, train_dataset = read_dir(train_path)
test_clients, test_dataset = read_dir(test_path)
assert train_clients.sort() == test_clients.sort()
all_clients = train_clients


class FEMNIST_DATASET(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]
        assert len(x) == 784
        x = np.array(x)
        x = x.reshape((28, 28, 1))
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.tensor(x)
        x = x.float()
        y = torch.tensor(y).long()
        return x, y

    def __len__(self):
        return len(self.labels)


def get_femnist_dataloaders(batch_size=10, transform=None, poison_swap_label=2):
    setup_seed(rs=24)
    train_loaders, test_loaders, attack_test_loaders = [], [], []
    for client in all_clients:
        train_x = train_dataset[client]['x']
        train_y = train_dataset[client]['y']
        test_x = test_dataset[client]['x']
        test_y = test_dataset[client]['y']
        attack_test_ids = np.where(np.array(test_y) != poison_swap_label)[0]
        attack_test_x = [x for (i, x) in enumerate(test_x) if i in attack_test_ids]
        attack_test_y = [y for (i, y) in enumerate(test_y) if i in attack_test_ids]

        train_d = FEMNIST_DATASET(data=train_x, labels=train_y, transform=transform)
        test_d = FEMNIST_DATASET(data=test_x, labels=test_y, transform=transform)
        attack_d = FEMNIST_DATASET(data=attack_test_x, labels=attack_test_y, transform=transform)
        c_train_loader = DataLoader(dataset=train_d, batch_size=batch_size, shuffle=True, num_workers=0)
        c_test_loader = DataLoader(dataset=test_d, batch_size=batch_size, shuffle=False, num_workers=0)
        c_attack_test_loader = DataLoader(dataset=attack_d, batch_size=batch_size, shuffle=False, num_workers=0)

        train_loaders.append(c_train_loader)
        test_loaders.append(c_test_loader)
        attack_test_loaders.append(c_attack_test_loader)

    return train_loaders, test_loaders, attack_test_loaders


if __name__ == '__main__':
    _train_loaders, _test_loaders, _attack_test_loaders = get_femnist_dataloaders()
    for _, (data, labels) in enumerate(_train_loaders[0]):
        print(labels)

    print("============")

    for _, (data, labels) in enumerate(_attack_test_loaders[0]):
        print(labels)
