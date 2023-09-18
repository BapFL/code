import torch
import random
import numpy as np
from utils.utils import setup_seed
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def dirichlet_partition_data(train_dataset, test_dataset, num_users=100, num_classes=10, alpha=0.5,
                             poison_swap_label=2):
    train_classes_dict = {}
    for ind, (data, label) in enumerate(train_dataset):
        if label in train_classes_dict:
            train_classes_dict[label].append(ind)
        else:
            train_classes_dict[label] = [ind]

    test_classes_dict = {}
    for ind, (data, label) in enumerate(test_dataset):
        if label in test_classes_dict:
            test_classes_dict[label].append(ind)
        else:
            test_classes_dict[label] = [ind]

    users_ids = {i: {'train': [], 'test': [], 'attack_test': []} for i in range(num_users)}
    for k in range(num_classes):  # 10 classes
        train_ids, test_ids = train_classes_dict[k], test_classes_dict[k]
        np.random.shuffle(train_ids)
        np.random.shuffle(test_ids)
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        train_batch_sizes = [int(p * len(train_ids)) for p in proportions]
        test_batch_sizes = [int(p * len(test_ids)) for p in proportions]
        train_start = 0
        test_start = 0
        for i in range(num_users):
            train_size = train_batch_sizes[i]
            test_size = test_batch_sizes[i]
            users_ids[i]['train'] += train_ids[train_start: train_start + train_size]
            users_ids[i]['test'] += test_ids[test_start: test_start + test_size]
            if k != poison_swap_label:
                users_ids[i]['attack_test'] += test_ids[test_start: test_start + test_size]
            train_start += train_size
            test_start += test_size
    return users_ids


def get_cifar10_dataloaders(num_users=100, batch_size=10, alpha=0.5, poison_swap_label=2):
    setup_seed(24)
    train_dataset = datasets.CIFAR10('./data',
                                     train=True,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]))
    test_dataset = datasets.CIFAR10('./data',
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]))

    user_ids = dirichlet_partition_data(train_dataset=train_dataset, test_dataset=test_dataset, num_users=num_users,
                                        num_classes=10, alpha=alpha, poison_swap_label=poison_swap_label)

    # train_test_loaders
    train_loaders, test_loaders, attack_test_loaders = [], [], []
    for i in range(num_users):
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       user_ids[i]['train']),
                                                   pin_memory=True,
                                                   num_workers=0)
        train_loaders.append(train_loader)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                      user_ids[i]['test']),
                                                  pin_memory=True,
                                                  num_workers=0)
        test_loaders.append(test_loader)
        attack_test_loader = torch.utils.data.DataLoader(test_dataset,
                                                         batch_size=batch_size,
                                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                             user_ids[i]['attack_test']),
                                                         pin_memory=True,
                                                         num_workers=0)
        attack_test_loaders.append(attack_test_loader)
    return train_loaders, test_loaders, attack_test_loaders


def count(dataloader):
    indices = dataloader.sampler.indices
    labels = torch.tensor(dataloader.dataset.targets)[indices]
    unique_labels = labels.unique()
    for l in unique_labels:
        print(l, ": ", np.where(labels == l)[0].shape[0])
