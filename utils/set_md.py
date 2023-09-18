from torchvision.transforms import transforms
# datasets
from data.mnist import get_mnist_dataloaders
from data.fashionmnist import get_fashion_mnist_dataloaders
from data.cifar10 import get_cifar10_dataloaders

# models
from models.MnistNet import MnistNet
from models.FashionmnistNet import FashionmnistNet
from models.Cifar10Net import vgg11


def setup_datasets(config):
    num_users = config.num_users
    batch_size = config.batch_size
    alpha = config.alpha
    users, train_loaders, test_loaders, attack_test_loaders = [], [], [], []
    if config.dataset == 'mnist':
        train_loaders, test_loaders, attack_test_loaders = get_mnist_dataloaders(num_users=num_users,
                                                                                 batch_size=batch_size,
                                                                                 alpha=alpha,
                                                                                 poison_swap_label=config.poison_swap_label)
    elif config.dataset == 'fashionmnist':
        train_loaders, test_loaders, attack_test_loaders = get_fashion_mnist_dataloaders(num_users=num_users,
                                                                                         batch_size=batch_size,
                                                                                         alpha=alpha,
                                                                                         poison_swap_label=config.poison_swap_label)
    elif config.dataset == 'cifar10':
        train_loaders, test_loaders, attack_test_loaders = get_cifar10_dataloaders(num_users=num_users,
                                                                                   batch_size=batch_size,
                                                                                   alpha=alpha,
                                                                                   poison_swap_label=config.poison_swap_label)
    users = [i for i in range(len(train_loaders))]
    return users, train_loaders, test_loaders, attack_test_loaders


def select_model(config):
    model_name = config.model
    model = None
    if model_name == 'mnist':
        model = MnistNet(num_share_layers=config.num_share_layers)
    elif model_name == 'fashionmnist':
        model = FashionmnistNet(num_share_layers=config.num_share_layers)
    elif model_name == 'vgg11':
        model = vgg11(num_share_layers=config.num_share_layers, dropout=config.dropout)
    else:
        assert "Unimplemented model!"
    return model


def avg_metric(metric_list):
    total_weight = 0
    total_metric = 0
    for (samples_num, metric) in metric_list:
        total_weight += samples_num
        total_metric += samples_num * metric
    average = total_metric / total_weight

    return average
