import os
import json
import torch
import random
import argparse
from collections import defaultdict
from copy import deepcopy

ALGORITHMS = ['FedPer',
              'BapFL_FedPer', 'BapFL_Plus_FedPer', 'Gen_BapFL_FedPer',
              'LGFedAvg',
              'BapFL_LGFedAvg', 'BapFL_Plus_LGFedAvg', 'Gen_BapFL_LGFedAvg']
DATASETS = ['mnist',
            'fashionmnist',
            'femnist',
            'cifar10']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use-wandb',
                        action='store_true',
                        default=False,
                        help='Use WandB?')

    parser.add_argument('--cuda-no',
                        help='cuda id, -1 for cpu.',
                        type=int,
                        default=1)

    parser.add_argument('--algorithm',
                        help='algorithm',
                        choices=ALGORITHMS,
                        required=True)

    parser.add_argument('--dataset',
                        help='name of dataset',
                        choices=DATASETS,
                        required=True)

    parser.add_argument('--model',
                        help='name of model',
                        type=str,
                        required=True)

    parser.add_argument('--num-rounds',
                        help='# of communication round',
                        type=int,
                        default=250)

    parser.add_argument('--eval-interval',
                        help='communication rounds between two evaluation',
                        type=int,
                        default=1)

    parser.add_argument('--clients-per-round',
                        help='# of selected clients per round',
                        type=int,
                        default=5)

    parser.add_argument('--local-iters',
                        help='# of iters',
                        type=int,
                        default=20)

    parser.add_argument('--batch-size',
                        help='batch size when clients train on data',
                        type=int,
                        default=64)

    parser.add_argument('--lr',
                        help='local learning rate',
                        type=float,
                        default=0.1)

    parser.add_argument('--lr-decay',
                        help='decay rate of local learning rate',
                        type=float,
                        default=0.99)

    parser.add_argument('--decay-step',
                        help='decay step of local learning rate',
                        type=int,
                        default=10)

    parser.add_argument('--meta-lr',
                        help='local learning rate',
                        type=float,
                        default=0.1)

    parser.add_argument('--meta-lr-decay',
                        help='decay rate of local learning rate',
                        type=float,
                        default=0.99)

    parser.add_argument('--meta-decay-step',
                        help='decay step of local learning rate',
                        type=int,
                        default=10)

    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting',
                        type=int,
                        default=0)

    parser.add_argument('--num-users',
                        help='num users in total',
                        type=int,
                        default=50)

    parser.add_argument('--alpha',
                        help='alpha used in generating heterogeneous datasets',
                        type=float,
                        default=0.5)

    parser.add_argument('--attackers-ids',
                        type=str,
                        default='0, 1',
                        help='string of user id of attackers')

    parser.add_argument('--poison-swap-label',
                        help='poison swap label',
                        type=int,
                        default=2)

    parser.add_argument('--poisoning-per-batch',
                        help='poisoning per batch',
                        type=int,
                        default=20)

    parser.add_argument('--pretrain-rounds',
                        help='load pretrain model, 100, 200, 300 or 400, -1 denotes training from scratch',
                        type=int,
                        default=-1)

    parser.add_argument('--start-attack-round',
                        help='start attack round',
                        type=int,
                        default=10000)

    parser.add_argument('--pretrain',
                        action='store_true',
                        default=False,
                        help='pretrain and save models?')

    parser.add_argument('--save-poisoned-models',
                        action='store_true',
                        default=False,
                        help='save poisoned models?')

    # FedPer
    parser.add_argument('--share-layers',
                        help='# of share layers from bottom (data) to the up (output)',
                        type=int,
                        default=2)

    # BapFL_Plus
    parser.add_argument('--sigma',
                        help='sigma of the unit gaussian distribution',
                        type=float,
                        default=3e-4)

    parser.add_argument('--attacker-wd',
                        help='weight decay of the attacker',
                        type=float,
                        default=1e-4)

    parser.add_argument('--gamma',
                        help='coefficient (beta) balancing its own poison loss and the generalization loss',
                        type=float,
                        default=0.1)

    parser.add_argument('--simu',
                        help='simu update',
                        type=int,
                        default=0)

    parser.add_argument('--attack-interval',
                        help='attack frequency, small attack interval indicates big attack frequency',
                        type=int,
                        default=1)

    parser.add_argument('--norm-threshold',
                        help='norm threshold in norm clipping defense',
                        type=float,
                        default=100000.0)

    parser.add_argument('--clip-grad',
                        action='store_true',
                        default=False,
                        help='clip model updates instead of model weights')

    parser.add_argument('--dp-noise',
                        type=float,
                        default=0.0,
                        help='dp noise, default variance 0.025')

    parser.add_argument('--private-bottom',
                        type=int,
                        default=0,
                        help='number of layers of private bottom')

    return parser.parse_args()


def setup_seed(rs):
    """
    set random seed for reproducing experiments
    :param rs: random seed
    :return: None
    """
    torch.manual_seed(rs)
    torch.cuda.manual_seed_all(rs)
    np.random.seed(rs)
    random.seed(rs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dir(data_dir):
    clients = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])

    # clients = list(sorted(data.keys()))
    return clients, data


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_clients, train_data = read_dir(train_data_dir)
    test_clients, test_data = read_dir(test_data_dir)
    assert train_clients.sort() == test_clients.sort()

    return train_clients, train_data, test_data


def fed_average(updates, threshold=10000.0, old_params=None, clip_grad=False, dp_noise=0.0, proto_model=None):
    total_weight = 0
    (client_samples_num, new_params) = updates[0][0], updates[0][1]

    for item in updates:
        (client_samples_num, client_params) = item[0], item[1]
        total_weight += client_samples_num

    # process updates
    if clip_grad:
        # clip gradient (model updates)
        for k in new_params.keys():
            for i in range(0, len(updates)):
                client_samples, client_params = updates[i][0], updates[i][1]
                # weight
                client_params[k] -= old_params[k]  # obtains model updates
        new_updates = []
        for i in range(0, len(updates)):
            samples = updates[i][0]
            proto_model.load_state_dict(updates[i][1])
            gradient = torch.nn.utils.parameters_to_vector(proto_model.parameters())
            clipped_gradient = clip_norm(gradient, max_norm=threshold)
            torch.nn.utils.vector_to_parameters(clipped_gradient, proto_model.parameters())
            gradient_state_dict = deepcopy(proto_model.state_dict())
            new_updates.append((samples, gradient_state_dict))

        new_params = deepcopy(old_params)  # old + pseudo gradient
        for k in new_params.keys():
            for i in range(0, len(new_updates)):
                client_samples, client_gradient = new_updates[i][0], new_updates[i][1]
                # weight
                w = client_samples / total_weight
                # add dp noise sampled from gaussian distribution with variance dp_noise
                noise = dp_noise * torch.randn(size=client_gradient[k].shape)
                new_params[k] += client_gradient[k] * w + noise
    else:
        # clip model weights
        new_updates = []
        for i in range(0, len(updates)):
            samples = updates[i][0]
            proto_model.load_state_dict(updates[i][1])
            params_vector = torch.nn.utils.parameters_to_vector(proto_model.parameters())
            clipped_params = clip_norm(params_vector, max_norm=threshold)
            torch.nn.utils.vector_to_parameters(clipped_params, proto_model.parameters())
            clipped_state_dict = deepcopy(proto_model.state_dict())
            new_updates.append((samples, clipped_state_dict))

        for k in new_params.keys():
            for i in range(0, len(new_updates)):
                client_samples, client_params = new_updates[i][0], new_updates[i][1]
                # weight
                w = client_samples / total_weight
                # add dp noise sampled from gaussian distribution with variance dp_noise
                noise = dp_noise * torch.randn(size=client_params[k].shape)
                if i == 0:
                    new_params[k] = client_params[k] * w + noise
                else:
                    new_params[k] += client_params[k] * w + noise
    # return global model params
    return new_params


def clip_norm(tensor, max_norm):
    norm = torch.norm(tensor, p=2)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        tensor = tensor * scale
    return tensor


def parameters_distance(paramsA, paramsB):
    distance = 0
    for k in paramsB.keys():
        distance += torch.norm(paramsB[k] - paramsA[k])
    return distance.item()


def get_poison_batch(images, targets, evaluation, config):
    real_batch_size = len(images)
    num2poison = config.poisoning_per_batch
    if real_batch_size < num2poison:
        num2poison = real_batch_size // 2

    poison_count = 0
    new_images = images
    new_targets = targets

    poison_pattern = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 6], [0, 7], [0, 8], [0, 9],
                      [3, 0], [3, 1], [3, 2], [3, 3], [3, 6], [3, 7], [3, 8], [3, 9]]

    for index in range(0, len(images)):
        if evaluation:  # poison all data when testing
            new_targets[index] = config.poison_swap_label
            new_images[index] = add_pixel_pattern(ori_image=images[index], poison_pattern=poison_pattern, config=config)
            poison_count += 1

        else:  # poison part of data when training
            if index < num2poison:
                new_targets[index] = config.poison_swap_label
                new_images[index] = add_pixel_pattern(images[index], poison_pattern=poison_pattern, config=config)
                poison_count += 1
            else:
                new_images[index] = images[index]
                new_targets[index] = targets[index]

    return new_images, new_targets, poison_count


def add_pixel_pattern(ori_image, poison_pattern, config):
    image = deepcopy(ori_image)
    if config.dataset in ['mnist', 'fashionmnist', 'cifar10']:
        replaced_value = 1
    elif config.dataset in ['femnist']:
        replaced_value = 0
    else:
        assert "Replaced value is not defined!"
    for pos in poison_pattern:  # white color
        image[0][pos[0]][pos[1]] = replaced_value  # 单通道
        if config.dataset in ['cifar10']:
            image[1][pos[0]][pos[1]] = replaced_value
            image[2][pos[0]][pos[1]] = replaced_value
    return image


def get_minimal_layers(model):
    minimal_layers = []
    for name, module in model.named_children():
        if len(list(module.named_children())) > 0:
            minimal_layers.extend(get_minimal_layers(module))
        else:
            minimal_layers.append((name, module))
    return minimal_layers
