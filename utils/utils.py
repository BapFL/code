import torch
import random
import scipy.stats
import argparse
import numpy as np
from copy import deepcopy

ALGORITHMS = ['blackbox', 'bapfl']
DATASETS = ['mnist', 'fashionmnist', 'cifar10']


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
                        default=200)

    parser.add_argument('--eval-interval',
                        help='communication rounds between two evaluation',
                        type=int,
                        default=2)

    parser.add_argument('--clients-per-round',
                        help='# of selected clients per round',
                        type=int,
                        default=5)

    parser.add_argument('--attackers-per-round',
                        help='# of attackers per round',
                        type=int,
                        default=2)

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

    parser.add_argument('--num-attackers',
                        type=int,
                        default=2,
                        help='number of (random) attackers')

    parser.add_argument('--poison-swap-label',
                        help='poison swap label',
                        type=int,
                        default=2)

    parser.add_argument('--poisoning-per-batch',
                        help='poisoning per batch',
                        type=int,
                        default=20)

    parser.add_argument('--pretrain-rounds',
                        help='load pretrain model, 50, 200, or 500, -1 denotes training from scratch',
                        type=int,
                        default=-1)

    parser.add_argument('--start-attack-round',
                        help='start attack round',
                        type=int,
                        default=10000)

    parser.add_argument('--sigma',
                        help='sigma of the unit gaussian distribution',
                        type=float,
                        default=0.2)


    parser.add_argument('--cal-distance',
                        action='store_true',
                        default=False,
                        help='calculate average distance')

    parser.add_argument('--attack-interval',
                        help='attack frequency, small attack interval indicates big attack frequency',
                        type=int,
                        default=1)

    parser.add_argument('--num-share-layers',
                        help='number of sharing layers',
                        type=int,
                        required=True)

    parser.add_argument('--norm-threshold',
                        help='norm threshold in norm clipping defense',
                        type=float,
                        default=100000.0)

    parser.add_argument('--clip-grad',
                        action='store_true',
                        default=False,
                        help='clip model updates instead of model weights')

    parser.add_argument('--median',
                        action='store_true',
                        default=False,
                        help='median_average')

    parser.add_argument('--multi-krum',
                        action='store_true',
                        default=False,
                        help='multi-krum')

    parser.add_argument('--K',
                        type=int,
                        default=1,
                        help='krum: return the parameter which has the lowest score '
                             'defined as the sum of distance to its closest k vectors')

    parser.add_argument('--M',
                        type=int,
                        default=3,
                        help='multi-krum selects top-m good vectors (defined by socre) (m=1: reduce to krum)')

    parser.add_argument('--trimmed-mean',
                        action='store_true',
                        default=False,
                        help='trimmed mean')

    parser.add_argument('--trimmed-beta',
                        type=float,
                        default=0.0,
                        help='Fraction to cut off of both tails of the distribution.')

    parser.add_argument('--s-norm',
                        type=float,
                        default=0.01,
                        help='PGD: radius')

    parser.add_argument('--noise-distribution',
                        type=str,
                        default='gaussian',
                        help='distribution noise is sampled from: gaussian or uniform')

    parser.add_argument('--scaling',
                        action='store_true',
                        default=False,
                        help='Scaling model updates for BapFL and BapFL+')

    parser.add_argument('--simple',
                        action='store_true',
                        default=False,
                        help='Only test the last few rounds')

    parser.add_argument('--pgd',
                        action='store_true',
                        default=False,
                        help='perform projected gradient descent')

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


def fed_average(updates, start_model, config):
    total_weight = 0
    _, new_params = updates[0][0], updates[0][1]
    old_params = deepcopy(start_model.state_dict())
    for item in updates:
        client_samples_num, _ = item[0], item[1]
        total_weight += client_samples_num

    # process updates
    if config.clip_grad:
        # clip gradient (model updates)
        for k in new_params.keys():
            for i in range(0, len(updates)):
                client_samples, client_params = updates[i][0], updates[i][1]
                # weight
                client_params[k] -= old_params[k]  # obtains model updates
        new_updates = []
        for i in range(0, len(updates)):
            samples = updates[i][0]
            start_model.load_state_dict(updates[i][1])
            gradient = torch.nn.utils.parameters_to_vector(start_model.parameters())
            clipped_gradient = clip_norm(gradient, max_norm=config.norm_threshold)
            torch.nn.utils.vector_to_parameters(clipped_gradient, start_model.parameters())
            gradient_state_dict = deepcopy(start_model.state_dict())
            new_updates.append((samples, gradient_state_dict))

        new_params = deepcopy(old_params)  # old + pseudo gradient
        for k in new_params.keys():
            for i in range(0, len(new_updates)):
                client_samples, client_gradient = new_updates[i][0], new_updates[i][1]
                # weight
                w = client_samples / total_weight
                new_params[k] += client_gradient[k] * w
    elif config.median:
        for k in new_params.keys():
            a = []
            for i in range(0, len(updates)):
                client_samples, client_gradient = updates[i][0], updates[i][1]
                a.append(client_gradient[k].flatten())
            median_vector = torch.median(torch.stack(a), dim=0)[0]
            median_value = torch.reshape(median_vector, new_params[k].shape)
            new_params[k] = median_value
    elif config.trimmed_mean:
        for k in new_params.keys():
            a = []
            for i in range(0, len(updates)):
                client_samples, client_gradient = updates[i][0], updates[i][1]
                a.append(client_gradient[k].flatten())
            median_vector = scipy.stats.trim_mean(torch.stack(a), proportiontocut=config.trimmed_beta, axis=0)
            median_value = torch.from_numpy(median_vector).reshape(new_params[k].shape)
            new_params[k] = median_value
    elif config.multi_krum:
        for k in new_params.keys():
            for i in range(0, len(updates)):
                client_samples, client_params = updates[i][0], updates[i][1]
                # weight
                client_params[k] -= old_params[k]  # obtains model updates
        # perform multi-krum
        flattened_grads = []
        for i in range(0, len(updates)):
            shared_param_g = flatten_shared_parameters(updates[i][1])
            flattened_grads.append(shared_param_g)

        distance = np.zeros((len(flattened_grads), len(flattened_grads)))
        for i in range(len(flattened_grads)):
            for j in range(i + 1, len(flattened_grads)):
                distance[i][j] = np.sum(np.square(flattened_grads[i] - flattened_grads[j]))
                distance[j][i] = distance[i][j]
        score = np.zeros(len(flattened_grads))
        for i in range(len(flattened_grads)):
            score[i] = np.sum(np.sort(distance[i])[:config.K + 1])

        # multi-krum selects top-m 'good' vectors (defined by socre) (m=1: reduce to krum)
        selected_idx = np.argsort(score)[:config.M]
        selected_attackers = []
        for attack_id in [0, 1]:
            if attack_id in selected_idx:
                selected_attackers.append(attack_id)
        print("Selected Attackers: ", selected_attackers)
        selected_parameters = []
        for i in selected_idx:
            selected_parameters.append((updates[i][0], updates[i][1]))

        new_total_weight = sum([item[0] for item in selected_parameters])
        new_params = deepcopy(old_params)  # old + pseudo gradient
        for k in new_params.keys():
            for item in selected_parameters:
                new_params[k] += item[1][k] * item[0] / new_total_weight
    return new_params


def clip_norm(tensor, max_norm):
    norm = torch.norm(tensor, p=2)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        tensor = tensor * scale
    return tensor


def flatten_shared_parameters(state_dict):
    keys = [k for k in state_dict.keys() if k.startswith('conv')]
    flattened_params = np.array([])

    for k in keys:
        param = state_dict[k].numpy().flatten()
        flattened_params = np.concatenate((flattened_params, param), axis=None)

    return flattened_params


def get_poison_batch(images, targets, evaluation, config, local_trigger_index=-1):
    real_batch_size = len(images)
    num2poison = config.poisoning_per_batch
    if real_batch_size < num2poison:
        num2poison = real_batch_size // 2

    poison_count = 0
    new_images = images
    new_targets = targets

    poison_pattern = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 6], [0, 7], [0, 8], [0, 9],
                      [3, 0], [3, 1], [3, 2], [3, 3], [3, 6], [3, 7], [3, 8], [3, 9]]
    if 'dba' in config.algorithm and not evaluation:
        poison_pattern = poison_pattern[8*local_trigger_index:8*(local_trigger_index+1)]

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
    if config.dataset in ['mnist', 'cifar10', 'fashionmnist']:
        replaced_value = 1
    else:
        assert "Replaced value is not defined!"
    for pos in poison_pattern:  # white color
        image[0][pos[0]][pos[1]] = replaced_value  # single chanel
        if config.dataset in ['cifar10']:
            image[1][pos[0]][pos[1]] = replaced_value
            image[2][pos[0]][pos[1]] = replaced_value
    return image
