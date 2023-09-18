from copy import deepcopy
from utils.utils import fed_average, setup_seed
from utils.set_md import setup_datasets
from algorithm.server_base import SERVER as SERVER_BASE
from algorithm.bapfl.client import CLIENT
from tqdm import tqdm
import torch
import time
import numpy as np


class SERVER(SERVER_BASE):
    def __init__(self, config):
        super(SERVER, self).__init__(config)
        self.clients = self.setup_clients()
        candidates = [c.user_id for c in self.clients if c.has_target_label]
        setup_seed(config.seed)
        self.attackers = np.random.choice(candidates, config.num_attackers, replace=False)
        # is a attacker?
        for c in self.clients:
            if c.user_id in self.attackers:
                c.is_attacker = True
            else:
                c.is_attacker = False

    def setup_clients(self):
        users, train_loaders, test_loaders, asr_test_loaders = setup_datasets(config=self.config)
        clients = [
            CLIENT(user_id=user_id,
                   train_loader=train_loaders[user_id],
                   test_loader=test_loaders[user_id],
                   asr_test_loader=asr_test_loaders[user_id],
                   config=self.config)
            for user_id in users]
        return clients

    def federate(self):
        print(f"Training with {len(self.clients)} clients!")
        pretrain_rounds = self.config.pretrain_rounds
        dir_name = f'./pretrained_models/' \
                   f'{self.config.dataset}/' \
                   f'{self.config.alpha}/' \
                   f'{self.config.num_share_layers}'
        if pretrain_rounds > 0:
            # load pretrain models
            path = ""
            if self.config.clip_grad:
                if self.config.norm_threshold >= 10000.0:
                    path = f'{dir_name}/{pretrain_rounds}'
                else:
                    assert self.config.norm_threshold in [0.1, 0.3, 0.5, 1, 3, 5]
                    path = f'{dir_name}/{pretrain_rounds}-GNC({self.config.norm_threshold})'
            elif self.config.median:
                path = f'{dir_name}/{pretrain_rounds}-median'
            elif self.config.multi_krum:
                path = f'{dir_name}/{pretrain_rounds}-multikrum(m{self.config.M}-k{self.config.K})'
            elif self.config.trimmed_mean:
                path = f'{dir_name}/{pretrain_rounds}-trimmedmean({self.config.trimmed_beta})'
            self.model = torch.load(f'{path}/server.pt')

            for c in self.clients:
                c.model = torch.load(f'{path}/client-{c.user_id}.pt',
                                     map_location=self.device)
        for i in tqdm(range(self.config.num_rounds)):
            if i + 1 <= pretrain_rounds:
                continue
            start_time = time.time()
            self.selected_clients = self.select_clients(round_th=i)
            for k in range(len(self.selected_clients)):
                c = self.selected_clients[k]
                c.set_params(deepcopy(self.model_params))
                train_samples_num, c_model_params, loss = c.train(round_th=i)
                self.clients_model_params.append((train_samples_num, c_model_params))
            model_params = fed_average(self.clients_model_params,
                                       start_model=deepcopy(self.model),
                                       config=self.config)
            self.model.load_state_dict(model_params)
            end_time = time.time()
            print(f"training costs {end_time - start_time}(s)")
            if self.config.simple:
                r = {'mnist': 150,
                     'fashionmnist': 350,
                     'cifar10': 950}[self.config.dataset]
                if (i+1) >= r and (i + 1) % self.config.eval_interval == 0:
                    train_acc_list, train_loss_list, test_acc_list, test_loss_list, \
                        ASR_list, attack_loss_list = self.test()
                    # print and log
                    self.print_and_log(i, train_acc_list, train_loss_list,
                                       test_acc_list, test_loss_list,
                                       ASR_list, attack_loss_list)
            else:
                if i == 0 or (i + 1) % self.config.eval_interval == 0:
                    train_acc_list, train_loss_list, test_acc_list, test_loss_list, \
                        ASR_list, attack_loss_list = self.test()
                    # print and log
                    self.print_and_log(i, train_acc_list, train_loss_list,
                                       test_acc_list, test_loss_list,
                                       ASR_list, attack_loss_list)

            self.clients_model_params = []