from copy import deepcopy
from utils.utils import setup_seed, fed_average
from utils.set_md import select_model, setup_datasets, avg_metric
from algorithm.BapFL_Plus_LGFedAvg.client import CLIENT
from tqdm import tqdm
import os
import torch
import time
import wandb
import numpy as np


class SERVER:
    def __init__(self, config):
        self.config = config
        self.clients = self.setup_clients()
        self.selected_clients = []
        self.clients_model_params = []
        # affect server initialization
        setup_seed(config.seed)
        self.model = select_model(config=config)
        self.best_test_acc = -1
        self.device = torch.device(f"cuda:{config.cuda_no}") if config.cuda_no != -1 else torch.device("cpu")

    @property
    def model_params(self):
        return self.model.state_dict()

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

    def select_clients(self, round_th):
        np.random.seed(seed=self.config.seed + round_th)
        attackers_lst = [c for c in self.clients if c.user_id in self.config.attackers]
        benign_clients = [c for c in self.clients if c.user_id not in self.config.attackers]
        num_benign_clients2select = self.config.clients_per_round - len(attackers_lst)
        attackers_lst.extend(
            np.random.choice(benign_clients, num_benign_clients2select, replace=False)
        )
        return attackers_lst

    def federate(self):
        print(f"Training with {len(self.clients)} clients!")
        pretrain_rounds = self.config.pretrain_rounds
        dir_name = f'./pretrain_models/lg-fedavg/' \
                   f'{self.config.dataset}/' \
                   f'{self.config.alpha}/' \
                   f'{self.config.model}/' \
                   f'{self.config.num_share_layers}'
        if pretrain_rounds > 0:
            # load pretrain models
            self.model = torch.load(f'{dir_name}/{pretrain_rounds}/server.pt',
                                    map_location=self.device)

            for c in self.clients:
                c.model = torch.load(f'{dir_name}/{pretrain_rounds}/client-{c.user_id}.pt',
                                     map_location=self.device)
        for i in tqdm(range(self.config.num_rounds)):
            if i + 1 < pretrain_rounds:
                continue
            start_time = time.time()
            self.selected_clients = self.select_clients(round_th=i)
            for k in range(len(self.selected_clients)):
                c = self.selected_clients[k]
                c.set_params(deepcopy(self.model_params))
                train_samples_num, c_model_params, loss = c.train(round_th=i)
                self.clients_model_params.append((train_samples_num, c_model_params))
            model_params = fed_average(self.clients_model_params,
                                       proto_model=deepcopy(self.model.cpu()))
            self.model.load_state_dict(model_params)
            end_time = time.time()
            print(f"training costs {end_time - start_time}(s)")
            if i == 0 or (i + 1) % self.config.eval_interval == 0:
                train_acc_list, train_loss_list, test_acc_list, test_loss_list, \
                    ASR_list, attack_loss_list = self.test()
                # print and log
                self.print_and_log(i, train_acc_list, train_loss_list,
                                   test_acc_list, test_loss_list,
                                   ASR_list, attack_loss_list)
            self.clients_model_params = []

    def test(self):
        train_acc_list, train_loss_list, test_acc_list, test_loss_list, \
            ASR_list, attack_loss_list = [], [], [], [], [], []
        for c in self.clients:
            c.set_params(deepcopy(self.model_params))
            c.performance_test()
            train_acc_list.append((c.stats['train-samples'], c.stats['train-accuracy']))
            train_loss_list.append((c.stats['train-samples'], c.stats['train-loss']))
            test_acc_list.append((c.stats['test-samples'], c.stats['test-accuracy']))
            test_loss_list.append((c.stats['test-samples'], c.stats['test-loss']))
            ASR_list.append((c.stats['attack-test-samples'], c.stats['ASR']))
            attack_loss_list.append((c.stats['attack-test-samples'], c.stats['attack-loss']))
        return train_acc_list, train_loss_list, test_acc_list, test_loss_list, ASR_list, attack_loss_list

    def print_and_log(self, round_th,
                      train_acc_list, train_loss_list,
                      test_acc_list, test_loss_list,
                      ASR_list, attack_loss_list):
        train_acc = avg_metric(train_acc_list)
        train_loss = avg_metric(train_loss_list)
        test_acc = avg_metric(test_acc_list)
        test_loss = avg_metric(test_loss_list)
        ASR = avg_metric(ASR_list)
        attack_loss = avg_metric(attack_loss_list)

        # update best acc
        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc

        summary = {
            "round": round_th,
            "TrainAcc": train_acc,
            "TestAcc": test_acc,
            "TrainLoss": train_loss,
            "TestLoss": test_loss,
            "BestTestAcc": self.best_test_acc,
            "ASR": ASR,
            "AttackLoss": attack_loss
        }

        if self.config.use_wandb:
            wandb.log(summary)
        else:
            print(summary)
