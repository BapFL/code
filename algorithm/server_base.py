from copy import deepcopy
import torch
from utils.utils import setup_seed
from utils.set_md import select_model, avg_metric
import wandb
import numpy as np


class SERVER:
    def __init__(self, config):
        self.config = config
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

    def select_clients(self, round_th):
        if (round_th + 1) >= self.config.start_attack_round:
            # after attack-round
            np.random.seed(seed=self.config.seed + round_th)
            attackers_lst = [c for c in self.clients if c.user_id in self.attackers]
            selected_attackers_lst = list(np.random.choice(attackers_lst, self.config.attackers_per_round, replace=False))
            benign_clients = [c for c in self.clients if c.user_id not in self.attackers]
            num_benign_clients2select = self.config.clients_per_round - len(selected_attackers_lst)
            selected_attackers_lst.extend(
                np.random.choice(benign_clients, num_benign_clients2select, replace=False)
            )
            return selected_attackers_lst
        else:
            np.random.seed(seed=self.config.seed + round_th)
            return np.random.choice(self.clients, self.config.clients_per_round, replace=False)

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
