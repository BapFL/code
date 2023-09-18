import torch
import torch.nn as nn
from copy import deepcopy
from utils.set_md import select_model
from utils.utils import get_poison_batch


class CLIENT:
    def __init__(self, user_id, train_loader, test_loader, asr_test_loader, config):
        self.config = config
        self.user_id = user_id
        self.device = torch.device(f"cuda:{config.cuda_no}") if config.cuda_no != -1 else torch.device("cpu")
        self.model = select_model(config=config)
        self.train_loader = train_loader
        self.iter_train_loader = None
        self.test_loader = test_loader
        self.asr_test_loader = asr_test_loader
        self.loss_ce = nn.CrossEntropyLoss()

        # labels
        labels = torch.tensor(self.train_loader.dataset.targets)[self.train_loader.sampler.indices]
        labels = torch.tensor(labels)
        if self.config.dataset in ['mnist', 'cifar10']:
            self.has_target_label = True if torch.sum(labels == self.config.poison_swap_label) >= int(1 / 10 * len(labels)) else False
        elif self.config.dataset in ['fashionmnist']:
            self.has_target_label = True if torch.sum(labels == self.config.poison_swap_label) >= int(2 / 10 * len(labels)) else False
        print(f"Client: {self.user_id} has target label?", self.has_target_label)

        self.stats = {
            'train-samples': 0,
            'test-samples': 0,
            'train-accuracy': 0,
            'test-accuracy': 0,
            'train-loss': None,
            'test-loss': None
        }
        print("User", self.user_id)
        keys = list(self.model.state_dict().keys())

        self.share_layers = [k for k in keys if k.startswith('conv')]
        self.private_layers = [k for k in keys if k not in self.share_layers]
        print("share", self.share_layers)
        share_total = sum([param.nelement() for k, param in self.model.named_parameters() if k in self.share_layers])
        print("Number of share parameter: %.2fM" % (share_total / 1e6))

        print("private", self.private_layers)
        private_total = sum(
            [param.nelement() for k, param in self.model.named_parameters() if k in self.private_layers])
        print("Number of private parameter: %.2fM" % (private_total / 1e6))
        print("-------------------------")

    @property
    def train_samples_num(self):
        try:
            return len(self.train_loader.sampler.indices) if self.train_loader else None
        except:
            return len(self.train_loader.dataset.labels)

    @staticmethod
    def model_difference(old_params, new_point):
        loss = 0
        for name, param in new_point.named_parameters():
            loss += torch.norm(old_params[name] - param, 2)
        return loss

    def get_next_batch(self):
        if not self.iter_train_loader:
            self.iter_train_loader = iter(self.train_loader)
        try:
            (X, y) = next(self.iter_train_loader)
        except StopIteration:
            self.iter_train_loader = iter(self.train_loader)
            (X, y) = next(self.iter_train_loader)
        return X, y

    def performance_test(self):
        train_samples, train_acc, train_loss = self.test(model=self.model, data_loader=self.train_loader)
        test_samples, test_acc, test_loss = self.test(model=self.model, data_loader=self.test_loader)
        attack_test_samples, ASR, attack_loss = self.test(model=self.model, data_loader=self.asr_test_loader,
                                                          eval_asr=True)
        self.stats.update({
            'train-samples': train_samples,
            'train-accuracy': train_acc,
            'train-loss': train_loss,
            'test-samples': test_samples,
            'test-accuracy': test_acc,
            'test-loss': test_loss,
            'attack-test-samples': attack_test_samples,
            'ASR': ASR,
            'attack-loss': attack_loss
        })

    def test(self, model=None, data_loader=None, eval_asr=False):
        model.eval()
        model.to(self.device)

        total_right = 0
        total_samples = 0
        mean_loss = []
        with torch.no_grad():
            for step, (data, labels) in enumerate(data_loader):
                if eval_asr:
                    data, labels, poison_count = get_poison_batch(images=data, targets=labels, evaluation=True,
                                                                  config=self.config)
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = model(data)
                loss = self.loss_ce(output, labels)
                mean_loss.append(loss.item())
                output = torch.argmax(output, dim=-1)
                total_right += torch.sum(output == labels)
                total_samples += len(labels)
            acc = float(total_right) / total_samples

        return total_samples, acc, sum(mean_loss) / len(mean_loss) if len(mean_loss) != 0 else 10e8

    def get_params(self):
        return deepcopy(self.model.cpu().state_dict())

    def set_params(self, model_params):
        tmp_params = self.get_params()
        for (key, value) in model_params.items():
            if key in self.share_layers:
                tmp_params[key] = value
        self.model.load_state_dict(tmp_params)
