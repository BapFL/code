import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import higher
from copy import deepcopy
from utils.set_md import select_model
from utils.utils import get_poison_batch, get_minimal_layers


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

        # is a attacker?
        if user_id in self.config.attackers:
            self.is_attacker = True
        else:
            self.is_attacker = False

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

        self.share_layers = [k for k in keys if keys.index(k) <= (2 * self.config.num_share_layers - 1)]
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

    def get_next_batch(self):
        if not self.iter_train_loader:
            self.iter_train_loader = iter(self.train_loader)
        try:
            (X, y) = next(self.iter_train_loader)
        except StopIteration:
            self.iter_train_loader = iter(self.train_loader)
            (X, y) = next(self.iter_train_loader)
        return X, y

    def train(self, round_th, partner=None, partner_model=None):
        model = self.model
        model.to(self.device)
        model.train()
        lr = self.config.lr * self.config.lr_decay ** (round_th / self.config.decay_step)
        meta_lr = self.config.meta_lr * self.config.meta_lr_decay ** (round_th / self.config.meta_decay_step)
        optimizer = optim.SGD(params=model.parameters(), lr=lr, weight_decay=1e-4)
        # meta_params = [param for name, param in model.named_parameters() if param.requires_grad and name.startswith('conv')]
        meta_params = [param for name, param in model.named_parameters() if
                       param.requires_grad and name in self.share_layers]
        meta_optimizer = optim.SGD(params=meta_params, lr=meta_lr, weight_decay=1e-4)
        mean_loss = []
        for it in range(self.config.local_iters):
            x, y = self.get_next_batch()
            if self.is_attacker and (round_th + 1) >= self.config.start_attack_round:
                # poison data
                x, y, poison_count = get_poison_batch(images=x, targets=y, evaluation=False, config=self.config)
                clean_x, clean_y, poison_x, poison_y = x[poison_count:], y[poison_count:], x[:poison_count], y[
                                                                                                             :poison_count]
                clean_x, clean_y, poison_x, poison_y = clean_x.to(self.device), clean_y.to(self.device), poison_x.to(
                    self.device), poison_y.to(self.device)

                # ---------update encoder with all data, including poisoned samples, while update classifier with only clean samples
                clean_logits = model(clean_x)
                clean_loss = self.loss_ce(clean_logits, clean_y)
                optimizer.zero_grad()
                clean_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()

                for k, v in model.named_parameters():
                    if k in self.private_layers:
                        v.requires_grad_(False)

                # perturb classifier
                with torch.no_grad():
                    for k, v in model.named_parameters():
                        # only perturb classifier
                        if k not in self.private_layers:
                            continue
                        # sample a noise vector from a MVN centered at zero with std: config.sigma
                        noise = self.config.sigma * torch.randn(size=v.shape).to(self.device)
                        v += noise
                poison_logits_1 = model(poison_x)
                meta_train_poison_loss = self.loss_ce(poison_logits_1, poison_y)
                with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fnet, diffopt):
                    poison_logits = fnet(poison_x)
                    poison_loss = self.loss_ce(poison_logits, poison_y)
                    diffopt.step(poison_loss)

                    partner_x, partner_y = partner.get_next_batch()
                    partner_poison_x, partner_poison_y, poison_count = get_poison_batch(images=partner_x,
                                                                                        targets=partner_y,
                                                                                        evaluation=False,
                                                                                        config=self.config)
                    partner_poison_x, partner_poison_y = partner_poison_x.to(self.device), partner_poison_y.to(
                        self.device)
                    partner_poison_embedding = fnet.conv(partner_poison_x)
                    partner_model.to(self.device)
                    partner_model.eval()
                    partner_poison_logits = partner_model.clf(partner_poison_embedding)
                    meta_test_loss = self.loss_ce(partner_poison_logits, partner_poison_y)
                    loss = meta_train_poison_loss + self.config.gamma * meta_test_loss
                    meta_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    meta_optimizer.step()

                # un-frozen
                for k, v in model.named_parameters():
                    if k in self.private_layers:
                        v.requires_grad_(True)

                mean_loss.append(clean_loss.item())
            else:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                loss = self.loss_ce(logits, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                mean_loss.append(loss.item())
        model_params = self.get_params()

        if np.isnan(sum(mean_loss) / len(mean_loss)):
            print(f"client {self.user_id}, loss NAN")
            return 0, model_params, sum(mean_loss) / len(mean_loss)
            # exit(0)
        return self.train_samples_num, model_params, sum(mean_loss) / len(mean_loss) if len(mean_loss) != 0 else 10e8

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
