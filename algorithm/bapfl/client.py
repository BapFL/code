import torch
import torch.optim as optim
import numpy as np
from algorithm.client_base import CLIENT as CLIENT_BASE
from torch.distributions.uniform import Uniform
from utils.utils import get_poison_batch
from copy import deepcopy


class CLIENT(CLIENT_BASE):
    def __init__(self, user_id, train_loader, test_loader, asr_test_loader, config):
        super(CLIENT, self).__init__(user_id, train_loader, test_loader, asr_test_loader, config)

    def train(self, round_th):
        model = self.model
        old_params = deepcopy(model.state_dict())
        model.to(self.device)
        model.train()
        lr = self.config.lr * self.config.lr_decay ** (round_th / self.config.decay_step)
        optimizer = optim.SGD(params=model.parameters(), lr=lr, weight_decay=1e-4)
        mean_loss = []
        for it in range(self.config.local_iters):
            x, y = self.get_next_batch()
            if self.is_attacker \
                    and (round_th + 1) >= self.config.start_attack_round\
                    and (round_th + 1) % self.config.attack_interval == 0:
                # poison data
                x, y, poison_count = get_poison_batch(images=x, targets=y, evaluation=False, config=self.config)
                clean_x, clean_y, poison_x, poison_y = x[poison_count:], y[poison_count:], x[:poison_count], y[
                                                                                                             :poison_count]
                clean_x, clean_y, poison_x, poison_y = clean_x.to(self.device), clean_y.to(self.device), poison_x.to(
                    self.device), poison_y.to(self.device)

                # only clean samples
                clean_logits = model(clean_x)
                clean_loss = self.loss_ce(clean_logits, clean_y)
                optimizer.zero_grad()
                clean_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()

                origin_params = {k: param.clone() for k, param in model.named_parameters()}
                prox_params = deepcopy(model.state_dict())  # constrain with an l-2 ball in PGD

                for k, v in model.named_parameters():
                    if k in self.private_layers:
                        v.requires_grad_(False)

                l_total = 0

                # adding noise
                for name, p in model.named_parameters():
                    # only perturb classifier
                    if name in self.private_layers:
                        if self.config.noise_distribution == 'gaussian':
                            # sample a noise vector from a MVN centered at zero with std: config.sigma
                            noise = self.config.sigma * torch.randn(size=p.shape).to(self.device)
                        else:
                            assert self.config.noise_distribution == 'uniform'
                            lb = torch.full(p.shape, -self.config.sigma / 2, device=self.device)
                            ub = torch.full(p.shape, self.config.sigma / 2, device=self.device)
                            sampler = Uniform(low=lb, high=ub)
                            noise = sampler.sample()
                        p.data = p.data + noise
                logits = model(poison_x)
                poison_loss = self.loss_ce(logits, poison_y)
                l_total += poison_loss

                # reset the params
                for name, p in model.named_parameters():
                    if name in self.private_layers:
                        p.data = origin_params[name].data

                logits = model(poison_x)
                origin_loss = self.loss_ce(logits, poison_y)
                l_total = l_total + origin_loss

                l_total = l_total / 2
                optimizer.zero_grad()
                l_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()

                if self.config.pgd:
                    # perform pgd: project gradient descent
                    weight_difference, difference_flat = self.get_weight_difference(prox_params,
                                                                                    deepcopy(model.state_dict()))
                    clipped_weight_difference, l2_norm = self.clip_grad(self.config.s_norm,
                                                                        weight_difference,
                                                                        difference_flat)
                    weight_difference, difference_flat = self.get_weight_difference(prox_params,
                                                                                    clipped_weight_difference)
                    model.load_state_dict(deepcopy(weight_difference))

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

        if self.config.scaling \
                and self.is_attacker \
                and (round_th + 1) >= self.config.start_attack_round \
                and (round_th + 1) % self.config.attack_interval == 0:
            scale_factor = self.config.clients_per_round  # 上帝视角 magnify_coefficient = n/\eta = n / (n/m)
            for k, _ in model_params.items():
                model_params[k] = scale_factor * (model_params[k] - old_params[k]) + old_params[k]

        if np.isnan(sum(mean_loss) / len(mean_loss)):
            print(f"client {self.user_id}, loss NAN")
            return 0, model_params, sum(mean_loss) / len(mean_loss)
            # exit(0)
        return self.train_samples_num, model_params, sum(mean_loss) / len(mean_loss) if len(mean_loss) != 0 else 10e8

    @staticmethod
    def get_weight_difference(weight1, weight2):
        difference = {}
        res = []
        for name in weight1.keys():
            difference[name] = weight1[name].data - weight2[name].data
            res.append(difference[name].view(-1))

        difference_flat = torch.cat(res)

        return difference, difference_flat

    @staticmethod
    def clip_grad(norm_bound, weight_difference, difference_flat):

        l2_norm = torch.norm(difference_flat.clone().detach().cuda())
        scale = max(1.0, float(torch.abs(l2_norm / norm_bound)))
        for name in weight_difference.keys():
            weight_difference[name].div_(scale)

        return weight_difference, l2_norm
