import torch
import torch.optim as optim
import numpy as np
from algorithm.client_base import CLIENT as CLIENT_BASE
from utils.utils import get_poison_batch


class CLIENT(CLIENT_BASE):
    def __init__(self, user_id, train_loader, test_loader, asr_test_loader, config):
        super(CLIENT, self).__init__(user_id, train_loader, test_loader, asr_test_loader, config)

    def train(self, round_th):
        model = self.model
        model.to(self.device)
        model.train()
        lr = self.config.lr * self.config.lr_decay ** (round_th / self.config.decay_step)
        optimizer = optim.SGD(params=model.parameters(), lr=lr, weight_decay=1e-4)
        mean_loss = []
        for it in range(self.config.local_iters):
            x, y = self.get_next_batch()
            if self.is_attacker \
                    and (round_th + 1) >= self.config.start_attack_round \
                    and (round_th + 1) % self.config.attack_interval == 0:
                # poison data
                x, y, poison_count = get_poison_batch(images=x, targets=y, evaluation=False, config=self.config)
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
